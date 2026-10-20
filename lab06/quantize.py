from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from pathlib import Path
import json

from tensors import (
    SCALE_BYTES,
    ZERO_BYTES,
    Sample,
    Tensor,
    TensorError,
    load_model,
    dtype_bytes,
    elements,
    total_parameters,
    materialise_samples,
)

from quantize_helper import (
    _channel_slice,
    _qmin_qmax,
    _range_of,
    _round_half_up,
    _scale_zero,
    CALIBRATIONS,
    CLIP_METHODS,
    DEFAULT_BITS,
    DEFAULT_PERCENTILE,
    GRANULARITIES,
    MSE_CANDIDATES,
    PRECISIONS,
)


# helper class and functions to carry the quantized tensor and its metadata around together
@dataclass(frozen=True)
class Quantized:
    name: str
    shape: tuple[int, ...]
    codes: tuple[int, ...]
    scales: tuple[float, ...]
    zeros: tuple[int, ...]
    bits: int
    granularity: str
    symmetric: bool
    clipped: int = 0

    @property
    def parameters(self) -> int:
        return elements(self.shape)

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def channel_stride(self) -> int:
        return elements(self.shape[1:]) if len(self.shape) > 1 else 1

    @property
    def scale_count(self) -> int:
        """How many (scale, zero) pairs this tensor carries on disk."""
        return len(self.scales)

    @property
    def clipped_fraction(self) -> float:
        return round(self.clipped / len(self.codes), 8) if self.codes else 0.0

def to_fp16(t: Tensor) -> Tensor:
    out = []
    for v in t.data:
        if not -65504.0 <= v <= 65504.0:
            raise TensorError(
                f"{t.name}: {v} is outside binary16's range; FP16 would store "
                "infinity and every number after this one would be a lie"
            )
        out.append(round(struct.unpack("e", struct.pack("e", float(v)))[0], 8))
    return Tensor(name=t.name, shape=t.shape, data=tuple(out), dtype="fp16")


def as_tensor(name: str, values: Sequence[float]) -> Tensor:
    vals = tuple(float(v) for v in values)
    if not vals:
        raise TensorError(f"{name}: no values")
    return Tensor(name=name, shape=(len(vals),), data=vals, dtype="fp32")


def select_calibration(activations: Sequence[Sample],
                       policy: str = "all") -> tuple[float, ...]:
    acts = tuple(activations)
    if policy == "all":
        chosen = acts
    elif policy == "two_class":
        chosen = tuple(s for s in acts if s.cls in (0, 1))
    elif policy == "first_8":
        chosen = acts[:8]
    elif policy == "one_class":
        chosen = tuple(s for s in acts if s.cls == 0)
    else:
        raise TensorError(
            f"unknown calibration policy {policy!r}; known: {list(CALIBRATIONS)}"
        )
    if not chosen:
        raise TensorError(f"calibration policy {policy!r} selected no samples")
    return tuple(v for s in chosen for v in s.data)


########################################
# start filling in the code from here.
#########################################

def quantize_tensor(t: Tensor, bits: int = DEFAULT_BITS,
                    granularity: str = "per_tensor", symmetric: bool = True,
                    clip: str = "minmax",
                    percentile: float = DEFAULT_PERCENTILE,
                    given_range: tuple[float, float] | None = None) -> Quantized:
    if granularity not in GRANULARITIES:
        raise TensorError(
            f"unknown granularity {granularity!r}; known: {list(GRANULARITIES)}"
        )
    qmin, qmax = _qmin_qmax(bits, symmetric)

    # Each group of values gets its own (scale, zero) pair: one group for the
    # whole tensor, or one per output channel.
    if granularity == "per_tensor":
        groups = [(0, t.parameters)]
    else:
        groups = [_channel_slice(t, c) for c in range(t.channels)]

    codes: list[int] = []
    scales: list[float] = []
    zeros: list[int] = []
    clipped = 0
    for start, end in groups:
        values = t.data[start:end]
        if given_range is not None:
            lo, hi = given_range
        else:
            lo, hi = _range_of(values, clip, percentile, bits)
        s, z = _scale_zero(lo, hi, bits, symmetric)
        scales.append(s)
        zeros.append(z)
        for v in values:
            q = _round_half_up(v / s) + z
            if q < qmin or q > qmax:
                clipped += 1
                q = max(qmin, min(qmax, q))
            codes.append(q)

    return Quantized(
        name=t.name,
        shape=t.shape,
        codes=tuple(codes),
        scales=tuple(scales),
        zeros=tuple(zeros),
        bits=bits,
        granularity=granularity,
        symmetric=symmetric,
        clipped=clipped,
    )


def dequantize(qt: Quantized) -> Tensor:
    # Per-tensor uses scale/zero 0 for every code; per-channel picks the pair
    # belonging to the channel the code sits in.
    stride = qt.channel_stride if qt.granularity == "per_channel" else len(qt.codes)
    out = []
    for i, q in enumerate(qt.codes):
        g = i // stride
        out.append(round(qt.scales[g] * (q - qt.zeros[g]), 8))
    return Tensor(name=qt.name, shape=qt.shape, data=tuple(out), dtype="fp32")



def quantization_error(before: Sequence[Tensor],
                       after: Sequence[Tensor]) -> dict[str, float]:
    if len(before) != len(after):
        raise TensorError(f"{len(before)} tensors before, {len(after)} after")
    diff_sq = 0.0
    orig_sq = 0.0
    max_abs = 0.0
    for b, a in zip(before, after):
        if b.shape != a.shape:
            raise TensorError(f"{b.name}: shape {b.shape} became {a.shape}")
        for x, y in zip(b.data, a.data):
            d = x - y
            diff_sq += d * d
            orig_sq += x * x
            max_abs = max(max_abs, abs(d))
    frobenius = math.sqrt(diff_sq)
    relative = frobenius / math.sqrt(orig_sq) if orig_sq > 0 else 0.0
    return {
        "frobenius": round(frobenius, 8),
        "relative": round(relative, 8),
        "max_abs": round(max_abs, 8),
    }


def bytes_stored(tensors: Sequence[Tensor], precision: str = "fp32",
                 granularity: str = "per_tensor") -> int:
    if precision not in PRECISIONS:
        raise TensorError(f"unknown precision {precision!r}; known: {list(PRECISIONS)}")
    params = total_parameters(tensors)
    payload = params * dtype_bytes(precision)
    if precision != "int8":
        return int(payload)
    # Integer codes are useless without the (scale, zero) pairs to decode them.
    if granularity == "per_tensor":
        pairs = len(tensors)
    elif granularity == "per_channel":
        pairs = sum(t.channels for t in tensors)
    else:
        raise TensorError(
            f"unknown granularity {granularity!r}; known: {list(GRANULARITIES)}"
        )
    return int(payload + pairs * (SCALE_BYTES + ZERO_BYTES))


def effective_bits(tensors: Sequence[Tensor], precision: str = "fp32",
                   granularity: str = "per_tensor") -> float:
    params = total_parameters(tensors)
    return round(bytes_stored(tensors, precision, granularity) * 8 / params, 4)


def precision_row(model_name: str, config: Mapping[str, Any],
                  tensors: Sequence[Tensor],
                  calibration_values: Sequence[float],
                  all_values: Sequence[float]) -> dict[str, Any]:
    precision = config.get("precision", "fp32")
    if precision not in PRECISIONS:
        raise TensorError(f"unknown precision {precision!r}; known: {list(PRECISIONS)}")
    acts = as_tensor("activations", all_values)

    if precision in ("fp32", "fp16"):
        convert = to_fp16 if precision == "fp16" else (lambda t: t)
        w_after = [convert(t) for t in tensors]
        a_after = convert(acts)
        lo, hi = _range_of(calibration_values, "minmax")
        granularity, symmetric, calibration, clip = "n/a", None, "not calibrated", "n/a"
        clipped_fraction, scale_count = 0.0, 0
        size_gran = "per_tensor"
    else:
        granularity = config.get("granularity", "per_tensor")
        symmetric = bool(config.get("symmetric", True))
        calibration = config.get("calibration", "all")
        clip = config.get("clip", "minmax")
        bits = int(config.get("bits", DEFAULT_BITS))
        percentile = float(config.get("percentile", DEFAULT_PERCENTILE))

        # Weights: every value is known ahead of time, so each tensor is
        # quantized against its own range.
        qs = [quantize_tensor(t, bits, granularity, symmetric, clip, percentile)
              for t in tensors]
        w_after = [dequantize(q) for q in qs]
        scale_count = sum(q.scale_count for q in qs)

        # Activations: the range comes from the calibration set, and then is
        # applied to all the activations the model actually sees. They sit
        # behind a ReLU, so they are always quantized asymmetrically.
        lo, hi = _range_of(calibration_values, clip, percentile, bits)
        qa = quantize_tensor(acts, bits, "per_tensor", symmetric=False,
                             given_range=(lo, hi))
        a_after = dequantize(qa)
        clipped_fraction = qa.clipped_fraction
        size_gran = granularity

    return {
        "model": model_name,
        "label": config.get("label", precision),
        "precision": precision,
        "granularity": granularity,
        "symmetric": symmetric,
        "calibration": calibration,
        "clip": clip,
        "parameters": total_parameters(tensors),
        "weight_error": quantization_error(tensors, w_after)["relative"],
        "activation_error": quantization_error([acts], [a_after])["relative"],
        "clipped_fraction": clipped_fraction,
        "activation_range": [round(lo, 8), round(hi, 8)],
        "scale_count": scale_count,
        "bytes_stored": bytes_stored(tensors, precision, size_gran),
        "effective_bits": effective_bits(tensors, precision, size_gran),
    }


def pareto_front(points: Sequence[Mapping[str, Any]],
                 objectives: Mapping[str, float]) -> tuple[int, ...]:
    if not objectives:
        raise TensorError("a Pareto front needs at least one objective")
    # Multiply by the sense so that "bigger is better" holds for every
    # objective: +1 means maximise, -1 means minimise.
    scored = [[float(p[k]) * float(sense) for k, sense in objectives.items()]
              for p in points]

    def dominates(a: list[float], b: list[float]) -> bool:
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    return tuple(
        i for i, s in enumerate(scored)
        if not any(dominates(other, s) for j, other in enumerate(scored) if j != i)
    )


if __name__ == "__main__":
    json_path = Path("model.json")

    model_spec = load_model(json_path)
    model_tensors = model_spec["tensors"]
    model_name = model_spec["name"]
    total_params = total_parameters(model_tensors)

    test_tensor = model_tensors[0]  # conv1: shape (20, 3, 3, 3)

    raw_samples = materialise_samples(layer="conv3", width=40, count=32, seed=model_spec["seed"] + 100)
    all_act_vals = tuple(v for s in raw_samples for v in s.data)
    calib_all = select_calibration(raw_samples, policy="all")
    calib_one_cls = select_calibration(raw_samples, policy="one_class")

    # -----------------------------------------------------------------
    # 1. Test quantize_tensor
    # -----------------------------------------------------------------
    print("\n[1/7] Testing quantize_tensor...")
    q_pt = quantize_tensor(test_tensor, bits=8, granularity="per_tensor", symmetric=True)
    q_pc = quantize_tensor(test_tensor, bits=8, granularity="per_channel", symmetric=True)

    assert len(q_pt.codes) == test_tensor.parameters
    assert q_pt.scale_count == 1
    assert q_pc.scale_count == test_tensor.channels
    print(f"  ✓ Per-tensor quantized:  {len(q_pt.codes)} codes, {q_pt.scale_count} scale")
    print(f"  ✓ Per-channel quantized: {len(q_pc.codes)} codes, {q_pc.scale_count} scales")

    # -----------------------------------------------------------------
    # 2. Test dequantize
    # -----------------------------------------------------------------
    print("\n[2/7] Testing dequantize...")
    deq_pt = dequantize(q_pt)
    deq_pc = dequantize(q_pc)

    assert deq_pt.shape == test_tensor.shape
    assert len(deq_pt.data) == len(test_tensor.data)
    assert deq_pc.shape == test_tensor.shape
    print(f"  ✓ Reconstructed tensor shape: {deq_pt.shape}, dtype: {deq_pt.dtype}")

    # -----------------------------------------------------------------
    # 3. Test quantization_error
    # -----------------------------------------------------------------
    print("\n[3/7] Testing quantization_error...")
    err_pt = quantization_error([test_tensor], [deq_pt])
    err_pc = quantization_error([test_tensor], [deq_pc])

    assert err_pt["relative"] >= 0.0 and err_pc["relative"] >= 0.0
    print(f"  ✓ Per-tensor relative error:  {err_pt['relative']:.6f} (max_abs: {err_pt['max_abs']:.6f})")
    print(f"  ✓ Per-channel relative error: {err_pc['relative']:.6f} (max_abs: {err_pc['max_abs']:.6f})")

    # -----------------------------------------------------------------
    # 4. Test bytes_stored
    # -----------------------------------------------------------------
    print("\n[4/7] Testing bytes_stored...")
    bytes_fp32 = bytes_stored(model_tensors, precision="fp32")
    bytes_fp16 = bytes_stored(model_tensors, precision="fp16")
    bytes_int8_pt = bytes_stored(model_tensors, precision="int8", granularity="per_tensor")
    bytes_int8_pc = bytes_stored(model_tensors, precision="int8", granularity="per_channel")

    assert bytes_fp32 > bytes_fp16 > bytes_int8_pt
    assert bytes_int8_pc >= bytes_int8_pt  # per-channel carries scale/zero per output channel
    print(f"  ✓ FP32 bytes:            {bytes_fp32} B")
    print(f"  ✓ FP16 bytes:            {bytes_fp16} B")
    print(f"  ✓ INT8 per-tensor bytes: {bytes_int8_pt} B")
    print(f"  ✓ INT8 per-channel bytes:{bytes_int8_pc} B")

    # -----------------------------------------------------------------
    # 5. Test effective_bits
    # -----------------------------------------------------------------
    print("\n[5/7] Testing effective_bits...")
    eff_fp32 = effective_bits(model_tensors, precision="fp32")
    eff_fp16 = effective_bits(model_tensors, precision="fp16")
    eff_int8_pt = effective_bits(model_tensors, precision="int8", granularity="per_tensor")
    eff_int8_pc = effective_bits(model_tensors, precision="int8", granularity="per_channel")

    assert eff_fp32 == 32.0
    assert eff_fp16 == 16.0
    assert eff_int8_pc > 8.0  # accounts for scales and zeros overhead
    print(f"  ✓ FP32 effective bits:            {eff_fp32:.4f}")
    print(f"  ✓ FP16 effective bits:            {eff_fp16:.4f}")
    print(f"  ✓ INT8 per-tensor effective bits: {eff_int8_pt:.4f}")
    print(f"  ✓ INT8 per-channel effective bits:{eff_int8_pc:.4f}")

    # -----------------------------------------------------------------
    # 6. Test precision_row
    # -----------------------------------------------------------------
    print("\n[6/7] Testing precision_row...")
    configs = [
        {"label": "FP32 Baseline", "precision": "fp32"},
        {"label": "FP16", "precision": "fp16"},
        {
            "label": "INT8 Per-Tensor (All)",
            "precision": "int8",
            "granularity": "per_tensor",
            "symmetric": False,
            "clip": "minmax",
            "calibration": "all",
        },
        {
            "label": "INT8 Per-Channel (All)",
            "precision": "int8",
            "granularity": "per_channel",
            "symmetric": True,
            "clip": "minmax",
            "calibration": "all",
        },
        {
            "label": "INT8 Per-Channel (Class 0 Only)",
            "precision": "int8",
            "granularity": "per_channel",
            "symmetric": True,
            "clip": "minmax",
            "calibration": "one_class",
        },
    ]

    rows = []
    for cfg in configs:
        calib_vals = calib_one_cls if cfg.get("calibration") == "one_class" else calib_all
        row = precision_row(
            model_name=model_name,
            config=cfg,
            tensors=model_tensors,
            calibration_values=calib_vals,
            all_values=all_act_vals,
        )
        rows.append(row)
        print(f"  ✓ {row['label']:<32} | W_Err: {row['weight_error']:<8.6f} | Act_Err: {row['activation_error']:<8.6f} | Bits: {row['effective_bits']:<6.3f}")

    # -----------------------------------------------------------------
    # 7. Test pareto_front & Save Output
    # -----------------------------------------------------------------
    print("\n[7/7] Testing pareto_front...")
    # Minimize effective_bits and minimize weight_error (sense = -1.0)
    objectives = {
        "effective_bits": -1.0,
        "weight_error": -1.0,
    }
    front_indices = pareto_front(rows, objectives)

    assert len(front_indices) > 0
    print(f"  ✓ Pareto optimal candidate indices: {front_indices}")
    for idx in front_indices:
        p = rows[idx]
        print(f"    * Index {idx}: {p['label']} (Bits: {p['effective_bits']}, Weight Err: {p['weight_error']})")

    # Output to JSON
    report_path = Path("quantization_test_results.json")
    results = {
        "model": model_name,
        "total_parameters": total_params,
        "objectives": objectives,
        "pareto_front_indices": list(front_indices),
        "configurations": rows,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Saved] Full output written to {report_path.resolve()}")

