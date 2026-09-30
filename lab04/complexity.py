from __future__ import annotations

from typing import Any

from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
    is_answered,
    unknown,
    measured,
)


FLOPS_PER_MAC = 2

# The conventions `to_flops` will honour by name. Anything else is unknown
# rather than an assumption, because the whole point of the parameter is that
# the caller has to say which one they mean.
FLOP_CONVENTIONS = {
    "mac_is_two_flops": 2,
    "mac_is_one_flop": 1,
}

# Batch normalisation holds two learnable vectors per channel (scale and shift)
# and two non-learnable ones (running mean and variance). The first pair are
# parameters; the second pair are buffers. Both are in the file.
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2

# Buffers are kept in FP32 even when the weights are not. Halving them saves
# nothing worth having and a denormal running variance is a real failure mode.
BUFFER_DTYPE = "fp32"

# Below this many models there is no line to fit and no residual to report.
MIN_MODELS_FOR_FIT = 3

# Two floats are the same MAC count when they are the same integer. There is no
# tolerance here on purpose: MAC counts are integers, and a tolerance would let
# two genuinely different architectures be reported as tied.
TIE_EXACT = True

# Layer kinds that are known to hold no parameters and no buffers. A kind that
# is in neither this set nor the parametric branches below is *unknown*, not
# zero: "I have never heard of this layer" is not evidence that it is empty.
PARAMETER_FREE = frozenset({"pool", "relu", "add", "flatten"})
 
# Name used for the network's own input in the liveness pass. Angle brackets
# keep it from colliding with any real layer name.
NETWORK_INPUT = "<input>"

def _duplicate_names(graph: Graph) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for ly in graph:
        if ly.name in seen:
            dupes.append(ly.name)
        seen.add(ly.name)
    return dupes
 
 
def _layer_parameters(ly: Layer) -> tuple[int | None, str]:
    """Learnable parameters of one layer, or (None, reason) if not computable."""
    if ly.kind == "conv":
        if ly.kernel is None or len(ly.kernel) != 2:
            return None, f"{ly.name}: conv has no 2-D kernel"
        if not ly.in_shape or not ly.out_shape:
            return None, f"{ly.name}: conv has an empty shape"
        c_in, c_out = ly.in_shape[0], ly.out_shape[0]
        g = ly.groups
        if g < 1 or c_in % g or c_out % g:
            return None, (
                f"{ly.name}: groups={g} does not divide C_in={c_in} / C_out={c_out}"
            )
        kh, kw = ly.kernel
        # Each output channel sees only C_in / groups input channels.
        n = c_out * (c_in // g) * kh * kw
        if ly.bias:
            n += c_out
        return n, ""
 
    if ly.kind == "linear":
        if not ly.in_shape or not ly.out_shape:
            return None, f"{ly.name}: linear has an empty shape"
        # A linear layer acts on the last dimension, as nn.Linear does.
        f_in, f_out = ly.in_shape[-1], ly.out_shape[-1]
        n = f_in * f_out
        if ly.bias:
            n += f_out
        return n, ""
 
    if ly.kind == "bn":
        if not ly.in_shape:
            return None, f"{ly.name}: bn has an empty shape"
        # Scale and shift only. Running mean / variance are buffers.
        return BN_PARAMS_PER_CHANNEL * ly.in_shape[0], ""
 
    if ly.kind in PARAMETER_FREE:
        return 0, ""
 
    return None, f"{ly.name}: unknown layer kind {ly.kind!r}"
 
 
def _layer_buffers(ly: Layer) -> int:
    """Non-learnable stored values of one layer (BN running statistics)."""
    if ly.kind == "bn" and ly.in_shape:
        return BN_BUFFERS_PER_CHANNEL * ly.in_shape[0]
    return 0


# ===========================================================================
# 1. How many numbers are stored
# ===========================================================================

def count_parameters(graph: Graph) -> dict[str, Any]:
    source = f"{graph.name}: {len(graph)} layers, shapes from the description"
 
    if len(graph) == 0:
        return unknown(source, "graph has no layers")
    dupes = _duplicate_names(graph)
    if dupes:
        return unknown(source, f"duplicate layer names {dupes}; per-layer keys would collide")
 
    per_layer: dict[str, int] = {}
    for ly in graph:
        n, why = _layer_parameters(ly)
        if n is None:
            return unknown(source, why)
        per_layer[ly.name] = n
 
    return computed(
        sum(per_layer.values()),
        source,
        per_layer=per_layer,
        includes_bias=True,
        excludes_bn_buffers=True,
        bn_params_per_channel=BN_PARAMS_PER_CHANNEL,
    )


# ===========================================================================
# 2. What those numbers weigh, which is not the size of the file
# ===========================================================================


def model_size_bytes(graph: Graph) -> dict[str, Any]:
    """Bytes of stored tensors: parameters plus buffers, at their own dtypes.

    Lecture 04 slide 8 gives the formula as `#Parameters × bit width` and slide
    9 spends a page on why the file on disk is not that number. Three reasons,
    two of which this function has to get right:

      * a model is not stored in one dtype. `Layer.weight_dtype` is per layer
        and a network with FP16 weights and FP32 normalisation is completely
        ordinary. Multiplying a single total by a single bit width is the
        mistake, and on these four descriptions it is worth several per cent
      * buffers are in the file. Batch norm's running statistics are two
        vectors per channel that no optimiser ever touched, and they are still
        bytes you have to ship
      * the container is in the file too — the pickle framing, the state-dict
        keys, the archive directory. This function does *not* try to model
        that, and it says so in `container_overhead_excluded` rather than
        quietly letting the caller assume it did

    Returns a `computed` finding whose value is bytes, with the per-dtype
    breakdown that makes the first bullet checkable.
    """
    source = f"{graph.name}: per-layer dtypes, buffers at {BUFFER_DTYPE}"
 
    params = count_parameters(graph)
    if not is_answered(params):
        return unknown(source, f"parameter count is unknown: {params.get('detail')}")
 
    buffer_width = dtype_bytes(BUFFER_DTYPE)
 
    per_layer: dict[str, float] = {}
    per_dtype: dict[str, float] = {}
    buffer_total = 0.0
 
    for ly in graph:
        try:
            weight_width = dtype_bytes(ly.weight_dtype)
        except KeyError as exc:
            return unknown(source, f"{ly.name}: {exc.args[0]}")
 
        # Each layer's parameters at that layer's own dtype, never a global one.
        param_bytes = params["per_layer"][ly.name] * weight_width
        buf_bytes = _layer_buffers(ly) * buffer_width
 
        per_layer[ly.name] = param_bytes + buf_bytes
        if param_bytes:
            per_dtype[ly.weight_dtype] = per_dtype.get(ly.weight_dtype, 0.0) + param_bytes
        if buf_bytes:
            per_dtype[BUFFER_DTYPE] = per_dtype.get(BUFFER_DTYPE, 0.0) + buf_bytes
        buffer_total += buf_bytes
 
    return computed(
        sum(per_layer.values()),
        source,
        per_layer=per_layer,
        per_dtype=per_dtype,
        buffer_bytes=buffer_total,
        container_overhead_excluded=True,
        note="not the size of the file on disk; see the handout, Stage A step 3",
    )

# ===========================================================================
# 3. The memory nobody puts in the table
# ===========================================================================

def count_activations(graph: Graph) -> dict[str, Any]:
    """Total and peak activation footprint, in elements and in bytes.

    UNC COMP 790-150 Lec 2 p. 70 gives AlexNet as total 932,264 and peak
    440,928, and the two numbers answer two different questions. Total is what
    the whole forward pass produced. Peak is how much had to be resident at
    once, and peak is the one that decides whether the model runs.

    Peak is not `max(out_elements)`. Three things make it larger than that:

      * a layer's input is still resident while its output is being written.
        The live set at layer *i* contains both
      * a tensor consumed by a later layer stays resident in between. `add`
        layers name two inputs in `Layer.reads`, and the earlier one has been
        sitting in memory across every layer of the block. This is the residual
        connection and it is the single largest contributor to peak in
        ResNet-shaped networks
      * the network's own input is a tensor too

    The implementation is a liveness pass: work out the last layer that reads
    each tensor, then walk forward keeping a live set and taking the maximum of
    its total size. Anything simpler than that is wrong on any graph with a
    skip connection, and it is wrong quietly, in the direction that says the
    model fits.

    Returns a `computed` finding whose value is peak *bytes*, because bytes are
    what a memory budget is denominated in, with elements and the layer where
    the peak occurs alongside.
    """
    n_layers = len(graph)
    source = f"{graph.name}: liveness over {n_layers} layers, input included"
 
    if n_layers == 0:
        return unknown(source, "graph has no layers")
    dupes = _duplicate_names(graph)
    if dupes:
        return unknown(source, f"duplicate layer names {dupes}; tensors would be ambiguous")
 
    layers = list(graph)
    produced_at = {ly.name: i for i, ly in enumerate(layers)}
 
    # --- sizes of every tensor: the network input plus one output per layer
    in_elements = 1
    for d in graph.input_shape:
        in_elements *= d
    try:
        elements = {NETWORK_INPUT: in_elements}
        nbytes = {NETWORK_INPUT: in_elements * dtype_bytes(graph.precision)}
        for ly in layers:
            elements[ly.name] = ly.out_elements
            nbytes[ly.name] = ly.out_elements * dtype_bytes(ly.act_dtype)
    except KeyError as exc:
        return unknown(source, exc.args[0])
 
    # --- pass 1: the last step at which each tensor is read
    # A layer with no explicit `reads` consumes the previous layer's output
    # (or the network input, for the first layer).
    last_use: dict[str, int] = {}
    for i, ly in enumerate(layers):
        if ly.reads:
            sources = ly.reads
        else:
            sources = (layers[i - 1].name,) if i else (NETWORK_INPUT,)
        for t in sources:
            if t != NETWORK_INPUT and (t not in produced_at or produced_at[t] >= i):
                return unknown(source, f"{ly.name}: reads {t!r}, which is not an earlier layer")
            last_use[t] = max(last_use.get(t, i), i)
 
    # A tensor lives from the step that produces it to the step that last reads
    # it. One that nobody reads lives only for its own step; the final output
    # lives to the end, which is the same thing.
    free_after: dict[int, list[str]] = {}
    free_after.setdefault(last_use.get(NETWORK_INPUT, 0), []).append(NETWORK_INPUT)
    for i, ly in enumerate(layers):
        free_after.setdefault(max(last_use.get(ly.name, i), i), []).append(ly.name)
 
    # --- pass 2: walk forward with a running live set
    live_elements = elements[NETWORK_INPUT]
    live_bytes = nbytes[NETWORK_INPUT]
    peak_bytes = None
    peak_elements = 0
    peak_at = None
 
    for i, ly in enumerate(layers):
        # The output is allocated while every input is still resident.
        live_elements += elements[ly.name]
        live_bytes += nbytes[ly.name]
        if peak_bytes is None or live_bytes > peak_bytes:
            peak_bytes, peak_elements, peak_at = live_bytes, live_elements, ly.name
        # Then release everything whose last reader was this layer.
        for t in free_after.get(i, []):
            live_elements -= elements[t]
            live_bytes -= nbytes[t]
 
    # Total is what the forward pass produced: every layer's output. The input
    # was handed to the network, not produced by it.
    total_elements = sum(elements[ly.name] for ly in layers)
    total_bytes = 0.0
    for ly in layers:
        total_bytes += nbytes[ly.name]
 
    return computed(
        peak_bytes,
        source,
        peak_at=peak_at,
        peak_elements=peak_elements,
        total_elements=total_elements,
        total_bytes=total_bytes,
        includes_network_input=True,
        note="peak is the resident set, not the largest single tensor",
    )

# ===========================================================================
# 4. The factor of two that halves everybody's numbers
# ===========================================================================

def to_flops(macs: dict[str, Any], convention: str = "mac_is_two_flops") -> dict[str, Any]:
    """Convert a MAC finding to a FLOP finding, naming the convention used.

    A multiply-accumulate is one multiply and one add, so it is two
    floating-point operations. Roughly half the published literature calls a
    MAC one FLOP anyway, and the two conventions differ by exactly the factor
    that makes two papers' numbers incomparable.

    Three requirements, and the third is the graded one:

      * multiply once. `FLOPS_PER_MAC` exists so that the number 2 appears in
        this file exactly once
      * an unknown MAC count converts to an unknown FLOP count. It does not
        convert to zero and it does not raise
      * the convention goes in the finding. A FLOP count that does not say
        which convention produced it is not a FLOP count, it is a number, and
        `to_flops(x, "mac_is_one_flop")` has to be as clearly labelled as the
        default

    An unrecognised convention is `unknown`, not a default. The caller asked
    for something this function does not know how to do.
    """
    source = (macs or {}).get("source", "unattributed")
 
    if convention not in FLOP_CONVENTIONS:
        out = unknown(
            source,
            f"unrecognised FLOP convention {convention!r}; "
            f"known: {sorted(FLOP_CONVENTIONS)}",
        )
        out["convention"] = convention
        return out
 
    if not is_answered(macs) or macs.get("value") is None:
        why = (macs or {}).get("detail", "no MAC finding supplied")
        out = unknown(source, f"MAC count is unknown ({why}), so the FLOP count is too")
        out["convention"] = convention
        return out
 
    factor = FLOP_CONVENTIONS[convention]
 
    # The single place a MAC becomes a FLOP. Unknown per-layer entries stay
    # unknown instead of turning into zero.
    def scale(m: Any) -> Any:
        return None if m is None else m * factor
 
    extra: dict[str, Any] = {"convention": convention, "flops_per_mac": factor}
    if isinstance(macs.get("per_layer"), dict):
        extra["per_layer"] = {k: scale(v) for k, v in macs["per_layer"].items()}
    extra["note"] = "a count of operations contains no unit of time"
 
    # Converting units does not change provenance: a measured MAC count gives a
    # measured FLOP count.
    make = measured if macs.get("status") == "measured" else computed
    return make(scale(macs["value"]), source, **extra)