from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json
from pathlib import Path

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

MODALITY_TRIM = 0.05

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"

SAMPLE_SOURCE = "bench.clock around workload.run() and workload.synchronied()"

# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    wl = bench.workload
    wl.synchronize()
    samples: list[float] = []

    for _ in range(repeats):
        t0 = bench.clock()
        wl.run()
        wl.synchronize()
        t1 = bench.clock()
        samples.append((t1 - t0) / 1_000_000.0)

    return samples


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return unknown(samples, "no samples to examine")

    if len(samples) < 4:
        return unknown(samples, "too few samples")

    settled = statistics.median(samples[len(samples) // 2 : ])
    threshold = settled * (1.0 + WARMUP_TOL)

    discarded = 0
    for s in samples:
        if s > threshold:
            discarded += 1
        else:
            break

    return measured(
        discarded,
        SAMPLE_SOURCE,
        settled_ms = round(settled, 4),
        threshold_ms = round(threshold, 4),
        tolerance = WARMUP_TOL,
        retained = len(samples) - discarded,
    )

def _percentile(s: list[float], q: float) -> float:
    h = (len(s) - 1) * q / 100
    i = int(h)
    if i + 1 >= len(s):
        return s[i]
    return s[i] + (h - i) * (s[i + 1] - s[i])

def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }

    s = sorted(samples)
    n = len(s)
    std = statistics.stdev(s) if n > 2 else 0.0

    result: dict[str, Any] = {
            "n": n,
            "mean": round(statistics.fmean(s), 4),
            "std": round(std, 4),
            "min": round(s[0], 4),
            "max": round(s[-1], 4),
    }

    for q in PERCENTILES:
        result[f"p{q}"] = round(_percentile(s, q), 4)

    return result
    

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    n = len(samples)
    if n < MIN_SAMPLES_FOR_MODALITY:
        return unknown(SAMPLE_SOURCE, f"not enough smaples {n}; need at least {MIN_SAMPLES_FOR_MODALITY}")

    s = sorted(samples)
    k = int(n * MODALITY_TRIM)
    trimmed = s[k : n - k]

    gaps = [b - a for a, b in zip(trimmed, trimmed[1:])]
    median_gap = statistics.median(gaps)

    if median_gap <= 0:
        return unknown(SAMPLE_SOURCE, "timer resolution is too course")

    widest = max(gaps)
    ratio = widest/median_gap
    split = gaps.index(widest)

    left = trimmed[ : split + 1]
    right = trimmed[ split - 1 : ]
    total = len(trimmed)
    left_frac = len(left) / total
    right_frac = len(right) / total

    multimodal = ratio >= MULTIMODAL_GAP_RATIO and left_frac >= MIN_MODE_FRACTION and right_frac >= MIN_MODE_FRACTION

    return measured(
        multimodal,
        SAMPLE_SOURCE,
        widest_gap_ms = round(widest, 4),
        median_gap_ms = round(median_gap, 4),
        gap_ratio = round(ratio, 4),
        ratio_threshold = MULTIMODAL_GAP_RATIO,
        split_between_ms = [round(trimmed[split], 4), round(trimmed[split + 1], 4)],
        trimmed_per_side = k,
        lower_group = {
            "count": len(left),
            "fraction": round(left_frac, 4),
            "mean_ms": round(statistics.fmean(left), 4)
        },
        upper_group = {
            "count": len(right),
            "fraction": round(right_frac, 4),
            "mean_ms": round(statistics.fmean(right), 4)
        },
    )

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================

def _parse_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        return int(text.split()[0])
    except (ValueError, IndexError):
        return None

def probe_power_state(bench: Bench) -> dict[str, Any]:
    res = bench.runner(["nvpmodel", "-q"])

    if not res.ok:
        return unknown(res.source, f"could not run nvpmodel: {res.error}")
    if res.returncode != 0:
        return unknown(res.source, f"nvpmodel exited with status {res.returncode}")

    lines = [l.strip() for l in res.stdout.splitlines()]
    mode_name = None
    mode_id = None

    for i, line in enumerate(lines):
        if "NV Power Mode" in line:
            mode_name = line.split("NV Power Mode:", 1)[1].strip()
            if i + 1 < len(lines):
                mode_id = _parse_int(lines[i + 1])
            break

    if mode_name is None:
        return unknown (res.source, "no 'NV Power Mode:' line in nvpmodel output")

    root = bench.telemetry
    fmin = (read_text(root, CPUFREQ_MIN))
    fmax = (read_text(root, CPUFREQ_MAX))

    if fmin is None or fmax is None:
        jetson_clocks = False
    else:
        jetson_clocks = fmin == fmax

    return measured(
        mode_name,
        res.source,
        mode_id = mode_id,
        cpu_min_freq_khz = fmin,
        cpu_max_freq_khz = fmax,
        jetson_clocks = jetson_clocks,
    )


def _temperature(root: Path) -> dict[str, Any]:
    base = Path(root) / THERMAL_ZONES
    try:
        zones = sorted(p.name for p in base.iterdir() if p.name.startswith("thermal_zone"))
    except OSError:
        return unknown(THERMAL_ZONES, "thermal directory missing or unreadable")

    readings: dict[str, float] = {}
    hottest: tuple[float, str, str] | None = None

    for zone in zones:
        rel = f"{THERMAL_ZONES}/{zone}/temp"
        raw = _parse_int(read_text(root, rel))
        if raw is None or raw <= -1000:
            continue
        temp_c = raw / 1000.0
        name = read_text(root, f"{THERMAL_ZONES}/{zone}/type")
        readings[name] = temp_c
        if hottest is None or temp_c > hottest[0]:
            hottest = (temp_c, rel, name)

    if hottest is None:
        return unknown(THERMAL_ZONES, "no valid thermal zone readings")

    temp_c, rel, name = hottest
    return measured(temp_c, rel, unit="C", zone=name, all_zones_c=readings)

def _power(root: Path) -> dict[str, Any]:
    hit = read_first(root, POWER_RAIL_CANDIDATES)
    if hit is None:
        return unknown(root, "no INA3221 power node found")
    path, text = hit
    power_mw = _parse_int(text)
    if power_mw is None:
        return unknown(root, f"could not parse {text!r} as an integer")

    return measured(power_mw, path, unit="mW")

def _gpu_load(root: Path) -> dict[str, Any]:
    hit = read_first(root, GPU_LOAD_CANDIDATES)
    if hit is None:
        return unknown(root, "no GPU load node found")

    path, text = hit
    raw = _parse_int(text)
    if raw is None:
        return unknown(root, f"could not parse {text!r} as an integer")

    return measured(raw / 10.0, path, unit="%", raw = raw)

def probe_telemetry(bench: Bench) -> dict[str, Any]:
    root = bench.telemetry
    return {
        "temperature_c": _temperature(root),
        "power_mw": _power(root),
        "gpu_load_pct": _gpu_load(root),
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=str)