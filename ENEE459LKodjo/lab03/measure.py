from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

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


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    samples = []
    bench.workload.synchronize()

    for _ in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()

        samples.append((end - start) / 1_000_000.0)

    return samples


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    source = (
        "leading prefix above (1 + 0.5) x median "
        "of the run's second half"
    )
    n = len(samples)

    if n < 4:
        return unknown(source, "too few samples; need at least 4")

    settled_rate = statistics.median(samples[n // 2:])

    if settled_rate <= 0:
        return unknown(source, "settled median must be positive")

    threshold = settled_rate * (1 + WARMUP_TOL)

    boundary = 0
    for sample in samples:
        if sample > threshold:
            boundary += 1
        else:
            break

    return measured(
        boundary,
        source,
        settled_rate_ms=round(settled_rate, 4),
        threshold_ms=round(threshold, 4),
        tolerance=WARMUP_TOL,
        retained=n - boundary,
    )



def summarize(samples: list[float]) -> dict[str, Any]:
    n = len(samples)
    keys = ("mean", "std", "min", "max", "p50", "p95", "p99")

    if n == 0:
        return {"n": 0, **{key: None for key in keys}}

    s = sorted(samples)

    result = {
        "n": n,
        "mean": statistics.fmean(s),
        "std": statistics.stdev(s) if n > 2 else 0.0,
        "min": s[0],
        "max": s[-1],
    }

    for q in PERCENTILES:
        h = (n - 1) * q / 100
        i = int(h)
        j = min(i + 1, n - 1)
        result[f"p{q}"] = s[i] + (h - i) * (s[j] - s[i])

    return {
        key: value if key == "n" else round(value, 4)
        for key, value in result.items()
    }

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    source = (
        "widest trimmed gap >= 20.0x the median gap, "
        "with >= 10% of samples on each side"
    )
    n = len(samples)

    if n < MIN_SAMPLES_FOR_MODALITY:
        return unknown(source, "too few samples; need at least 20")

    s = sorted(samples)
    trim = int(n * 0.05)
    trimmed = s[trim:n - trim]

    gaps = [
        trimmed[i + 1] - trimmed[i]
        for i in range(len(trimmed) - 1)
    ]
    typical_gap = statistics.median(gaps)

    if typical_gap <= 0:
        return unknown(source, "timer resolution is too coarse")

    widest_gap = max(gaps)
    ratio = widest_gap / typical_gap

    # Map the trimmed split back to the full sorted sample list.
    split = trim + gaps.index(widest_gap) + 1
    left = s[:split]
    right = s[split:]

    multimodal = (
        ratio >= MULTIMODAL_GAP_RATIO
        and len(left) / n >= MIN_MODE_FRACTION
        and len(right) / n >= MIN_MODE_FRACTION
    )

    modes = [
        {
            "n": len(group),
            "share": round(len(group) / n, 4),
            "median_ms": round(statistics.median(group), 4),
        }
        for group in (left, right)
    ]

    return measured(
        multimodal,
        source,
        gap_ratio=round(ratio, 2),
        widest_gap_ms=round(widest_gap, 4),
        typical_gap_ms=round(typical_gap, 5),
        modes=modes,
    )

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    source = "nvpmodel -q"
    result = bench.runner(["nvpmodel", "-q"])

    if not result.ok or result.returncode != 0:
        return unknown(
            source,
            result.error or f"command exited with code {result.returncode}",
        )

    lines = result.stdout.splitlines()
    mode_name = None
    mode_index = None

    for i, line in enumerate(lines):
        if "NV Power Mode:" in line:
            mode_name = line.split("NV Power Mode:", 1)[1].strip()
            if i + 1 < len(lines):
                try:
                    mode_index = int(lines[i + 1].strip())
                except ValueError:
                    pass
            break

    if not mode_name or mode_index is None:
        return unknown(source, "could not parse power mode name and index")

    finding = measured(mode_name, source, mode_index=mode_index)
    clock_source = f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}"

    minimum = read_text(bench.telemetry, CPUFREQ_MIN)
    maximum = read_text(bench.telemetry, CPUFREQ_MAX)

    try:
        min_freq = int(minimum)
        max_freq = int(maximum)
        if min_freq <= 0 or max_freq <= 0:
            raise ValueError("non-positive frequency")
    except (TypeError, ValueError):
        finding["jetson_clocks"] = None
        finding["jetson_clocks_source"] = unknown(
            clock_source, "CPU frequency limits missing or invalid"
        )
    else:
        finding["jetson_clocks"] = min_freq == max_freq
        finding["jetson_clocks_source"] = measured(
            f"scaling_min_freq={min_freq}, scaling_max_freq={max_freq}",
            clock_source,
        )

    return finding



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    root = bench.telemetry
    temperature_source = f"{THERMAL_ZONES}/*/temp"
    temperatures = []

    # Read valid temperature sensors.
    try:
        zones = sorted((root / THERMAL_ZONES).glob("thermal_zone*"))
    except OSError:
        zones = []

    for zone in zones:
        relative = f"{THERMAL_ZONES}/{zone.name}"
        try:
            raw = read_text(root, f"{relative}/temp")
            temperature = int(raw)
        except (OSError, TypeError, ValueError):
            continue

        if temperature <= -1000:
            continue

        name = read_text(root, f"{relative}/type") or zone.name
        temperatures.append((temperature / 1000.0, name))

    if temperatures:
        hottest, zone_name = max(temperatures, key=lambda item: item[0])
        temperature_record = measured(
            round(hottest, 2),
            temperature_source,
            zone=zone_name,
            zones_read=len(temperatures),
        )
    else:
        temperature_record = unknown(
            temperature_source, "no valid thermal zones could be read"
        )

    # Only use the documented power paths, not hwmon voltage inputs.
    power_candidates = tuple(
        path for path in POWER_RAIL_CANDIDATES
        if path.endswith("/in_power0_input")
    )
    power = read_first(root, power_candidates)

    if power is None:
        power_record = unknown(
            " | ".join(power_candidates),
            "no documented power path could be read; "
            "hwmon in1_input is voltage, not power",
        )
    else:
        path, raw = power
        try:
            power_mw = int(raw)
            if power_mw < 0:
                raise ValueError("negative power")
        except (TypeError, ValueError):
            power_record = unknown(path, "invalid power reading")
        else:
            power_record = measured(power_mw, path)

    # GPU load is reported in per-mille.
    gpu = read_first(root, GPU_LOAD_CANDIDATES)

    if gpu is None:
        gpu_record = unknown(
            " | ".join(GPU_LOAD_CANDIDATES),
            "none of the documented GPU load paths could be read",
        )
    else:
        path, raw = gpu
        try:
            load = int(raw)
            if not 0 <= load <= 1000:
                raise ValueError("GPU load outside valid range")
        except (TypeError, ValueError):
            gpu_record = unknown(path, "invalid GPU load reading")
        else:
            gpu_record = measured(
                load / 10.0,
                path,
                units="per-mille / 10",
            )

    return {
        "temperature_c": temperature_record,
        "power_mw": power_record,
        "gpu_utilization_percent": gpu_record,
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
        json.dump(report, f, indent=4)