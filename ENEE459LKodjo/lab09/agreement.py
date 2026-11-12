
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    samples = []

    for i, value in enumerate(ms):
        value = _finite(value, f"ms[{i}]")
        if value < 0:
            raise AgreementError("Latency cannot be negative.")
        samples.append(value)

    if not samples:
        raise AgreementError("No latency samples provided.")

    n = len(samples)
    mean = sum(samples) / n
    worst = max(samples)

    p50 = _percentile(samples, 50)
    p95 = _percentile(samples, 95)
    p99 = _percentile(samples, 99)

    return {
        "n": n,
        "mean": mean,
        "worst": worst,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "jitter": p95 - p50,
    }


def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    base = _finite(base_ms, "base_ms")
    other = _finite(other_ms, "other_ms")

    if base <= 0 or other <= 0:
        raise AgreementError("Execution times must be positive.")

    factor = base / other

    return {
        "factor": factor,
        "faster": factor > 1.0,
        "saved_ms": base - other,
    }


def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    if len(a) == 0 or len(a) != len(b):
        raise AgreementError("Arrays must be nonempty and have equal lengths.")

    values_a = [_finite(v, f"a[{i}]") for i, v in enumerate(a)]
    values_b = [_finite(v, f"b[{i}]") for i, v in enumerate(b)]

    best_i = 0
    best = abs(values_a[0] - values_b[0])

    for i in range(1, len(values_a)):
        diff = abs(values_a[i] - values_b[i])
        if diff > best:
            best = diff
            best_i = i

    return {
        "max_abs": best,
        "index": best_i,
        "a": values_a[best_i],
        "b": values_b[best_i],
        "n": len(values_a),
    }


def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    floor = _finite(floor, "floor")

    if floor <= 0:
        raise AgreementError("Floor must be positive.")

    if len(a) == 0 or len(a) != len(b):
        raise AgreementError("Arrays must be nonempty and have equal lengths.")

    values_a = [_finite(v, f"a[{i}]") for i, v in enumerate(a)]
    values_b = [_finite(v, f"b[{i}]") for i, v in enumerate(b)]

    best = -1.0
    best_i = 0
    floored = 0
    denom_min = float("inf")

    for i in range(len(values_a)):
        d = abs(values_b[i])
        denom_min = min(denom_min, d)

        if d < floor:
            floored += 1

        denom = max(d, floor)
        rel = abs(values_a[i] - values_b[i]) / denom

        if rel > best:
            best = rel
            best_i = i

    return {
        "max_rel": best,
        "index": best_i,
        "floored": floored,
        "denom_min": denom_min,
        "n": len(values_a),
    }


def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    if len(rows_a) == 0 or len(rows_a) != len(rows_b):
        raise AgreementError("Row counts must be nonzero and equal.")

    agree = 0
    first_disagreement = None

    for k, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        if len(row_a) == 0 or len(row_a) != len(row_b):
            raise AgreementError(f"Invalid row lengths at row {k}.")

        a = [_finite(v, f"rows_a[{k}][{i}]")
             for i, v in enumerate(row_a)]
        b = [_finite(v, f"rows_b[{k}][{i}]")
             for i, v in enumerate(row_b)]

        ia = max(range(len(a)), key=lambda i: a[i])
        ib = max(range(len(b)), key=lambda i: b[i])

        if ia == ib:
            agree += 1
        elif first_disagreement is None:
            first_disagreement = k

    n = len(rows_a)

    return {
        "n": n,
        "agree": agree,
        "disagree": n - agree,
        "rate": agree / n,
        "first_disagreement": first_disagreement,
    }


def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    tol = _finite(tol, "tol")

    if tol < 0:
        raise AgreementError("Tolerance cannot be negative.")

    if len(rows) == 0:
        raise AgreementError("No rows provided.")

    margins = []

    for k, row in enumerate(rows):
        if len(row) < 2:
            raise AgreementError(f"Row {k} must contain at least two classes.")

        values = [_finite(v, f"rows[{k}][{i}]")
                  for i, v in enumerate(row)]

        scores = sorted(values, reverse=True)
        margins.append(scores[0] - scores[1])

    minimum = min(margins)
    argmin = margins.index(minimum)
    below_tol = sum(m < tol for m in margins)

    return {
        "n": len(margins),
        "min": minimum,
        "argmin": argmin,
        "median": _percentile(margins, 50),
        "mean": sum(margins) / len(margins),
        "below_tol": below_tol,
        "safe": below_tol == 0,
    }

#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()