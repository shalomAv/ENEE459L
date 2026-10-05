from typing import Any
from statistics import median

MIN_SAMPLES_FOR_STATIONARITY = 12
STATIONARITY_TOL = 0.10


def is_stationary(samples: list[float]) -> dict[str, Any]:
    # 1. Validate sample count
    if len(samples) < MIN_SAMPLES_FOR_STATIONARITY:
        return {
            "status": "unknown",
             "value": None,
            "detail": "too few samples to divide into thirds"
        }

    # 2. Compute overall median
    overall_median = median(samples)

    if overall_median <= 0:
        return {
            "status": "unknown",
             "value": None,
            "detail": "median is not positive"
        }

    # 3. Split into thirds
    n = len(samples)
    k = n // 3

    first_third = samples[:k]
    last_third = samples[-k:]

    first_median = median(first_third)
    last_median = median(last_third)

    # 4. Calculate drift
    drift = last_median - first_median
    relative_drift = abs(drift) / overall_median

    # 5. Classify direction
    if drift > 0:
        direction = "slower"
    elif drift < 0:
        direction = "faster"
    else:
        direction = "flat"

    # 6. Stationarity result
    stationary = relative_drift <= STATIONARITY_TOL

    return {
        "status": "measured",
        "value": stationary,
        "first_third_median_ms": round(first_median, 4),
        "last_third_median_ms": round(last_median, 4),
        "drift_ms": round(drift, 4),
        "drift_relative": round(relative_drift, 4),
        "direction": direction,
        "tolerance": STATIONARITY_TOL,
    }
if __name__ == "__main__":
    tests = {
        "flat": [10.0] * 12,

        "slightly_slower": [
            10.0, 10.0, 10.0, 10.0,
            10.2, 10.2, 10.2, 10.2,
            10.5, 10.5, 10.5, 10.5
        ],

        "too_much_drift": [
            10.0, 10.0, 10.0, 10.0,
            11.0, 11.0, 11.0, 11.0,
            12.0, 12.0, 12.0, 12.0
        ],

        "too_few": [10.0, 10.1, 10.2],

        "faster": [
            12.0, 12.0, 12.0, 12.0,
            11.0, 11.0, 11.0, 11.0,
            10.0, 10.0, 10.0, 10.0
],

        "non_positive_median": [
          -1.0, -1.0, -1.0, -1.0,
           0.0, 0.0, 0.0, 0.0,
           1.0, 1.0, 1.0, 1.0
],
    }

    for name, samples in tests.items():
        print(name)
        print(is_stationary(samples))
        print()