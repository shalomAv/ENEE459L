from __future__ import annotations

from typing import Any, Sequence

from tensors import (
    INDEX_BYTES,
    Tensor,
    TensorError,
    dtype_bytes,
    total_parameters,
)

# The four meanings of "removed", Lecture 05 slide 8, in staircase order.
REMOVALS = ("masked", "patterned", "stored sparse", "structurally absent")

# How a tensor is written to disk. Three forms, and `classify_removal` reads
# this alongside the shapes to decide which of the four it got.
STORAGE_FORMS = ("dense", "masked", "sparse")

# What each granularity produces if you implement it the obvious way. Fine
# grained pruning in a framework leaves a mask behind; channel pruning leaves a
# smaller dense tensor. `sweep_model` uses this unless told otherwise, and
# `sweep.py --storage` is how you tell it otherwise.
GRANULARITY_STORAGE = {"fine": "masked", "channel": "dense"}

GRANULARITIES = tuple(GRANULARITY_STORAGE)

# A layer with no output channels is not a smaller layer, it is a broken one.
MIN_CHANNELS = 1

# N:M, for `classify_removal`. 2:4 is the case Ampere implements and the only
# one Lecture 05 slide 18 draws.
NM_N, NM_M = 2, 4


def _drop_count(n: int, ratio: float) -> int:
    if not 0.0 <= ratio <= 1.0:
        raise TensorError(f"ratio {ratio} is not in [0, 1]")
    return max(0, min(int(n), int(int(n) * float(ratio) + 0.5)))


def _smallest_indices(values: Sequence[float], k: int) -> tuple[int, ...]:
    if k <= 0:
        return ()
    order = sorted(range(len(values)), key=lambda i: (abs(values[i]), i))
    return tuple(sorted(order[:k]))


def _group_scores(t: Tensor, p: float = 2.0) -> tuple[float, ...]:
    if p <= 0:
        raise TensorError(f"p must be positive, got {p}")
    scores = []
    for c in range(t.channels):
        lo, hi = _channel_slice(t, c)
        acc = 0.0
        for v in t.data[lo:hi]:
            acc += abs(v) ** p
        scores.append(round(acc ** (1.0 / p), 10))
    return tuple(scores)


def _channel_slice(t: Tensor, c: int) -> tuple[int, int]:
    if not 0 <= c < t.channels:
        raise TensorError(f"{t.name}: no channel {c} in {t.channels}")
    stride = t.channel_stride
    return c * stride, (c + 1) * stride


# ---------------------------------------------------------------------------
# 1. the fine-grained criterion
# ---------------------------------------------------------------------------

def magnitude_mask(t: Tensor, ratio: float) -> tuple[int, ...]:
    """Which elements survive a magnitude prune at `ratio`, element by element.

    Returns a tuple the same length as `t.data`, `1` to keep and `0` to zero.
    Importance is `|w|`.
    """
    n = len(t.data)
    k = _drop_count(n, ratio)

    drop = set(_smallest_indices(t.data, k))

    mask = tuple(
        0 if i in drop else 1
        for i in range(n)
    )

    return mask


# ---------------------------------------------------------------------------
# 2. the structured criterion
# ---------------------------------------------------------------------------

def channel_keep(t: Tensor, ratio: float, p: float = 2.0) -> tuple[int, ...]:
    scores = _group_scores(t, p)

    requested_drop = _drop_count(t.channels, ratio)
    drop = min(requested_drop, t.channels - MIN_CHANNELS)

    drop_set = set(_smallest_indices(scores, drop))

    keep = tuple(
        c for c in range(t.channels)
        if c not in drop_set
    )

    return keep


# ---------------------------------------------------------------------------
# 3. masking — the removal that changes no shape
# ---------------------------------------------------------------------------

def apply_mask(t: Tensor, mask: Sequence[int]) -> Tensor:
    """Return `t` with the masked elements set to zero. Same shape. Same name."""

    if len(mask) != len(t.data):
        raise TensorError(
            f"{t.name}: mask length {len(mask)} does not match tensor length {len(t.data)}"
        )

    new_data = tuple(
        v if m == 1 else 0.0
        for v, m in zip(t.data, mask)
    )

    return Tensor(
        name=t.name,
        shape=t.shape,
        dtype=t.dtype,
        data=new_data,
    )

# ---------------------------------------------------------------------------
# 4. channel removal — the removal that changes the shape
# ---------------------------------------------------------------------------

def drop_channels(t: Tensor, keep: Sequence[int]) -> Tensor:
    if len(keep) == 0:
        raise TensorError(f"{t.name}: keep cannot be empty")

    if len(set(keep)) != len(keep):
        raise TensorError(f"{t.name}: keep contains duplicates")

    for c in keep:
        if not 0 <= c < t.channels:
            raise TensorError(
                f"{t.name}: channel {c} is out of bounds for {t.channels} channels"
            )

    kept = sorted(keep)

    new_data = []
    for c in kept:
        lo, hi = _channel_slice(t, c)
        new_data.extend(t.data[lo:hi])

    new_shape = (len(kept),) + t.shape[1:]

    return Tensor(
        name=t.name,
        shape=new_shape,
        dtype=t.dtype,
        data=tuple(new_data),
    )


# ---------------------------------------------------------------------------
# 5. the file
# ---------------------------------------------------------------------------

def bytes_stored(tensors: Sequence[Tensor], storage: str = "dense",
                 mask_encoding: str = "framework") -> int:

    if storage not in STORAGE_FORMS:
        raise TensorError(f"unknown storage form: {storage}")

    if mask_encoding not in ("framework", "bitmap"):
        raise TensorError(f"unknown mask encoding: {mask_encoding}")

    total = 0

    for t in tensors:
        n = len(t.data)
        db = dtype_bytes(t.dtype)

        if storage == "dense":
            total += n * db

        elif storage == "masked":
            total += n * db

            if mask_encoding == "framework":
                total += n * db
            else:
                total += (n + 7) // 8

        elif storage == "sparse":
            nnz = sum(1 for v in t.data if v != 0.0)
            total += nnz * db
            total += nnz * INDEX_BYTES

    return int(total)


# ---------------------------------------------------------------------------
# 6. one row of the accounting table
# ---------------------------------------------------------------------------

def sparsity_row(model: str, ratio: float, granularity: str,
                 before: Sequence[Tensor], after: Sequence[Tensor],
                 storage: str = "dense",
                 mask_encoding: str = "framework") -> dict[str, Any]:

    if granularity not in GRANULARITIES:
        raise TensorError(f"unknown granularity: {granularity}")

    pbefore = total_parameters(before)
    pafter = total_parameters(after)

    nnz_after = sum(
        1
        for t in after
        for v in t.data
        if v != 0.0
    )

    zeroed = pafter - nnz_after

    zeroed_fraction = (
        zeroed / pbefore
        if pbefore > 0
        else 0.0
    )

    achieved_reduction = (
        1.0 - (pafter / pbefore)
        if pbefore > 0
        else 0.0
    )

    dense_bytes = bytes_stored(before, storage="dense")

    stored_bytes = bytes_stored(
        after,
        storage=storage,
        mask_encoding=mask_encoding,
    )

    removal = classify_removal(
        before,
        after,
        storage=storage,
    )

    return {
        "model": model,
        "nominal_ratio": ratio,
        "granularity": granularity,
        "removal": removal,
        "parameters_before": pbefore,
        "parameters_after": pafter,
        "values_zeroed": zeroed,
        "zeroed_fraction": zeroed_fraction,
        "achieved_reduction": achieved_reduction,
        "bytes_dense": dense_bytes,
        "bytes_stored": stored_bytes,
        "storage": storage,
        "mask_encoding": mask_encoding if storage == "masked" else None,
    }


# ---------------------------------------------------------------------------
# 7. which of the four you got
# ---------------------------------------------------------------------------

def classify_removal(before: Sequence[Tensor], after: Sequence[Tensor],
                     storage: str = "dense") -> str:

    if len(before) != len(after):
        return "structurally absent"

    for b, a in zip(before, after):
        if b.shape != a.shape:
            return "structurally absent"

    if storage == "sparse":
        return "stored sparse"

    all_values = [
        v
        for t in after
        for v in t.data
    ]

    if not any(v == 0.0 for v in all_values):
        return "dense"

    patterned = True

    for t in after:
        n = len(t.data)

        if n % NM_M != 0:
            patterned = False
            break

        for i in range(0, n, NM_M):
            block = t.data[i:i + NM_M]
            nonzeros = sum(1 for v in block if v != 0.0)

            if nonzeros > NM_N:
                patterned = False
                break

        if not patterned:
            break

    if patterned:
        return "patterned"

    return "masked"


def sweep_model(model: dict[str, Any], ratios: Sequence[float],
                granularities: Sequence[str] = GRANULARITIES,
                storage: str | None = None,
                mask_encoding: str = "framework",
                p: float = 2.0) -> list[dict[str, Any]]:

    before = model["tensors"]
    results = []

    for g in granularities:
        if g not in GRANULARITIES:
            raise TensorError(f"unknown granularity: {g}")

        form = storage if storage is not None else GRANULARITY_STORAGE[g]

        for r in sorted(ratios):

            if g == "fine":
                after = []

                for t in before:
                    mask = magnitude_mask(t, r)
                    after.append(apply_mask(t, mask))

            elif g == "channel":
                after = []

                for t in before:
                    keep = channel_keep(t, r, p)
                    after.append(drop_channels(t, keep))

            row = sparsity_row(
                model=model["name"],
                ratio=r,
                granularity=g,
                before=before,
                after=after,
                storage=form,
                mask_encoding=mask_encoding,
            )

            results.append(row)

    return results