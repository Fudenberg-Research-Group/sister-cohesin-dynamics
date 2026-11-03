"""Metrics describing cis and trans P(s) signatures."""

import numpy as np


def _prepare_curve(distance_bp, ps):
    """Return sorted, unique distances and mean positive P(s) values."""
    distance_bp = np.asarray(distance_bp, dtype=float)
    ps = np.asarray(ps, dtype=float)
    if distance_bp.shape != ps.shape:
        raise ValueError("distance_bp and ps must have the same shape.")

    valid = (
        np.isfinite(distance_bp)
        & np.isfinite(ps)
        & (distance_bp > 0)
        & (ps > 0)
    )
    distances = distance_bp[valid]
    values = ps[valid]
    if distances.size == 0:
        return distances, values

    unique_distances, inverse = np.unique(distances, return_inverse=True)
    mean_values = np.bincount(inverse, weights=values) / np.bincount(inverse)
    return unique_distances, mean_values


def calculate_gap(
    cis_distance_bp,
    cis_ps,
    trans_distance_bp,
    trans_ps,
    target_bp=40_000,
):
    """Return the cis/trans P(s) ratio at ``target_bp``.

    Cis and trans values are independently interpolated to the target
    distance. The returned gap is dimensionless.
    """
    if target_bp <= 0:
        raise ValueError("target_bp must be positive.")

    cis_distance_bp, cis_ps = _prepare_curve(cis_distance_bp, cis_ps)
    trans_distance_bp, trans_ps = _prepare_curve(trans_distance_bp, trans_ps)
    if cis_distance_bp.size < 2 or trans_distance_bp.size < 2:
        return np.nan
    if not (
        cis_distance_bp[0] <= target_bp <= cis_distance_bp[-1]
        and trans_distance_bp[0] <= target_bp <= trans_distance_bp[-1]
    ):
        return np.nan

    cis_at_target = np.interp(target_bp, cis_distance_bp, cis_ps)
    trans_at_target = np.interp(target_bp, trans_distance_bp, trans_ps)
    if trans_at_target <= 0:
        return np.nan

    return cis_at_target / trans_at_target


def calculate_slope(
    distance_bp,
    ps,
    target_bp=40_000,
    max_gap_bp=50_000,
):
    """Return the local log-log P(s) slope at or above ``target_bp``.

    The slope is calculated from the first two valid distance points at or
    above the target. Values at duplicate distances are averaged. The result
    is the signed slope ``d log(P) / d log(s)``; it is therefore normally
    negative for a decreasing P(s) curve.

    Parameters
    ----------
    distance_bp, ps : array-like
        Genomic distances in bp and the corresponding positive P(s) values.
    target_bp : float
        Lower distance bound for selecting the two points.
    max_gap_bp : float
        Maximum permitted separation between the selected points.

    Returns
    -------
    float
        Local log-log slope, or ``np.nan`` if it cannot be calculated.
    """
    if target_bp <= 0 or max_gap_bp <= 0:
        raise ValueError("target_bp and max_gap_bp must be positive.")

    distance_bp, ps = _prepare_curve(distance_bp, ps)
    eligible = distance_bp >= target_bp
    distance_bp = distance_bp[eligible]
    ps = ps[eligible]
    if distance_bp.size < 2:
        return np.nan

    distance_1, distance_2 = distance_bp[:2]
    if distance_2 - distance_1 > max_gap_bp:
        return np.nan

    value_1, value_2 = ps[:2]
    return np.log(value_2 / value_1) / np.log(distance_2 / distance_1)


def calculate_shoulder(
    distance_bp,
    cis_ps,
    search_min_bp=40_000,
    search_max_bp=10_000_000,
):
    """Return the cis P(s) shoulder position in bp.

    The log-log derivative is calculated over the complete valid curve. The
    shoulder is the distance at which that derivative is maximal within the
    requested search window.
    """
    if search_min_bp <= 0 or search_max_bp <= search_min_bp:
        raise ValueError(
            "Require 0 < search_min_bp < search_max_bp."
        )

    distance_bp, cis_ps = _prepare_curve(distance_bp, cis_ps)
    if distance_bp.size < 2:
        return np.nan

    derivative = np.gradient(np.log(cis_ps), np.log(distance_bp))
    in_window = (
        (distance_bp >= search_min_bp)
        & (distance_bp <= search_max_bp)
    )
    if not np.any(in_window):
        return np.nan

    window_indices = np.flatnonzero(in_window)
    maximum_index = window_indices[np.argmax(derivative[in_window])]
    return distance_bp[maximum_index]
