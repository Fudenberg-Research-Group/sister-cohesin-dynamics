"""Reusable scoring helpers for parameter inference from cis and trans P(s).

The functions implement the combined and trans-only log-space R-squared
calculations used for the initial sister-configuration, WT residence-time, and
WAPL-depletion parameter sweeps. Plotting and condition-specific file I/O
remain outside this module.
"""

import numpy as np


def combined_log_r2(
    dist_cis_sim,
    ps_cis_sim,
    dist_trans_sim,
    ps_trans_sim,
    dist_cis_exp,
    ps_cis_exp,
    dist_trans_exp,
    ps_trans_exp,
    sim_cis_norm_factor,
    exp_cis_norm_factor,
    start_dist=40_000,
    end_dist=5_000_000,
    n_log_points=500,
):
    """Calculate one combined cis/trans R-squared value in log P(s) space.

    Experimental and simulated curves are each normalized by their own cis
    P(s) value at the normalization distance supplied by the caller. The same
    cis normalization factor is applied to the corresponding trans curve.

    For cis and trans separately, experimental P(s) is averaged in bins around
    a log-spaced distance grid. Simulated P(s) is linearly interpolated in its
    original (non-log-transformed) units onto that grid. Valid cis and trans
    log10 P(s) values are then concatenated before calculating one R-squared.

    Parameters
    ----------
    dist_cis_sim, dist_trans_sim, dist_cis_exp, dist_trans_exp : array-like
        Genomic separations in bp.
    ps_cis_sim, ps_trans_sim, ps_cis_exp, ps_trans_exp : array-like
        Contact probabilities corresponding to the distance arrays.
    sim_cis_norm_factor, exp_cis_norm_factor : float
        Cis P(s) normalization factors, normally evaluated at 40 kb.
    start_dist, end_dist : float
        Inclusive genomic-distance range in bp used for scoring.
    n_log_points : int
        Number of points in the common log-spaced distance grid.

    Returns
    -------
    float
        Combined log-space R-squared, or ``np.nan`` when it cannot be
        calculated.
    """
    if sim_cis_norm_factor <= 0 or exp_cis_norm_factor <= 0:
        return np.nan
    if start_dist <= 0 or end_dist <= start_dist or n_log_points < 2:
        raise ValueError(
            "Require 0 < start_dist < end_dist and n_log_points >= 2."
        )

    def prepare_one(dist_sim, ps_sim_raw, dist_exp, ps_exp_raw):
        ps_sim_norm = np.asarray(ps_sim_raw) / sim_cis_norm_factor
        ps_exp_norm = np.asarray(ps_exp_raw) / exp_cis_norm_factor
        dist_sim = np.asarray(dist_sim)
        dist_exp = np.asarray(dist_exp)

        mask_exp = (
            (dist_exp >= start_dist)
            & (dist_exp <= end_dist)
            & (~np.isnan(ps_exp_norm))
            & (ps_exp_norm > 0)
        )
        if mask_exp.sum() < 2:
            return None, None

        mask_sim = (~np.isnan(ps_sim_norm)) & (ps_sim_norm > 0)
        if mask_sim.sum() < 2:
            return None, None

        grid_min = max(start_dist, dist_exp[mask_exp].min())
        grid_max = min(end_dist, dist_exp[mask_exp].max())
        if grid_min >= grid_max:
            return None, None

        log_x_grid = np.logspace(
            np.log10(grid_min), np.log10(grid_max), n_log_points
        )
        bin_edges = np.concatenate(
            (
                [log_x_grid[0] / 2],
                np.sqrt(log_x_grid[:-1] * log_x_grid[1:]),
                [log_x_grid[-1] * 2],
            )
        )

        exp_dist_filtered = dist_exp[mask_exp]
        exp_ps_filtered = ps_exp_norm[mask_exp]
        ps_exp_binned = np.full(n_log_points, np.nan)
        for index in range(n_log_points):
            in_bin = (
                (exp_dist_filtered >= bin_edges[index])
                & (exp_dist_filtered < bin_edges[index + 1])
            )
            if in_bin.sum() > 0:
                ps_exp_binned[index] = np.mean(exp_ps_filtered[in_bin])

        sort_index = np.argsort(dist_sim[mask_sim])
        sim_dist_sorted = dist_sim[mask_sim][sort_index]
        sim_ps_sorted = ps_sim_norm[mask_sim][sort_index]
        ps_sim_interpolated = np.interp(
            log_x_grid, sim_dist_sorted, sim_ps_sorted
        )

        valid = (
            (~np.isnan(ps_exp_binned))
            & (ps_exp_binned > 0)
            & (ps_sim_interpolated > 0)
        )
        if valid.sum() < 2:
            return None, None

        return (
            np.log10(ps_exp_binned[valid]),
            np.log10(ps_sim_interpolated[valid]),
        )

    log_exp_cis, log_sim_cis = prepare_one(
        dist_cis_sim, ps_cis_sim, dist_cis_exp, ps_cis_exp
    )
    log_exp_trans, log_sim_trans = prepare_one(
        dist_trans_sim, ps_trans_sim, dist_trans_exp, ps_trans_exp
    )
    if log_exp_cis is None or log_exp_trans is None:
        return np.nan

    log_exp_all = np.concatenate((log_exp_cis, log_exp_trans))
    log_sim_all = np.concatenate((log_sim_cis, log_sim_trans))

    residual_sum_squares = np.sum((log_exp_all - log_sim_all) ** 2)
    total_sum_squares = np.sum(
        (log_exp_all - np.mean(log_exp_all)) ** 2
    )
    if total_sum_squares == 0:
        return np.nan

    return 1 - residual_sum_squares / total_sum_squares


def trans_log_r2(
    dist_trans_sim,
    ps_trans_sim,
    dist_trans_exp,
    ps_trans_exp,
    sim_cis_norm_factor,
    exp_cis_norm_factor,
    start_dist=40_000,
    end_dist=10_000_000,
    n_log_points=1_000,
):
    """Calculate trans-only R-squared in log P(s) space.

    Experimental and simulated trans curves are normalized by their
    corresponding cis P(s) values, normally evaluated at 40 kb. Experimental
    trans P(s) is averaged in bins around a log-spaced distance grid, while
    simulated trans P(s) is linearly interpolated onto that grid before the
    log10 transformation and R-squared calculation.
    """
    if sim_cis_norm_factor <= 0 or exp_cis_norm_factor <= 0:
        return np.nan
    if start_dist <= 0 or end_dist <= start_dist or n_log_points < 2:
        raise ValueError(
            "Require 0 < start_dist < end_dist and n_log_points >= 2."
        )

    dist_sim = np.asarray(dist_trans_sim)
    ps_sim = np.asarray(ps_trans_sim) / sim_cis_norm_factor
    dist_exp = np.asarray(dist_trans_exp)
    ps_exp = np.asarray(ps_trans_exp) / exp_cis_norm_factor

    mask_exp = (
        (dist_exp >= start_dist)
        & (dist_exp <= end_dist)
        & np.isfinite(dist_exp)
        & np.isfinite(ps_exp)
        & (ps_exp > 0)
    )
    mask_sim = (
        np.isfinite(dist_sim)
        & np.isfinite(ps_sim)
        & (ps_sim > 0)
    )
    if mask_exp.sum() < 2 or mask_sim.sum() < 2:
        return np.nan

    grid_min = max(start_dist, dist_exp[mask_exp].min())
    grid_max = min(end_dist, dist_exp[mask_exp].max())
    if grid_min >= grid_max:
        return np.nan

    distance_grid = np.logspace(
        np.log10(grid_min), np.log10(grid_max), n_log_points
    )
    bin_edges = np.concatenate(
        (
            [distance_grid[0] / 2],
            np.sqrt(distance_grid[:-1] * distance_grid[1:]),
            [distance_grid[-1] * 2],
        )
    )

    exp_dist_filtered = dist_exp[mask_exp]
    exp_ps_filtered = ps_exp[mask_exp]
    exp_ps_binned = np.full(n_log_points, np.nan)
    for index in range(n_log_points):
        in_bin = (
            (exp_dist_filtered >= bin_edges[index])
            & (exp_dist_filtered < bin_edges[index + 1])
        )
        if in_bin.sum() > 0:
            exp_ps_binned[index] = np.mean(exp_ps_filtered[in_bin])

    sort_index = np.argsort(dist_sim[mask_sim])
    sim_dist_sorted = dist_sim[mask_sim][sort_index]
    sim_ps_sorted = ps_sim[mask_sim][sort_index]
    sim_ps_interpolated = np.interp(
        distance_grid, sim_dist_sorted, sim_ps_sorted
    )

    valid = (
        np.isfinite(exp_ps_binned)
        & (exp_ps_binned > 0)
        & np.isfinite(sim_ps_interpolated)
        & (sim_ps_interpolated > 0)
    )
    if valid.sum() < 2:
        return np.nan

    log_exp = np.log10(exp_ps_binned[valid])
    log_sim = np.log10(sim_ps_interpolated[valid])
    residual_sum_squares = np.sum((log_exp - log_sim) ** 2)
    total_sum_squares = np.sum((log_exp - np.mean(log_exp)) ** 2)
    if total_sum_squares == 0:
        return np.nan

    return 1 - residual_sum_squares / total_sum_squares


def score_trans_parameter_grid(
    ps_data,
    parameter_axes,
    dist_trans_exp,
    ps_trans_exp,
    exp_cis_norm_factor,
    start_dist=40_000,
    end_dist=10_000_000,
    n_log_points=1_000,
):
    """Calculate trans-only log-space R-squared on an N-dimensional grid.

    ``parameter_axes`` is an insertion-ordered mapping from parameter names to
    sampled values. ``ps_data`` must be keyed by tuples in that same order.
    Each entry must contain ``dist_trans``, ``ps_trans``, and ``norm_factor``.
    """
    if not parameter_axes:
        raise ValueError("parameter_axes must contain at least one axis.")

    parameter_names = list(parameter_axes)
    parameter_values = [list(parameter_axes[name]) for name in parameter_names]
    if any(len(values) == 0 for values in parameter_values):
        raise ValueError("Parameter axes must not be empty.")

    shape = tuple(len(values) for values in parameter_values)
    r2_matrix = np.full(shape, np.nan, dtype=float)

    for index in np.ndindex(shape):
        key = tuple(
            parameter_values[axis][value_index]
            for axis, value_index in enumerate(index)
        )
        data = ps_data.get(key)
        if data is None:
            continue

        r2_matrix[index] = trans_log_r2(
            dist_trans_sim=data["dist_trans"],
            ps_trans_sim=data["ps_trans"],
            dist_trans_exp=dist_trans_exp,
            ps_trans_exp=ps_trans_exp,
            sim_cis_norm_factor=data["norm_factor"],
            exp_cis_norm_factor=exp_cis_norm_factor,
            start_dist=start_dist,
            end_dist=end_dist,
            n_log_points=n_log_points,
        )

    if np.all(np.isnan(r2_matrix)):
        best_parameters = None
        best_r2 = np.nan
        relative_difference_from_best = np.full_like(r2_matrix, np.nan)
    else:
        best_index = np.unravel_index(
            np.nanargmax(r2_matrix), r2_matrix.shape
        )
        best_parameters = {
            name: parameter_values[axis][best_index[axis]]
            for axis, name in enumerate(parameter_names)
        }
        best_r2 = r2_matrix[best_index]
        if best_r2 == 0:
            relative_difference_from_best = np.full_like(r2_matrix, np.nan)
        else:
            relative_difference_from_best = (
                best_r2 - r2_matrix
            ) / abs(best_r2)

    return {
        "parameter_names": parameter_names,
        "parameter_values": dict(zip(parameter_names, parameter_values)),
        "start_dist": start_dist,
        "end_dist": end_dist,
        "n_log_points": n_log_points,
        "r2_matrix": r2_matrix,
        "best_parameters": best_parameters,
        "best_r2": best_r2,
        "relative_difference_from_best": relative_difference_from_best,
    }


def score_two_parameter_grid(
    ps_data,
    row_values,
    column_values,
    dist_cis_exp,
    ps_cis_exp,
    dist_trans_exp,
    ps_trans_exp,
    exp_cis_norm_factor,
    row_name="row_parameter",
    column_name="column_parameter",
    start_dist=40_000,
    end_dist=5_000_000,
    n_log_points=500,
):
    """Calculate combined log-space R-squared across a two-parameter grid.

    ``ps_data`` must be keyed by ``(row_value, column_value)``. Each value
    must contain ``dist_cis``, ``ps_cis``, ``dist_trans``, ``ps_trans``, and
    ``norm_factor``. The parameter names are labels for returned metadata and
    do not affect the calculation.

    Returns a dictionary containing the score matrix, its best sampled point,
    and each point's fractional decrease from the maximum. Missing parameter
    combinations remain ``np.nan``.
    """
    row_values = list(row_values)
    column_values = list(column_values)
    r2_matrix = np.full(
        (len(row_values), len(column_values)), np.nan, dtype=float
    )

    for row_index, row_value in enumerate(row_values):
        for column_index, column_value in enumerate(column_values):
            data = ps_data.get((row_value, column_value))
            if data is None:
                continue

            r2_matrix[row_index, column_index] = combined_log_r2(
                dist_cis_sim=data["dist_cis"],
                ps_cis_sim=data["ps_cis"],
                dist_trans_sim=data["dist_trans"],
                ps_trans_sim=data["ps_trans"],
                dist_cis_exp=dist_cis_exp,
                ps_cis_exp=ps_cis_exp,
                dist_trans_exp=dist_trans_exp,
                ps_trans_exp=ps_trans_exp,
                sim_cis_norm_factor=data["norm_factor"],
                exp_cis_norm_factor=exp_cis_norm_factor,
                start_dist=start_dist,
                end_dist=end_dist,
                n_log_points=n_log_points,
            )

    if np.all(np.isnan(r2_matrix)):
        best_row_value = None
        best_column_value = None
        best_r2 = np.nan
        relative_difference_from_best = np.full_like(r2_matrix, np.nan)
    else:
        best_index = np.unravel_index(
            np.nanargmax(r2_matrix), r2_matrix.shape
        )
        best_row_value = row_values[best_index[0]]
        best_column_value = column_values[best_index[1]]
        best_r2 = r2_matrix[best_index]
        if best_r2 == 0:
            relative_difference_from_best = np.full_like(r2_matrix, np.nan)
        else:
            relative_difference_from_best = (
                best_r2 - r2_matrix
            ) / abs(best_r2)

    best_parameters = None
    if best_row_value is not None:
        best_parameters = {
            row_name: best_row_value,
            column_name: best_column_value,
        }

    return {
        "row_name": row_name,
        "column_name": column_name,
        "row_values": row_values,
        "column_values": column_values,
        "start_dist": start_dist,
        "end_dist": end_dist,
        "n_log_points": n_log_points,
        "r2_matrix": r2_matrix,
        "best_parameters": best_parameters,
        "best_r2": best_r2,
        "relative_difference_from_best": relative_difference_from_best,
    }


def score_parameter_grid(
    ps_data,
    alphas,
    residence_times,
    dist_cis_exp,
    ps_cis_exp,
    dist_trans_exp,
    ps_trans_exp,
    exp_cis_norm_factor,
    start_dist=40_000,
    end_dist=5_000_000,
    n_log_points=500,
):
    """Calculate combined log-space R-squared across an alpha/tau grid.

    This backward-compatible wrapper retains the WT-specific result keys. New
    analyses can call :func:`score_two_parameter_grid` directly.
    """
    result = score_two_parameter_grid(
        ps_data=ps_data,
        row_values=alphas,
        column_values=residence_times,
        dist_cis_exp=dist_cis_exp,
        ps_cis_exp=ps_cis_exp,
        dist_trans_exp=dist_trans_exp,
        ps_trans_exp=ps_trans_exp,
        exp_cis_norm_factor=exp_cis_norm_factor,
        row_name="alpha",
        column_name="residence_time",
        start_dist=start_dist,
        end_dist=end_dist,
        n_log_points=n_log_points,
    )
    result.update({
        "alphas": alphas,
        "residence_times": residence_times,
        "best_alpha": (
            None
            if result["best_parameters"] is None
            else result["best_parameters"]["alpha"]
        ),
        "best_residence_time": (
            None
            if result["best_parameters"] is None
            else result["best_parameters"]["residence_time"]
        ),
    })
    return result
