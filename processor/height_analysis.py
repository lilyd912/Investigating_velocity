#height analysis

from __future__ import annotations

from dataclasses import dataclass
import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Point

from cross_river import (
    CrossSectionGeometry,
    select_cross_section_sample,
)

logger = logging.getLogger("river_height_analysis")

@dataclass(frozen=True)
class LocalHeightSummary:
    """
    Local PIXC height statistics for one anchor, plus an explicit NaN
    audit distinguishing "no points selected geometrically" from
    "points selected but height is unusable" from "too few valid
    points".
    """

    anchor_label: str
    radius_m: float

    n_candidates_before_height_filter: int
    n_valid_height_points: int
    n_nan_height_points: int

    mean_height_m: float
    median_height_m: float
    standard_deviation_m: float
    minimum_height_m: float
    maximum_height_m: float
    mad_m: float
    standard_error_m: float

    anchor_x_m: float
    anchor_y_m: float

    sample_status: str  # "ok" | "insufficient_points" | "no_points"


def summarise_local_height(
    sample: gpd.GeoDataFrame,
    anchor_label: str,
    radius_m: float,
    height_column: str,
    anchor: Point,
    minimum_sample_points: int,
) -> LocalHeightSummary:
    """
    Compute robust local height statistics for one anchor's PIXC
    sample, with an explicit NaN audit and no NumPy warnings.

    Behaviour is fully explicit at every point count:
      - 0 valid heights  -> all statistics NaN, sample_status="no_points".
      - 1 valid height   -> mean/median are that value; standard
        deviation and standard error are NaN (ddof=1 needs >= 2
        points); sample_status reflects minimum_sample_points.
      - >= 2 valid heights -> full statistics using ddof=1.
      - Any count below minimum_sample_points still returns whatever
        statistics could be calculated (never hidden), but is marked
        sample_status="insufficient_points" so quality can be judged
        separately from raw availability.
    """

    n_candidates_before_height_filter = int(len(sample))

    if n_candidates_before_height_filter == 0:
        return LocalHeightSummary(
            anchor_label=anchor_label,
            radius_m=radius_m,
            n_candidates_before_height_filter=0,
            n_valid_height_points=0,
            n_nan_height_points=0,
            mean_height_m=np.nan,
            median_height_m=np.nan,
            standard_deviation_m=np.nan,
            minimum_height_m=np.nan,
            maximum_height_m=np.nan,
            mad_m=np.nan,
            standard_error_m=np.nan,
            anchor_x_m=float(anchor.x),
            anchor_y_m=float(anchor.y),
            sample_status="no_points",
        )

    raw_heights = pd.to_numeric(sample[height_column], errors="coerce")
    heights = raw_heights.replace([np.inf, -np.inf], np.nan).dropna()

    n_valid_height_points = int(len(heights))
    n_nan_height_points = (
        n_candidates_before_height_filter - n_valid_height_points
    )

    if n_valid_height_points == 0:
        return LocalHeightSummary(
            anchor_label=anchor_label,
            radius_m=radius_m,
            n_candidates_before_height_filter=n_candidates_before_height_filter,
            n_valid_height_points=0,
            n_nan_height_points=n_nan_height_points,
            mean_height_m=np.nan,
            median_height_m=np.nan,
            standard_deviation_m=np.nan,
            minimum_height_m=np.nan,
            maximum_height_m=np.nan,
            mad_m=np.nan,
            standard_error_m=np.nan,
            anchor_x_m=float(anchor.x),
            anchor_y_m=float(anchor.y),
            sample_status="no_points",
        )

    mean_height_m = float(heights.mean())
    median_height_m = float(heights.median())
    minimum_height_m = float(heights.min())
    maximum_height_m = float(heights.max())
    mad_m = float((heights - median_height_m).abs().median())

    if n_valid_height_points >= 2:
        standard_deviation_m = float(heights.std(ddof=1))
        standard_error_m = standard_deviation_m / np.sqrt(n_valid_height_points)
    else:
        # A single point has no meaningful spread; return NaN rather
        # than a misleading 0.0 or a ddof<=0 warning.
        standard_deviation_m = np.nan
        standard_error_m = np.nan

    if n_valid_height_points >= minimum_sample_points:
        sample_status = "ok"
    else:
        sample_status = "insufficient_points"

    return LocalHeightSummary(
        anchor_label=anchor_label,
        radius_m=radius_m,
        n_candidates_before_height_filter=n_candidates_before_height_filter,
        n_valid_height_points=n_valid_height_points,
        n_nan_height_points=n_nan_height_points,
        mean_height_m=mean_height_m,
        median_height_m=median_height_m,
        standard_deviation_m=standard_deviation_m,
        minimum_height_m=minimum_height_m,
        maximum_height_m=maximum_height_m,
        mad_m=mad_m,
        standard_error_m=standard_error_m,
        anchor_x_m=float(anchor.x),
        anchor_y_m=float(anchor.y),
        sample_status=sample_status,
    )


def _count_shared_points(
    sample_a: gpd.GeoDataFrame, sample_b: gpd.GeoDataFrame
) -> int:
    """Count PIXC points (by index) shared between two anchor samples."""

    return int(len(sample_a.index.intersection(sample_b.index)))


def _safe_delta(a: float, b: float) -> float:
    """Return a - b, or NaN if either input is NaN (sign is meaningful,
    so no absolute value is taken)."""

    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a - b)


def _safe_propagated_sem(sem_a: float, sem_b: float) -> float:
    """
    Approximate propagated standard error for a difference of two
    means, assuming independent samples:
        sem(a - b) = sqrt(sem_a^2 + sem_b^2)
    This does NOT capture spatially correlated SWOT measurement
    errors and is not a full instrument-uncertainty budget.
    """

    if sem_a is None or sem_b is None or np.isnan(sem_a) or np.isnan(sem_b):
        return np.nan
    return float(np.sqrt(sem_a ** 2 + sem_b ** 2)) #can fix actually, need to sep


@dataclass(frozen=True)
class DeltaHeightResult:
    """Full local-sampling and delta-height result for one reach."""

    radius_m: float

    negative_edge: LocalHeightSummary
    centre: LocalHeightSummary
    positive_edge: LocalHeightSummary

    # Mean-based deltas (primary statistic used in the output table).
    delta_h_centre_minus_negative_m: float
    delta_h_centre_minus_positive_m: float
    delta_h_positive_minus_negative_m: float

    # Median-based deltas (more robust to anomalous PIXC heights).
    delta_h_centre_minus_negative_median_m: float
    delta_h_centre_minus_positive_median_m: float
    delta_h_positive_minus_negative_median_m: float

    # Propagated SEM on the mean-based deltas (see _safe_propagated_sem).
    delta_h_centre_minus_negative_sem_m: float
    delta_h_centre_minus_positive_sem_m: float
    delta_h_positive_minus_negative_sem_m: float

    negative_passes_minimum: bool
    centre_passes_minimum: bool
    positive_passes_minimum: bool

    quality_status: str  # "pass" | "fail"

    neighbourhoods_overlap: bool
    n_shared_negative_centre: int
    n_shared_centre_positive: int
    n_shared_negative_positive: int


def analyse_cross_section_height(
    pixc_points: gpd.GeoDataFrame,
    geometry: CrossSectionGeometry,
    radius_m: float,
    minimum_sample_points: int,
    height_column: str,
    centre_negative_cutoff_m: float,
    centre_positive_cutoff_m: float,
) -> DeltaHeightResult:
    """
    Select the three non-overlapping sampling regions for one reach,
    summarise local height at each, and compute signed height
    differences (delta h), using both the mean and the median.
    """

    negative_sample = select_cross_section_sample(
        points=pixc_points,
        geometry=geometry,
        anchor_name="negative_edge",
        radius_m=radius_m,
        centre_negative_cutoff_m=centre_negative_cutoff_m,
        centre_positive_cutoff_m=centre_positive_cutoff_m,
    )
    centre_sample = select_cross_section_sample(
        points=pixc_points,
        geometry=geometry,
        anchor_name="centre",
        radius_m=radius_m,
        centre_negative_cutoff_m=centre_negative_cutoff_m,
        centre_positive_cutoff_m=centre_positive_cutoff_m,
    )
    positive_sample = select_cross_section_sample(
        points=pixc_points,
        geometry=geometry,
        anchor_name="positive_edge",
        radius_m=radius_m,
        centre_negative_cutoff_m=centre_negative_cutoff_m,
        centre_positive_cutoff_m=centre_positive_cutoff_m,
    )

    negative_summary = summarise_local_height(
        negative_sample, "negative_edge", radius_m, height_column,
        geometry.negative_edge, minimum_sample_points,
    )
    centre_summary = summarise_local_height(
        centre_sample, "centre", radius_m, height_column,
        geometry.centre, minimum_sample_points,
    )
    positive_summary = summarise_local_height(
        positive_sample, "positive_edge", radius_m, height_column,
        geometry.positive_edge, minimum_sample_points,
    )

    delta_h_centre_minus_negative_m = _safe_delta(
        centre_summary.mean_height_m, negative_summary.mean_height_m
    )
    delta_h_centre_minus_positive_m = _safe_delta(
        centre_summary.mean_height_m, positive_summary.mean_height_m
    )
    delta_h_positive_minus_negative_m = _safe_delta(
        positive_summary.mean_height_m, negative_summary.mean_height_m
    )

    delta_h_centre_minus_negative_median_m = _safe_delta(
        centre_summary.median_height_m, negative_summary.median_height_m
    )
    delta_h_centre_minus_positive_median_m = _safe_delta(
        centre_summary.median_height_m, positive_summary.median_height_m
    )
    delta_h_positive_minus_negative_median_m = _safe_delta(
        positive_summary.median_height_m, negative_summary.median_height_m
    )

    delta_h_centre_minus_negative_sem_m = _safe_propagated_sem(
        centre_summary.standard_error_m, negative_summary.standard_error_m
    )
    delta_h_centre_minus_positive_sem_m = _safe_propagated_sem(
        centre_summary.standard_error_m, positive_summary.standard_error_m
    )
    delta_h_positive_minus_negative_sem_m = _safe_propagated_sem(
        positive_summary.standard_error_m, negative_summary.standard_error_m
    )

    negative_passes_minimum = negative_summary.sample_status == "ok"
    centre_passes_minimum = centre_summary.sample_status == "ok"
    positive_passes_minimum = positive_summary.sample_status == "ok"

    quality_status = (
        "pass"
        if (negative_passes_minimum and centre_passes_minimum and positive_passes_minimum)
        else "fail"
    )

    n_shared_negative_centre = _count_shared_points(negative_sample, centre_sample)
    n_shared_centre_positive = _count_shared_points(centre_sample, positive_sample)
    n_shared_negative_positive = _count_shared_points(negative_sample, positive_sample)

    neighbourhoods_overlap = (
        n_shared_negative_centre > 0
        or n_shared_centre_positive > 0
        or n_shared_negative_positive > 0
    )

    if neighbourhoods_overlap:
        # The three across-river intervals are mutually exclusive by
        # construction, so this should never actually fire; if it
        # does, it signals a bug rather than expected behaviour, and
        # is surfaced loudly rather than silently ignored.
        logger.warning(
            "Unexpected sample overlap detected (neg-centre=%d, "
            "centre-pos=%d, neg-pos=%d) despite mutually exclusive "
            "across-river cutoffs -- please investigate.",
            n_shared_negative_centre,
            n_shared_centre_positive,
            n_shared_negative_positive,
        )

    if quality_status != "pass":
        logger.warning(
            "Radius %.1f m: quality_status='fail' (n_negative=%d, "
            "n_centre=%d, n_positive=%d, minimum_sample_points=%d)",
            radius_m,
            negative_summary.n_valid_height_points,
            centre_summary.n_valid_height_points,
            positive_summary.n_valid_height_points,
            minimum_sample_points,
        )

    return DeltaHeightResult(
        radius_m=radius_m,
        negative_edge=negative_summary,
        centre=centre_summary,
        positive_edge=positive_summary,
        delta_h_centre_minus_negative_m=delta_h_centre_minus_negative_m,
        delta_h_centre_minus_positive_m=delta_h_centre_minus_positive_m,
        delta_h_positive_minus_negative_m=delta_h_positive_minus_negative_m,
        delta_h_centre_minus_negative_median_m=delta_h_centre_minus_negative_median_m,
        delta_h_centre_minus_positive_median_m=delta_h_centre_minus_positive_median_m,
        delta_h_positive_minus_negative_median_m=delta_h_positive_minus_negative_median_m,
        delta_h_centre_minus_negative_sem_m=delta_h_centre_minus_negative_sem_m,
        delta_h_centre_minus_positive_sem_m=delta_h_centre_minus_positive_sem_m,
        delta_h_positive_minus_negative_sem_m=delta_h_positive_minus_negative_sem_m,
        negative_passes_minimum=negative_passes_minimum,
        centre_passes_minimum=centre_passes_minimum,
        positive_passes_minimum=positive_passes_minimum,
        quality_status=quality_status,
        neighbourhoods_overlap=neighbourhoods_overlap,
        n_shared_negative_centre=n_shared_negative_centre,
        n_shared_centre_positive=n_shared_centre_positive,
        n_shared_negative_positive=n_shared_negative_positive,
    )


