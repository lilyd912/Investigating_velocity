#cross river

from __future__ import annotations

from dataclasses import dataclass
import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import (
    LineString,
    Point,
)

logger = logging.getLogger("river_height_analysis")

class ReachProcessingError(Exception):
    """
    Raised for a hard failure while processing one reach (e.g. no
    nearby PIXC points, no points on one side of the river). Carries a
    machine-readable failure_code alongside the human-readable
    message, so the batch processor can record both without having to
    parse exception text.
    """

    def __init__(self, failure_code: str, message: str):
        super().__init__(message)
        self.failure_code = failure_code
        self.message = message


def _unit_vector_deg(angle_deg: float) -> tuple[float, float]:
    """Return the (ux, uy) unit vector pointing along `angle_deg`."""

    return (
        float(np.cos(np.radians(angle_deg))),
        float(np.sin(np.radians(angle_deg))),
    )

def build_cross_section_line(
    centre_point: Point,
    river_angle_deg: float,
    half_length_m: float,
) -> LineString:
    """
    Construct a straight line through `centre_point`, perpendicular to
    the local river direction, extending `half_length_m` on each side.
    Used for plotting and as the reference direction for edge
    detection; edges themselves come from PIXC points, not from
    intersecting this line with any polygon.

    Angle convention: cross_section_angle_deg = river_angle_deg + 90.
    """

    if half_length_m <= 0:
        raise ValueError("half_length_m must be greater than zero.")

    if not isinstance(centre_point, Point) or centre_point.is_empty:
        raise ValueError("centre_point must be a non-empty shapely Point.")

    cross_section_angle_deg = river_angle_deg + 90.0
    ux, uy = _unit_vector_deg(cross_section_angle_deg)

    dx = half_length_m * ux
    dy = half_length_m * uy

    return LineString([
        (centre_point.x - dx, centre_point.y - dy),
        (centre_point.x + dx, centre_point.y + dy),
    ])


@dataclass(frozen=True)
class CrossSectionAnchors:
    """The three raw anchor locations found along one cross-section."""

    centre: Point
    negative_edge: Point
    positive_edge: Point


@dataclass(frozen=True)
class CrossSectionGeometry:
    """
    Validated cross-section geometry for one reach.

    `negative_edge_distance_m` / `positive_edge_distance_m` are the
    (positive) distances in metres from the centre to each PIXC-
    derived edge; `river_width_m` is their sum.
    """

    centre: Point
    negative_edge: Point
    positive_edge: Point
    cross_section_line: LineString
    river_angle_deg: float
    cross_section_angle_deg: float
    negative_edge_distance_m: float
    positive_edge_distance_m: float
    river_width_m: float
    n_anchor_candidates: int


def find_cross_section_anchors(
    pixc_points: gpd.GeoDataFrame,
    centre_point: Point,
    cross_section_angle_deg: float,
    strip_half_width_m: float,
    search_half_length_m: float,
    edge_method: str,
    edge_lower_percentile: float,
    edge_upper_percentile: float,
) -> tuple[CrossSectionAnchors, int]:
    """
    Derive a centre anchor and two river-edge anchors directly from
    nearby PIXC points, by projecting them onto the across-river axis
    and taking the extreme (or, by default, a robust percentile)
    signed distance on each side.

    Robustness
    -----------
    - Requires at least 2 candidate points in the search strip.
    - Requires at least 1 point strictly on each side of the centre.
    - `edge_method="percentile"` (the default) uses
      `edge_lower_percentile` / `edge_upper_percentile` instead of the
      true min/max, so a single extreme outlier point cannot swing the
      whole river-width estimate. `edge_method="minmax"` reproduces
      the original true-min/max definition.
    - Duplicate points do not need special handling: both minmax and
      percentile are well-defined for repeated values.
    """

    ux, uy = _unit_vector_deg(cross_section_angle_deg)
    vx, vy = -uy, ux  # unit vector along the cross-section line

    x = pixc_points.geometry.x.to_numpy()
    y = pixc_points.geometry.y.to_numpy()

    dx = x - centre_point.x
    dy = y - centre_point.y

    across_distance_m = dx * ux + dy * uy
    along_distance_m = dx * vx + dy * vy

    in_strip = (
        (np.abs(along_distance_m) <= strip_half_width_m)
        & (np.abs(across_distance_m) <= search_half_length_m)
    )

    candidate_across_distances = across_distance_m[in_strip]
    n_anchor_candidates = int(len(candidate_across_distances))

    if n_anchor_candidates < 2:
        raise ReachProcessingError(
            "NO_ANCHOR_CANDIDATES",
            "Too few PIXC points were found near the cross-section "
            f"(found {n_anchor_candidates}, need >= 2).",
        )

    negative_distances = candidate_across_distances[
        candidate_across_distances < 0
    ]
    positive_distances = candidate_across_distances[
        candidate_across_distances > 0
    ]

    if len(negative_distances) == 0:
        raise ReachProcessingError(
            "NO_NEGATIVE_SIDE_POINTS",
            "No PIXC points were found on the negative side of the "
            "cross-section centre.",
        )

    if len(positive_distances) == 0:
        raise ReachProcessingError(
            "NO_POSITIVE_SIDE_POINTS",
            "No PIXC points were found on the positive side of the "
            "cross-section centre.",
        )

    if edge_method == "minmax":
        negative_edge_distance = float(negative_distances.min())
        positive_edge_distance = float(positive_distances.max())
    else:  # "percentile"
        negative_edge_distance = float(
            np.percentile(negative_distances, edge_lower_percentile)
        )
        positive_edge_distance = float(
            np.percentile(positive_distances, edge_upper_percentile)
        )

    negative_edge = Point(
        centre_point.x + negative_edge_distance * ux,
        centre_point.y + negative_edge_distance * uy,
    )
    positive_edge = Point(
        centre_point.x + positive_edge_distance * ux,
        centre_point.y + positive_edge_distance * uy,
    )

    logger.info(
        "PIXC-derived edges (%s): %.2f m negative, %.2f m positive, "
        "using %d nearby candidates.",
        edge_method,
        negative_edge_distance,
        positive_edge_distance,
        n_anchor_candidates,
    )

    return CrossSectionAnchors(
        centre=centre_point,
        negative_edge=negative_edge,
        positive_edge=positive_edge,
    ), n_anchor_candidates


def validate_cross_section_geometry(
    anchors: CrossSectionAnchors,
    n_anchor_candidates: int,
    cross_section_line: LineString,
    river_angle_deg: float,
    cross_section_angle_deg: float,
) -> CrossSectionGeometry:
    """Validate raw anchors and package them into CrossSectionGeometry."""

    for label, point in (
        ("centre", anchors.centre),
        ("negative_edge", anchors.negative_edge),
        ("positive_edge", anchors.positive_edge),
    ):
        if not isinstance(point, Point) or point.is_empty:
            raise ReachProcessingError(
                "INVALID_GEOMETRY", f"Cross-section anchor '{label}' is invalid."
            )
        if not (np.isfinite(point.x) and np.isfinite(point.y)):
            raise ReachProcessingError(
                "INVALID_GEOMETRY",
                f"Cross-section anchor '{label}' has non-finite coordinates.",
            )

    ux, uy = _unit_vector_deg(cross_section_angle_deg)

    def _signed_distance(point: Point) -> float:
        return (
            (point.x - anchors.centre.x) * ux
            + (point.y - anchors.centre.y) * uy
        )

    negative_signed_distance_m = _signed_distance(anchors.negative_edge)
    positive_signed_distance_m = _signed_distance(anchors.positive_edge)

    if negative_signed_distance_m >= 0:
        raise ReachProcessingError(
            "INVALID_GEOMETRY",
            "negative_edge does not have a negative signed distance from "
            "the centre; cross-section geometry is inconsistent.",
        )

    if positive_signed_distance_m <= 0:
        raise ReachProcessingError(
            "INVALID_GEOMETRY",
            "positive_edge does not have a positive signed distance from "
            "the centre; cross-section geometry is inconsistent.",
        )

    negative_edge_distance_m = abs(negative_signed_distance_m)
    positive_edge_distance_m = abs(positive_signed_distance_m)
    river_width_m = negative_edge_distance_m + positive_edge_distance_m

    if not np.isfinite(river_width_m) or river_width_m <= 0:
        raise ReachProcessingError(
            "INVALID_GEOMETRY",
            f"Derived river width is not physically plausible: "
            f"{river_width_m!r} m.",
        )

    return CrossSectionGeometry(
        centre=anchors.centre,
        negative_edge=anchors.negative_edge,
        positive_edge=anchors.positive_edge,
        cross_section_line=cross_section_line,
        river_angle_deg=river_angle_deg,
        cross_section_angle_deg=cross_section_angle_deg,
        negative_edge_distance_m=negative_edge_distance_m,
        positive_edge_distance_m=positive_edge_distance_m,
        river_width_m=river_width_m,
        n_anchor_candidates=n_anchor_candidates,
    )

def select_points_within_radius(
    points: gpd.GeoDataFrame,
    centre: Point,
    radius_m: float,
) -> gpd.GeoDataFrame:
    """
    Return every point in `points` within `radius_m` metres of
    `centre` (planar distance in a projected CRS). Selects ALL valid
    points inside a fixed physical radius; never a fixed nearest-
    neighbour count.
    """

    if points.crs is None or not points.crs.is_projected:
        raise ValueError(
            "select_points_within_radius requires points in a projected "
            "CRS for metre-based distances."
        )

    if radius_m <= 0:
        raise ValueError("radius_m must be greater than zero.")

    distance_m = points.geometry.distance(centre)
    selected = points.loc[distance_m <= radius_m].copy()
    selected["distance_to_anchor_m"] = distance_m.loc[selected.index]

    return selected


def select_cross_section_sample(
    points: gpd.GeoDataFrame,
    geometry: CrossSectionGeometry,
    anchor_name: str,
    radius_m: float,
    centre_negative_cutoff_m: float,
    centre_positive_cutoff_m: float,
) -> gpd.GeoDataFrame:
    """
    Select PIXC points for one cross-section anchor ('negative_edge',
    'centre', or 'positive_edge').

    A point is retained only if it satisfies ALL of:
      1. lies within the anchor's circular sampling radius;
      2. lies inside the PIXC-observed river width (between the
         negative and positive edges derived in
         find_cross_section_anchors);
      3. lies inside the across-river interval configured for this
         anchor, split by centre_negative_cutoff_m /
         centre_positive_cutoff_m.

    Because the three anchors' across-river intervals
    (< negative cutoff, between the cutoffs, > positive cutoff) are
    mutually exclusive by construction, the three resulting samples
    cannot overlap -- this is verified downstream by counting actually
    shared point indices rather than assumed.
    """

    if centre_negative_cutoff_m >= centre_positive_cutoff_m:
        raise ValueError(
            "centre_negative_cutoff_m must be smaller than "
            "centre_positive_cutoff_m."
        )

    anchor_lookup = {
        "negative_edge": geometry.negative_edge,
        "centre": geometry.centre,
        "positive_edge": geometry.positive_edge,
    }
    if anchor_name not in anchor_lookup:
        raise ValueError(
            "anchor_name must be 'negative_edge', 'centre', or "
            "'positive_edge'."
        )

    anchor = anchor_lookup[anchor_name]

    # 1. Circular radius.
    selected = select_points_within_radius(points, anchor, radius_m)

    if selected.empty:
        selected["across_distance_m"] = pd.Series(dtype=float)
        return selected

    # Signed across-river coordinate, measured from the *centre*
    # (not from the anchor), so all three samples share one consistent
    # across-river axis.
    ux, uy = _unit_vector_deg(geometry.cross_section_angle_deg)
    dx = selected.geometry.x - geometry.centre.x
    dy = selected.geometry.y - geometry.centre.y
    selected["across_distance_m"] = dx * ux + dy * uy

    # 2. Inside the PIXC-observed river width.
    negative_river_limit_m = -geometry.negative_edge_distance_m
    positive_river_limit_m = geometry.positive_edge_distance_m

    inside_river = (
        (selected["across_distance_m"] >= negative_river_limit_m)
        & (selected["across_distance_m"] <= positive_river_limit_m)
    )
    selected = selected.loc[inside_river].copy()

    if selected.empty:
        return selected

    # 3. Inside this anchor's configured across-river region.
    if anchor_name == "negative_edge":
        region_mask = selected["across_distance_m"] < centre_negative_cutoff_m
    elif anchor_name == "centre":
        region_mask = (
            (selected["across_distance_m"] >= centre_negative_cutoff_m)
            & (selected["across_distance_m"] <= centre_positive_cutoff_m)
        )
    else:  # positive_edge
        region_mask = selected["across_distance_m"] > centre_positive_cutoff_m

    return selected.loc[region_mask].copy()

