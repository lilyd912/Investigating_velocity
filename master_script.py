# -*- coding: utf-8 -*-
"""investigating_height_across_river_17_07.py

Batch SWOT PIXC cross-river height analysis.

WORKFLOW SUMMARY
-----------------
1. Load PIXC heights, river centreline, river boundary/AOI and the
   satellite ground track; reproject everything into one projected
   (metre-based) CRS.
2. Clean the PIXC height values, auditing how many rows are removed
   at each stage (null/empty geometry, non-finite coordinates,
   non-finite height).
3. Split the river centreline into segments, compute each segment's
   orientation, and find every *stable* run of consecutive segments
   that is approximately perpendicular to the satellite ground track.
4. For every such run (not just the one nearest the AOI), build a
   local cross-section, derive the river edges directly from nearby
   PIXC points, and sample three genuinely non-overlapping regions:
   negative edge, centre, and positive edge.
5. Compute local height statistics and centre/edge height differences
   for every reach, continuing past individual reach failures and
   recording a machine-readable failure code and human-readable
   message for each one.
6. Save one row per attempted reach (successful or failed) to CSV,
   split successful/failed tables, print a terminal summary, and
   optionally save diagnostic plots for a handful of reaches.

THIS SCRIPT VS. YOUR PLANNED MODULE SPLIT
-----------------
This is one file so it is easy to run and sanity-check end-to-end,
but it is organised into clearly labelled sections that map directly
onto the module breakdown you mentioned:

    SECTION: config.py       -> HeightAnalysisConfig + validation
    SECTION: io_utils.py     -> file validation, loading, projection,
                                 height cleaning
    SECTION: geometry.py     -> centreline segmentation, angles,
                                 cross-section + edge-anchor geometry
    SECTION: sampling.py     -> radius- and cutoff-based point
                                 selection
    SECTION: statistics.py   -> local height summaries and delta-h
    SECTION: processor.py    -> per-reach orchestration and the
                                 all-reaches batch loop
    SECTION: plotting.py     -> optional diagnostic figures
    SECTION: run_analysis.py -> the single entry point,
                                 run_height_analysis(config)

Every function takes its inputs as explicit arguments and returns its
outputs; nothing depends on notebook state, module-level side effects,
or hidden globals. The only thing executed at import time is function
and dataclass *definitions*. All the actual work happens inside
run_height_analysis(config), called from the `if __name__ == "__main__"`
block at the bottom.

KEY SCIENTIFIC / METHODOLOGICAL NOTES
-----------------
- "Perpendicular" means the undirected angle difference between a
  river segment and the satellite ground track is within
  `angle_tolerance_deg` of 90 degrees, sustained for at least
  `minimum_run_length` consecutive segments (isolated perpendicular
  segments are treated as noise and discarded).
- River edges are derived directly from nearby PIXC points projected
  onto the across-river axis (NOT from intersecting the AOI polygon).
  By default this uses a robust percentile (edge_method="percentile")
  rather than the raw min/max, because a single stray/outlier PIXC
  point can otherwise swing the whole river-width estimate; true
  min/max (edge_method="minmax") remains available since that was the
  original requested definition.
- The three sampling regions (negative edge / centre / positive edge)
  are defined by two configurable cutoff lines
  (`centre_negative_cutoff_m`, `centre_positive_cutoff_m`) measured
  across the river from the centreline. Because the three regions are
  built from mutually exclusive across-river intervals, they cannot
  overlap by construction -- but shared-point counts are still
  computed explicitly from the final selected point indices, as a
  concrete check rather than an assumption.
- A comparison sample must additionally lie inside the PIXC-observed
  river width (between the derived negative/positive edges) and
  inside its configured circular search radius.
- No claim is made that any measured height difference is caused by
  water velocity; this script measures height only.
"""

from __future__ import annotations

# ============================================================
# SECTION: config.py -- imports
# ============================================================

from dataclasses import dataclass
from pathlib import Path
import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shapely.geometry import LineString, Point

logger = logging.getLogger("river_height_analysis")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ============================================================
# SECTION: config.py -- configuration dataclass
# ============================================================

@dataclass(frozen=True)
class HeightAnalysisConfig:
    """
    Centralised, immutable configuration for the batch SWOT PIXC
    cross-river height analysis. All tunable parameters live here so
    the pipeline can be adapted to another river or PIXC file by
    editing only this object. Validated in __post_init__ so a bad
    configuration fails immediately with a clear message, rather than
    partway through a long batch run.
    """

    # --- input files ---
    pixc_file: Path
    centreline_file: Path
    river_boundary_file: Path
    ground_track_file: Path

    # --- output location ---
    output_directory: Path

    # --- coordinate reference system ---
    # A projected (metre-based) CRS is required for every distance,
    # angle and area calculation in this pipeline.
    target_crs: str = "EPSG:32632"

    # --- PIXC columns ---
    height_column: str = "height"

    # --- perpendicular-reach identification ---
    angle_tolerance_deg: float = 20.0
    minimum_run_length: int = 3

    # --- local PIXC subset used for one reach ---
    # PIXC points further than this from a reach's centre point are
    # excluded before any anchor-finding or sampling for that reach.
    local_search_radius_m: float = 500.0

    # --- cross-section line construction (used for plotting only;
    # edges are now derived from PIXC points, not from intersecting
    # this line with the AOI boundary) ---
    cross_section_half_length_m: float = 500.0

    # --- PIXC-derived edge detection ---
    # Half-width (m) of the narrow strip either side of the
    # cross-section line within which nearby PIXC points are
    # considered candidates for edge detection.
    anchor_search_strip_half_width_m: float = 10.0

    # "minmax" uses the true minimum/maximum signed across-river
    # coordinate of candidate points (the original requested
    # definition, but sensitive to a single outlier point).
    # "percentile" uses a robust percentile instead, which is the
    # safer default.
    edge_method: str = "percentile"
    edge_lower_percentile: float = 2.0
    edge_upper_percentile: float = 98.0

    # --- local height sampling ---
    sample_radius_m: float = 20.0
    minimum_sample_points: int = 5

    # Across-river cutoff positions (m) measured from the centreline
    # along the cross-section direction. Negative values are on the
    # negative side; positive values are on the positive side. These
    # two lines partition the river into three mutually exclusive
    # regions: negative edge (< centre_negative_cutoff_m), centre
    # (between the two cutoffs), and positive edge
    # (> centre_positive_cutoff_m).
    centre_negative_cutoff_m: float = -10.0
    centre_positive_cutoff_m: float = 10.0

    # --- diagnostic plotting ---
    save_figures: bool = True
    # Zoom half-width (m) used for the spatial diagnostic plots.
    cross_section_plot_zoom_m: float = 200.0
    # Upper limit on how many per-reach diagnostic figures to save,
    # so a batch of ~40 reaches does not silently dump ~40 figures.
    # Ignored if diagnostic_reach_segment_indices is set.
    max_diagnostic_reach_plots: int = 5
    # Explicit set of segment_index values to plot instead of the
    # first max_diagnostic_reach_plots successful reaches. None means
    # "use max_diagnostic_reach_plots".
    diagnostic_reach_segment_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _validate_config(self)


def _validate_config(config: HeightAnalysisConfig) -> None:
    """Raise a clear ValueError for any inconsistent configuration."""

    if not config.target_crs:
        raise ValueError("target_crs must be a non-empty CRS string.")

    if not config.height_column:
        raise ValueError("height_column must be a non-empty column name.")

    if not (0.0 < config.angle_tolerance_deg <= 90.0):
        raise ValueError(
            "angle_tolerance_deg must be in the range (0, 90], got "
            f"{config.angle_tolerance_deg!r}."
        )

    if config.minimum_run_length < 1:
        raise ValueError(
            "minimum_run_length must be >= 1, got "
            f"{config.minimum_run_length!r}."
        )

    if config.local_search_radius_m <= 0:
        raise ValueError("local_search_radius_m must be > 0.")

    if config.cross_section_half_length_m <= 0:
        raise ValueError("cross_section_half_length_m must be > 0.")

    if config.anchor_search_strip_half_width_m <= 0:
        raise ValueError("anchor_search_strip_half_width_m must be > 0.")

    if config.edge_method not in ("minmax", "percentile"):
        raise ValueError(
            "edge_method must be 'minmax' or 'percentile', got "
            f"{config.edge_method!r}."
        )

    if config.edge_method == "percentile":
        if not (0.0 <= config.edge_lower_percentile < 50.0):
            raise ValueError(
                "edge_lower_percentile must be in [0, 50), got "
                f"{config.edge_lower_percentile!r}."
            )
        if not (50.0 < config.edge_upper_percentile <= 100.0):
            raise ValueError(
                "edge_upper_percentile must be in (50, 100], got "
                f"{config.edge_upper_percentile!r}."
            )

    if config.sample_radius_m <= 0:
        raise ValueError("sample_radius_m must be > 0.")

    if config.minimum_sample_points < 1:
        raise ValueError("minimum_sample_points must be >= 1.")

    if config.centre_negative_cutoff_m >= config.centre_positive_cutoff_m:
        raise ValueError(
            "centre_negative_cutoff_m must be smaller than "
            "centre_positive_cutoff_m, got "
            f"{config.centre_negative_cutoff_m!r} >= "
            f"{config.centre_positive_cutoff_m!r}."
        )

    if config.cross_section_plot_zoom_m <= 0:
        raise ValueError("cross_section_plot_zoom_m must be > 0.")

    if config.max_diagnostic_reach_plots < 0:
        raise ValueError("max_diagnostic_reach_plots must be >= 0.")


# ============================================================
# SECTION: io_utils.py -- output dirs, validation, loading, cleaning
# ============================================================

def prepare_output_directories(output_directory: Path) -> dict[str, Path]:
    """Create (if needed) and return the figures/tables output folders."""

    directories = {
        "root": output_directory,
        "figures": output_directory / "figures",
        "tables": output_directory / "tables",
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def validate_file(file_path: Path, description: str) -> None:
    """Confirm that an expected input file exists and is a file."""

    if not file_path.exists():
        raise FileNotFoundError(f"{description} does not exist:\n{file_path}")

    if not file_path.is_file():
        raise ValueError(f"{description} is not a file:\n{file_path}")

    logger.info("%s found: %s", description, file_path)


def require_columns(
    gdf: gpd.GeoDataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Confirm that a GeoDataFrame contains all required columns."""

    missing_columns = required_columns.difference(gdf.columns)

    if missing_columns:
        raise KeyError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    logger.info("%s contains all required columns", dataset_name)


def load_and_project(
    file_path: Path,
    target_crs: str,
    dataset_name: str,
    driver: str | None = None,
) -> gpd.GeoDataFrame:
    """Load a geospatial file, validate it, and reproject to target_crs."""

    logger.info("Loading %s", dataset_name)

    gdf = gpd.read_file(file_path) if driver is None else gpd.read_file(
        file_path, driver=driver
    )

    if gdf.empty:
        raise ValueError(
            f"{dataset_name} loaded successfully but contains no rows."
        )

    if gdf.crs is None:
        raise ValueError(f"{dataset_name} does not have a CRS.")

    projected = gdf.to_crs(target_crs)

    if not projected.crs.is_projected:
        raise ValueError(
            f"{dataset_name} must use a projected CRS for metre-based "
            "distances."
        )

    logger.info("%s loaded: %d rows", dataset_name, len(projected))
    logger.info("%s projected to %s", dataset_name, projected.crs)

    return projected


def load_ground_track(
    file_path: Path,
    target_crs: str,
    longitude_column: str = "lon_sat",
    latitude_column: str = "lat_sat",
) -> gpd.GeoDataFrame:
    """Load satellite ground-track points from CSV and reproject them."""

    validate_file(file_path, "Satellite ground-track CSV")

    table = pd.read_csv(file_path)

    missing = {longitude_column, latitude_column}.difference(table.columns)
    if missing:
        raise KeyError(
            f"Ground-track CSV is missing columns: {sorted(missing)}"
        )

    track = gpd.GeoDataFrame(
        table,
        geometry=gpd.points_from_xy(
            table[longitude_column], table[latitude_column]
        ),
        crs="EPSG:4326",
    ).to_crs(target_crs)

    if len(track) < 2:
        raise ValueError("At least two ground-track points are required.")

    logger.info("Loaded %d satellite ground-track points", len(track))

    return track


def clean_height_values(
    pixc_gdf: gpd.GeoDataFrame,
    height_column: str,
) -> gpd.GeoDataFrame:
    """
    Remove PIXC rows with invalid geometry, non-finite coordinates, or
    a non-finite height, auditing how many rows are removed at each
    stage so it is clear where NaNs in the raw data actually came
    from.

    This performs basic data hygiene only; it applies no scientific
    quality filtering beyond removing values that cannot be used in
    any calculation at all.
    """

    n_start = len(pixc_gdf)
    cleaned = pixc_gdf.copy()

    # Stage 1: null or empty geometry.
    has_geometry = cleaned.geometry.notna() & ~cleaned.geometry.is_empty
    n_missing_geometry = int((~has_geometry).sum())
    cleaned = cleaned.loc[has_geometry].copy()

    # Stage 2: non-finite x/y coordinates (e.g. a Point(nan, nan), or a
    # geometry type without a simple .x/.y that would break later
    # distance calculations).
    finite_xy = np.isfinite(cleaned.geometry.x) & np.isfinite(
        cleaned.geometry.y
    )
    n_non_finite_xy = int((~finite_xy).sum())
    cleaned = cleaned.loc[finite_xy].copy()

    # Stage 3: height column, coerced to numeric.
    cleaned[height_column] = pd.to_numeric(
        cleaned[height_column], errors="coerce"
    )
    finite_height = np.isfinite(cleaned[height_column])
    n_invalid_height = int((~finite_height).sum())
    cleaned = cleaned.loc[finite_height].copy()

    n_removed_total = n_start - len(cleaned)

    logger.info(
        "PIXC cleaning: %d rows in, %d removed (missing/empty geometry="
        "%d, non-finite coordinates=%d, non-finite height=%d), %d rows "
        "remain",
        n_start,
        n_removed_total,
        n_missing_geometry,
        n_non_finite_xy,
        n_invalid_height,
        len(cleaned),
    )

    return cleaned


def combine_geometries(gdf: gpd.GeoDataFrame, dataset_name: str):
    """Combine all geometries in a GeoDataFrame into a single geometry."""

    combined = gdf.geometry.union_all()

    if combined.is_empty:
        raise ValueError(f"{dataset_name} produced an empty geometry.")

    logger.info(
        "%s combined geometry type: %s", dataset_name, combined.geom_type
    )

    return combined


# ============================================================
# SECTION: geometry.py -- segments, angles, cross-section geometry
# ============================================================

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


def extract_centreline_segments(
    centreline_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Break the river centreline into individual segments and calculate
    each segment's midpoint, length and orientation angle.

    Each row of the centreline, and each line part within a
    MultiLineString, is processed independently so that no false
    segment is created bridging two disconnected parts.
    """

    records = []

    for geometry_id, geometry in enumerate(centreline_gdf.geometry):
        if geometry.geom_type == "LineString":
            parts = [geometry]
        elif geometry.geom_type == "MultiLineString":
            parts = list(geometry.geoms)
        else:
            raise TypeError(
                "Centreline geometry must be a LineString or "
                "MultiLineString."
            )

        for part_id, part in enumerate(parts):
            coordinates = np.asarray(part.coords)

            if len(coordinates) < 2:
                continue

            x = coordinates[:, 0]
            y = coordinates[:, 1]

            dx = np.diff(x)
            dy = np.diff(y)

            angles_deg = np.degrees(np.arctan2(dy, dx))

            mid_x = (x[:-1] + x[1:]) / 2
            mid_y = (y[:-1] + y[1:]) / 2

            lengths_m = np.hypot(dx, dy)

            for local_index in range(len(angles_deg)):
                records.append({
                    "geometry_id": geometry_id,
                    "part_id": part_id,
                    "local_segment_index": local_index,
                    "river_angle_deg": float(angles_deg[local_index]),
                    "segment_length_m": float(lengths_m[local_index]),
                    "geometry": Point(
                        mid_x[local_index], mid_y[local_index]
                    ),
                })

    if not records:
        raise ValueError("No valid centreline segments were created.")

    segments = gpd.GeoDataFrame(
        records, geometry="geometry", crs=centreline_gdf.crs
    )
    segments["segment_index"] = np.arange(len(segments))

    logger.info("Created %d centreline segments", len(segments))

    return segments


def calculate_track_angle_deg(track: gpd.GeoDataFrame) -> float:
    """
    Estimate the overall satellite-track direction using the first and
    last projected track points.
    """

    if len(track) < 2:
        raise ValueError("At least two ground-track points are required.")

    x = track.geometry.x.to_numpy()
    y = track.geometry.y.to_numpy()

    dx = x[-1] - x[0]
    dy = y[-1] - y[0]

    if np.isclose(dx, 0) and np.isclose(dy, 0):
        raise ValueError("Ground-track first and last points are identical.")

    angle_deg = float(np.degrees(np.arctan2(dy, dx)))

    logger.info("Satellite-track angle: %.2f deg", angle_deg)

    return angle_deg


def undirected_angle_difference_deg(
    angles_deg: np.ndarray,
    reference_angle_deg: float,
) -> np.ndarray:
    """
    Return the undirected angle difference between each angle and a
    reference line, in the range 0-90 degrees (0 = parallel,
    90 = perpendicular).
    """

    difference = np.abs(angles_deg - reference_angle_deg)
    difference = difference % 180.0
    difference = np.where(difference > 90.0, 180.0 - difference, difference)

    return difference


def identify_perpendicular_runs(
    segments: gpd.GeoDataFrame,
    minimum_run_length: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Retain stable runs of consecutive perpendicular segments.

    Segments are grouped independently by (geometry_id, part_id) and
    sorted by local_segment_index before consecutive perpendicular
    candidates are split into runs. Only runs of at least
    `minimum_run_length` consecutive segments are accepted.

    Returns
    -------
    accepted_segments
        Every segment belonging to an accepted run.
    representative_midpoints
        One midpoint (from the middle of the run) representing each
        accepted run. This is the *full* set of accepted runs -- use
        this directly for batch processing rather than filtering
        further to a single "nearest" reach.
    """

    segments = segments.copy()
    segments["perpendicular_run_id"] = pd.array(
        [pd.NA] * len(segments), dtype="Int64"
    )

    accepted_indices = []
    representative_indices = []
    run_id = 0

    grouped = segments.groupby(["geometry_id", "part_id"], sort=False)

    for _, group in grouped:
        group = group.sort_values("local_segment_index")

        candidates = group.loc[group["is_perpendicular_candidate"]]
        if candidates.empty:
            continue

        local_indices = candidates["local_segment_index"].to_numpy()
        split_locations = np.where(np.diff(local_indices) != 1)[0] + 1
        runs = np.split(local_indices, split_locations)

        for run in runs:
            if len(run) < minimum_run_length:
                continue

            run_rows = group.loc[group["local_segment_index"].isin(run)]
            accepted_indices.extend(run_rows.index.tolist())

            middle_local_index = run[len(run) // 2]
            middle_row_index = group.loc[
                group["local_segment_index"] == middle_local_index
            ].index[0]
            representative_indices.append(middle_row_index)

            segments.loc[run_rows.index, "perpendicular_run_id"] = run_id
            run_id += 1

    if not accepted_indices:
        raise ValueError(
            "No stable perpendicular reaches were found. Try increasing "
            "angle_tolerance_deg or reducing minimum_run_length."
        )

    accepted_segments = segments.loc[accepted_indices].copy()
    representative_midpoints = segments.loc[representative_indices].copy()

    logger.info(
        "Found %d stable perpendicular runs", len(representative_midpoints)
    )

    return accepted_segments, representative_midpoints


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


# ============================================================
# SECTION: sampling.py -- radius- and cutoff-based point selection
# ============================================================

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


# ============================================================
# SECTION: statistics.py -- local height summaries and delta-h
# ============================================================

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
    return float(np.sqrt(sem_a ** 2 + sem_b ** 2))


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


# ============================================================
# SECTION: processor.py -- per-reach orchestration and batch loop
# ============================================================

def _reach_id(segment_index: int, perpendicular_run_id) -> str:
    """Build a stable, human-readable identifier for one reach."""

    return f"segment_{segment_index}_run_{perpendicular_run_id}"


def _empty_all_nan_row(
    reach_id: str,
    segment_index: int,
    perpendicular_run_id,
    river_angle_deg: float,
    centre_point: Point,
    nearest_pixc_distance_m: float,
    failure_code: str,
    failure_message: str,
) -> dict:
    """Build one failed-reach output row with the full schema, NaN-filled."""

    row = {
        "reach_id": reach_id,
        "segment_index": segment_index,
        "perpendicular_run_id": perpendicular_run_id,
        "processing_status": "failed",
        "quality_status": "not_evaluated",
        "failure_code": failure_code,
        "failure_message": failure_message,
        "centre_x_m": float(centre_point.x),
        "centre_y_m": float(centre_point.y),
        "river_angle_deg": river_angle_deg,
        "cross_section_angle_deg": river_angle_deg + 90.0,
        "nearest_pixc_distance_m": nearest_pixc_distance_m,
        "observed_pixc_width_m": np.nan,
        "negative_edge_distance_m": np.nan,
        "positive_edge_distance_m": np.nan,
        "n_anchor_candidates": np.nan,
        "n_negative": 0,
        "n_centre": 0,
        "n_positive": 0,
        "n_candidates_negative": 0,
        "n_candidates_centre": 0,
        "n_candidates_positive": 0,
        "n_nan_height_negative": 0,
        "n_nan_height_centre": 0,
        "n_nan_height_positive": 0,
        "negative_mean_height": np.nan,
        "centre_mean_height": np.nan,
        "positive_mean_height": np.nan,
        "negative_median_height": np.nan,
        "centre_median_height": np.nan,
        "positive_median_height": np.nan,
        "negative_std_height": np.nan,
        "centre_std_height": np.nan,
        "positive_std_height": np.nan,
        "negative_sem_height": np.nan,
        "centre_sem_height": np.nan,
        "positive_sem_height": np.nan,
        "delta_h_centre_minus_negative": np.nan,
        "delta_h_centre_minus_positive": np.nan,
        "delta_h_positive_minus_negative": np.nan,
        "delta_h_centre_minus_negative_median": np.nan,
        "delta_h_centre_minus_positive_median": np.nan,
        "delta_h_positive_minus_negative_median": np.nan,
        "delta_h_centre_minus_negative_sem": np.nan,
        "delta_h_centre_minus_positive_sem": np.nan,
        "delta_h_positive_minus_negative_sem": np.nan,
        "negative_sample_status": "not_evaluated",
        "centre_sample_status": "not_evaluated",
        "positive_sample_status": "not_evaluated",
        "negative_passes_minimum": False,
        "centre_passes_minimum": False,
        "positive_passes_minimum": False,
        "n_shared_negative_centre": 0,
        "n_shared_centre_positive": 0,
        "n_shared_negative_positive": 0,
        "neighbourhoods_overlap": False,
    }
    return row


def analyse_one_reach(
    reach_row: pd.Series,
    pixc_points: gpd.GeoDataFrame,
    config: HeightAnalysisConfig,
) -> tuple[dict, CrossSectionGeometry | None, dict[str, gpd.GeoDataFrame] | None]:
    """
    Run the full cross-section + sampling + statistics pipeline for
    one representative perpendicular reach.

    Always returns a full-schema output row (never raises for
    ordinary reach-level failures such as missing PIXC coverage; those
    are caught and encoded as processing_status="failed" with a
    failure_code/failure_message). Also returns the cross-section
    geometry and the three final samples when successful, purely so
    the caller can optionally build a diagnostic plot without
    recomputing everything.
    """

    segment_index = int(reach_row["segment_index"])
    perpendicular_run_id = reach_row["perpendicular_run_id"]
    centre_point: Point = reach_row.geometry
    river_angle_deg = float(reach_row["river_angle_deg"])
    cross_section_angle_deg = river_angle_deg + 90.0
    nearest_pixc_distance_m = float(reach_row.get("nearest_pixc_distance_m", np.nan))

    reach_id = _reach_id(segment_index, perpendicular_run_id)

    logger.info(
        "Starting reach %s (segment %d, run %s)",
        reach_id, segment_index, perpendicular_run_id,
    )

    try:
        pixc_local = select_points_within_radius(
            points=pixc_points,
            centre=centre_point,
            radius_m=config.local_search_radius_m,
        )
        if pixc_local.empty:
            raise ReachProcessingError(
                "NO_LOCAL_PIXC",
                "No PIXC points were found within local_search_radius_m="
                f"{config.local_search_radius_m:.1f} m of the reach centre.",
            )

        cross_section_line = build_cross_section_line(
            centre_point=centre_point,
            river_angle_deg=river_angle_deg,
            half_length_m=config.cross_section_half_length_m,
        )

        anchors, n_anchor_candidates = find_cross_section_anchors(
            pixc_points=pixc_local,
            centre_point=centre_point,
            cross_section_angle_deg=cross_section_angle_deg,
            strip_half_width_m=config.anchor_search_strip_half_width_m,
            search_half_length_m=config.cross_section_half_length_m,
            edge_method=config.edge_method,
            edge_lower_percentile=config.edge_lower_percentile,
            edge_upper_percentile=config.edge_upper_percentile,
        )

        cross_section_geometry = validate_cross_section_geometry(
            anchors=anchors,
            n_anchor_candidates=n_anchor_candidates,
            cross_section_line=cross_section_line,
            river_angle_deg=river_angle_deg,
            cross_section_angle_deg=cross_section_angle_deg,
        )

        result = analyse_cross_section_height(
            pixc_points=pixc_local,
            geometry=cross_section_geometry,
            radius_m=config.sample_radius_m,
            minimum_sample_points=config.minimum_sample_points,
            height_column=config.height_column,
            centre_negative_cutoff_m=config.centre_negative_cutoff_m,
            centre_positive_cutoff_m=config.centre_positive_cutoff_m,
        )

        row = {
            "reach_id": reach_id,
            "segment_index": segment_index,
            "perpendicular_run_id": perpendicular_run_id,
            "processing_status": "success",
            "quality_status": result.quality_status,
            "failure_code": None,
            "failure_message": None,
            "centre_x_m": float(centre_point.x),
            "centre_y_m": float(centre_point.y),
            "river_angle_deg": river_angle_deg,
            "cross_section_angle_deg": cross_section_angle_deg,
            "nearest_pixc_distance_m": nearest_pixc_distance_m,
            "observed_pixc_width_m": cross_section_geometry.river_width_m,
            "negative_edge_distance_m": cross_section_geometry.negative_edge_distance_m,
            "positive_edge_distance_m": cross_section_geometry.positive_edge_distance_m,
            "n_anchor_candidates": cross_section_geometry.n_anchor_candidates,
            "n_negative": result.negative_edge.n_valid_height_points,
            "n_centre": result.centre.n_valid_height_points,
            "n_positive": result.positive_edge.n_valid_height_points,
            "n_candidates_negative": result.negative_edge.n_candidates_before_height_filter,
            "n_candidates_centre": result.centre.n_candidates_before_height_filter,
            "n_candidates_positive": result.positive_edge.n_candidates_before_height_filter,
            "n_nan_height_negative": result.negative_edge.n_nan_height_points,
            "n_nan_height_centre": result.centre.n_nan_height_points,
            "n_nan_height_positive": result.positive_edge.n_nan_height_points,
            "negative_mean_height": result.negative_edge.mean_height_m,
            "centre_mean_height": result.centre.mean_height_m,
            "positive_mean_height": result.positive_edge.mean_height_m,
            "negative_median_height": result.negative_edge.median_height_m,
            "centre_median_height": result.centre.median_height_m,
            "positive_median_height": result.positive_edge.median_height_m,
            "negative_std_height": result.negative_edge.standard_deviation_m,
            "centre_std_height": result.centre.standard_deviation_m,
            "positive_std_height": result.positive_edge.standard_deviation_m,
            "negative_sem_height": result.negative_edge.standard_error_m,
            "centre_sem_height": result.centre.standard_error_m,
            "positive_sem_height": result.positive_edge.standard_error_m,
            "delta_h_centre_minus_negative": result.delta_h_centre_minus_negative_m,
            "delta_h_centre_minus_positive": result.delta_h_centre_minus_positive_m,
            "delta_h_positive_minus_negative": result.delta_h_positive_minus_negative_m,
            "delta_h_centre_minus_negative_median": result.delta_h_centre_minus_negative_median_m,
            "delta_h_centre_minus_positive_median": result.delta_h_centre_minus_positive_median_m,
            "delta_h_positive_minus_negative_median": result.delta_h_positive_minus_negative_median_m,
            "delta_h_centre_minus_negative_sem": result.delta_h_centre_minus_negative_sem_m,
            "delta_h_centre_minus_positive_sem": result.delta_h_centre_minus_positive_sem_m,
            "delta_h_positive_minus_negative_sem": result.delta_h_positive_minus_negative_sem_m,
            "negative_sample_status": result.negative_edge.sample_status,
            "centre_sample_status": result.centre.sample_status,
            "positive_sample_status": result.positive_edge.sample_status,
            "negative_passes_minimum": result.negative_passes_minimum,
            "centre_passes_minimum": result.centre_passes_minimum,
            "positive_passes_minimum": result.positive_passes_minimum,
            "n_shared_negative_centre": result.n_shared_negative_centre,
            "n_shared_centre_positive": result.n_shared_centre_positive,
            "n_shared_negative_positive": result.n_shared_negative_positive,
            "neighbourhoods_overlap": result.neighbourhoods_overlap,
        }

        logger.info(
            "Finished reach %s: quality_status=%s (n=%d/%d/%d)",
            reach_id, result.quality_status,
            result.negative_edge.n_valid_height_points,
            result.centre.n_valid_height_points,
            result.positive_edge.n_valid_height_points,
        )

        plot_payload = {
            "pixc_local": pixc_local,
            "negative_sample": select_cross_section_sample(
                pixc_local, cross_section_geometry, "negative_edge",
                config.sample_radius_m, config.centre_negative_cutoff_m,
                config.centre_positive_cutoff_m,
            ),
            "centre_sample": select_cross_section_sample(
                pixc_local, cross_section_geometry, "centre",
                config.sample_radius_m, config.centre_negative_cutoff_m,
                config.centre_positive_cutoff_m,
            ),
            "positive_sample": select_cross_section_sample(
                pixc_local, cross_section_geometry, "positive_edge",
                config.sample_radius_m, config.centre_negative_cutoff_m,
                config.centre_positive_cutoff_m,
            ),
        }

        return row, cross_section_geometry, plot_payload

    except ReachProcessingError as error:
        logger.warning("Reach %s failed [%s]: %s", reach_id, error.failure_code, error.message)
        row = _empty_all_nan_row(
            reach_id, segment_index, perpendicular_run_id, river_angle_deg,
            centre_point, nearest_pixc_distance_m,
            error.failure_code, error.message,
        )
        return row, None, None

    except (ValueError, TypeError, KeyError) as error:
        logger.warning("Reach %s failed [UNEXPECTED_ERROR]: %s", reach_id, error)
        row = _empty_all_nan_row(
            reach_id, segment_index, perpendicular_run_id, river_angle_deg,
            centre_point, nearest_pixc_distance_m,
            "UNEXPECTED_ERROR", str(error),
        )
        return row, None, None


def analyse_all_perpendicular_reaches(
    reaches_to_analyse: gpd.GeoDataFrame,
    pixc_points: gpd.GeoDataFrame,
    config: HeightAnalysisConfig,
) -> tuple[pd.DataFrame, dict[str, tuple[CrossSectionGeometry, dict]]]:
    """
    Run the cross-section height analysis for every representative
    perpendicular reach in `reaches_to_analyse`. One row is produced
    per reach whether it succeeds or fails, so a single bad reach
    never stops the batch.

    Returns the results table plus a dict of successful reaches'
    geometry/plot payloads (keyed by reach_id) for optional plotting.
    """

    if reaches_to_analyse.empty:
        raise ValueError("There were no perpendicular reaches to analyse.")

    rows: list[dict] = []
    plot_payloads: dict[str, tuple[CrossSectionGeometry, dict]] = {}

    for _, reach_row in reaches_to_analyse.iterrows():
        row, geometry, payload = analyse_one_reach(reach_row, pixc_points, config)
        rows.append(row)
        if geometry is not None and payload is not None:
            plot_payloads[row["reach_id"]] = (geometry, payload)

    results = pd.DataFrame(rows)

    n_success = int((results["processing_status"] == "success").sum())
    n_failed = int((results["processing_status"] == "failed").sum())
    n_quality_pass = int((results["quality_status"] == "pass").sum())

    logger.info(
        "Finished all reaches: %d attempted, %d successful geometry "
        "analyses, %d fully quality-approved, %d failed",
        len(results), n_success, n_quality_pass, n_failed,
    )

    return results, plot_payloads


# ============================================================
# SECTION: plotting.py -- optional diagnostic figures
# ============================================================

def plot_data_overview(
    pixc_clean: gpd.GeoDataFrame,
    river_boundary: gpd.GeoDataFrame,
    river_centreline: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Save a simple overview map of the cleaned PIXC data and inputs."""

    fig, ax = plt.subplots(figsize=(10, 10))

    pixc_clean.plot(ax=ax, markersize=2, alpha=0.5, label="PIXC points")
    river_boundary.boundary.plot(ax=ax, linewidth=2, label="AOI")
    river_centreline.plot(ax=ax, linewidth=2, label="River centreline")

    ax.set_title("PIXC data, river centreline and boundary")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved overview figure to %s", output_path)


def plot_perpendicular_reaches(
    river_centreline: gpd.GeoDataFrame,
    perpendicular_segments: gpd.GeoDataFrame,
    all_perpendicular_reaches: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Save a map of every stable perpendicular run and its representative point."""

    fig, ax = plt.subplots(figsize=(10, 10))

    river_centreline.plot(ax=ax, linewidth=1.5, label="River centreline")
    perpendicular_segments.plot(
        ax=ax, markersize=14, color="tab:red", label="Accepted perpendicular segments"
    )
    all_perpendicular_reaches.plot(
        ax=ax, markersize=90, marker="x", linewidth=2.5, color="black",
        label="Representative reach midpoint",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(
        f"Stable perpendicular reaches (n={len(all_perpendicular_reaches)})"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved perpendicular-reaches figure to %s", output_path)


def select_reaches_to_plot(
    all_reach_results: pd.DataFrame,
    config: HeightAnalysisConfig,
) -> list[str]:
    """Choose which successful reaches' reach_id values to plot."""

    successful = all_reach_results.loc[
        all_reach_results["processing_status"] == "success"
    ]

    if config.diagnostic_reach_segment_indices is not None:
        wanted_segments = set(config.diagnostic_reach_segment_indices)
        chosen = successful.loc[successful["segment_index"].isin(wanted_segments)]
    else:
        chosen = successful.head(config.max_diagnostic_reach_plots)

    return chosen["reach_id"].tolist()


def plot_reach_diagnostics(
    reach_row: pd.Series,
    geometry: CrossSectionGeometry,
    pixc_local: gpd.GeoDataFrame,
    negative_sample: gpd.GeoDataFrame,
    centre_sample: gpd.GeoDataFrame,
    positive_sample: gpd.GeoDataFrame,
    config: HeightAnalysisConfig,
    output_path: Path,
) -> None:
    """
    Save a two-panel diagnostic figure for one reach:
      left  -- spatial map: local PIXC points, cross-section line,
               anchors, retained samples, and rejected points;
      right -- height vs. signed across-river distance, with the
               PIXC-derived river limits and the configured cutoff
               lines drawn in, so it is obvious whether the cutoff
               geometry is sensible for this reach.
    """

    ux, uy = _unit_vector_deg(geometry.cross_section_angle_deg)
    dx = pixc_local.geometry.x - geometry.centre.x
    dy = pixc_local.geometry.y - geometry.centre.y
    pixc_local = pixc_local.copy()
    pixc_local["across_distance_m"] = dx * ux + dy * uy

    retained_index = negative_sample.index.union(centre_sample.index).union(
        positive_sample.index
    )
    within_radius = select_points_within_radius(
        pixc_local, geometry.centre, config.sample_radius_m
    )
    rejected = within_radius.loc[~within_radius.index.isin(retained_index)]

    fig, (ax_map, ax_height) = plt.subplots(1, 2, figsize=(18, 9))

    # --- left panel: spatial map ---
    pixc_local.plot(ax=ax_map, markersize=3, alpha=0.2, color="lightgrey", label="Local PIXC")
    rejected.plot(ax=ax_map, markersize=8, marker="x", color="grey", label="Rejected (in radius)")

    gpd.GeoSeries([geometry.cross_section_line], crs=pixc_local.crs).plot(
        ax=ax_map, linewidth=1.5, color="black", label="Cross-section line"
    )

    negative_sample.plot(ax=ax_map, markersize=12, color="tab:orange", label="Negative-edge sample")
    centre_sample.plot(ax=ax_map, markersize=12, color="tab:red", label="Centre sample")
    positive_sample.plot(ax=ax_map, markersize=12, color="tab:purple", label="Positive-edge sample")

    for point, colour, marker, label in (
        (geometry.negative_edge, "tab:orange", "v", "Negative edge anchor"),
        (geometry.centre, "tab:red", "*", "Centre anchor"),
        (geometry.positive_edge, "tab:purple", "^", "Positive edge anchor"),
    ):
        gpd.GeoSeries([point], crs=pixc_local.crs).plot(
            ax=ax_map, markersize=150, marker=marker, color=colour, label=label
        )

    zoom_m = config.cross_section_plot_zoom_m
    ax_map.set_xlim(geometry.centre.x - zoom_m, geometry.centre.x + zoom_m)
    ax_map.set_ylim(geometry.centre.y - zoom_m, geometry.centre.y + zoom_m)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("Easting (m)")
    ax_map.set_ylabel("Northing (m)")
    ax_map.set_title(f"{reach_row['reach_id']} -- spatial view")
    ax_map.legend(loc="best", fontsize=7)

    # --- right panel: height vs across-river distance ---
    height_column = config.height_column

    def _plot_group(sample, colour, label):
        if sample.empty:
            return
        heights = pd.to_numeric(sample[height_column], errors="coerce")
        ax_height.scatter(
            sample["across_distance_m"], heights, s=14, color=colour, label=label, alpha=0.8
        )

    _plot_group(rejected, "grey", "Rejected")
    _plot_group(negative_sample, "tab:orange", "Negative-edge sample")
    _plot_group(centre_sample, "tab:red", "Centre sample")
    _plot_group(positive_sample, "tab:purple", "Positive-edge sample")

    ax_height.axvline(-geometry.negative_edge_distance_m, color="black", linestyle="-", linewidth=1, label="PIXC river limit")
    ax_height.axvline(geometry.positive_edge_distance_m, color="black", linestyle="-", linewidth=1)
    ax_height.axvline(config.centre_negative_cutoff_m, color="grey", linestyle="--", linewidth=1, label="Configured cutoff")
    ax_height.axvline(config.centre_positive_cutoff_m, color="grey", linestyle="--", linewidth=1)

    ax_height.set_xlabel("Signed across-river distance (m)")
    ax_height.set_ylabel(f"PIXC height, '{height_column}' (m)")
    ax_height.set_title(
        f"{reach_row['reach_id']} -- height vs across-river distance | "
        f"width={geometry.river_width_m:.1f} m | quality={reach_row['quality_status']}"
    )
    ax_height.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved reach diagnostic figure to %s", output_path)


# ============================================================
# SECTION: run_analysis.py -- the single entry point
# ============================================================

@dataclass
class HeightAnalysisRunResult:
    """Everything produced by one call to run_height_analysis(config)."""

    all_reach_results: pd.DataFrame
    successful_reach_results: pd.DataFrame
    failed_reach_results: pd.DataFrame

    pixc_clean: gpd.GeoDataFrame
    pixc_in_boundary: gpd.GeoDataFrame
    river_centreline: gpd.GeoDataFrame
    river_boundary: gpd.GeoDataFrame
    centreline_segments: gpd.GeoDataFrame
    perpendicular_segments: gpd.GeoDataFrame
    all_perpendicular_reaches: gpd.GeoDataFrame

    output_dirs: dict[str, Path]


def _print_terminal_summary(all_reach_results: pd.DataFrame) -> None:
    """Print (and log) a concise scientific/processing summary."""

    n_total = len(all_reach_results)
    n_success = int((all_reach_results["processing_status"] == "success").sum())
    n_failed = int((all_reach_results["processing_status"] == "failed").sum())
    n_quality_pass = int((all_reach_results["quality_status"] == "pass").sum())

    lines = [
        "",
        "=== Batch reach-processing summary ===",
        f"Total reaches attempted        : {n_total}",
        f"Successful geometry analyses   : {n_success}",
        f"Fully quality-approved reaches : {n_quality_pass}",
        f"Failed reaches                 : {n_failed}",
    ]

    if n_failed > 0:
        lines.append("")
        lines.append("Failures grouped by failure code:")
        failure_counts = (
            all_reach_results.loc[all_reach_results["processing_status"] == "failed"]
            ["failure_code"].value_counts()
        )
        for code, count in failure_counts.items():
            lines.append(f"  {code}: {count}")

    summary_text = "\n".join(lines)
    print(summary_text)
    for line in lines:
        if line:
            logger.info(line)


def run_height_analysis(config: HeightAnalysisConfig) -> HeightAnalysisRunResult:
    """
    The single entry point for the whole pipeline.

    Loads every input, cleans and projects the data, identifies every
    stable perpendicular reach, analyses all of them (recording
    failures rather than crashing), saves CSV result tables and
    optional diagnostic figures, prints a terminal summary, and
    returns everything as a HeightAnalysisRunResult for further
    interactive use.

    This function relies on no notebook state, no module-level
    execution side effects, and no hidden globals: every value it
    needs is either passed in via `config` or computed locally.
    """

    output_dirs = prepare_output_directories(config.output_directory)
    logger.info("Outputs will be saved to %s", output_dirs["root"])

    # --- validate and load inputs ---
    validate_file(config.pixc_file, "PIXC GeoJSON")
    validate_file(config.centreline_file, "River centreline")
    validate_file(config.river_boundary_file, "River boundary/AOI")
    validate_file(config.ground_track_file, "Satellite ground-track CSV")

    pixc = load_and_project(config.pixc_file, config.target_crs, "PIXC data")
    river_centreline = load_and_project(
        config.centreline_file, config.target_crs, "River centreline", driver="KML"
    )
    river_boundary = load_and_project(
        config.river_boundary_file, config.target_crs, "River boundary/AOI"
    )
    satellite_track = load_ground_track(config.ground_track_file, config.target_crs)

    require_columns(pixc, {config.height_column, "geometry"}, "PIXC data")

    # --- clean heights ---
    pixc_clean = clean_height_values(pixc, config.height_column)

    if config.save_figures:
        plot_data_overview(
            pixc_clean, river_boundary, river_centreline,
            output_dirs["figures"] / "01_data_overview.png",
        )

    combine_geometries(river_centreline, "River centreline")  # sanity check only
    combine_geometries(river_boundary, "River boundary/AOI")  # sanity check only

    # --- centreline segments and angles ---
    centreline_segments = extract_centreline_segments(river_centreline)

    satellite_angle_deg = calculate_track_angle_deg(satellite_track)

    centreline_segments["angle_difference_deg"] = undirected_angle_difference_deg(
        centreline_segments["river_angle_deg"].to_numpy(), satellite_angle_deg,
    )
    centreline_segments["is_perpendicular_candidate"] = (
        np.abs(centreline_segments["angle_difference_deg"] - 90.0)
        <= config.angle_tolerance_deg
    )

    perpendicular_segments, all_perpendicular_reaches = identify_perpendicular_runs(
        centreline_segments, config.minimum_run_length,
    )

    if config.save_figures:
        plot_perpendicular_reaches(
            river_centreline, perpendicular_segments, all_perpendicular_reaches,
            output_dirs["figures"] / "02_perpendicular_reaches.png",
        )

    # --- clip PIXC to the AOI ---
    pixc_in_boundary = gpd.clip(pixc_clean, river_boundary).copy()
    if pixc_in_boundary.empty:
        raise ValueError("No valid PIXC points lie inside the boundary/AOI.")

    # --- use every stable perpendicular reach, not a single "nearest" one ---
    reaches_to_analyse = all_perpendicular_reaches.copy()
    reaches_to_analyse["nearest_pixc_distance_m"] = reaches_to_analyse.geometry.apply(
        lambda reach_point: pixc_in_boundary.geometry.distance(reach_point).min()
    )

    logger.info("Number of perpendicular reaches to analyse: %d", len(reaches_to_analyse))

    all_reach_results, plot_payloads = analyse_all_perpendicular_reaches(
        reaches_to_analyse=reaches_to_analyse,
        pixc_points=pixc_in_boundary,
        config=config,
    )

    # --- save result tables ---
    all_path = output_dirs["tables"] / "all_reach_results.csv"
    all_reach_results.to_csv(all_path, index=False)
    logger.info("Saved all-reach results to %s", all_path)

    successful_reach_results = all_reach_results.loc[
        all_reach_results["processing_status"] == "success"
    ].copy()
    failed_reach_results = all_reach_results.loc[
        all_reach_results["processing_status"] == "failed"
    ].copy()

    successful_path = output_dirs["tables"] / "successful_reach_results.csv"
    successful_reach_results.to_csv(successful_path, index=False)
    logger.info("Saved successful-reach results to %s", successful_path)

    failed_path = output_dirs["tables"] / "failed_reach_results.csv"
    failed_reach_results.to_csv(failed_path, index=False)
    logger.info("Saved failed-reach results to %s", failed_path)

    # --- optional diagnostic plots for a handful of reaches ---
    if config.save_figures:
        reach_ids_to_plot = select_reaches_to_plot(all_reach_results, config)
        for reach_id in reach_ids_to_plot:
            geometry, payload = plot_payloads[reach_id]
            reach_row = all_reach_results.loc[
                all_reach_results["reach_id"] == reach_id
            ].iloc[0]
            output_path = (
                output_dirs["figures"]
                / f"reach_{reach_row['segment_index']:04d}_diagnostics.png"
            )
            plot_reach_diagnostics(
                reach_row=reach_row,
                geometry=geometry,
                pixc_local=payload["pixc_local"],
                negative_sample=payload["negative_sample"],
                centre_sample=payload["centre_sample"],
                positive_sample=payload["positive_sample"],
                config=config,
                output_path=output_path,
            )

    _print_terminal_summary(all_reach_results)

    return HeightAnalysisRunResult(
        all_reach_results=all_reach_results,
        successful_reach_results=successful_reach_results,
        failed_reach_results=failed_reach_results,
        pixc_clean=pixc_clean,
        pixc_in_boundary=pixc_in_boundary,
        river_centreline=river_centreline,
        river_boundary=river_boundary,
        centreline_segments=centreline_segments,
        perpendicular_segments=perpendicular_segments,
        all_perpendicular_reaches=all_perpendicular_reaches,
        output_dirs=output_dirs,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    config = HeightAnalysisConfig(
        pixc_file=Path(
            "/content/Orco/Orco_processed/PIXC_029_234R_20250408.geojson"
        ),
        centreline_file=Path("/content/Orco/Orco_CL_25832.kml"),
        river_boundary_file=Path("/content/Orco_shapefile_uav.shp"),
        ground_track_file=Path(
            "/content/sat_paths/sat_paths/"
            "SWOT_L2_HR_PIXC_031_029_234R_"
            "20250408T044534_20250408T044545_PGD0_01.csv"
        ),
        output_directory=Path("/content/orco_height_outputs"),
        target_crs="EPSG:32632",
        height_column="height",
        angle_tolerance_deg=20.0,
        minimum_run_length=3,
        local_search_radius_m=500.0,
        cross_section_half_length_m=500.0,
        anchor_search_strip_half_width_m=10.0,
        edge_method="percentile",
        edge_lower_percentile=2.0,
        edge_upper_percentile=98.0,
        sample_radius_m=20.0,
        minimum_sample_points=5,
        centre_negative_cutoff_m=-10.0,
        centre_positive_cutoff_m=10.0,
        save_figures=True,
        max_diagnostic_reach_plots=5,
    )

    results = run_height_analysis(config)
