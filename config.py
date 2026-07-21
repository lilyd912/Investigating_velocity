#configurables

from __future__ import annotations

# ============================================================
# SECTION: config.py -- imports
# ============================================================

from dataclasses import dataclass
from pathlib import Path
import logging




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
