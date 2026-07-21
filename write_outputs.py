#outputs

from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger("river_height_analysis")

logger = logging.getLogger("river_height_analysis")
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


def write_result_tables(
    all_reach_results: pd.DataFrame,
    tables_directory: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Save the full, successful and failed reach-result tables.
    """

    all_path = tables_directory / "all_reach_results.csv"
    all_reach_results.to_csv(all_path, index=False)
    logger.info(
        "Saved all-reach results to %s",
        all_path,
    )

    successful_reach_results = all_reach_results.loc[
        all_reach_results["processing_status"] == "success"
    ].copy()

    failed_reach_results = all_reach_results.loc[
        all_reach_results["processing_status"] == "failed"
    ].copy()

    successful_path = (
        tables_directory
        / "successful_reach_results.csv"
    )
    successful_reach_results.to_csv(
        successful_path,
        index=False,
    )
    logger.info(
        "Saved successful-reach results to %s",
        successful_path,
    )

    failed_path = (
        tables_directory
        / "failed_reach_results.csv"
    )
    failed_reach_results.to_csv(
        failed_path,
        index=False,
    )
    logger.info(
        "Saved failed-reach results to %s",
        failed_path,
    )

    return (
        successful_reach_results,
        failed_reach_results,
    )


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




