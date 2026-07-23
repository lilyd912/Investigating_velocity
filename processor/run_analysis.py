#run_analysis


from __future__ import annotations

from write_outputs import write_reach_results_netcdf
from corrections import apply_height_corrections

from dataclasses import dataclass
from pathlib import Path
import logging

import geopandas as gpd
import pandas as pd

from config import HeightAnalysisConfig

from read_data import (
    combine_geometries,
    load_and_project,
    load_ground_track,
    require_columns,
    validate_file,
)

from segmentation import (
    segment_pixc,
)

from angles import (
    identify_perpendicular_reaches,
)

from processor import (
    analyse_all_perpendicular_reaches,
)

from write_outputs import (
    prepare_output_directories,
    write_result_tables,
)

from reporting import (
    configure_logging,
    plot_data_overview,
    plot_perpendicular_reaches,
    plot_reach_diagnostics,
    print_terminal_summary,
    select_reaches_to_plot,
)

logger = logging.getLogger("river_height_analysis")

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
    configure_logging()
    output_dirs = prepare_output_directories(config.output_directory)
    logger.info("Outputs will be saved to %s", output_dirs["root"])

    # --- validate and load inputs ---
    validate_file(config.pixc_file, "PIXC GeoJSON")
    validate_file(config.centreline_file, "River centreline")
    validate_file(config.river_boundary_file, "River boundary/AOI")
    validate_file(config.ground_track_file, "Satellite ground-track CSV")

    pixc = load_and_project(
        config.pixc_file,
        config.target_crs,
        "PIXC data",
    )

    # Check that the raw height and correction columns exist.
    require_columns(
        pixc,
        {"height", "pole_tide", "geoid", "geometry"},
        "PIXC data",
    )

    # Apply the signed corrections:
    # corrected height = height + pole_tide + geoid
    pixc = apply_height_corrections(
        pixc=pixc,
        raw_height_column="height",
        pole_tide_column="pole_tide",
        geoid_column="geoid",
        output_column="height_corrected",
    )

    river_centreline = load_and_project(
        config.centreline_file,
        config.target_crs,
        "River centreline",
        driver="KML",
    )

    river_boundary = load_and_project(
        config.river_boundary_file,
        config.target_crs,
        "River boundary/AOI",
    )

    satellite_track = load_ground_track(
        config.ground_track_file,
        config.target_crs,
    )

    # --- segmentation: clean, AOI, classification and flags ---
    pixc_in_boundary = segment_pixc(
        pixc=pixc,
        river_boundary=river_boundary,
        height_column=config.height_column,
        allowed_classifications=None,
    )

    # Retain this name for the returned run-result object and overview plot.
    pixc_clean = pixc_in_boundary
    

    if config.save_figures:
        plot_data_overview(
            pixc_clean, river_boundary, river_centreline,
            output_dirs["figures"] / "01_data_overview.png",
        )

    combine_geometries(river_centreline, "River centreline")  # sanity check only
    combine_geometries(river_boundary, "River boundary/AOI")  # sanity check only

   

    (
        centreline_segments,
        perpendicular_segments,
        all_perpendicular_reaches,
    ) = identify_perpendicular_reaches(
        river_centreline=river_centreline,
        satellite_track=satellite_track,
        angle_tolerance_deg=config.angle_tolerance_deg,        minimum_run_length=config.minimum_run_length,
    )

    if config.save_figures:
        plot_perpendicular_reaches(
            river_centreline, perpendicular_segments, all_perpendicular_reaches,
            output_dirs["figures"] / "02_perpendicular_reaches.png",
        )

    

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

    (
        successful_reach_results,
        failed_reach_results,
    ) = write_result_tables(
        all_reach_results=all_reach_results,
        tables_directory=output_dirs["tables"],
    )

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

    print_terminal_summary(all_reach_results)

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


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    config = HeightAnalysisConfig(
        pixc_file=(
            project_root
            / "inputs"
            / "pixc"
            / "PIXC_029_234R_20250408.geojson"
        ),
        centreline_file=(
            project_root
            / "inputs"
            / "centreline"
            / "Orco_CL_25832.kml"
        ),
        river_boundary_file=(
            project_root
            / "inputs"
            / "river_boundary"
            / "Orco_shapefile_uav.shp"
        ),
        ground_track_file=(
            project_root
            / "inputs"
            / "ground_track"
            / "SWOT_L2_HR_PIXC_031_029_234R_"
            "20250408T044534_20250408T044545_PGD0_01.csv"
        ),
        output_directory=project_root / "outputs",
        target_crs="EPSG:32632",
        height_column="height_corrected",
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

    




write_reach_results_netcdf(
    reach_results=results.all_reach_results,
    output_path=config.output_directory / "reach_height_results.nc",
)