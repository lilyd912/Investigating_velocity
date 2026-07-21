#processor 

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Point

from config import HeightAnalysisConfig

from cross_river import (
    CrossSectionGeometry,
    ReachProcessingError,
    build_cross_section_line,
    find_cross_section_anchors,
    select_cross_section_sample,
    select_points_within_radius,
    validate_cross_section_geometry,
)

from height_analysis import (
    analyse_cross_section_height,
)

logger = logging.getLogger("river_height_analysis")

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


