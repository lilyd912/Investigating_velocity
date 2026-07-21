#angles

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Point

logger = logging.getLogger("river_height_analysis")

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

def identify_perpendicular_reaches(
    river_centreline: gpd.GeoDataFrame,
    satellite_track: gpd.GeoDataFrame,
    angle_tolerance_deg: float,
    minimum_run_length: int,
) -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """
    Create centreline segments, calculate their angle to the satellite
    track, and identify stable perpendicular runs.
    """

    centreline_segments = extract_centreline_segments(
        river_centreline
    )

    satellite_angle_deg = calculate_track_angle_deg(
        satellite_track
    )

    centreline_segments["angle_difference_deg"] = (
        undirected_angle_difference_deg(
            centreline_segments[
                "river_angle_deg"
            ].to_numpy(),
            satellite_angle_deg,
        )
    )

    centreline_segments["is_perpendicular_candidate"] = (
        np.abs(
            centreline_segments["angle_difference_deg"]
            - 90.0
        )
        <= angle_tolerance_deg
    )

    (
        perpendicular_segments,
        all_perpendicular_reaches,
    ) = identify_perpendicular_runs(
        segments=centreline_segments,
        minimum_run_length=minimum_run_length,
    )

    return (
        centreline_segments,
        perpendicular_segments,
        all_perpendicular_reaches,
    )


