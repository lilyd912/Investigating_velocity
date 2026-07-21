#read data

from __future__ import annotations

from pathlib import Path
import logging

import geopandas as gpd
import pandas as pd

logger = logging.getLogger("river_height_analysis")


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

def combine_geometries(gdf: gpd.GeoDataFrame, dataset_name: str):
    """Combine all geometries in a GeoDataFrame into a single geometry."""

    combined = gdf.geometry.union_all()

    if combined.is_empty:
        raise ValueError(f"{dataset_name} produced an empty geometry.")

    logger.info(
        "%s combined geometry type: %s", dataset_name, combined.geom_type
    )

    return combined


