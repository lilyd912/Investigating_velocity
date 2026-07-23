from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd


logger = logging.getLogger("river_height_analysis")


def apply_height_corrections(
    pixc: gpd.GeoDataFrame,
    raw_height_column: str = "height",
    pole_tide_column: str = "pole_tide",
    geoid_column: str = "geoid",
    output_column: str = "height_corrected",
) -> gpd.GeoDataFrame:
    """
    Apply pole-tide and geoid corrections to PIXC heights.

    The correction values are already signed, so they are added directly:

        corrected height = raw height + pole tide + geoid
    """

    required_columns = [
        raw_height_column,
        pole_tide_column,
        geoid_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in pixc.columns
    ]

    if missing_columns:
        raise KeyError(
            "Cannot apply height corrections. Missing columns: "
            + ", ".join(missing_columns)
        )

    corrected = pixc.copy()

    raw_height = pd.to_numeric(
        corrected[raw_height_column],
        errors="coerce",
    )

    pole_tide = pd.to_numeric(
        corrected[pole_tide_column],
        errors="coerce",
    )

    geoid = pd.to_numeric(
        corrected[geoid_column],
        errors="coerce",
    )

    corrected[output_column] = (
        raw_height
        + pole_tide
        + geoid
    )

    logger.info(
        "Applied pole-tide and geoid corrections to %d PIXC points.",
        int(corrected[output_column].notna().sum()),
    )

    return corrected