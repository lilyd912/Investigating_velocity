#segmentation

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger("river_height_analysis")

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

def clip_pixc_to_boundary(
    pixc: gpd.GeoDataFrame,
    river_boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Clip PIXC points to the supplied AOI or river boundary.

    Both inputs must already use the same projected CRS.
    """

    if pixc.crs != river_boundary.crs:
        raise ValueError(
            "PIXC data and river boundary must use the same CRS "
            "before clipping."
        )

    pixc_in_boundary = gpd.clip(
        pixc,
        river_boundary,
    ).copy()

    if pixc_in_boundary.empty:
        raise ValueError(
            "No valid PIXC points lie inside the boundary/AOI."
        )

    logger.info(
        "PIXC AOI clipping: %d points entered, %d points remain",
        len(pixc),
        len(pixc_in_boundary),
    )

    return pixc_in_boundary

def filter_pixc_classifications(
    pixc: gpd.GeoDataFrame,
    allowed_classifications: tuple[int, ...] | None,
    classification_column: str = "classification",
) -> gpd.GeoDataFrame:
    """
    Retain only configured PIXC classification values.

    If allowed_classifications is None, no classification filter is
    applied.
    """

    if allowed_classifications is None:
        logger.info("No PIXC classification filter applied")
        return pixc.copy()

    if classification_column not in pixc.columns:
        raise KeyError(
            f"PIXC data does not contain '{classification_column}'."
        )

    filtered = pixc.loc[
        pixc[classification_column].isin(allowed_classifications)
    ].copy()

    logger.info(
        "Classification filtering: %d points entered, %d points remain",
        len(pixc),
        len(filtered),
    )

    return filtered

def filter_pixc_flags(
    pixc: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Placeholder for the agreed SWOT PIXC flag filtering.

    No flags are removed until the relevant flag definitions and
    accepted values have been confirmed.
    """

    logger.info("No PIXC flag filter currently applied")
    return pixc.copy()

def segment_pixc(
    pixc: gpd.GeoDataFrame,
    river_boundary: gpd.GeoDataFrame,
    height_column: str,
    allowed_classifications: tuple[int, ...] | None = None,
) -> gpd.GeoDataFrame:
    """Run the complete PIXC segmentation stage."""

    segmented = clean_height_values(
        pixc_gdf=pixc,
        height_column=height_column,
    )

    segmented = clip_pixc_to_boundary(
        pixc=segmented,
        river_boundary=river_boundary,
    )

    segmented = filter_pixc_classifications(
        pixc=segmented,
        allowed_classifications=allowed_classifications,
    )

    segmented = filter_pixc_flags(segmented)

    if segmented.empty:
        raise ValueError(
            "No PIXC points remain after segmentation."
        )

    return segmented

allowed_classifications=None


