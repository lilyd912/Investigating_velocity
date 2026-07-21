#reporting

from __future__ import annotations

from pathlib import Path
import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from config import HeightAnalysisConfig
from cross_river import (
    CrossSectionGeometry,
    _unit_vector_deg,
    select_points_within_radius,
)

logger = logging.getLogger("river_height_analysis")

configure_logging()


def configure_logging() -> logging.Logger:
    """Configure and return the pipeline logger."""

    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(levelname)s | %(message)s"
            )
        )
        logger.addHandler(handler)

    return logger

def print_terminal_summary(all_reach_results: pd.DataFrame) -> None:
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
