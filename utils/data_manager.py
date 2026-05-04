"""Persistence layer: saving sweep data and fit results to disk.

Formats supported
-----------------
* CSV  — two-column data with comment-block metadata header.
* JSON — structured payload with timestamps, units, and fit metadata.

This module has no GUI or hardware dependencies.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from core.analysis import FitResult

logger = logging.getLogger(__name__)


class DataManager:
    """Handles all file I/O for SMU sweep data.

    Example
    -------
    dm = DataManager()
    dm.save_csv("results.csv", sweep_array, fit_result)
    dm.save_json("results.json", sweep_array, fit_result, metadata={"device": "1N4148"})
    """

    # CSV comment lines start with '#' so they round-trip cleanly with
    # numpy.genfromtxt(fname, comments='#', delimiter=',')
    _CSV_COMMENT = "#"

    # ------------------------------------------------------------------
    # Filename helpers
    # ------------------------------------------------------------------

    @staticmethod
    def generate_filename(prefix: str = "smu_sweep", extension: str = "csv") -> str:
        """Return a timestamped filename, e.g. ``smu_sweep_20260504_152300.csv``."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}.{extension.lstrip('.')}"

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def save_csv(
        self,
        path: str | Path,
        data: np.ndarray,
        fit: Optional[FitResult] = None,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Write sweep data to a CSV file with a metadata comment block.

        The output is compatible with numpy.genfromtxt and most
        spreadsheet applications.

        Parameters
        ----------
        path:
            Destination path.  The ``.csv`` extension is added if absent.
        data:
            Structured NumPy array with fields ``voltage`` (V) and
            ``current`` (A), as returned by B2901A.run_iv_sweep().
        fit:
            Optional FitResult to embed in the comment header.
        metadata:
            Optional dict of arbitrary key/value strings (device name,
            operator, etc.) to include in the header.

        Returns
        -------
        Path
            Resolved path of the written file.
        """
        path = self._resolve_path(path, ".csv")
        timestamp = datetime.now().isoformat(timespec="seconds")

        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)

            # --- metadata header ---
            writer.writerow([f"{self._CSV_COMMENT} SMU IV Sweep"])
            writer.writerow([f"{self._CSV_COMMENT} Timestamp : {timestamp}"])
            writer.writerow([f"{self._CSV_COMMENT} Points    : {len(data)}"])

            if metadata:
                for key, val in metadata.items():
                    writer.writerow([f"{self._CSV_COMMENT} {key} : {val}"])

            if fit is not None:
                writer.writerow([f"{self._CSV_COMMENT} Model     : {fit.model.name}"])
                writer.writerow([f"{self._CSV_COMMENT} Equation  : {fit.equation}"])
                writer.writerow([f"{self._CSV_COMMENT} R_squared : {fit.r_squared:.8f}"])
                coeff_str = "  ".join(f"{c:.10g}" for c in fit.coefficients)
                writer.writerow([f"{self._CSV_COMMENT} Coefficients: {coeff_str}"])

            # --- column headers ---
            writer.writerow(["Voltage (V)", "Current (A)"])

            # --- data rows ---
            for row in data:
                writer.writerow([
                    f"{row['voltage']:.10E}",
                    f"{row['current']:.10E}",
                ])

        logger.info("CSV saved: %s  (%d rows)", path, len(data))
        return path

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def save_json(
        self,
        path: str | Path,
        data: np.ndarray,
        fit: Optional[FitResult] = None,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Write sweep data and fit results to a structured JSON file.

        Parameters
        ----------
        path:
            Destination path.  The ``.json`` extension is added if absent.
        data, fit, metadata:
            Same semantics as save_csv().

        Returns
        -------
        Path
            Resolved path of the written file.
        """
        path = self._resolve_path(path, ".json")
        timestamp = datetime.now().isoformat(timespec="seconds")

        payload: dict = {
            "timestamp": timestamp,
            "metadata": metadata or {},
            "sweep": {
                "points": int(len(data)),
                "voltage": {
                    "unit": "V",
                    "values": data["voltage"].tolist(),
                },
                "current": {
                    "unit": "A",
                    "values": data["current"].tolist(),
                },
            },
        }

        if fit is not None:
            payload["fit"] = {
                "model": fit.model.name,
                "coefficients": list(fit.coefficients),
                "r_squared": fit.r_squared,
                "equation": fit.equation,
            }

        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        logger.info("JSON saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(path: str | Path, default_suffix: str) -> Path:
        p = Path(path)
        if p.suffix.lower() != default_suffix:
            p = p.with_suffix(default_suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.resolve()
