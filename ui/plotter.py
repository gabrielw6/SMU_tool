"""PyQtGraph-based IV-sweep plot widget.

Responsibilities
----------------
* Render raw data points (blue circles).
* Overlay the fitted model curve (red line).
* Export the plot to a PNG via PyQtGraph's ImageExporter.

This module must not import anything from core.driver (no VISA / hardware).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
    from PySide6.QtCore import Qt
except ImportError:
    from PySide2.QtWidgets import QWidget, QVBoxLayout, QLabel  # type: ignore[no-redef]
    from PySide2.QtCore import Qt  # type: ignore[no-redef]

try:
    import pyqtgraph as pg
    from pyqtgraph.exporters import ImageExporter

    # Ensure PyQtGraph uses a white background by default
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

from core.analysis import FitResult, ModelType

logger = logging.getLogger(__name__)


class SweepPlotWidget(QWidget):
    """Embeddable IV-sweep plot widget backed by PyQtGraph.

    When pyqtgraph is not installed the widget shows a plain text
    install instruction instead of crashing the application.

    Usage
    -----
    widget = SweepPlotWidget()
    layout.addWidget(widget)
    widget.update_data(voltage_array, current_array, fit_result)
    """

    # Pen / brush specs
    _DATA_PEN = None          # no connecting line for raw data
    _DATA_BRUSH = (0, 80, 200)
    _DATA_SYMBOL = "o"
    _DATA_SYMBOL_SIZE = 5
    _FIT_PEN_WIDTH = 2
    _FIT_COLOR = (200, 0, 0)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _HAS_PG:
            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setLabel("left", "Current", units="A")
            self._plot_widget.setLabel("bottom", "Voltage", units="V")
            self._plot_widget.setTitle("IV Characteristic")
            self._plot_widget.addLegend(offset=(10, 10))
            self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
            layout.addWidget(self._plot_widget)
            logger.debug("PyQtGraph plot widget initialised")
        else:
            self._plot_widget = None
            msg = QLabel(
                "pyqtgraph is not installed.\n"
                "Install it with:  pip install pyqtgraph"
            )
            msg.setAlignment(Qt.AlignCenter)
            layout.addWidget(msg)
            logger.warning(
                "pyqtgraph not found — install with: pip install pyqtgraph"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_data(
        self,
        voltages: np.ndarray,
        currents: np.ndarray,
        fit: Optional[FitResult] = None,
    ) -> None:
        """Redraw the plot with new sweep data and an optional fit curve.

        Parameters
        ----------
        voltages:
            1-D array of voltage values in volts.
        currents:
            1-D array of measured current values in amperes.
        fit:
            FitResult from core.analysis, or None to plot data only.
        """
        if not _HAS_PG or self._plot_widget is None:
            return

        self._plot_widget.clear()

        v = np.asarray(voltages, dtype=float)
        i = np.asarray(currents, dtype=float)

        # --- raw data scatter ---
        self._plot_widget.plot(
            v, i,
            pen=self._DATA_PEN,
            symbol=self._DATA_SYMBOL,
            symbolSize=self._DATA_SYMBOL_SIZE,
            symbolBrush=pg.mkBrush(*self._DATA_BRUSH),
            symbolPen=pg.mkPen(None),
            name="Measured data",
        )

        # --- fit curve ---
        if fit is not None:
            fit_pen = pg.mkPen(color=self._FIT_COLOR, width=self._FIT_PEN_WIDTH)
            label = fit.model.name.replace("_", " ").title() + " fit"
            self._plot_widget.plot(
                fit.fitted_voltages,
                fit.fitted_currents,
                pen=fit_pen,
                name=label,
            )
            logger.debug(
                "Plot updated with %s fit (R²=%.4f)", fit.model.name, fit.r_squared
            )

        # Auto-range after drawing
        self._plot_widget.autoRange()

    def export_image(self, path: str) -> None:
        """Export the current plot to a PNG file.

        Parameters
        ----------
        path:
            Destination file path.  The ``.png`` extension is appended if
            not already present.
        """
        if not _HAS_PG or self._plot_widget is None:
            raise RuntimeError(
                "Cannot export image: pyqtgraph is not installed."
            )
        if not path.lower().endswith(".png"):
            path += ".png"
        exporter = ImageExporter(self._plot_widget.plotItem)
        exporter.export(path)
        logger.info("Plot exported to %s", path)
