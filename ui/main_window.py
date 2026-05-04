"""Main application window for the SMU Control Interface.

Design rules
------------
* Never calls pyvisa.ResourceManager() directly — all hardware access
  goes through an injected core.driver.B2901A instance.
* Long-running IV sweeps execute on a QThread so the UI stays responsive.
* Fit model selection and recalculation are decoupled from the sweep.
* SCPI traffic is surfaced in the UI via a logging.Handler → Qt Signal
  pipeline, so the driver stays free of any GUI imports.
"""

from __future__ import annotations

import logging
from html import escape as _html_escape
from typing import Optional

import numpy as np

try:
    from PySide6.QtCore import Qt, QThread, Signal, QObject
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QDoubleSpinBox,
        QRadioButton,
        QButtonGroup,
        QGroupBox,
        QFormLayout,
        QMessageBox,
        QTextEdit,
        QFileDialog,
        QSplitter,
    )
except ImportError:
    from PySide2.QtCore import Qt, QThread, Signal, QObject  # type: ignore[no-redef]
    from PySide2.QtWidgets import (  # type: ignore[no-redef]
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QDoubleSpinBox,
        QRadioButton,
        QButtonGroup,
        QGroupBox,
        QFormLayout,
        QMessageBox,
        QTextEdit,
        QFileDialog,
        QSplitter,
    )

from core.analysis import fit_diode, fit_linear, fit_quadratic, FitResult
from ui.plotter import SweepPlotWidget
from utils.data_manager import DataManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SCPI log handler — driver → Qt signal bridge
# ---------------------------------------------------------------------------

class _LogRelay(QObject):
    """Carries formatted log strings safely across thread boundaries."""
    message = Signal(str)


class _ScpiLogHandler(logging.Handler):
    """Logging handler that forwards records to a Qt signal.

    Only records whose message starts with 'SCPI' (raw SCPI traffic) or
    whose level is INFO or above (high-level driver events) are forwarded,
    so the panel stays readable without drowning in internal debug noise.
    """

    def __init__(self, relay: _LogRelay) -> None:
        super().__init__(level=logging.DEBUG)
        self._relay = relay
        self.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        )

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        return record.getMessage().startswith("SCPI")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._relay.message.emit(self.format(record))
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Background sweep worker
# ---------------------------------------------------------------------------

class _SweepWorker(QObject):
    """Runs a single IV sweep on a worker thread.

    Signals
    -------
    finished(np.ndarray):
        Emitted with the structured array on success.
    error(str):
        Emitted with an error message on failure.
    """

    finished: Signal = Signal(object)
    error: Signal = Signal(str)

    def __init__(
        self,
        driver,
        start: float,
        stop: float,
        steps: int,
        compliance: float,
    ) -> None:
        super().__init__()
        self._driver = driver
        self._start = start
        self._stop = stop
        self._steps = steps
        self._compliance = compliance

    def run(self) -> None:
        try:
            data = self._driver.run_iv_sweep(
                self._start, self._stop, self._steps, self._compliance
            )
            self.finished.emit(data)
        except Exception as exc:
            logger.exception("IV sweep failed in worker thread")
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class SMUMainWindow(QMainWindow):
    """Top-level window for the SMU Control Interface.

    Parameters
    ----------
    driver:
        Pre-connected B2901A instance, or None to let the user connect
        from within the UI.
    """

    def __init__(self, driver=None) -> None:
        super().__init__()
        self._driver = driver
        self._sweep_data: Optional[np.ndarray] = None
        self._fit_result: Optional[FitResult] = None
        self._sweep_thread: Optional[QThread] = None
        self._sweep_worker: Optional[_SweepWorker] = None
        self._data_manager = DataManager()
        self.setWindowTitle("SMU Control Interface")
        self._build_ui()
        self._setup_log_handler()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)

        self._control_panel = self._build_control_panel()

        # Right side: plot on top, SCPI log below — user-resizable
        right_splitter = QSplitter(Qt.Vertical)
        self._plot_widget = SweepPlotWidget()
        right_splitter.addWidget(self._plot_widget)
        right_splitter.addWidget(self._build_scpi_log_panel())
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        root_layout.addWidget(self._control_panel, stretch=1)
        root_layout.addWidget(right_splitter, stretch=2)
        self.setCentralWidget(root)

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(self._build_connectivity_group())
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_acquisition_group())
        layout.addWidget(self._build_equation_group())
        layout.addLayout(self._build_export_row())
        layout.addStretch()
        return panel

    def _build_connectivity_group(self) -> QGroupBox:
        group = QGroupBox("Device Connectivity")
        form = QFormLayout()

        self._ip_input = QLineEdit("11.1.0.2")
        self._ip_input.setPlaceholderText("e.g., 192.168.1.100")
        form.addRow("IP Address:", self._ip_input)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(5025)
        form.addRow("Port:", self._port_spin)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._toggle_connection)
        form.addRow(self._connect_btn)

        self._status_label = QLabel("Not connected")
        self._status_label.setStyleSheet("color: #CCAA00;")
        form.addRow("Status:", self._status_label)

        group.setLayout(form)
        return group

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Model Configuration")
        layout = QVBoxLayout()

        type_row = QHBoxLayout()
        self._type_group = QButtonGroup()
        self._resistive_radio = QRadioButton("Resistive")
        self._resistive_radio.setChecked(True)
        self._semiconductor_radio = QRadioButton("Semiconductor")
        self._type_group.addButton(self._resistive_radio)
        self._type_group.addButton(self._semiconductor_radio)
        type_row.addWidget(self._resistive_radio)
        type_row.addWidget(self._semiconductor_radio)
        layout.addLayout(type_row)

        order_row = QHBoxLayout()
        self._order_group = QButtonGroup()
        self._first_radio = QRadioButton("1st Order")
        self._second_radio = QRadioButton("2nd Order")
        self._second_radio.setChecked(True)
        self._order_group.addButton(self._first_radio)
        self._order_group.addButton(self._second_radio)
        order_row.addWidget(self._first_radio)
        order_row.addWidget(self._second_radio)
        layout.addLayout(order_row)

        self._recalc_btn = QPushButton("Recalculate Model")
        self._recalc_btn.clicked.connect(self._recalculate)
        layout.addWidget(self._recalc_btn)

        self._semiconductor_radio.toggled.connect(self._on_model_type_changed)
        group.setLayout(layout)
        return group

    def _build_acquisition_group(self) -> QGroupBox:
        group = QGroupBox("Acquisition Parameters")
        form = QFormLayout()

        self._points_spin = QSpinBox()
        self._points_spin.setRange(2, 1000)
        self._points_spin.setValue(101)
        form.addRow("Sweep Points:", self._points_spin)

        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0.0, 210.0)
        self._start_spin.setValue(0.1)
        self._start_spin.setDecimals(4)
        self._start_spin.setSingleStep(0.01)
        form.addRow("Start (V):", self._start_spin)

        self._stop_spin = QDoubleSpinBox()
        self._stop_spin.setRange(0.0, 210.0)
        self._stop_spin.setValue(1.0)
        self._stop_spin.setDecimals(4)
        self._stop_spin.setSingleStep(0.01)
        form.addRow("Stop (V):", self._stop_spin)

        self._compliance_spin = QDoubleSpinBox()
        self._compliance_spin.setRange(1e-9, 3.0)
        self._compliance_spin.setValue(0.1)
        self._compliance_spin.setDecimals(6)
        self._compliance_spin.setSingleStep(0.01)
        form.addRow("Compliance (A):", self._compliance_spin)

        btn_row = QHBoxLayout()
        self._config_btn = QPushButton("Send Config")
        self._config_btn.clicked.connect(self._send_config)
        self._config_btn.setEnabled(False)
        self._sweep_btn = QPushButton("Start Sweep")
        self._sweep_btn.clicked.connect(self._start_sweep)
        self._sweep_btn.setEnabled(False)
        btn_row.addWidget(self._config_btn)
        btn_row.addWidget(self._sweep_btn)
        form.addRow(btn_row)

        self._sweep_status = QLabel("")
        self._sweep_status.setAlignment(Qt.AlignCenter)
        self._sweep_status.setStyleSheet("font-size: 9pt;")
        form.addRow(self._sweep_status)

        group.setLayout(form)
        return group

    def _build_equation_group(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._equation_edit = QTextEdit()
        self._equation_edit.setReadOnly(True)
        self._equation_edit.setMaximumHeight(90)
        self._equation_edit.setPlaceholderText("Fit equation will appear here…")
        layout.addWidget(self._equation_edit)

        copy_row = QHBoxLayout()
        copy_row.addStretch()
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(
                self._equation_edit.toPlainText()
            )
        )
        copy_row.addWidget(copy_btn)
        layout.addLayout(copy_row)
        return container

    def _build_export_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._png_btn = QPushButton("Export PNG")
        self._png_btn.clicked.connect(self._export_png)
        self._csv_btn = QPushButton("Export CSV")
        self._csv_btn.clicked.connect(self._export_csv)
        self._json_btn = QPushButton("Export JSON")
        self._json_btn.clicked.connect(self._export_json)
        row.addWidget(self._png_btn)
        row.addWidget(self._csv_btn)
        row.addWidget(self._json_btn)
        return row

    def _build_scpi_log_panel(self) -> QGroupBox:
        """SCPI traffic monitor — sits below the plot in the right splitter."""
        group = QGroupBox("SCPI Log")
        layout = QVBoxLayout()

        self._scpi_log = QTextEdit()
        self._scpi_log.setReadOnly(True)
        self._scpi_log.setStyleSheet(
            "QTextEdit {"
            "  font-family: 'Courier New', 'Consolas', monospace;"
            "  font-size: 9pt;"
            "  background-color: #1e1e1e;"
            "  color: #d4d4d4;"
            "}"
        )
        layout.addWidget(self._scpi_log)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._scpi_log.clear)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

        group.setLayout(layout)
        return group

    # ------------------------------------------------------------------
    # SCPI log handler wiring
    # ------------------------------------------------------------------

    def _setup_log_handler(self) -> None:
        """Attach a logging handler to core.driver so all SCPI traffic
        flows into the on-screen log panel."""
        self._log_relay = _LogRelay()
        self._log_relay.message.connect(self._append_log)

        self._log_handler = _ScpiLogHandler(self._log_relay)

        driver_log = logging.getLogger("core.driver")
        driver_log.setLevel(logging.DEBUG)
        driver_log.addHandler(self._log_handler)

    _LOG_RE = __import__("re").compile(
        r"^(\S+)\s+SCPI\s*(<<|\?\?|>>)\s*(.*)", __import__("re").DOTALL
    )

    def _append_log(self, msg: str) -> None:
        """Parse the raw SCPI string out of the log record and display it.

        Format produced:
            HH:MM:SS  >>  *RST                   (TX write,  cyan)
            HH:MM:SS  >>  :SYST:ERR?             (TX query,  cyan)
            HH:MM:SS  <<  +0,"No error"          (RX response, green)
        """
        m = self._LOG_RE.match(msg)
        if m:
            ts, arrow, content = m.group(1), m.group(2), m.group(3).strip()

            if arrow in ("<<", "??"):   # anything sent to the instrument
                direction = ">>"
                color = "#4ec9b0"       # cyan
            else:                       # ">>" — response from instrument
                direction = "<<"
                color = "#b5cea8"       # yellow-green

            ts_html = f'<span style="color:#555555;">{_html_escape(ts)}</span>'
            dir_html = f'<span style="color:{color};font-weight:bold;"> {direction} </span>'
            val_html = f'<span style="color:{color};">{_html_escape(content)}</span>'
            html = ts_html + dir_html + val_html
        else:
            html = f'<span style="color:#666666;">{_html_escape(msg)}</span>'

        self._scpi_log.append(html)

        sb = self._scpi_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _toggle_connection(self) -> None:
        if self._driver is not None:
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        from core.driver import B2901A  # late import — no driver at module load time

        ip = self._ip_input.text().strip()
        if not ip:
            QMessageBox.warning(self, "Input Error", "Enter an IP address.")
            return

        resource = f"{ip}:{self._port_spin.value()}"
        try:
            self._driver = B2901A(resource)
        except Exception as exc:
            logger.error("Connection failed: %s", exc)
            QMessageBox.critical(self, "Connection Error", str(exc))
            self._status_label.setText("Failed")
            self._status_label.setStyleSheet("color: #CC0000;")
            return

        self._status_label.setText("Connected")
        self._status_label.setStyleSheet("color: #00CC00;")
        self._connect_btn.setText("Disconnect")
        self._ip_input.setEnabled(False)
        self._port_spin.setEnabled(False)
        self._config_btn.setEnabled(True)
        self._sweep_btn.setEnabled(True)
        logger.info("Connected to %s", resource)

    def _disconnect(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception as exc:
                logger.warning("Error during disconnect: %s", exc)
            self._driver = None

        self._status_label.setText("Not connected")
        self._status_label.setStyleSheet("color: #CCAA00;")
        self._connect_btn.setText("Connect")
        self._ip_input.setEnabled(True)
        self._port_spin.setEnabled(True)
        self._config_btn.setEnabled(False)
        self._sweep_btn.setEnabled(False)
        self._sweep_status.setText("")
        logger.info("Disconnected")

    # ------------------------------------------------------------------
    # Instrument actions
    # ------------------------------------------------------------------

    def _send_config(self) -> None:
        if self._driver is None:
            return
        try:
            self._driver.configure_iv_sweep(
                self._start_spin.value(),
                self._stop_spin.value(),
                self._points_spin.value(),
                self._compliance_spin.value(),
            )
            self._sweep_status.setText("Config sent ✓")
            self._sweep_status.setStyleSheet("color: #00CC00; font-size: 9pt;")
        except Exception as exc:
            logger.error("configure_iv_sweep failed: %s", exc)
            self._sweep_status.setText(f"Config failed: {str(exc)[:50]}")
            self._sweep_status.setStyleSheet("color: #CC0000; font-size: 9pt;")

    def _start_sweep(self) -> None:
        if self._driver is None:
            return

        start = self._start_spin.value()
        stop = self._stop_spin.value()
        if start >= stop:
            self._sweep_status.setText("Invalid range (start ≥ stop)")
            self._sweep_status.setStyleSheet("color: #CC0000; font-size: 9pt;")
            return

        self._set_sweep_buttons_enabled(False)
        self._sweep_status.setText("Running…")
        self._sweep_status.setStyleSheet("color: #CCAA00; font-size: 9pt;")

        self._sweep_worker = _SweepWorker(
            self._driver,
            start,
            stop,
            self._points_spin.value(),
            self._compliance_spin.value(),
        )
        self._sweep_thread = QThread()
        self._sweep_worker.moveToThread(self._sweep_thread)

        self._sweep_thread.started.connect(self._sweep_worker.run)
        self._sweep_worker.finished.connect(self._on_sweep_done)
        self._sweep_worker.error.connect(self._on_sweep_error)
        self._sweep_worker.finished.connect(self._sweep_thread.quit)
        self._sweep_worker.error.connect(self._sweep_thread.quit)
        self._sweep_thread.finished.connect(self._sweep_worker.deleteLater)

        self._sweep_thread.start()

    def _on_sweep_done(self, data: np.ndarray) -> None:
        self._sweep_data = data
        self._set_sweep_buttons_enabled(True)
        self._sweep_status.setText(f"Complete: {len(data)} points ✓")
        self._sweep_status.setStyleSheet("color: #00CC00; font-size: 9pt;")
        logger.info("Sweep done: %d points", len(data))
        self._apply_fit_and_plot()

    def _on_sweep_error(self, msg: str) -> None:
        self._set_sweep_buttons_enabled(True)
        short = msg[:60] if len(msg) > 60 else msg
        self._sweep_status.setText(f"Error: {short}")
        self._sweep_status.setStyleSheet("color: #CC0000; font-size: 9pt;")
        logger.error("Sweep error: %s", msg)

    def _set_sweep_buttons_enabled(self, enabled: bool) -> None:
        self._sweep_btn.setEnabled(enabled and self._driver is not None)
        self._config_btn.setEnabled(enabled and self._driver is not None)

    # ------------------------------------------------------------------
    # Model / fitting
    # ------------------------------------------------------------------

    def _recalculate(self) -> None:
        if self._sweep_data is None:
            QMessageBox.warning(self, "No Data", "Run a sweep first.")
            return
        self._apply_fit_and_plot()

    def _apply_fit_and_plot(self) -> None:
        data = self._sweep_data
        v = data["voltage"]
        i = data["current"]

        if self._semiconductor_radio.isChecked():
            result = fit_diode(v, i)
        elif self._first_radio.isChecked():
            result = fit_linear(v, i)
        else:
            result = fit_quadratic(v, i)

        self._fit_result = result
        self._plot_widget.update_data(v, i, result)
        self._equation_edit.setPlainText(
            f"{result.equation}\nR² = {result.r_squared:.6f}"
        )
        logger.info(
            "Fit: %s  R²=%.4f  coefficients=%s",
            result.model.name,
            result.r_squared,
            result.coefficients,
        )

    def _on_model_type_changed(self, semiconductor: bool) -> None:
        self._first_radio.setEnabled(not semiconductor)
        self._second_radio.setEnabled(not semiconductor)

    # ------------------------------------------------------------------
    # Export actions
    # ------------------------------------------------------------------

    def _export_png(self) -> None:
        if self._sweep_data is None:
            QMessageBox.warning(self, "No Data", "No sweep data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", "", "PNG image (*.png)"
        )
        if not path:
            return
        try:
            self._plot_widget.export_image(path)
            QMessageBox.information(self, "Saved", f"Plot saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_csv(self) -> None:
        if self._sweep_data is None:
            QMessageBox.warning(self, "No Data", "No sweep data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV file (*.csv)"
        )
        if not path:
            return
        try:
            self._data_manager.save_csv(path, self._sweep_data, self._fit_result)
            QMessageBox.information(self, "Saved", f"Data saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_json(self) -> None:
        if self._sweep_data is None:
            QMessageBox.warning(self, "No Data", "No sweep data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", "", "JSON file (*.json)"
        )
        if not path:
            return
        try:
            self._data_manager.save_json(path, self._sweep_data, self._fit_result)
            QMessageBox.information(self, "Saved", f"Data saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        # Detach the log handler before any teardown so no signals fire
        # into a partially-destroyed widget
        logging.getLogger("core.driver").removeHandler(self._log_handler)

        if self._driver is not None:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "The instrument is still connected.\nDisconnect and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                # Re-attach the handler if the user cancels
                logging.getLogger("core.driver").addHandler(self._log_handler)
                event.ignore()
                return
            self._disconnect()

        event.accept()
        QApplication.quit()
