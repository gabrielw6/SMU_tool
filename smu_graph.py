"""Configure a Keysight B2901A SMU to measure resistance via PyVISA.

This script initializes the instrument, sources a user-defined voltage,
applies a current compliance limit, and reads the resulting resistance.
Provides both CLI and GUI interfaces.
"""

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

from smu_interface import (
    configure_b2901a_for_resistance,
    configure_b2901a_for_resistance_sweep,
    interactive_scpi_console,
    measure_resistance,
    measure_voltage_sweep,
    open_instrument,
)

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtGui import QFont, QPainter, QPen, QColor
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QRadioButton,
        QButtonGroup, QGroupBox, QFormLayout, QMessageBox, QProgressDialog, QTextEdit, QFileDialog
    )
except ImportError:
    try:
        from PySide2.QtCore import Qt, QThread, Signal
        from PySide2.QtGui import QFont, QPainter, QPen, QColor
        from PySide2.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
            QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QRadioButton,
            QButtonGroup, QGroupBox, QFormLayout, QMessageBox, QProgressDialog, QTextEdit, QFileDialog
        )
    except ImportError as exc:
        raise ImportError(
            "PySide6 or PySide2 is required for GUI mode. Install one of them with `pip install PySide6` or `pip install PySide2`."
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configure a Keysight B2901A to measure resistance via PyVISA."
    )
    parser.add_argument(
        "--resource",
        default=None,
        help=("VISA resource string or TCP/IP address for the B2901A instrument, "
              "e.g. 192.168.1.100 or TCPIP0::192.168.1.100::INSTR"),
    )
    parser.add_argument(
        "--source-level",
        type=float,
        default=None,
        help="Source level: voltage in volts for voltage mode, current in amperes for current mode.",
    )
    parser.add_argument(
        "--compliance",
        type=float,
        default=None,
        help="Compliance limit in amperes for voltage source, or volts for current source.",
    )
    parser.add_argument(
        "--source",
        choices=["voltage", "current"],
        default="voltage",
        help="Source mode for the measurement. Default is voltage.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Enable sweep mode using start, stop, and step count.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Starting source level for sweep mode.",
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=None,
        help="Stopping source level for sweep mode.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of steps for sweep mode.",
    )
    parser.add_argument(
        "-g",
        "--gui",
        action="store_true",
        help="Open a PySide GUI window with the sweep plot after data fetch.",
    )
    parser.add_argument(
        "-x",
        "--export",
        nargs="?",
        const="",
        default=None,
        help=(
            "Export measurement data to a CSV file. "
            "If a filename is provided, save there; otherwise a default .csv name is generated."
        ),
    )
    parser.add_argument(
        "--order",
        type=int,
        choices=[1, 2],
        default=2,
        help="Model order to fit to the sweep data: 1 for linear, 2 for quadratic.",
    )
    parser.add_argument(
        "--diode",
        action="store_true",
        help="Fit diode model: ln(I) = ln(I_0) + V/V_T, displayed as I = I_0 * exp(V/V_T).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Settling delay in seconds before reading the measurement.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5000,
        help="VISA timeout in milliseconds.",
    )
    parser.add_argument(
        "-t",
        "--terminal",
        action="store_true",
        help="Open an interactive SCPI command line after the resistance measurement.",
    )
    return parser.parse_args()


def _build_default_export_path(prefix="smu", sweep=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "sweep" if sweep else "measurement"
    filename = f"{prefix}_{mode}_{timestamp}.csv"
    return Path(filename).resolve()


def _write_csv(path, rows, headers=None):
    path = Path(path)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if headers:
            writer.writerow(headers)
        writer.writerows(rows)
    return path


def _export_data(export_flag, rows, headers=None):
    if export_flag is None:
        return None
    if export_flag == "":
        export_path = _build_default_export_path()
    else:
        export_path = Path(export_flag)
    saved_path = _write_csv(export_path, rows, headers=headers)
    print(f"Exported CSV data to {saved_path}")
    return saved_path


class SweepThread(QThread):
    """Thread for running sweep measurements without blocking UI."""
    
    finished = Signal(list)  # Emits list of (voltage, current) tuples
    error = Signal(str)      # Emits error message
    
    def __init__(self, instrument, start_val, stop_val, points, compliance):
        super().__init__()
        self.instrument = instrument
        self.start_val = start_val
        self.stop_val = stop_val
        self.points = points
        self.compliance = compliance
    
    def run(self):
        try:
            configure_b2901a_for_resistance_sweep(
                self.instrument,
                self.start_val,
                self.stop_val,
                self.points,
                self.compliance,
            )
            result = measure_voltage_sweep(self.instrument)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class SMUControlPanel(QWidget):
    """Left panel with device connectivity and acquisition parameters."""

    def __init__(self, parent=None, plot_callback=None, recalc_callback=None):
        super().__init__(parent)
        self.instrument = None
        self.plot_callback = plot_callback  # Callback to update plot in main window
        self.recalc_callback = recalc_callback
        self.sweep_thread = None
        self.model_type = "resistive"
        self.model_order = 2
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Device Connectivity Section
        conn_group = QGroupBox("Device Connectivity")
        conn_layout = QFormLayout()

        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("e.g., 192.168.1.100")
        self.ip_input.setText("11.1.0.2")  # Default IP
        conn_layout.addRow("IP Address:", self.ip_input)

        self.port_input = QSpinBox()
        self.port_input.setMinimum(1)
        self.port_input.setMaximum(65535)
        self.port_input.setValue(5025)
        conn_layout.addRow("Port:", self.port_input)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect_toggle)
        conn_layout.addRow(self.connect_btn)

        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet("color: #CCAA00;")  # Yellow
        conn_layout.addRow("Status:", self.status_label)

        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        # Model Configuration Section
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()

        # Model type
        type_layout = QHBoxLayout()
        self.type_group = QButtonGroup()
        self.resistive_radio = QRadioButton("Resistive")
        self.resistive_radio.setChecked(True)
        self.semiconductor_radio = QRadioButton("Semiconductor")
        self.type_group.addButton(self.resistive_radio)
        self.type_group.addButton(self.semiconductor_radio)
        type_layout.addWidget(self.resistive_radio)
        type_layout.addWidget(self.semiconductor_radio)
        model_layout.addLayout(type_layout)

        # Model order
        order_layout = QHBoxLayout()
        self.order_group = QButtonGroup()
        self.first_radio = QRadioButton("First Order")
        self.second_radio = QRadioButton("Second Order")
        self.second_radio.setChecked(True)
        self.order_group.addButton(self.first_radio)
        self.order_group.addButton(self.second_radio)
        order_layout.addWidget(self.first_radio)
        order_layout.addWidget(self.second_radio)
        model_layout.addLayout(order_layout)

        # Recalculate using currently selected model settings and cached sweep data
        self.recalc_model_btn = QPushButton("Recalculate Model")
        self.recalc_model_btn.clicked.connect(self.on_recalculate_model_clicked)
        model_layout.addWidget(self.recalc_model_btn)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Acquisition Parameters Section
        acq_group = QGroupBox("Acquisition Parameters")
        acq_layout = QFormLayout()

        self.sweep_points_spin = QSpinBox()
        self.sweep_points_spin.setMinimum(2)
        self.sweep_points_spin.setMaximum(1000)
        self.sweep_points_spin.setValue(101)
        acq_layout.addRow("Sweep Points:", self.sweep_points_spin)

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setMinimum(0.0)
        self.start_spin.setMaximum(10.0)
        self.start_spin.setValue(0.1)
        self.start_spin.setDecimals(4)
        self.start_spin.setSingleStep(0.01)
        acq_layout.addRow("Start Value (V):", self.start_spin)

        self.stop_spin = QDoubleSpinBox()
        self.stop_spin.setMinimum(0.0)
        self.stop_spin.setMaximum(10.0)
        self.stop_spin.setValue(1.0)
        self.stop_spin.setDecimals(4)
        self.stop_spin.setSingleStep(0.01)
        acq_layout.addRow("Stop Value (V):", self.stop_spin)

        # Source selection
        source_layout = QHBoxLayout()
        self.source_group = QButtonGroup()
        self.voltage_radio = QRadioButton("Voltage")
        self.voltage_radio.setChecked(True)
        self.current_radio = QRadioButton("Current")
        self.source_group.addButton(self.voltage_radio, 0)
        self.source_group.addButton(self.current_radio, 1)
        source_layout.addWidget(self.voltage_radio)
        source_layout.addWidget(self.current_radio)
        acq_layout.addRow("Source Mode:", source_layout)

        self.compliance_spin = QDoubleSpinBox()
        self.compliance_spin.setMinimum(0.001)
        self.compliance_spin.setMaximum(10.0)
        self.compliance_spin.setValue(0.5)
        self.compliance_spin.setDecimals(4)
        self.compliance_spin.setSingleStep(0.01)
        acq_layout.addRow("Compliance (A):", self.compliance_spin)

        # Control Buttons
        button_layout = QHBoxLayout()
        
        config_button_group = QVBoxLayout()
        self.send_config_btn = QPushButton("Send Configuration")
        self.send_config_btn.clicked.connect(self.on_send_config)
        self.send_config_btn.setEnabled(False)
        config_button_group.addWidget(self.send_config_btn)
        
        self.config_status_label = QLabel("")
        self.config_status_label.setAlignment(Qt.AlignCenter)
        self.config_status_label.setStyleSheet("font-size: 9pt;")
        config_button_group.addWidget(self.config_status_label)
        button_layout.addLayout(config_button_group)
        
        sweep_button_group = QVBoxLayout()
        self.start_sweep_btn = QPushButton("Start Sweep")
        self.start_sweep_btn.clicked.connect(self.on_start_sweep)
        self.start_sweep_btn.setEnabled(False)
        sweep_button_group.addWidget(self.start_sweep_btn)
        
        self.sweep_status_label = QLabel("")
        self.sweep_status_label.setAlignment(Qt.AlignCenter)
        self.sweep_status_label.setStyleSheet("font-size: 9pt;")
        sweep_button_group.addWidget(self.sweep_status_label)
        button_layout.addLayout(sweep_button_group)
        
        acq_layout.addRow(button_layout)

        acq_group.setLayout(acq_layout)
        layout.addWidget(acq_group)

        # Connect model signals
        self.resistive_radio.toggled.connect(self.on_model_type_changed)
        self.semiconductor_radio.toggled.connect(self.on_model_type_changed)
        self.first_radio.toggled.connect(self.on_model_order_changed)
        self.second_radio.toggled.connect(self.on_model_order_changed)

        layout.addStretch()
        self.setLayout(layout)

    def on_connect_toggle(self):
        """Toggle between connect and disconnect based on current state."""
        if self.is_connected():
            self.on_disconnect()
        else:
            self.on_connect()

    def on_connect(self):
        """Attempt to connect to the device."""
        if not self.ip_input.text().strip():
            QMessageBox.warning(self, "Input Error", "Please enter an IP address.")
            return

        resource = f"{self.ip_input.text()}:{self.port_input.value()}"
        try:
            self.instrument = open_instrument(resource, timeout_ms=5000)
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("color: #00CC00;")  # Green
            self.connect_btn.setText("Disconnect")
            self.ip_input.setEnabled(False)
            self.port_input.setEnabled(False)
            self.send_config_btn.setEnabled(True)
            self.start_sweep_btn.setEnabled(True)
        except Exception as exc:
            QMessageBox.critical(self, "Connection Error", f"Failed to connect: {exc}")
            self.status_label.setText("Connection failed")
            self.status_label.setStyleSheet("color: #CC0000;")  # Red

    def on_disconnect(self):
        """Disconnect from the device."""
        if self.instrument:
            try:
                self.instrument.close()
            except Exception as exc:
                QMessageBox.warning(self, "Disconnect Warning", f"Error during disconnect: {exc}")
            self.instrument = None
        
        self.status_label.setText("Not connected")
        self.status_label.setStyleSheet("color: #CCAA00;")  # Yellow
        self.connect_btn.setText("Connect")
        self.ip_input.setEnabled(True)
        self.port_input.setEnabled(True)
        self.send_config_btn.setEnabled(False)
        self.start_sweep_btn.setEnabled(False)
        self.config_status_label.setText("")
        self.sweep_status_label.setText("")

    def on_recalculate_model_clicked(self):
        """Request a model recalculation using the current settings."""
        if self.recalc_callback:
            self.recalc_callback()

    def on_send_config(self):
        """Send configuration to the instrument."""
        if not self.is_connected():
            self.config_status_label.setText("Not Connected")
            self.config_status_label.setStyleSheet("color: #CC0000; font-size: 9pt;")
            return
        
        try:
            start = self.start_spin.value()
            stop = self.stop_spin.value()
            points = self.sweep_points_spin.value()
            compliance = self.compliance_spin.value()
            
            if start >= stop:
                self.config_status_label.setText("Invalid Range")
                self.config_status_label.setStyleSheet("color: #CC0000; font-size: 9pt;")
                return
            
            configure_b2901a_for_resistance_sweep(
                self.instrument,
                start,
                stop,
                points,
                compliance,
            )
            self.config_status_label.setText("Configuration Sent ✓")
            self.config_status_label.setStyleSheet("color: #00CC00; font-size: 9pt;")
        except Exception as exc:
            self.config_status_label.setText(f"Failed: {str(exc)[:40]}")
            self.config_status_label.setStyleSheet("color: #CC0000; font-size: 9pt;")

    def on_start_sweep(self):
        """Initiate the sweep measurement in a separate thread."""
        if not self.is_connected():
            self.sweep_status_label.setText("Not Connected")
            self.sweep_status_label.setStyleSheet("color: #CC0000; font-size: 9pt;")
            return
        
        start = self.start_spin.value()
        stop = self.stop_spin.value()
        points = self.sweep_points_spin.value()
        compliance = self.compliance_spin.value()
        
        if start >= stop:
            self.sweep_status_label.setText("Invalid Range")
            self.sweep_status_label.setStyleSheet("color: #CC0000; font-size: 9pt;")
            return
        
        # Clear status and show running
        self.sweep_status_label.setText("Running...")
        self.sweep_status_label.setStyleSheet("color: #CCAA00; font-size: 9pt;")
        
        # Disable buttons during sweep
        self.start_sweep_btn.setEnabled(False)
        self.send_config_btn.setEnabled(False)
        
        # Start sweep in separate thread
        self.sweep_thread = SweepThread(self.instrument, start, stop, points, compliance)
        self.sweep_thread.finished.connect(self.on_sweep_finished)
        self.sweep_thread.error.connect(self.on_sweep_error)
        self.sweep_thread.start()

    def on_sweep_finished(self, result):
        """Handle sweep completion."""
        self.start_sweep_btn.setEnabled(True)
        self.send_config_btn.setEnabled(True)
        
        # Extract voltages and currents
        voltages = [float(v) for v, _ in result if v != ""]
        currents = [float(i) for _, i in result if i != ""]
        
        if self.plot_callback:
            self.plot_callback(voltages, currents)
        
        self.sweep_status_label.setText(f"Complete: {len(voltages)} points ✓")
        self.sweep_status_label.setStyleSheet("color: #00CC00; font-size: 9pt;")

    def on_sweep_error(self, error_msg):
        """Handle sweep error."""
        self.start_sweep_btn.setEnabled(True)
        self.send_config_btn.setEnabled(True)
        truncated_error = error_msg[:40] if len(error_msg) > 40 else error_msg
        self.sweep_status_label.setText(f"Error: {truncated_error}")
        self.sweep_status_label.setStyleSheet("color: #CC0000; font-size: 9pt;")

    def is_connected(self):
        return self.instrument is not None

    def on_model_type_changed(self):
        if self.semiconductor_radio.isChecked():
            self.model_type = "semiconductor"
            self.first_radio.setEnabled(False)
            self.second_radio.setEnabled(False)
        else:
            self.model_type = "resistive"
            self.first_radio.setEnabled(True)
            self.second_radio.setEnabled(True)

    def on_model_order_changed(self):
        if self.first_radio.isChecked():
            self.model_order = 1
        else:
            self.model_order = 2

    def disconnect(self):
        """Public disconnect method for external callers (e.g., window close event)."""
        self.on_disconnect()


class SMUMainWindow(QMainWindow):
    """Main GUI window combining control panel and plot area."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMU Control Interface")
        # Window geometry will be set in gui_main based on screen size

        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Right plot area (will be updated dynamically)
        self.plot_area = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_label = QLabel("Plot area\n(Results will appear here after measurement)")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_layout.addWidget(self.plot_label)
        self.plot_area.setLayout(self.plot_layout)

        # Left control panel with plot callback
        self.current_voltages = []
        self.current_currents = []
        self.control_panel = SMUControlPanel(parent=self, plot_callback=self.on_plot_sweep_data, recalc_callback=self.recalculate_model)
        main_layout.addWidget(self.control_panel, 1)
        main_layout.addWidget(self.plot_area, 2)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def recalculate_model(self):
        """Recalculate the model curve using the last acquired sweep data."""
        if not self.current_voltages or not self.current_currents:
            QMessageBox.warning(self, "No Data", "No sweep data available to recalculate the model.")
            return
        self.on_plot_sweep_data(self.current_voltages, self.current_currents)

    def on_plot_sweep_data(self, voltages, currents):
        """Update plot area with sweep data and model curve."""
        try:
            from smu_plot import plot_sweep_data
        except ImportError:
            QMessageBox.critical(self, "Import Error", "Could not import smu_plot module.")
            return
        
        # Clear previous layout
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    sublayout = item.layout()
                    if sublayout:
                        clear_layout(sublayout)
        
        clear_layout(self.plot_layout)
        
        # Create a new plot widget (in-process, not a separate window)
        # We'll use the SweepPlotWidget from smu_plot
        try:
            from smu_plot import SweepPlotWidget, fit_linear, fit_quadratic, fit_diode
            
            model_type = self.control_panel.model_type
            model_order = self.control_panel.model_order
            
            if model_type == "resistive":
                if model_order == 1:
                    coefficients = fit_linear(voltages, currents)
                    slope, intercept = coefficients
                    order = 1
                    diode = False
                    if abs(slope) > 1e-18:
                        equation_text = (
                            f"Linear Fit:\n"
                            f"y(x) = {slope:.6g} x + {intercept:.6g}\n"
                            f"R [Ω] = {1.0 / slope:.6g}"
                        )
                    else:
                        equation_text = f"Linear Fit:\ny(x) = {slope:.6g} x + {intercept:.6g}\nR [Ω] = undefined"
                else:
                    coefficients = fit_quadratic(voltages, currents)
                    a, b, c = coefficients
                    order = 2
                    diode = False
                    equation_text = (
                        f"Quadratic Fit:\ny(x) = {a:.6g} x² + {b:.6g} x + {c:.6g}"
                        f"\nR [Ω] = {1.0 / b:.6g}"
                    )
            else:  # semiconductor
                coefficients = fit_diode(voltages, currents)
                slope, intercept = coefficients
                I0 = math.exp(intercept)
                VT = 1.0 / slope if slope != 0 else float('inf')
                order = 1
                diode = True
                equation_text = f"Diode Model:\nI = {I0:.6g} * exp(V / {VT:.6g})"
            
            plot_widget = SweepPlotWidget(
                voltages, currents, coefficients=coefficients, order=order, diode=diode
            )
            self.plot_widget = plot_widget
            self.current_voltages = voltages
            self.current_currents = currents
            self.plot_layout.addWidget(plot_widget)
            self.current_voltages = voltages
            self.current_currents = currents
            
            # Equation display as editable text
            self.equation_text_edit = QTextEdit()
            self.equation_text_edit.setPlainText(equation_text)
            self.equation_text_edit.setReadOnly(True)
            self.equation_text_edit.setMaximumHeight(80)
            self.plot_layout.addWidget(self.equation_text_edit)
            
            # Copy button
            copy_layout = QHBoxLayout()
            copy_layout.addStretch()
            copy_button = QPushButton("Copy")
            copy_button.clicked.connect(lambda: QApplication.clipboard().setText(equation_text))
            copy_layout.addWidget(copy_button)
            self.plot_layout.addLayout(copy_layout)
            
            # Export buttons
            export_layout = QHBoxLayout()
            png_btn = QPushButton("Export PNG")
            png_btn.clicked.connect(self.export_png)
            csv_btn = QPushButton("Export CSV")
            csv_btn.clicked.connect(self.export_csv)
            export_layout.addWidget(png_btn)
            export_layout.addWidget(csv_btn)
            self.plot_layout.addLayout(export_layout)
            
        except Exception as exc:
            error_label = QLabel(f"Error displaying plot: {exc}")
            self.plot_layout.addWidget(error_label)

    def export_png(self):
        """Export the plot as PNG image."""
        if not hasattr(self, 'plot_widget'):
            QMessageBox.warning(self, "No Plot", "No plot available to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png)")
        if filename:
            pixmap = self.plot_widget.grab()
            if not pixmap.save(filename, "PNG"):
                QMessageBox.critical(self, "Export Error", "Failed to save PNG file.")

    def export_csv(self):
        """Export the data as CSV file."""
        if not hasattr(self, 'current_voltages'):
            QMessageBox.warning(self, "No Data", "No data available to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")
        if filename:
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Voltage (V)", "Current (A)"])
                    for v, c in zip(self.current_voltages, self.current_currents):
                        writer.writerow([v, c])
                QMessageBox.information(self, "Export Success", f"Data exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save CSV: {e}")

    def closeEvent(self, event):
        """Handle window close event with graceful instrument disconnect."""
        # If connected, ask for confirmation
        if self.control_panel.is_connected():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "The instrument is still connected. Disconnect and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

        # Gracefully disconnect from the instrument
        self.control_panel.disconnect()
        event.accept()

        # Cleanly exit the application
        QApplication.quit()


def gui_main():
    """Launch the GUI interface."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = SMUMainWindow()
    
    # Set window size to 90% of available screen area
    screen = app.primaryScreen()
    available_geometry = screen.availableGeometry()
    width = int(available_geometry.width() * 0.9)
    height = int(available_geometry.height() * 0.9)
    x = (available_geometry.width() - width) // 2
    y = (available_geometry.height() - height) // 2
    window.setGeometry(x, y, width, height)
    
    window.show()
    sys.exit(app.exec())


def cli_main(args):
    """Execute CLI mode with provided arguments."""
    if args.resource is None or args.source_level is None or args.compliance is None:
        print("Error: --resource, --source-level, and --compliance are required for CLI mode.", file=sys.stderr)
        sys.exit(1)

    try:
        instrument = open_instrument(args.resource, timeout_ms=args.timeout)
    except Exception as exc:
        print(f"Failed to open instrument: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.sweep:
            if args.start is None or args.stop is None or args.steps is None:
                raise ValueError("Sweep mode requires --start, --stop, and --steps.")
            if args.source != "voltage":
                raise ValueError("Sweep mode currently supports voltage source only.")
            configure_b2901a_for_resistance_sweep(
                instrument,
                args.start,
                args.stop,
                args.steps,
                args.compliance,
            )
            result = measure_voltage_sweep(instrument)
            print("Sweep measurement result:")
            print("Voltage (V), Current (A)")
            for voltage, current in result:
                print(f"{voltage}, {current}")
            if args.gui:
                print("Opening GUI plot...")
                try:
                    from smu_plot import plot_sweep_data
                except ImportError as exc:
                    raise RuntimeError(
                        "PySide is required to show the GUI plot. Install PySide6 or PySide2."
                    ) from exc

                voltages = [float(v) for v, _ in result if v != ""]
                currents = [float(i) for _, i in result if i != ""]
                plot_sweep_data(voltages, currents, order=args.order, diode=args.diode)

            if args.export is not None:
                _export_data(args.export, result, headers=["Voltage (V)", "Current (A)"])
        else:
            configure_b2901a_for_resistance(instrument, args.source, args.source_level, args.compliance)
            result = measure_resistance(instrument, args.delay)
            print("Resistance measurement result:")
            for label, value in result:
                print(f"  {label}: {value}")

            if args.export is not None:
                _export_data(args.export, result, headers=["Label", "Value"])

        if args.terminal:
            interactive_scpi_console(instrument)
    except Exception as exc:
        print(f"Error during measurement: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            instrument.close()
        except Exception:
            pass


def main():
    """Entry point that determines GUI or CLI mode."""
    args = parse_args()

    # Check if running in CLI mode (has required CLI arguments) or GUI mode
    is_cli_mode = (
        args.resource is not None or
        args.source_level is not None or
        args.compliance is not None or
        args.sweep or
        args.terminal or
        args.export is not None
    )

    if is_cli_mode:
        cli_main(args)
    else:
        gui_main()


if __name__ == "__main__":
    main()
