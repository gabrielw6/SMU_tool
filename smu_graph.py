"""Configure a Keysight B2901A SMU to measure resistance via PyVISA.

This script initializes the instrument, sources a user-defined voltage,
applies a current compliance limit, and reads the resulting resistance.
Provides both CLI and GUI interfaces.
"""

import argparse
import csv
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
        QButtonGroup, QGroupBox, QFormLayout, QMessageBox, QProgressDialog
    )
except ImportError:
    try:
        from PySide2.QtCore import Qt, QThread, Signal
        from PySide2.QtGui import QFont, QPainter, QPen, QColor
        from PySide2.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
            QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QRadioButton,
            QButtonGroup, QGroupBox, QFormLayout, QMessageBox, QProgressDialog
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

    def __init__(self, parent=None, plot_callback=None):
        super().__init__(parent)
        self.instrument = None
        self.plot_callback = plot_callback  # Callback to update plot in main window
        self.sweep_thread = None
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
        
        self.send_config_btn = QPushButton("Send Configuration")
        self.send_config_btn.clicked.connect(self.on_send_config)
        self.send_config_btn.setEnabled(False)
        button_layout.addWidget(self.send_config_btn)
        
        self.start_sweep_btn = QPushButton("Start Sweep")
        self.start_sweep_btn.clicked.connect(self.on_start_sweep)
        self.start_sweep_btn.setEnabled(False)
        button_layout.addWidget(self.start_sweep_btn)
        
        acq_layout.addRow(button_layout)

        acq_group.setLayout(acq_layout)
        layout.addWidget(acq_group)

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

    def on_send_config(self):
        """Send configuration to the instrument."""
        if not self.is_connected():
            QMessageBox.warning(self, "Not Connected", "Please connect to the instrument first.")
            return
        
        try:
            start = self.start_spin.value()
            stop = self.stop_spin.value()
            points = self.sweep_points_spin.value()
            compliance = self.compliance_spin.value()
            
            if start >= stop:
                QMessageBox.warning(self, "Invalid Range", "Start value must be less than stop value.")
                return
            
            configure_b2901a_for_resistance_sweep(
                self.instrument,
                start,
                stop,
                points,
                compliance,
            )
            QMessageBox.information(self, "Configuration Sent", "Sweep configuration has been sent to the instrument.")
        except Exception as exc:
            QMessageBox.critical(self, "Configuration Error", f"Failed to send configuration: {exc}")

    def on_start_sweep(self):
        """Initiate the sweep measurement in a separate thread."""
        if not self.is_connected():
            QMessageBox.warning(self, "Not Connected", "Please connect to the instrument first.")
            return
        
        start = self.start_spin.value()
        stop = self.stop_spin.value()
        points = self.sweep_points_spin.value()
        compliance = self.compliance_spin.value()
        
        if start >= stop:
            QMessageBox.warning(self, "Invalid Range", "Start value must be less than stop value.")
            return
        
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
        
        QMessageBox.information(self, "Sweep Complete", f"Sweep completed. {len(voltages)} data points collected.")

    def on_sweep_error(self, error_msg):
        """Handle sweep error."""
        self.start_sweep_btn.setEnabled(True)
        self.send_config_btn.setEnabled(True)
        QMessageBox.critical(self, "Sweep Error", f"Sweep failed: {error_msg}")

    def is_connected(self):
        return self.instrument is not None

    def disconnect(self):
        """Public disconnect method for external callers (e.g., window close event)."""
        self.on_disconnect()


class SMUMainWindow(QMainWindow):
    """Main GUI window combining control panel and plot area."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMU Control Interface")
        self.setGeometry(100, 100, 1400, 700)

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
        self.control_panel = SMUControlPanel(parent=None, plot_callback=self.on_plot_sweep_data)
        main_layout.addWidget(self.control_panel, 1)
        main_layout.addWidget(self.plot_area, 2)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def on_plot_sweep_data(self, voltages, currents):
        """Update plot area with sweep data and model curve."""
        try:
            from smu_plot import plot_sweep_data
        except ImportError:
            QMessageBox.critical(self, "Import Error", "Could not import smu_plot module.")
            return
        
        # Clear previous layout
        while self.plot_layout.count():
            self.plot_layout.takeAt(0).widget().deleteLater()
        
        # Create a new plot widget (in-process, not a separate window)
        # We'll use the SweepPlotWidget from smu_plot
        try:
            from smu_plot import SweepPlotWidget, fit_linear, fit_quadratic
            
            # Fit linear model for now (default order=2 logic would fit quadratic)
            coefficients = fit_linear(voltages, currents)
            slope, intercept = coefficients
            
            plot_widget = SweepPlotWidget(
                voltages, currents, coefficients=coefficients, order=1, diode=False
            )
            self.plot_layout.addWidget(plot_widget)
            
            # Add equation display
            if abs(slope) > 1e-18:
                equation_text = (
                    f"Linear Fit:\n"
                    f"y(x) = {slope:.6g} x + {intercept:.6g}\n"
                    f"R [Ω] = {1.0 / slope:.6g}"
                )
            else:
                equation_text = f"Linear Fit:\ny(x) = {slope:.6g} x + {intercept:.6g}\nR [Ω] = undefined"
            
            equation_label = QLabel(equation_text)
            equation_label.setStyleSheet("font-family: Courier; font-size: 10pt;")
            equation_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            self.plot_layout.addWidget(equation_label)
            
        except Exception as exc:
            error_label = QLabel(f"Error displaying plot: {exc}")
            self.plot_layout.addWidget(error_label)

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
