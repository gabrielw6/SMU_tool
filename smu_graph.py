"""Configure a Keysight B2901A SMU to measure resistance via PyVISA.

This script initializes the instrument, sources a user-defined voltage,
applies a current compliance limit, and reads the resulting resistance.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configure a Keysight B2901A to measure resistance via PyVISA."
    )
    parser.add_argument(
        "--resource",
        required=True,
        help=("VISA resource string or TCP/IP address for the B2901A instrument, "
              "e.g. 192.168.1.100 or TCPIP0::192.168.1.100::INSTR"),
    )
    parser.add_argument(
        "--source-level",
        type=float,
        required=True,
        help="Source level: voltage in volts for voltage mode, current in amperes for current mode.",
    )
    parser.add_argument(
        "--compliance",
        type=float,
        required=True,
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
        help="Starting source level for sweep mode.",
    )
    parser.add_argument(
        "--stop",
        type=float,
        help="Stopping source level for sweep mode.",
    )
    parser.add_argument(
        "--steps",
        type=int,
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


def _build_default_export_path(args, prefix="smu"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "sweep" if args.sweep else "measurement"
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


def _export_data(args, rows, headers=None):
    if args.export is None:
        return None
    if args.export == "":
        export_path = _build_default_export_path(args)
    else:
        export_path = Path(args.export)
    saved_path = _write_csv(export_path, rows, headers=headers)
    print(f"Exported CSV data to {saved_path}")
    return saved_path


def main():
    args = parse_args()
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

            _export_data(args, result, headers=["Voltage (V)", "Current (A)"])
        else:
            configure_b2901a_for_resistance(instrument, args.source, args.source_level, args.compliance)
            result = measure_resistance(instrument, args.delay)
            print("Resistance measurement result:")
            for label, value in result:
                print(f"  {label}: {value}")

            _export_data(args, result, headers=["Label", "Value"])

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


if __name__ == "__main__":
    main()
