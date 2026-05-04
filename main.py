"""Entry point for the Keysight B2901A SMU Control Tool.

Modes
-----
GUI (default)
    Run without --resource to open the graphical interface.  The user
    connects to the instrument from within the UI.

CLI
    Pass --resource to run a headless sweep, fit, and optional export.

Examples
--------
# Launch GUI
python main.py

# CLI sweep, quadratic fit, export CSV
python main.py --resource 192.168.1.100:5025 \\
               --start 0.1 --stop 1.0 --steps 101 \\
               --compliance 0.1 --model quadratic --export sweep.csv

# CLI diode characterisation, export JSON
python main.py --resource 11.1.0.2:5025 \\
               --start 0.3 --stop 0.9 --steps 61 \\
               --compliance 0.5 --model diode --export-format json --export diode.json
"""

from __future__ import annotations

import argparse
import logging
import sys

# Configure root logger before any package imports so all modules inherit it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smu_tool",
        description="Keysight B2901A SMU Control & Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    hw = p.add_argument_group("Hardware")
    hw.add_argument(
        "--resource",
        default=None,
        metavar="IP[:PORT]",
        help="VISA resource string or IP address of the B2901A.  "
             "Omit to launch the GUI.",
    )
    hw.add_argument("--timeout", type=int, default=5000, metavar="MS",
                    help="VISA timeout in milliseconds.")

    sweep = p.add_argument_group("Sweep parameters (CLI mode)")
    sweep.add_argument("--start", type=float, default=0.1, metavar="V",
                       help="Sweep start voltage.")
    sweep.add_argument("--stop", type=float, default=1.0, metavar="V",
                       help="Sweep stop voltage.")
    sweep.add_argument("--steps", type=int, default=101,
                       help="Number of sweep points.")
    sweep.add_argument("--compliance", type=float, default=0.1, metavar="A",
                       help="Current compliance limit.")

    fit = p.add_argument_group("Fitting")
    fit.add_argument(
        "--model",
        choices=["linear", "quadratic", "diode"],
        default="quadratic",
        help="Curve-fit model to apply.",
    )

    out = p.add_argument_group("Output")
    out.add_argument(
        "--export",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Export data.  Omit PATH to auto-generate a timestamped filename.",
    )
    out.add_argument(
        "--export-format",
        choices=["csv", "json"],
        default="csv",
        help="File format for --export.",
    )

    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    p.add_argument(
        "-g", "--gui",
        action="store_true",
        help="Force GUI mode even when --resource is given.",
    )

    return p


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main(args: argparse.Namespace) -> None:
    from core.driver import B2901A
    from core.analysis import fit_diode, fit_linear, fit_quadratic
    from utils.data_manager import DataManager

    logger.info(
        "CLI sweep: resource=%s  start=%g  stop=%g  steps=%d  compliance=%g",
        args.resource, args.start, args.stop, args.steps, args.compliance,
    )

    driver = B2901A(args.resource, timeout_ms=args.timeout)
    try:
        data = driver.run_iv_sweep(
            args.start, args.stop, args.steps, args.compliance
        )
    finally:
        driver.close()

    v, i = data["voltage"], data["current"]

    fit_fn = {"linear": fit_linear, "quadratic": fit_quadratic, "diode": fit_diode}[
        args.model
    ]
    result = fit_fn(v, i)

    print(f"\nCollected {len(data)} points.")
    print(f"Model     : {result.model.name}")
    print(f"Equation  : {result.equation}")
    print(f"R²        : {result.r_squared:.6f}\n")

    if args.export is not None:
        dm = DataManager()
        path = args.export or dm.generate_filename(extension=args.export_format)
        if args.export_format == "json":
            dm.save_json(path, data, result)
        else:
            dm.save_csv(path, data, result)


# ---------------------------------------------------------------------------
# GUI entry point
# ---------------------------------------------------------------------------

def _gui_main(pre_connected_driver=None) -> None:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        from PySide2.QtWidgets import QApplication  # type: ignore[no-redef]

    from ui.main_window import SMUMainWindow

    app = QApplication.instance() or QApplication(sys.argv)

    window = SMUMainWindow(driver=pre_connected_driver)

    # Size the window to 90 % of the available screen area, centred
    screen = app.primaryScreen()
    geo = screen.availableGeometry()
    w = int(geo.width() * 0.9)
    h = int(geo.height() * 0.9)
    x = (geo.width() - w) // 2
    y = (geo.height() - h) // 2
    window.setGeometry(x, y, w, h)

    window.show()
    sys.exit(app.exec())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Apply chosen log level
    logging.getLogger().setLevel(args.log_level)

    if args.resource is not None and not args.gui:
        _cli_main(args)
    else:
        _gui_main()


if __name__ == "__main__":
    main()
