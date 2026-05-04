"""Hardware abstraction layer for the Keysight B2901A SMU.

All SCPI communication is encapsulated here.  No GUI or plotting imports.
Dependency-inject an instance of B2901A into higher-level components.
"""

from __future__ import annotations

import logging
from enum import StrEnum

import numpy as np

try:
    import pyvisa
except ImportError:
    pyvisa = None

logger = logging.getLogger(__name__)


class ScpiCmd(StrEnum):
    """Static SCPI commands used by the B2901A driver.

    Parametric commands (those requiring a numeric argument) are built
    inline with f-strings so the enum stays clean string constants only.
    """

    RESET = "*RST"
    CLEAR = "*CLS"
    OUTPUT_ON = "OUTP ON"
    OUTPUT_OFF = "OUTP OFF"
    FORM_ELEMENTS_VOLT_CURR = "FORM:ELEMENTS:SENSE VOLT,CURR"
    SOUR_FUNC_VOLT = "SOUR:FUNC:MODE VOLT"
    SOUR_VOLT_MODE_SWEEP = "SOUR:VOLT:MODE SWE"
    SENS_FUNC_CURR = "SENS:FUNC 'CURR'"
    SENS_CURR_RANG_AUTO = "SENS:CURR:RANG:AUTO ON"
    TRIG_DEL_ZERO = "TRIG:DEL 0"
    READ_ARRAY = "READ:ARRAY? (@1)"
    SYST_ERR = ":SYST:ERR?"


class B2901A:
    """Hardware driver for the Keysight B2901A Source-Measure Unit.

    Opens a PyVISA session on construction.  Call close() (or use as a
    context manager) to release the VISA resource when finished.

    Example
    -------
    with B2901A("11.1.0.2:5025") as smu:
        data = smu.run_iv_sweep(0.1, 1.0, 101, compliance=0.1)
    """

    _DEFAULT_NPLC: float = 1.0
    _DEFAULT_CURRENT_LIMIT: float = 0.1
    _DEFAULT_VOLTAGE_LIMIT: float = 10.0

    def __init__(self, resource: str, timeout_ms: int = 5000) -> None:
        if pyvisa is None:
            raise RuntimeError(
                "PyVISA is not installed.  Run: pip install pyvisa pyvisa-py"
            )
        resource = self._normalize_resource(resource)
        logger.info("Opening VISA resource: %s", resource)
        self._rm = pyvisa.ResourceManager()
        self._inst = self._rm.open_resource(resource)
        self._inst.timeout = timeout_ms
        self._inst.write_termination = "\n"
        self._inst.read_termination = "\n"

        # Cached shadow values — authoritative until a setter changes them
        self._nplc: float = self._DEFAULT_NPLC
        self._current_limit: float = self._DEFAULT_CURRENT_LIMIT
        self._voltage_limit: float = self._DEFAULT_VOLTAGE_LIMIT
        logger.info("Connected to %s", resource)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> B2901A:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_resource(resource: str) -> str:
        """Convert a bare IP or IP:port string to a full VISA resource string."""
        if "::" in resource:
            return resource
        if ":" in resource and resource.count(":") == 1:
            host, port = resource.split(":", 1)
            if port.isdigit():
                return f"TCPIP0::{host}::{port}::SOCKET"
        return f"TCPIP0::{resource}::INSTR"

    def _write(self, cmd: str) -> None:
        logger.debug("SCPI << %s", cmd)
        self._inst.write(cmd)

    def _query(self, cmd: str) -> str:
        logger.debug("SCPI ?? %s", cmd)
        response = self._inst.query(cmd)
        logger.debug("SCPI >> %s", response.strip())
        return response

    def _check_error(self) -> None:
        """Query :SYST:ERR? and raise RuntimeError if the instrument reports one."""
        response = self._query(ScpiCmd.SYST_ERR)
        parts = response.split(",", 1)
        try:
            code = int(parts[0].strip())
        except ValueError:
            logger.warning("Unexpected SYST:ERR? response: %s", response.strip())
            return
        if code != 0:
            msg = parts[1].strip().strip('"') if len(parts) > 1 else "unknown"
            raise RuntimeError(f"SCPI error {code}: {msg}")

    # ------------------------------------------------------------------
    # Properties with live SCPI setters
    # ------------------------------------------------------------------

    @property
    def nplc(self) -> float:
        """Integration time in power-line cycles for current sensing."""
        return self._nplc

    @nplc.setter
    def nplc(self, value: float) -> None:
        self._nplc = float(value)
        self._write(f"SENS:CURR:NPLC {self._nplc}")
        self._check_error()
        logger.debug("NPLC set to %g", self._nplc)

    @property
    def current_limit(self) -> float:
        """Compliance current limit in amperes (voltage-source mode)."""
        return self._current_limit

    @current_limit.setter
    def current_limit(self, value: float) -> None:
        self._current_limit = float(value)
        self._write(f"SENS:CURR:PROT:LEVEL {self._current_limit}")
        self._check_error()
        logger.debug("Current limit set to %g A", self._current_limit)

    @property
    def voltage_limit(self) -> float:
        """Compliance voltage limit in volts (current-source mode)."""
        return self._voltage_limit

    @voltage_limit.setter
    def voltage_limit(self, value: float) -> None:
        self._voltage_limit = float(value)
        self._write(f"SOUR:VOLT:ILIM {self._voltage_limit}")
        self._check_error()
        logger.debug("Voltage limit set to %g V", self._voltage_limit)

    # ------------------------------------------------------------------
    # Instrument operations
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Send *RST and *CLS to return the instrument to a known state."""
        self._write(ScpiCmd.RESET)
        self._write(ScpiCmd.CLEAR)
        logger.info("Instrument reset")

    def configure_iv_sweep(
        self,
        start: float,
        stop: float,
        steps: int,
        compliance: float,
    ) -> None:
        """Configure the instrument for a voltage-source IV sweep.

        Does not trigger the sweep.  Call run_iv_sweep() to configure
        and trigger in one step, or call this followed by a manual trigger.
        """
        logger.info(
            "Configuring IV sweep: start=%g V, stop=%g V, steps=%d, compliance=%g A",
            start, stop, steps, compliance,
        )
        self.reset()
        self._write(ScpiCmd.FORM_ELEMENTS_VOLT_CURR)
        self._write(ScpiCmd.SOUR_FUNC_VOLT)
        self._write(ScpiCmd.SOUR_VOLT_MODE_SWEEP)
        self._write(f"SOUR:VOLT:START {start}")
        self._write(f"SOUR:VOLT:STOP {stop}")
        self._write(f"SOUR:VOLT:POIN {steps}")
        self._write(f"SENS:CURR:PROT:LEVEL {compliance}")
        self._write(ScpiCmd.SENS_FUNC_CURR)
        self._write(ScpiCmd.SENS_CURR_RANG_AUTO)
        self._write(f"SENS:CURR:NPLC {self._nplc}")
        self._write(f"TRIG:COUN {steps}")
        self._write(ScpiCmd.TRIG_DEL_ZERO)
        self._write(ScpiCmd.OUTPUT_OFF)
        self._check_error()

    def run_iv_sweep(
        self,
        start: float,
        stop: float,
        steps: int,
        compliance: float,
    ) -> np.ndarray:
        """Configure and run a complete IV sweep.

        Returns a NumPy structured array with fields ``voltage`` (V) and
        ``current`` (A), shape ``(steps,)``.
        """
        self.configure_iv_sweep(start, stop, steps, compliance)
        logger.info("Triggering IV sweep (%d points)", steps)
        self._write(ScpiCmd.OUTPUT_ON)
        try:
            response = self._query(ScpiCmd.READ_ARRAY)
        finally:
            self._write(ScpiCmd.OUTPUT_OFF)

        tokens = [t.strip() for t in response.split(",") if t.strip()]
        pairs = [
            (float(tokens[i]), float(tokens[i + 1]))
            for i in range(0, len(tokens) - 1, 2)
        ]
        dtype = np.dtype([("voltage", np.float64), ("current", np.float64)])
        data = np.array(pairs, dtype=dtype)
        logger.info("Sweep complete: %d points acquired", len(data))
        return data

    def close(self) -> None:
        """Release the VISA instrument and resource manager."""
        logger.info("Closing instrument connection")
        try:
            self._inst.close()
        except Exception:
            logger.debug("Exception closing instrument (ignored)", exc_info=True)
        try:
            self._rm.close()
        except Exception:
            logger.debug("Exception closing resource manager (ignored)", exc_info=True)
