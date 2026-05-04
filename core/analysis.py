"""Pure-math analysis layer: LMS fitting and diode modelling.

This module has zero GUI or hardware dependencies and can be used in
headless scripts, notebooks, or unit tests without a display.

Fitting hierarchy
-----------------
* Linear  (resistive, 1st order) — numpy.polyfit degree 1
* Quadratic (resistive, 2nd order) — numpy.polyfit degree 2
* Diode (exponential) — scipy.optimize.curve_fit with log-linear seed;
  falls back to numpy.polyfit on log(I) if scipy is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import numpy as np

try:
    from scipy.optimize import curve_fit as _scipy_curve_fit
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = logging.getLogger(__name__)

_CURVE_POINTS = 300  # number of x samples used for the rendered fit curve


class ModelType(Enum):
    LINEAR = auto()
    QUADRATIC = auto()
    DIODE = auto()


@dataclass
class FitResult:
    """Container for a completed curve-fit.

    Attributes
    ----------
    model:
        Which model was applied.
    coefficients:
        Raw fit coefficients (slope/intercept for linear; a/b/c for
        quadratic; I₀/Vₜ for diode).
    r_squared:
        Coefficient of determination evaluated on the original data points.
    fitted_voltages:
        Dense voltage array spanning the data range (for smooth plotting).
    fitted_currents:
        Model current evaluated at ``fitted_voltages``.
    equation:
        Human-readable equation string.
    """

    model: ModelType
    coefficients: Tuple[float, ...]
    r_squared: float
    fitted_voltages: np.ndarray
    fitted_currents: np.ndarray
    equation: str


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _r_squared(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_actual - y_pred) ** 2))
    ss_tot = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0


def _dense_x(v: np.ndarray, n: int = _CURVE_POINTS) -> np.ndarray:
    return np.linspace(v.min(), v.max(), n)


# ---------------------------------------------------------------------------
# Public fitting functions
# ---------------------------------------------------------------------------

def fit_linear(voltages, currents) -> FitResult:
    """Fit I = slope·V + intercept using numpy.polyfit.

    Returns a FitResult with dense curve data for smooth plotting.
    """
    v = np.asarray(voltages, dtype=float)
    i = np.asarray(currents, dtype=float)

    if len(v) < 2:
        logger.warning("fit_linear: only %d point(s) — returning zero model", len(v))
        return FitResult(
            model=ModelType.LINEAR,
            coefficients=(0.0, 0.0),
            r_squared=0.0,
            fitted_voltages=v.copy(),
            fitted_currents=np.zeros_like(v),
            equation="I(V) = 0  [insufficient data]",
        )

    coeffs = np.polyfit(v, i, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    r2 = _r_squared(i, np.polyval(coeffs, v))

    v_dense = _dense_x(v)
    i_dense = np.polyval(coeffs, v_dense)

    resistance = 1.0 / slope if abs(slope) > 1e-18 else float("inf")
    if np.isfinite(resistance):
        eq = f"I(V) = {slope:.6g}·V + {intercept:.6g}   [R = {resistance:.6g} Ω]"
    else:
        eq = f"I(V) = {slope:.6g}·V + {intercept:.6g}   [R = ∞]"

    logger.debug("Linear fit: slope=%g, intercept=%g, R²=%g", slope, intercept, r2)
    return FitResult(
        model=ModelType.LINEAR,
        coefficients=(slope, intercept),
        r_squared=r2,
        fitted_voltages=v_dense,
        fitted_currents=i_dense,
        equation=eq,
    )


def fit_quadratic(voltages, currents) -> FitResult:
    """Fit I = a·V² + b·V + c using numpy.polyfit.

    Falls back to fit_linear when fewer than 3 data points are available.
    The linear coefficient b approximates the small-signal conductance.
    """
    v = np.asarray(voltages, dtype=float)
    i = np.asarray(currents, dtype=float)

    if len(v) < 3:
        logger.warning("fit_quadratic: only %d point(s) — falling back to linear", len(v))
        return fit_linear(v, i)

    coeffs = np.polyfit(v, i, 2)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    r2 = _r_squared(i, np.polyval(coeffs, v))

    v_dense = _dense_x(v)
    i_dense = np.polyval(coeffs, v_dense)

    resistance = 1.0 / b if abs(b) > 1e-18 else float("inf")
    r_str = f"{resistance:.6g} Ω" if np.isfinite(resistance) else "∞"
    eq = f"I(V) = {a:.6g}·V² + {b:.6g}·V + {c:.6g}   [R ≈ {r_str}]"

    logger.debug("Quadratic fit: a=%g, b=%g, c=%g, R²=%g", a, b, c, r2)
    return FitResult(
        model=ModelType.QUADRATIC,
        coefficients=(a, b, c),
        r_squared=r2,
        fitted_voltages=v_dense,
        fitted_currents=i_dense,
        equation=eq,
    )


def _diode_curve(v: np.ndarray, i0: float, vt: float) -> np.ndarray:
    """Shockley diode equation: I = I₀ · exp(V / Vₜ)."""
    return i0 * np.exp(np.clip(v / vt, -500, 500))


def fit_diode(voltages, currents) -> FitResult:
    """Fit the Shockley diode model I = I₀ · exp(V / Vₜ).

    Strategy
    --------
    1. Seed I₀ and Vₜ from a log-linear (numpy.polyfit) fit on points
       where I > 0.
    2. Refine with scipy.optimize.curve_fit for a non-linear least-squares
       result (if scipy is available).
    3. Fall back to the log-linear estimate when scipy is absent or fails.
    """
    v = np.asarray(voltages, dtype=float)
    i = np.asarray(currents, dtype=float)

    mask = i > 0
    v_pos, i_pos = v[mask], i[mask]

    if len(v_pos) < 2:
        logger.warning("fit_diode: fewer than 2 positive-current points")
        return FitResult(
            model=ModelType.DIODE,
            coefficients=(1e-12, 0.026),
            r_squared=0.0,
            fitted_voltages=v.copy(),
            fitted_currents=np.zeros_like(v),
            equation="I(V) = I₀·exp(V/Vₜ)  [insufficient data]",
        )

    # --- log-linear seed (always computed) ---
    log_i = np.log(i_pos)
    lin_coeffs = np.polyfit(v_pos, log_i, 1)
    slope_seed = float(lin_coeffs[0])
    i0_seed = float(np.exp(lin_coeffs[1]))
    vt_seed = 1.0 / slope_seed if abs(slope_seed) > 1e-18 else 0.026

    i0, vt = i0_seed, vt_seed

    if _HAS_SCIPY:
        try:
            popt, _ = _scipy_curve_fit(
                _diode_curve,
                v_pos,
                i_pos,
                p0=[max(i0_seed, 1e-20), max(abs(vt_seed), 1e-6)],
                maxfev=10_000,
                bounds=([0, 1e-6], [np.inf, 10.0]),
            )
            i0, vt = float(popt[0]), float(popt[1])
            logger.debug("Diode fit (scipy): I0=%g, Vt=%g", i0, vt)
        except Exception as exc:
            logger.warning("scipy curve_fit failed (%s); using log-linear seed", exc)

    # R² evaluated only over forward-bias (positive-current) points
    i_pred_pos = _diode_curve(v_pos, i0, vt)
    r2 = _r_squared(i_pos, i_pred_pos)

    v_dense = _dense_x(v)
    i_dense = _diode_curve(v_dense, i0, vt)

    eq = f"I(V) = {i0:.6g} · exp(V / {vt:.6g})   [R² = {r2:.4f}]"
    logger.debug("Diode fit final: I0=%g, Vt=%g, R²=%g", i0, vt, r2)
    return FitResult(
        model=ModelType.DIODE,
        coefficients=(i0, vt),
        r_squared=r2,
        fitted_voltages=v_dense,
        fitted_currents=i_dense,
        equation=eq,
    )
