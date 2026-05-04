"""Plotting support for SMU sweep data using PySide.

This module provides a lightweight window for displaying voltage/current sweep
results and a fitted quadratic model when the user passes the `-g` flag.
"""

import math

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPainter, QPen, QColor, QFont
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPlainTextEdit
except ImportError:
    try:
        from PySide2.QtCore import Qt
        from PySide2.QtGui import QPainter, QPen, QColor, QFont
        from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPlainTextEdit
    except ImportError as exc:
        raise ImportError(
            "PySide6 or PySide2 is required for GUI plotting. Install one of them to use the -g flag."
        ) from exc


def _safe_range(values):
    if not values:
        return 0.0, 1.0
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return minimum - 0.5, maximum + 0.5
    padding = (maximum - minimum) * 0.1
    return minimum - padding, maximum + padding


def fit_linear(x_values, y_values):
    n = len(x_values)
    if n < 2:
        return 0.0, 0.0

    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if abs(denominator) < 1e-18:
        return 0.0, y_mean
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def fit_quadratic(x_values, y_values):
    n = len(x_values)
    if n < 3:
        slope, intercept = fit_linear(x_values, y_values)
        return 0.0, slope, intercept

    # --- STEP 1: Center and Scale x-values ---
    avg_x = sum(x_values) / n
    std_x = (sum((x - avg_x) ** 2 for x in x_values) / n) ** 0.5
    if std_x == 0:
        return 0.0, 0.0, sum(y_values) / n

    scaled_x = [(x - avg_x) / std_x for x in x_values]

    # --- STEP 2: Build the System ---
    s0 = n
    s1 = sum(scaled_x)
    s2 = sum(xi * xi for xi in scaled_x)
    s3 = sum(xi * xi * xi for xi in scaled_x)
    s4 = sum(xi * xi * xi * xi for xi in scaled_x)
    t0 = sum(y_values)
    t1 = sum(xi * yi for xi, yi in zip(scaled_x, y_values))
    t2 = sum((xi * xi) * yi for xi, yi in zip(scaled_x, y_values))

    A = [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]]
    B = [t0, t1, t2]

    for i in range(3):
        max_row = i
        for k in range(i + 1, 3):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k
        A[i], A[max_row] = A[max_row], A[i]
        B[i], B[max_row] = B[max_row], B[i]

        pivot = A[i][i]
        if abs(pivot) < 1e-18:
            return 0.0, 0.0, sum(y_values) / n

        for j in range(i + 1, 3):
            factor = A[j][i] / pivot
            for k in range(i, 3):
                A[j][k] -= factor * A[i][k]
            B[j] -= factor * B[i]

    c = [0.0, 0.0, 0.0]
    for i in range(2, -1, -1):
        val = B[i]
        for j in range(i + 1, 3):
            val -= A[i][j] * c[j]
        c[i] = val / A[i][i]

    a = c[2] / (std_x ** 2)
    b = (c[1] / std_x) - (2 * c[2] * avg_x / (std_x ** 2))
    intercept = c[0] - (c[1] * avg_x / std_x) + (c[2] * avg_x * avg_x / (std_x ** 2))

    return a, b, intercept


def fit_diode(voltages, currents):
    """Fit diode model: ln(I) = ln(I_0) + V/V_T"""
    # Filter out zero or negative currents to avoid log issues
    valid = [(v, c) for v, c in zip(voltages, currents) if c > 0]
    if len(valid) < 2:
        return 0.0, 0.0  # fallback
    
    v_vals = [v for v, c in valid]
    log_c = [math.log(c) for v, c in valid]
    
    # Linear fit on v_vals, log_c
    n = len(v_vals)
    x_mean = sum(v_vals) / n
    y_mean = sum(log_c) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(v_vals, log_c))
    denominator = sum((x - x_mean) ** 2 for x in v_vals)
    if abs(denominator) < 1e-18:
        return 0.0, y_mean
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def evaluate_quadratic(x, a, b, c):
    return a * x * x + b * x + c


def evaluate_linear(x, b, c):
    return b * x + c


class SweepPlotWidget(QWidget):
    def __init__(self, voltages, currents, coefficients=None, order=2, diode=False, parent=None):
        super().__init__(parent)
        self.voltages = voltages
        self.currents = currents
        self.coefficients = coefficients
        self.order = order
        self.diode = diode
        self.setMinimumSize(700, 500)
        self.x_min, self.x_max = _safe_range(self.voltages)
        self.y_min, self.y_max = _safe_range(self.currents)

        # Use data points only for axis ranges (remove model curve adjustment)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        margin = 60
        rect = self.rect().adjusted(margin, margin, -margin, -margin)

        painter.setPen(QPen(QColor("black"), 2))
        painter.drawRect(rect)

        painter.setPen(QPen(QColor("black"), 1))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(self.width() // 2 - 40, 20, "Voltage vs Current")
        
        # Vertical Y axis label
        painter.save()
        painter.translate(15, self.height() // 2)
        painter.rotate(-90)
        painter.drawText(0, 0, "Current (A)")
        painter.restore()
        
        painter.drawText(self.width() // 2 - 40, self.height() - 10, "Voltage (V)")

        def map_x(x):
            if self.x_max == self.x_min:
                return rect.left()
            return rect.left() + (x - self.x_min) / (self.x_max - self.x_min) * rect.width()

        def map_y(y):
            if self.y_max == self.y_min:
                return rect.bottom()
            return rect.bottom() - (y - self.y_min) / (self.y_max - self.y_min) * rect.height()

        painter.setPen(QPen(QColor("gray"), 1, Qt.DashLine))
        for step in range(1, 5):
            x = rect.left() + rect.width() * step / 5
            painter.drawLine(x, rect.top(), x, rect.bottom())
            y = rect.top() + rect.height() * step / 5
            painter.drawLine(rect.left(), y, rect.right(), y)

        painter.setPen(QPen(QColor("black"), 1))
        # Extreme labels
        painter.drawText(rect.left() - 40, rect.top() + 10, f"{self.y_max:.3g}")
        painter.drawText(rect.left() - 40, rect.bottom(), f"{self.y_min:.3g}")
        painter.drawText(rect.left(), rect.bottom() + 20, f"{self.x_min:.3g}")
        painter.drawText(rect.right() - 40, rect.bottom() + 20, f"{self.x_max:.3g}")
        
        # Intermediate tick labels
        for step in range(1, 5):
            # Y axis intermediate
            y_val = self.y_max - (self.y_max - self.y_min) * step / 5
            y_pos = rect.top() + rect.height() * step / 5
            painter.drawText(rect.left() - 40, y_pos + 5, f"{y_val:.3g}")
            
            # X axis intermediate
            x_val = self.x_min + (self.x_max - self.x_min) * step / 5
            x_pos = rect.left() + rect.width() * step / 5
            painter.drawText(x_pos - 20, rect.bottom() + 20, f"{x_val:.3g}")

        if len(self.voltages) > 1:
            painter.setPen(QPen(QColor("blue"), 2))
            path = None
            for x, y in zip(self.voltages, self.currents):
                px = map_x(x)
                py = map_y(y)
                if path is None:
                    path = (px, py)
                else:
                    painter.drawLine(path[0], path[1], px, py)
                    path = (px, py)

        painter.setPen(QPen(QColor("red"), 2))
        if self.coefficients is not None and self.x_max != self.x_min:
            model_points = 200
            path = None
            for index in range(model_points + 1):
                x = self.x_min + (self.x_max - self.x_min) * index / model_points
                if self.diode:
                    slope, intercept = self.coefficients
                    y = math.exp(intercept + slope * x)
                elif self.order == 1:
                    b, c = self.coefficients
                    y = evaluate_linear(x, b, c)
                else:
                    a, b, c = self.coefficients
                    y = evaluate_quadratic(x, a, b, c)
                px = map_x(x)
                py = map_y(y)
                if path is None:
                    path = (px, py)
                else:
                    painter.drawLine(path[0], path[1], px, py)
                    path = (px, py)

        painter.setPen(QPen(QColor("darkblue"), 5))
        for x, y in zip(self.voltages, self.currents):
            px = map_x(x)
            py = map_y(y)
            painter.drawPoint(px, py)


def plot_sweep_data(voltages, currents, order=2, diode=False):
    if diode:
        # Filter out non-positive currents for log
        valid_data = [(v, i) for v, i in zip(voltages, currents) if i > 0]
        if not valid_data:
            equation = "No valid positive current data for diode model."
            coefficients = None
        else:
            v_vals, i_vals = zip(*valid_data)
            ln_i = [math.log(i) for i in i_vals]
            coefficients = fit_linear(v_vals, ln_i)
            slope, intercept = coefficients
            I_0 = math.exp(intercept)
            V_T = 1.0 / slope if abs(slope) > 1e-18 else float('inf')
            if V_T == float('inf'):
                equation = f"I(V) = {I_0:.6g} * exp(V / inf)"
            else:
                equation = f"I(V) = {I_0:.6g} * exp(V / {V_T:.6g})"
    elif order == 1:
        coefficients = fit_linear(voltages, currents)
        slope, intercept = coefficients
        if abs(slope) > 1e-18:
            equation = (
                f"y(x) = {slope:.6g} x + {intercept:.6g}\n"
                f"R [Ω] = {1.0 / slope:.6g}"
            )
        else:
            equation = f"y(x) = {slope:.6g} x + {intercept:.6g}\nR [Ω] = undefined"
    else:
        coefficients = fit_quadratic(voltages, currents)
        a, b, c = coefficients
        equation = f"y(x) = {a:.6g} x^2 + {b:.6g} x + {c:.6g}"
        if abs(b) > 1e-18:
            equation += f"\nR [Ω] = {1.0 / b:.6g}"
        else:
            equation += "\nR [Ω] = undefined"

    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    window.setWindowTitle("SMU Sweep Plot")

    scene = QWidget()
    layout = QVBoxLayout(scene)
    plot_widget = SweepPlotWidget(voltages, currents, coefficients=coefficients, order=order, diode=diode)
    layout.addWidget(plot_widget)

    equation_widget = QPlainTextEdit()
    equation_widget.setReadOnly(True)
    equation_widget.setPlainText(equation)
    equation_widget.setFixedHeight(80)
    equation_widget.setFont(QFont("Courier", 10))
    equation_widget.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
    layout.addWidget(equation_widget)

    window.setCentralWidget(scene)
    window.resize(800, 700)
    window.show()
    app.exec()
