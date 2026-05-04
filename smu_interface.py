"""Instrument command and SCPI interface for the Keysight B2901A SMU."""

try:
    import pyvisa
except ImportError:
    pyvisa = None

DISPLAY_LABELS = {
    "RES": "Resistance (Ω)",
    "R": "Resistance (Ω)",
    "VOLT": "Voltage (V)",
    "V": "Voltage (V)",
    "CURR": "Current (A)",
    "I": "Current (A)",
    "POW": "Power (W)",
    "TEMP": "Temperature",
    "STAT": "Status",
    "MODE": "Mode",
}


def open_instrument(resource, timeout_ms=5000):
    if pyvisa is None:
        raise RuntimeError(
            "PyVISA is not installed. Install it with `pip install pyvisa` before running this script."
        )

    if "::" not in resource:
        if ":" in resource and resource.count(":") == 1:
            host, port = resource.split(":", 1)
            if port.isdigit():
                resource = f"TCPIP0::{host}::{port}::SOCKET"
            else:
                resource = f"TCPIP0::{resource}::INSTR"
        else:
            resource = f"TCPIP0::{resource}::INSTR"

    rm = pyvisa.ResourceManager()
    instrument = rm.open_resource(resource)
    instrument.timeout = timeout_ms
    instrument.write_termination = "\n"
    instrument.read_termination = "\n"
    return instrument


def configure_b2901a_for_resistance(instrument, source, level, compliance):
    instrument.write("*RST")
    instrument.write("*CLS")

    if source == "current":
        instrument.write("SOUR:FUNC CURR")
        instrument.write(f"SOUR:CURR {level}")
        instrument.write(f"SOUR:CURR:VLIM {compliance}")
    else:
        instrument.write("SOUR:FUNC VOLT")
        instrument.write(f"SOUR:VOLT {level}")
        instrument.write(f"SOUR:VOLT:ILIM {compliance}")

    instrument.write("SENS:FUNC 'RES'")
    instrument.write("SENS:RES:RANG:AUTO ON")
    instrument.write("SENS:RES:APER 0.1")
    instrument.write("SENS:RES:NPLC 1")
    instrument.write("OUTP OFF")


def configure_b2901a_for_resistance_sweep(instrument, start, stop, steps, compliance):
    instrument.write("*RST")
    instrument.write("*CLS")
    instrument.write("FORM:ELEMENTS:SENSE VOLT,CURR")
    instrument.write("SOUR:FUNC:MODE VOLT")
    instrument.write("SOUR:VOLT:MODE SWE")
    instrument.write("SOUR:SWE:RANG BEST")
    instrument.write(f"SOUR:VOLT:START {start}")
    instrument.write(f"SOUR:VOLT:STOP {stop}")
    instrument.write(f"SOUR:VOLT:POIN {steps}")
    instrument.write(f"SENS:CURR:PROT:LEVEL {compliance}")
    instrument.write("SENS:FUNC CURR")
    instrument.write("SENS:CURR:RANG:BEST ON")
    instrument.write("SENS:CURR:APER 0.1")
    instrument.write("SENS:CURR:NPLC 1")
    instrument.write(f"TRIG:COUN {steps}")
    instrument.write("TRIG:DEL 0")
    instrument.write("OUTP OFF")


def interactive_scpi_console(instrument):
    print("Entering interactive SCPI mode. Type 'exit' or 'quit' to leave.")
    while True:
        try:
            command = input("SCPI> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not command:
            continue
        if command.lower() in {"exit", "quit", "q"}:
            break

        try:
            if command.endswith("?"):
                response = instrument.query(command)
                print(response.strip())
            else:
                instrument.write(command)
                print("OK")
        except Exception as exc:
            print(f"SCPI error: {exc}")


def normalize_element(element):
    return element.strip().strip("'\"").upper()


def pretty_label(element):
    key = normalize_element(element)
    return DISPLAY_LABELS.get(key, key.replace("_", " ").title())


def get_read_labels(instrument):
    try:
        response = instrument.query("FORM:ELEM:SENS?")
        elements = [normalize_element(item) for item in response.split(",") if item.strip()]
        if elements:
            return elements
    except Exception:
        pass
    return ["RES", "CURR", "VOLT"]


def measure_resistance(instrument, delay):
    instrument.write("OUTP ON")
    if delay > 0:
        import time
        time.sleep(delay)

    labels = get_read_labels(instrument)
    response = instrument.query("READ?")
    instrument.write("OUTP OFF")

    values = [value.strip() for value in response.split(",")]
    results = []
    for idx, value in enumerate(values):
        label = pretty_label(labels[idx]) if idx < len(labels) else f"Value {idx + 1}"
        results.append((label, value))
    return results


def measure_voltage_sweep(instrument):
    instrument.write("OUTP ON")
    response = instrument.query("READ:ARRAY? (@1)")
    instrument.write("OUTP OFF")

    values = [value.strip() for value in response.split(",") if value.strip()]
    results = []
    for i in range(0, len(values), 2):
        voltage = values[i]
        current = values[i + 1] if i + 1 < len(values) else ""
        results.append((voltage, current))
    return results
