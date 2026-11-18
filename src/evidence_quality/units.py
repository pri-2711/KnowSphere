"""Quantities and units: read numbers with units from text, and compare quantities across units.

Deterministic on purpose: claim quantities are read from the source quote, never
taken from a model's answer, so a number cannot be invented.
"""

from __future__ import annotations

import re
from typing import Iterable

from src.contracts.schemas import Quantity

# unit -> (dimension, factor to the SI base of that dimension)
UNITS: dict[str, tuple[str, float]] = {
    "kg": ("mass", 1.0), "g": ("mass", 1e-3), "mg": ("mass", 1e-6), "lb": ("mass", 0.45359237),
    "lbm": ("mass", 0.45359237), "t": ("mass", 1000.0),
    "m": ("length", 1.0), "cm": ("length", 0.01), "mm": ("length", 1e-3), "km": ("length", 1000.0),
    "ft": ("length", 0.3048), "in": ("length", 0.0254),
    "s": ("time", 1.0), "sec": ("time", 1.0), "min": ("time", 60.0), "h": ("time", 3600.0),
    "hr": ("time", 3600.0), "day": ("time", 86400.0), "d": ("time", 86400.0), "yr": ("time", 3.15576e7),
    "l": ("volume", 1e-3), "ml": ("volume", 1e-6), "m3": ("volume", 1.0), "gal": ("volume", 3.785411784e-3),
    "pa": ("pressure", 1.0), "kpa": ("pressure", 1e3), "mpa": ("pressure", 1e6), "psi": ("pressure", 6894.757),
    "psia": ("pressure", 6894.757), "atm": ("pressure", 101325.0), "bar": ("pressure", 1e5),
    "mmhg": ("pressure", 133.322),
    "w": ("power", 1.0), "kw": ("power", 1e3), "mw": ("power", 1e6),
    "j": ("energy", 1.0), "kj": ("energy", 1e3), "mj": ("energy", 1e6), "kwh": ("energy", 3.6e6),
    "kcal": ("energy", 4184.0), "btu": ("energy", 1055.06),
    "°c": ("temperature_c", 1.0), "°f": ("temperature_f", 1.0), "k": ("temperature_k", 1.0),
    "%": ("ratio", 0.01), "percent": ("ratio", 0.01), "ppm": ("ratio", 1e-6),
    "db": ("decibel", 1.0), "hz": ("frequency", 1.0), "khz": ("frequency", 1e3), "lux": ("illuminance", 1.0),
}
_UNIT_WORDS = {"days": "day", "hours": "h", "hrs": "hr", "minutes": "min", "seconds": "s", "liters": "l",
               "litres": "l", "kilograms": "kg", "grams": "g", "meters": "m", "metres": "m", "pounds": "lb",
               "watts": "w", "kilowatts": "kw", "years": "yr", "per cent": "percent", "deg c": "°c", "deg f": "°f"}
_UNIT_TOKEN = r"(?:°\s?[CF]|%|[A-Za-z][A-Za-z0-9]*(?:\s?/\s?[A-Za-z][A-Za-z0-9-]*)*)"
_NUMBER = re.compile(r"(?<![\w.])([-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)\s?(" + _UNIT_TOKEN + r")?")
_TOLERANCE = 0.02   # quantities within 2 percent are treated as the same value


def normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    value = re.sub(r"\s+", "", unit.strip()).lower().replace("-", "")
    value = _UNIT_WORDS.get(unit.strip().lower(), value)
    return value


def dimension(unit: str | None) -> tuple[str, float] | None:
    """(dimension, factor) for a simple or compound unit such as ``kg/day``; None when unknown."""
    unit = normalize_unit(unit)
    if unit is None:
        return ("number", 1.0)
    if unit in UNITS:
        return UNITS[unit]
    if "/" in unit:
        parts = [part for part in unit.split("/") if part]
        resolved = [UNITS.get(_UNIT_WORDS.get(part, part)) for part in parts]
        if len(parts) >= 2 and all(resolved):
            name = resolved[0][0] + "".join(f"/{dim}" for dim, _ in resolved[1:])
            factor = resolved[0][1]
            for _, divisor in resolved[1:]:
                factor /= divisor
            return name, factor
    return None


def find_quantities(text: str) -> list[Quantity]:
    """Numbers in ``text`` with the unit right after them when it is a known unit."""
    found = []
    for match in _NUMBER.finditer(text):
        number, unit = match.group(1), match.group(2)
        if unit and dimension(unit) is None:
            unit = None                        # a following word that is not a unit ("3 systems")
        raw = number + (f" {unit}" if unit and not unit.startswith(("%", "°")) else (unit or ""))
        found.append(Quantity(value=float(number.replace(",", "")), unit=normalize_unit(unit), raw=raw.strip()))
    return found


def to_base(quantity: Quantity) -> tuple[str, float] | None:
    resolved = dimension(quantity.unit)
    if resolved is None:
        return None
    name, factor = resolved
    return name, quantity.value * factor


def same_value(a: Quantity, b: Quantity) -> bool | None:
    """True or False when the two quantities are comparable (same dimension), None when they are not."""
    base_a, base_b = to_base(a), to_base(b)
    if base_a is None or base_b is None or base_a[0] != base_b[0]:
        return None
    larger = max(abs(base_a[1]), abs(base_b[1])) or 1.0
    return abs(base_a[1] - base_b[1]) / larger <= _TOLERANCE


def contains_quantity(quantities: Iterable[Quantity], target: Quantity) -> bool:
    return any(same_value(item, target) is True or (item.value == target.value and item.unit == target.unit)
               for item in quantities)


def convert(quantity: Quantity, unit: str) -> float | None:
    """The quantity expressed in ``unit`` (same dimension, linear units only), or None."""
    base, target = to_base(quantity), dimension(unit)
    if base is None or target is None or base[0] != target[0] or base[0].startswith("temperature") \
            or base[0] == "number":
        return None
    return base[1] / target[1]


_REQUESTED = re.compile(r"\b(?:in|into|as|to)\s+(" + _UNIT_TOKEN + r")\s*[?.!]*\s*$", re.IGNORECASE)


def requested_unit(question: str) -> str | None:
    """The unit a question asks for ("... per day in kg/day?"), when it is a known unit."""
    match = _REQUESTED.search(question.strip())
    if match and dimension(match.group(1)) not in (None, ("number", 1.0)):
        return normalize_unit(match.group(1))
    return None
