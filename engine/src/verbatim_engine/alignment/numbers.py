"""Spoken readings of English numbers, for acoustic scoring.

Whisper writes many numbers as digits ("1984", "5%", "$20", "3rd"), but a
CTC acoustic model can only spell words. To compare such a passage with the
audio, each digit token is expanded into the words a speaker would say. A
number can often be read in several ways ("1984": "nineteen eighty four" or
"one thousand nine hundred eighty four"), so every plausible reading is
returned and the scorer keeps the one the audio supports best.

Only English is handled; other languages' numbers stay unspellable.
"""

from __future__ import annotations

import re

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
_ORDINAL_WORDS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
    "twelve": "twelfth",
}
_CURRENCY = {"$": ("dollar", "dollars"), "£": ("pound", "pounds"), "€": ("euro", "euros")}
#: Readings per token are capped so a passage full of numbers stays cheap to score.
MAX_READINGS = 4

_NUMBER = re.compile(
    r"^(?P<cur>[$£€])?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?"
    r"(?P<suffix>%|st|nd|rd|th|s|'s)?$"
)


def _below_thousand(n: int) -> list[str]:
    words: list[str] = []
    if n >= 100:
        words += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        words.append(_TENS[n // 10])
        if n % 10:
            words.append(_ONES[n % 10])
    elif n or not words:
        words.append(_ONES[n])
    return words


def cardinal(n: int) -> list[str]:
    """``2024`` -> two thousand twenty four (no "and", as in American usage)."""
    if n < 1000:
        return _below_thousand(n)
    words: list[str] = []
    for value, name in _SCALES:
        if n >= value:
            words += cardinal(n // value) + [name]
            n %= value
    if n:
        words += _below_thousand(n)
    return words


def _pairs(n: int) -> list[str]:
    """Year-style reading: 1984 -> nineteen eighty four, 1905 -> nineteen oh five."""
    high, low = divmod(n, 100)
    if low == 0:
        return _below_thousand(high) + ["hundred"]
    tail = _below_thousand(low) if low >= 10 else ["oh", _ONES[low]]
    return _below_thousand(high) + tail


def _ordinal(words: list[str]) -> list[str]:
    last = words[-1]
    if last in _ORDINAL_WORDS:
        new = _ORDINAL_WORDS[last]
    elif last.endswith("y"):
        new = last[:-1] + "ieth"
    else:
        new = last + "th"
    return words[:-1] + [new]


def _plural(words: list[str]) -> list[str]:
    last = words[-1]
    return words[:-1] + [last[:-1] + "ies" if last.endswith("y") else last + "s"]


def readings(token: str) -> list[list[str]] | None:
    """Alternative word sequences for a numeric token, or None if it is not one."""
    match = _NUMBER.match(token)
    if match is None:
        return None
    digits = match["int"].replace(",", "")
    n = int(digits)
    options: list[list[str]] = [cardinal(n)]
    if 1000 <= n <= 2099 and "," not in match["int"] and not match["dec"]:
        options.append(_pairs(n))  # years
        if 2000 <= n <= 2009:
            options.append(["two", "thousand"] + ([] if n == 2000 else ["and"] + _below_thousand(n - 2000)))
    if len(digits) > 1 and digits.startswith("0"):
        options = [[_ONES[int(d)] for d in digits]]  # "007" read digit by digit
    if match["dec"]:
        fraction = [_ONES[int(d)] for d in match["dec"]]
        options = [o + ["point"] + fraction for o in options]
    suffix = match["suffix"]
    if suffix in ("st", "nd", "rd", "th"):
        options = [_ordinal(o) for o in options]
    elif suffix in ("s", "'s"):
        options = [_plural(o) for o in options]
    elif suffix == "%":
        options = [o + ["percent"] for o in options]
    if match["cur"]:
        singular, plural = _CURRENCY[match["cur"]]
        options = [o + [singular if n == 1 and not match["dec"] else plural] for o in options]
    unique: list[list[str]] = []
    for option in options:
        if option not in unique:
            unique.append(option)
    return unique[:MAX_READINGS]
