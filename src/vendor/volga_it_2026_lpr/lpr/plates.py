"""Plate grammar: alphabet, masks and post-processing rules (GOST R 50577-2018 + task statement)."""
from __future__ import annotations

import re
from typing import Optional, Tuple

LETTERS = "ABEKMHOPCTYX"
DIGITS = "0123456789"
UNK = "#"
ALPHABET = DIGITS + LETTERS + UNK          # index 0..22 ; CTC blank is appended by the model
CHAR2IDX = {c: i for i, c in enumerate(ALPHABET)}

# Cyrillic look-alikes → Latin (labels typed in Cyrillic are normalised with this table)
CYR2LAT = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")

# masks over the character *classes*: L = letter, D = digit, R = region digit
MASK_STD = "LDDDLL"          # type1 / type1a / type1b (task statement)   + region (2-3 digits)
MASK_TAXI = "LLDDD"          # type1b as defined in GOST (ММ 000 55)      + region
RE_STD = re.compile(r"^[ABEKMHOPCTYX#][0-9#]{3}[ABEKMHOPCTYX#]{2}([0-9#]{2}|[1-9#][0-9#]{2})$")
RE_TAXI = re.compile(r"^[ABEKMHOPCTYX#]{2}[0-9#]{3}([0-9#]{2}|[1-9#][0-9#]{2})$")

PLATE_TYPES = ["type1", "type1a", "type1b", "other"]


def normalise(text: str) -> str:
    return text.upper().translate(CYR2LAT).replace(" ", "")


def matches_standard(text: str) -> bool:
    return bool(RE_STD.match(text))


def matches_taxi(text: str) -> bool:
    return bool(RE_TAXI.match(text))


def valid_for_type(text: str, ptype: str) -> bool:
    if ptype in ("type1", "type1a"):
        return matches_standard(text)
    if ptype == "type1b":
        return matches_standard(text) or matches_taxi(text)
    return True


def split_region(text: str) -> Tuple[str, str]:
    """Return (body, region) for a standard-format string; region is 2 or 3 digits."""
    if len(text) == 9:
        return text[:6], text[6:]
    if len(text) == 8:
        return text[:6], text[6:]
    return text, ""


def fix_lookalikes(text: str, ptype: str) -> str:
    """Resolve digit/letter confusions position-wise for the standard mask (O<->0, B<->8, ...)."""
    if ptype not in ("type1", "type1a", "type1b"):
        return text
    d2l = {"0": "O", "8": "B", "4": "A", "3": "E", "6": "B", "1": "T", "7": "T", "5": "C", "2": "E", "9": "P"}
    l2d = {"O": "0", "B": "8", "A": "4", "E": "3", "T": "7", "C": "0", "P": "9", "H": "4", "K": "4", "M": "4", "X": "4", "Y": "4"}
    n = len(text)
    if ptype == "type1b" and n in (7, 8) and not matches_standard(text) and _looks_like(text, MASK_TAXI):
        mask = MASK_TAXI + "R" * (n - 5)
    elif n in (8, 9):
        mask = MASK_STD + "R" * (n - 6)
    else:
        return text
    out = []
    changed = 0
    for ch, m in zip(text, mask):
        if ch == UNK:
            out.append(ch)
        elif m == "L" and ch.isdigit():
            out.append(d2l.get(ch, UNK)); changed += 1
        elif m in "DR" and ch.isalpha():
            out.append(l2d.get(ch, UNK)); changed += 1
        else:
            out.append(ch)
    # more than one look-alike substitution means the string is probably not a Russian plate at all
    # (trailer "EC651450", Belarus "1234AB7" ...) - leave it as read so the grammar rejects it
    if changed > 1:
        return text
    return "".join(out)


def _looks_like(text: str, mask: str) -> bool:
    """Loose test: at least 60% of positions already agree with the mask (used to pick between two masks)."""
    mask = mask + "R" * (len(text) - len(mask))
    ok = sum(1 for ch, m in zip(text, mask) if ch == UNK or (m == "L") == ch.isalpha())
    return ok >= 0.6 * len(text)


def plausible_length(text: str, ptype: str) -> bool:
    if ptype in ("type1", "type1a"):
        return len(text) in (8, 9)
    if ptype == "type1b":
        return len(text) in (7, 8, 9)
    return True
