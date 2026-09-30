"""GRZ charset: digits + 12 lookalike letters + hash for unreadables."""

from __future__ import annotations

from pathlib import Path
import math

from src.utils.plate_mask import mask_invalid_chars, normalize_plate, validate_plate

# Spec order: digits, then the 12 lookalike letters, then unreadables.
CHARSET = "0123456789ABEKMHOPCTYX#"
CHARSET_SET = frozenset(CHARSET)

DEFAULT_DICT_PATH = Path(__file__).resolve().parents[2] / "configs" / "ocr_dict.txt"


def charset_from_config(raw: str | None) -> str:
    if not raw:
        return CHARSET
    return "".join(ch for ch in raw if ch in CHARSET_SET or ch == "#") or CHARSET


def write_dict_file(path: Path, charset: str = CHARSET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # One symbol per line, CTC-style (no blank token here)
    path.write_text("\n".join(charset) + "\n", encoding="utf-8")
    return path


def filter_ocr_text(text: str, charset: str = CHARSET) -> str:
    """Keep only charset glyphs after Cyrillic→Latin normalize."""
    allowed = frozenset(charset)
    norm = normalize_plate(text)
    return "".join(ch if ch in allowed else "#" for ch in norm)


def apply_char_threshold(text: str, char_scores: list[float], min_char_conf: float) -> str:
    """Replace low-confidence positions with '#' (conservative)."""
    if not text:
        return text
    out = []
    for i, ch in enumerate(text):
        score = char_scores[i] if i < len(char_scores) else 0.0
        out.append(ch if score >= min_char_conf else "#")
    return "".join(out)


def finalize_plate(text: str, char_scores: list[float] | None, min_char_conf: float) -> tuple[str, float]:
    """Charset filter + optional per-char hash + mask repair."""
    # Normalize characters and their confidences together. Missing scores are unknown.
    pairs = [(ch, char_scores[i] if char_scores and i < len(char_scores) else 0.0)
             for i, ch in enumerate(text) if not ch.isspace() and ch != "-"]
    filtered = "".join(filter_ocr_text(ch) for ch, _ in pairs)
    scores = [max(0.0, min(1.0, float(s))) if math.isfinite(float(s)) else 0.0
              for ch, s in pairs for _ in filter_ocr_text(ch)]
    if scores:
        filtered = apply_char_threshold(filtered, scores, min_char_conf)
        ocr_conf = float(min(scores[: len(filtered)] or scores or [0.0]))
    else:
        ocr_conf = 0.0
    masked = mask_invalid_chars(filtered)
    if len(masked) == 9 and masked[6] not in "127#":
        masked = masked[:6] + "#" + masked[7:]
        ocr_conf = 0.0
    ok, norm = validate_plate(masked, allow_hash=True)
    if not ok:
        # Unknown length: do not invent positions or a region.
        return "", 0.0
    if "#" in norm:
        ocr_conf = 0.0
    return norm, ocr_conf
