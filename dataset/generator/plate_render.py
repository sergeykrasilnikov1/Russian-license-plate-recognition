"""Plate renderer.

Type 1 is https://github.com/maerty1/Russia-number-plate-generator (`app.py` +
`License_plate_in_Russia.svg` + RoadNumbers 2.0). Type 1Б / 1А / other only
recolour or rearrange that same drawing.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import cairosvg
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils.geometry import PLATE_TYPES

ASSETS = Path(__file__).resolve().parent / "assets"
FONTS = Path(__file__).resolve().parent / "fonts"

# maerty1/Russia-number-plate-generator — app.py
FONT_PATH = FONTS / "RoadNumbers2.0.ttf"
SVG_PATH = ASSETS / "License_plate_in_Russia.svg"
PLATE_SIZE = (520, 112)
FONT_SIZE = 135
LEFT_OFFSET = 40
ELEMENT_OFFSET = -10
REGION_OFFSET = 40
LETTER1_WIDTH = 58
LETTER2_WIDTH = 76
LETTER3_WIDTH = 58
REGION_WIDTH = 95

TYPE1A_BLANK = ASSETS / "type1a_blank.png"
# Native blank geometry (1024×603 ChatGPT layout): region box left/top edges.
_TYPE1A_BOX_X0 = 617
_TYPE1A_BOX_Y0 = 294
_TYPE1A_INNER = (50, 40, 980, 555)  # left, top, right, bottom of drawable field

DEFAULT_PX_PER_MM = 3.0
DIPLOMATIC_RED = (180, 20, 20)



def resize_text(draw, text, font, target_width):
    """Возвращает шрифт, масштабированный для заданной ширины текста."""
    while True:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        if text_width <= target_width:
            break
        font_size = font.size - 1
        if font_size < 1:
            raise ValueError("Слишком малый размер шрифта.")
        font = ImageFont.truetype(str(FONT_PATH), font_size)
    return font


@lru_cache(maxsize=1)
def _svg_blank() -> Image.Image:
    png = cairosvg.svg2png(
        url=str(SVG_PATH),
        output_width=PLATE_SIZE[0],
        output_height=PLATE_SIZE[1],
    )
    return Image.open(io.BytesIO(png)).convert("RGB")


def _paint_elements(draw: ImageDraw.ImageDraw, number: str, font, y_position: int, fill="black") -> None:
    """Glyph loop from maerty1 draw_plate (without YOLO char annotations)."""
    x_offset = LEFT_OFFSET
    elements = [
        (number[0], LETTER1_WIDTH),
        (number[1:4], 3 * LETTER1_WIDTH),
        (number[4], LETTER2_WIDTH),
        (number[5], LETTER3_WIDTH),
    ]
    for text, target_width in elements:
        resized_font = resize_text(draw, text, font, target_width)
        text_bbox = draw.textbbox((0, 0), text, font=resized_font)
        text_width = text_bbox[2] - text_bbox[0]
        x_centered = x_offset + (target_width - text_width) // 2
        draw.text((x_centered, y_position), text, font=resized_font, fill=fill)
        x_offset += target_width + ELEMENT_OFFSET

    # Region glyphs are shorter than the main series (GOST 58 vs 76 mm).
    # maerty1 only shrinks by width — fine for 3-digit codes, but a 2-digit
    # region stays at FONT_SIZE=135 and covers RUS/flag. Use the size that a
    # full 3-digit region would get, then shrink further if needed.
    region_text = number[6:]
    region_font = resize_text(
        draw,
        "000",
        ImageFont.truetype(str(FONT_PATH), FONT_SIZE),
        REGION_WIDTH,
    )
    resized_font = resize_text(draw, region_text, region_font, REGION_WIDTH)
    text_bbox = draw.textbbox((0, 0), region_text, font=resized_font)
    text_width = text_bbox[2] - text_bbox[0]
    x_centered = PLATE_SIZE[0] - REGION_OFFSET - text_width
    draw.text((x_centered, y_position), region_text, font=resized_font, fill=fill)


def draw_plate(number: str, fill="black") -> Image.Image:
    """maerty1 draw_plate: SVG blank + RoadNumbers at FONT_SIZE=135, y=30."""
    img = _svg_blank().copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
    _paint_elements(draw, number, font, y_position=30, fill=fill)
    return img


def _recolor_yellow(img: Image.Image) -> Image.Image:
    """Type 1Б: same type-1 drawing, yellow field, keep flag colours."""
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    neutral = hsv[:, :, 1] < 50
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    yellow = np.zeros_like(bgr)
    yellow[:, :, 0] = (luma.astype(np.uint16) * 20 // 255).astype(np.uint8)
    yellow[:, :, 1] = (luma.astype(np.uint16) * 200 // 255).astype(np.uint8)
    yellow[:, :, 2] = luma
    out = bgr.copy()
    out[neutral] = yellow[neutral]
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def _draw_char_fixed(
    draw: ImageDraw.ImageDraw,
    ch: str,
    cx: float,
    y: float,
    font: ImageFont.FreeTypeFont,
    fill="black",
) -> None:
    """Draw one glyph centered on cx; letters and digits share the same font."""
    bb = draw.textbbox((0, 0), ch, font=font)
    tw = bb[2] - bb[0]
    draw.text((cx - tw / 2 - bb[0], y - bb[1]), ch, font=font, fill=fill)


def _draw_type1a(number: str) -> Image.Image:
    """Fill type1a_blank.png: same-size main glyphs, lowered; region in framed box."""
    if not TYPE1A_BLANK.is_file():
        raise FileNotFoundError(TYPE1A_BLANK)
    img = Image.open(TYPE1A_BLANK).convert("RGB")
    draw = ImageDraw.Draw(img)
    left, _top, right, _bottom = _TYPE1A_INNER
    box_x0, box_y0 = _TYPE1A_BOX_X0, _TYPE1A_BOX_Y0

    # Letters larger than digits (RoadNumbers).
    letter_font = ImageFont.truetype(str(FONT_PATH), 370)
    digit_font = ImageFont.truetype(str(FONT_PATH), 300)

    # Top row A123 — lowered, shifted right; extra gap after the first letter.
    top_chars = number[0] + number[1:4]
    top_y = 75
    top_left = left + 110  # сдвиг вправо
    step = (right - top_left) / (len(top_chars) + 3)
    letter_digit_gap = 100  # доп. расстояние между первой буквой и цифрами
    for i, ch in enumerate(top_chars):
        font = letter_font if ch.isalpha() else digit_font
        x = top_left + step * (i + 1)
        if i > 0:
            x += letter_digit_gap
        _draw_char_fixed(draw, ch, x, top_y, font)

    # Bottom-left series — letter size.
    series = number[4:6]
    series_right = box_x0 + 100
    series_y = 365
    step_s = (series_right - left) / (len(series) + 2)
    for i, ch in enumerate(series):
        _draw_char_fixed(draw, ch, left + step_s * (i + 1), series_y, letter_font)

    # Region digits in the upper half of the framed box (RUS/flag already on blank).
    region = number[6:]
    region_font = resize_text(
        draw,
        "000",
        ImageFont.truetype(str(FONT_PATH), 250),
        right - box_x0 - 50,
    )
    region_font = resize_text(draw, region, region_font, right - box_x0 - 50)
    step_r = (right - box_x0) / (len(region) + 1)
    region_y = box_y0 + 18
    for i, ch in enumerate(region):
        _draw_char_fixed(draw, ch, box_x0 + step_r * (i + 1), region_y, region_font)

    return img



def _to_bgra(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    alpha = np.zeros((h, w), dtype=np.uint8)
    radius = max(4, h // 16)
    cv2.rectangle(alpha, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(alpha, (0, radius), (w, h - radius), 255, -1)
    for cx, cy in ((radius, radius), (w - radius, radius), (radius, h - radius), (w - radius, h - radius)):
        cv2.circle(alpha, (cx, cy), radius, 255, -1)
    return np.dstack([bgr, alpha])


def render_plate(plate: str, plate_type: str, px_per_mm: float = DEFAULT_PX_PER_MM) -> np.ndarray:
    if plate_type not in PLATE_TYPES:
        raise KeyError(f"unknown plate type {plate_type!r}")
    if plate_type == "type1":
        pil = draw_plate(plate)
    elif plate_type == "type1b":
        pil = _recolor_yellow(draw_plate(plate))
    elif plate_type == "type1a":
        pil = _draw_type1a(plate)
    else:
        pil = draw_plate(plate, fill=DIPLOMATIC_RED)

    geom = PLATE_TYPES[plate_type]
    size = (int(round(geom.width_mm * px_per_mm)), int(round(geom.height_mm * px_per_mm)))
    if pil.size != size:
        pil = pil.resize(size, Image.Resampling.LANCZOS)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return _to_bgra(bgr)
