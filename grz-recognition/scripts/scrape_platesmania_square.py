#!/usr/bin/env python3
"""Download gallery photos whose plate preview is exactly 80 px high.

The plate preview and the vehicle photo are paired only inside the same
``.panel.panel-grey`` gallery card.  This prevents unrelated images from the
header, carousel, or sidebar from being downloaded.

Examples:
  python scripts/scrape_platesmania_square.py --start 1 --pages 1
  python scripts/scrape_platesmania_square.py --start 1 --pages 10 --output dataset/platesmania_square
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from PIL import Image, UnidentifiedImageError


BASE_URL = "https://platesmania.com"
GALLERY_URL = f"{BASE_URL}/ru/gallery.php"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
USER_AGENT = "square-plate-photo-downloader/1.0 (personal research; contact: local-user)"

log = logging.getLogger("platesmania-square")


class AccessBlocked(RuntimeError):
    """The gallery rejected a non-browser HTTP client."""


@dataclass(frozen=True)
class GalleryCard:
    """The two relevant images belonging to one gallery card."""

    photo_url: str
    plate_preview_url: str
    plate_text: str = ""


@dataclass
class ScrapeStats:
    pages: int = 0
    cards: int = 0
    square: int = 0
    saved: int = 0
    already_exists: int = 0
    errors: int = 0


class _GalleryParser(HTMLParser):
    """Small dependency-free parser for PlatesMania gallery cards."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cards: list[GalleryCard] = []
        self._div_depth = 0
        self._card_depth: int | None = None
        self._photo_url: str | None = None
        self._plate_url: str | None = None
        self._plate_text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "div":
            self._div_depth += 1
            classes = set((attrs_dict.get("class") or "").split())
            if self._card_depth is None and {"panel", "panel-grey"}.issubset(classes):
                self._card_depth = self._div_depth
                self._photo_url = None
                self._plate_url = None
                self._plate_text = ""
            return

        if tag != "img" or self._card_depth is None:
            return

        src = attrs_dict.get("src") or attrs_dict.get("data-src")
        if not src:
            return
        path = urlparse(src).path.lower()
        if "/m/" in path and self._photo_url is None:
            self._photo_url = urljoin(BASE_URL, src)
        elif "/inf/" in path and self._plate_url is None:
            self._plate_url = urljoin(BASE_URL, src)
            self._plate_text = (attrs_dict.get("alt") or "").strip()

    def handle_endtag(self, tag: str) -> None:
        if tag != "div":
            return
        if self._card_depth == self._div_depth:
            if self._photo_url and self._plate_url:
                self.cards.append(
                    GalleryCard(
                        photo_url=self._photo_url,
                        plate_preview_url=self._plate_url,
                        plate_text=self._plate_text,
                    )
                )
            self._card_depth = None
            self._photo_url = None
            self._plate_url = None
            self._plate_text = ""
        self._div_depth = max(0, self._div_depth - 1)


def parse_gallery(html: str) -> list[GalleryCard]:
    """Extract unique vehicle/plate-preview pairs from gallery HTML."""

    parser = _GalleryParser()
    parser.feed(html)
    parser.close()
    # Combined browser exports can contain repeated cards when an intermediate
    # response was cached. Never spend requests or disk writes on duplicates.
    unique: dict[str, GalleryCard] = {}
    for card in parser.cards:
        unique.setdefault(card.photo_url, card)
    return list(unique.values())


class PoliteClient:
    """HTTP client with a fixed minimum interval and bounded retries."""

    def __init__(self, *, delay: float, timeout: float, retries: int = 3) -> None:
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request_at = 0.0

    def _wait(self) -> None:
        remaining = self.delay - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def get(self, url: str, **kwargs) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            self._wait()
            try:
                response = self.session.get(url, timeout=self.timeout, **kwargs)
                self._last_request_at = time.monotonic()
                if response.status_code == 403 and urlparse(url).hostname == "platesmania.com":
                    raise AccessBlocked(
                        "PlatesMania отклонил прямой HTTP-запрос (403/Cloudflare). "
                        "Откройте нужные страницы в обычном браузере, сохраните их как HTML "
                        "и запустите скрипт с --html-file pages/*.html"
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("Retry-After", "")
                    wait = float(retry_after) if retry_after.isdigit() else 2**attempt
                    if attempt + 1 < self.retries:
                        time.sleep(min(wait, 30.0))
                        continue
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                self._last_request_at = time.monotonic()
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    def close(self) -> None:
        self.session.close()


def robots_allows(client: PoliteClient, url: str) -> bool:
    """Return False only when an accessible robots.txt explicitly forbids URL."""

    try:
        response = client.get(ROBOTS_URL)
    except requests.RequestException as exc:
        log.warning("robots.txt недоступен (%s); продолжаю без обхода ограничений", exc)
        return True
    parser = RobotFileParser()
    parser.set_url(ROBOTS_URL)
    parser.parse(response.text.splitlines())
    return parser.can_fetch(USER_AGENT, url)


def image_size(data: bytes) -> tuple[int, int]:
    """Read the intrinsic image size without relying on HTML attributes."""

    with Image.open(BytesIO(data)) as image:
        return image.size


def photo_filename(url: str) -> str:
    """Build a safe local filename from a gallery photo URL."""

    name = Path(urlparse(url).path).name
    stem = "".join(char for char in Path(name).stem if char.isalnum() or char in "-_")
    suffix = Path(name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    return f"{stem or 'photo'}{suffix}"


def process_card(
    card: GalleryCard,
    *,
    client: PoliteClient,
    output: Path,
    target_height: int,
    dry_run: bool,
    stats: ScrapeStats,
) -> None:
    """Check one plate preview and save its paired vehicle photo if selected."""

    try:
        preview = client.get(card.plate_preview_url).content
        width, height = image_size(preview)
    except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
        stats.errors += 1
        log.warning("Не удалось проверить %s: %s", card.plate_preview_url, exc)
        return

    if height != target_height:
        log.debug("Пропуск %s: размер номера %dx%d", card.plate_text or card.plate_preview_url, width, height)
        return

    stats.square += 1
    destination = output / photo_filename(card.photo_url)
    if destination.exists():
        stats.already_exists += 1
        log.info("Уже существует: %s", destination)
        return
    if dry_run:
        log.info("Подходит: %s (%dx%d) -> %s", card.plate_text or "без текста", width, height, destination)
        return

    try:
        photo = client.get(card.photo_url).content
        # Validate the response before it is persisted as an image file.
        image_size(photo)
        destination.write_bytes(photo)
    except (requests.RequestException, UnidentifiedImageError, OSError) as exc:
        stats.errors += 1
        log.warning("Не удалось сохранить %s: %s", card.photo_url, exc)
        return

    stats.saved += 1
    log.info("Сохранено: %s (номер: %s)", destination, card.plate_text or "не указан")


def scrape(args: argparse.Namespace, client: PoliteClient) -> ScrapeStats:
    output = args.output.resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    stats = ScrapeStats()
    if args.html_file:
        local_pages = [
            (args.start + offset, path.read_text(encoding="utf-8"))
            for offset, path in enumerate(args.html_file)
        ]
    else:
        local_pages = []
        if not robots_allows(client, GALLERY_URL):
            raise RuntimeError(f"robots.txt запрещает загрузку {GALLERY_URL}")

    page_count = len(local_pages) if local_pages else args.pages
    for offset in range(page_count):
        start = args.start + offset
        if local_pages:
            html = local_pages[offset][1]
        else:
            response = client.get(GALLERY_URL, params={"ctype": 1, "start": start})
            html = response.text

        cards = parse_gallery(html)
        stats.pages += 1
        stats.cards += len(cards)
        log.info("Страница start=%d: найдено карточек %d", start, len(cards))
        if not cards:
            log.warning("Карточки не найдены; останавливаюсь — возможно, разметка сайта изменилась")
            break

        client.session.headers["Referer"] = f"{GALLERY_URL}?ctype=1&start={start}"
        for card in cards:
            process_card(
                card,
                client=client,
                output=output,
                target_height=args.target_height,
                dry_run=args.dry_run,
                stats=stats,
            )
    return stats


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("значение должно быть не меньше 1")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("значение не может быть отрицательным")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Скачать фотографии PlatesMania, у которых PNG номера имеет высоту 80 px"
    )
    parser.add_argument("--start", type=positive_int, default=1, help="Первое значение start (по умолчанию: 1)")
    parser.add_argument("--pages", type=positive_int, default=1, help="Количество страниц (по умолчанию: 1)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/platesmania_square"),
        help="Каталог для фотографий",
    )
    parser.add_argument("--target-height", type=positive_int, default=80, help="Искомая высота PNG номера")
    parser.add_argument("--delay", type=nonnegative_float, default=1.0, help="Минимальная пауза между запросами")
    parser.add_argument("--timeout", type=positive_int, default=30, help="Таймаут HTTP-запроса в секундах")
    parser.add_argument("--dry-run", action="store_true", help="Показать совпадения, не сохраняя фотографии")
    parser.add_argument(
        "--html-file",
        type=Path,
        nargs="+",
        default=None,
        metavar="HTML",
        help="Разобрать один или несколько локальных HTML вместо загрузки страниц",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    client = PoliteClient(delay=args.delay, timeout=args.timeout)
    try:
        stats = scrape(args, client)
    except (requests.RequestException, OSError, RuntimeError) as exc:
        log.error("Парсер остановлен: %s", exc)
        return 1
    finally:
        client.close()

    print(
        f"Готово: страниц={stats.pages}, карточек={stats.cards}, "
        f"квадратных={stats.square}, сохранено={stats.saved}, "
        f"уже было={stats.already_exists}, ошибок={stats.errors}"
    )
    return 0 if stats.errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
