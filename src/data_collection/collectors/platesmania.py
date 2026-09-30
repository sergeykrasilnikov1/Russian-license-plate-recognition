"""PlatesMania collector — documented XML export only.

The competition rules forbid bypassing technical restrictions, so this
collector deliberately contains no CAPTCHA solving, no browser-impersonating
User-Agent, no cookie replay and no HTML parsing of gallery pages. It probes
the documented XML feed endpoints with an honest User-Agent and, if the site
refuses access, reports itself unavailable and the run continues without it.
"""

from __future__ import annotations

import logging
from typing import Iterator
from xml.etree import ElementTree

from ..licensing import normalize_license
from .base import Collector, RawItem

log = logging.getLogger(__name__)

# Documented feed endpoints; probed in order.
FEED_URLS: tuple[str, ...] = (
    "https://platesmania.com/ru/gallery.xml",
    "https://platesmania.com/xml/gallery.xml",
    "https://platesmania.com/informer.xml",
)

ROBOTS_URL = "https://platesmania.com/robots.txt"


class PlatesManiaCollector(Collector):
    name = "platesmania"
    access_note = (
        "Documented XML export only, honest User-Agent, robots.txt respected. "
        "No CAPTCHA bypass, no paywall bypass, no HTML gallery scraping."
    )

    def __init__(self, cache_dir, timeout: float = 30.0, feed_urls: tuple[str, ...] = FEED_URLS) -> None:
        super().__init__(cache_dir, timeout)
        self.feed_urls = feed_urls
        self._usable_feed: str | None = None

    def _robots_allows(self, session, url: str) -> bool:
        from urllib.robotparser import RobotFileParser

        try:
            resp = session.get(ROBOTS_URL, timeout=self.timeout)
            if resp.status_code != 200:
                return True
            parser = RobotFileParser()
            parser.parse(resp.text.splitlines())
            return parser.can_fetch(session.headers.get("User-Agent", "*"), url)
        except Exception:
            return True

    def check_available(self) -> tuple[bool, str]:
        session = self._session()
        failures: list[str] = []
        for url in self.feed_urls:
            if not self._robots_allows(session, url):
                failures.append(f"{url}:robots_disallow")
                continue
            try:
                resp = session.get(url, timeout=self.timeout)
            except Exception as exc:
                failures.append(f"{url}:{type(exc).__name__}")
                continue
            if resp.status_code == 200 and resp.text.lstrip().startswith("<"):
                self._usable_feed = url
                return True, f"ok:{url}"
            failures.append(f"{url}:http_{resp.status_code}")
        return False, "no_documented_feed_accessible(" + ", ".join(failures) + ")"

    def collect(self, limit: int) -> Iterator[RawItem]:
        if self._usable_feed is None:
            ok, reason = self.check_available()
            if not ok:
                log.warning("platesmania unavailable: %s", reason)
                return

        session = self._session()
        try:
            resp = session.get(self._usable_feed, timeout=self.timeout)
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)
        except Exception as exc:
            log.warning("platesmania feed parse failed: %s", exc)
            return

        produced = 0
        for item in root.iter("item"):
            if produced >= limit:
                return
            image_url = self._first_text(item, ("enclosure_url", "image", "photo", "link"))
            if not image_url:
                enclosure = item.find("enclosure")
                image_url = enclosure.get("url") if enclosure is not None else None
            if not image_url:
                continue
            plate_text = self._first_text(item, ("plate", "number", "title")) or ""
            license_raw = self._first_text(item, ("license", "rights")) or ""
            if normalize_license(license_raw).spdx == "UNKNOWN":
                continue

            try:
                img = session.get(image_url, timeout=self.timeout)
                img.raise_for_status()
            except Exception as exc:
                log.warning("platesmania image download failed %s: %s", image_url, exc)
                continue

            yield RawItem(
                source=self.name,
                source_url=self._first_text(item, ("link",)) or image_url,
                license_raw=license_raw,
                filename=f"pm_{produced:06d}.jpg",
                data=img.content,
                query=plate_text,
            )
            produced += 1

    @staticmethod
    def _first_text(item, tags: tuple[str, ...]) -> str | None:
        for tag in tags:
            node = item.find(tag)
            if node is not None and (node.text or "").strip():
                return node.text.strip()
        return None
