"""Wikimedia Commons collector via the public MediaWiki API.

Commons is used as the primary source for the long-tail types (1A squares,
1B yellow taxi/bus plates) that public plate datasets barely contain. The
MediaWiki API is a documented public endpoint that returns a per-file license
in `extmetadata`, which is exactly what `meta.csv` needs.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

from .base import Collector, RateLimited, RawItem, request_with_retry

log = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"

# Queries grouped by the plate type they are meant to surface
QUERIES: dict[str, tuple[str, ...]] = {
    "type1b": (
        "taxi Russia license plate",
        "yellow license plate Russia",
        "Yandex Taxi car",
        "Moscow taxi car",
        "russian bus registration plate",
        "маршрутное такси Россия",
    ),
    "type1a": (
        "square license plate Russia",
        "kei car Russia",
        "japanese car in Russia rear",
        "Toyota right hand drive Russia rear",
        "square number plate car rear",
    ),
    "type1": (
        "russian license plate",
        "Russia car registration plate",
        "автомобильный номер Россия",
    ),
    "other": (
        "trailer license plate Russia",
        "motorcycle license plate Russia",
        "diplomatic license plate Russia",
        "military vehicle plate Russia",
        "transit license plate Russia",
    ),
}

# Category traversal is far more precise than free-text search: a file in
# "Taxicabs in Moscow" is a taxi photo, whereas searching "moscow taxi car"
# also returns maps, buses and archive shots.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "type1b": (
        "Taxicabs in Russia",
        "Buses in Russia",
        "Minibuses in Russia",
    ),
    "type1a": (
        "Kei cars",
        "Japanese cars in Russia",
        "Right-hand-drive cars in Russia",
    ),
    "type1": (
        "Cars in Russia",
        "Vehicle registration plates of Russia",
    ),
    "other": (
        "Motorcycles of Russia",
        "Trailers in Russia",
        "Diplomatic vehicle registration plates",
    ),
}


class WikimediaCommonsCollector(Collector):
    name = "wikimedia_commons"
    access_note = (
        "Public MediaWiki API (action=query&list=search + prop=imageinfo). "
        "Per-file license read from extmetadata; no scraping of HTML, no bypass."
    )

    def __init__(
        self,
        cache_dir,
        timeout: float = 30.0,
        thumb_width: int = 1024,
        delay: float = 1.5,
        queries: dict[str, tuple[str, ...]] | None = None,
        categories: dict[str, tuple[str, ...]] | None = None,
        plate_types: tuple[str, ...] | None = None,
        mode: str = "category",
        category_depth: int = 2,
    ) -> None:
        super().__init__(cache_dir, timeout)
        self.thumb_width = thumb_width
        self.delay = delay
        self.mode = mode
        self.category_depth = category_depth
        queries = queries or QUERIES
        categories = categories or CATEGORIES
        if plate_types:
            queries = {k: v for k, v in queries.items() if k in plate_types}
            categories = {k: v for k, v in categories.items() if k in plate_types}
        self.queries = queries
        self.categories = categories

    def check_available(self) -> tuple[bool, str]:
        try:
            session = self._session()
            r = session.get(
                API_URL,
                params={"action": "query", "meta": "siteinfo", "format": "json"},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                return False, f"http_{r.status_code}"
            return True, "ok"
        except Exception as exc:
            return False, f"unreachable:{type(exc).__name__}"

    def _search(self, session, query: str, limit: int) -> list[str]:
        titles: list[str] = []
        offset = 0
        while len(titles) < limit:
            batch = min(50, limit - len(titles))
            r = request_with_retry(
                session,
                API_URL,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srnamespace": 6,
                    "srlimit": batch,
                    "sroffset": offset,
                    "maxlag": 5,
                    "format": "json",
                },
                timeout=self.timeout,
            )
            payload = r.json()
            hits = payload.get("query", {}).get("search", [])
            if not hits:
                break
            titles.extend(h["title"] for h in hits)
            offset += len(hits)
            if "continue" not in payload:
                break
            time.sleep(self.delay)
        return titles[:limit]

    def _category_members(self, session, category: str, member_type: str, limit: int) -> list[str]:
        titles: list[str] = []
        continue_token: str | None = None
        while len(titles) < limit:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmtype": member_type,
                "cmlimit": min(100, limit - len(titles)),
                "maxlag": 5,
                "format": "json",
            }
            if continue_token:
                params["cmcontinue"] = continue_token
            r = request_with_retry(session, API_URL, params=params, timeout=self.timeout)
            payload = r.json()
            members = payload.get("query", {}).get("categorymembers", [])
            titles.extend(m["title"] for m in members)
            continue_token = payload.get("continue", {}).get("cmcontinue")
            if not continue_token or not members:
                break
            time.sleep(self.delay)
        return titles[:limit]

    def _category_files(self, session, category: str, limit: int, depth: int) -> list[str]:
        """Collect file titles from a category, descending into subcategories."""
        files = self._category_members(session, category, "file", limit)
        if len(files) >= limit or depth <= 0:
            return files[:limit]

        subcats = self._category_members(session, category, "subcat", 12)
        for subcat in subcats:
            if len(files) >= limit:
                break
            name = subcat.removeprefix("Category:")
            try:
                files.extend(
                    self._category_files(session, name, limit - len(files), depth - 1)
                )
            except RateLimited:
                raise
            except Exception as exc:
                log.warning("commons subcategory %r failed: %s", name, exc)
        return files[:limit]

    def _image_info(self, session, titles: list[str]) -> list[dict]:
        infos: list[dict] = []
        for i in range(0, len(titles), 20):
            chunk = titles[i : i + 20]
            r = request_with_retry(
                session,
                API_URL,
                params={
                    "action": "query",
                    "titles": "|".join(chunk),
                    "prop": "imageinfo",
                    "iiprop": "url|size|extmetadata|mime",
                    "iiurlwidth": self.thumb_width,
                    "maxlag": 5,
                    "format": "json",
                },
                timeout=self.timeout,
            )
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                for info in page.get("imageinfo", []) or []:
                    info["_title"] = page.get("title", "")
                    infos.append(info)
            time.sleep(self.delay)
        return infos

    @staticmethod
    def _license_of(info: dict) -> str:
        meta = info.get("extmetadata", {}) or {}
        for key in ("LicenseShortName", "License", "UsageTerms"):
            value = (meta.get(key) or {}).get("value")
            if value:
                return str(value)
        return ""

    @staticmethod
    def _attribution_of(info: dict) -> str:
        meta = info.get("extmetadata", {}) or {}
        artist = (meta.get("Artist") or {}).get("value", "")
        return " ".join(artist.replace("\n", " ").split())[:300]

    def _work_items(self) -> list[tuple[str, str, str]]:
        """Build (plate_type, kind, term) tasks according to the active mode."""
        items: list[tuple[str, str, str]] = []
        if self.mode in ("category", "both"):
            for plate_type, categories in self.categories.items():
                items.extend((plate_type, "category", c) for c in categories)
        if self.mode in ("search", "both"):
            for plate_type, queries in self.queries.items():
                items.extend((plate_type, "search", q) for q in queries)
        return items

    def collect(self, limit: int) -> Iterator[RawItem]:
        session = self._session()
        work = self._work_items()
        if not work:
            return
        per_term = max(1, limit // len(work))
        yielded = 0

        for plate_type, kind, term in work:
            if yielded >= limit:
                return
            budget = min(per_term * 2, limit - yielded)
            try:
                if kind == "category":
                    titles = self._category_files(session, term, budget, self.category_depth)
                else:
                    titles = self._search(session, term, budget)
                infos = self._image_info(session, titles)
            except RateLimited as exc:
                log.warning("commons throttled on %s %r (%s); stopping this source", kind, term, exc)
                return
            except Exception as exc:
                log.warning("commons %s %r failed: %s", kind, term, exc)
                continue

            for info in infos:
                if yielded >= limit:
                    return
                mime = info.get("mime", "")
                if mime not in {"image/jpeg", "image/png"}:
                    continue

                # Wikimedia explicitly asks clients to fetch generated
                # thumbnails rather than unscaled originals. When no thumbnail
                # exists the file is a small diagram, not a usable photo, so
                # skipping it costs nothing.
                url = info.get("thumburl")
                if not url or url == info.get("url"):
                    continue

                try:
                    resp = request_with_retry(session, url, timeout=self.timeout)
                except RateLimited as exc:
                    log.warning("commons media throttled (%s); stopping this source", exc)
                    return
                except Exception as exc:
                    log.warning("commons download failed %s: %s", url, exc)
                    continue

                title = info["_title"].removeprefix("File:")
                ext = ".png" if mime == "image/png" else ".jpg"
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:80]
                yield RawItem(
                    source=self.name,
                    source_url=info.get("descriptionurl") or url,
                    license_raw=self._license_of(info),
                    filename=f"{safe}{ext}",
                    data=resp.content,
                    query=f"{kind}:{term}",
                    suggested_type=plate_type,
                    attribution=self._attribution_of(info),
                )
                yielded += 1
                time.sleep(self.delay)
