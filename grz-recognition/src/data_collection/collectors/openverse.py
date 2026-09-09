"""Openverse collector — CC-licensed photo search across public providers.

Openverse exposes a documented anonymous REST API and returns the license of
every result, so images can be admitted or rejected by license before they
touch the dataset.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator

from .base import Collector, RateLimited, RawItem, request_with_retry

log = logging.getLogger(__name__)

API_URL = "https://api.openverse.org/v1/images/"

QUERIES: dict[str, tuple[str, ...]] = {
    "type1b": ("russian taxi", "moscow taxi car", "russian bus front", "yandex taxi"),
    "type1a": ("kei car", "japanese car rear plate", "right hand drive car russia"),
    "type1": ("russian car", "car number plate russia", "lada car rear"),
    "other": ("car trailer plate", "motorcycle number plate", "truck trailer russia"),
}

# Licenses that permit derivatives (crops/augmentations) and non-commercial reuse
ALLOWED_LICENSES = ("by", "by-sa", "by-nc", "by-nc-sa", "cc0", "pdm")


class OpenverseCollector(Collector):
    name = "openverse"
    access_note = (
        "Documented REST API (api.openverse.org/v1/images) with server-side "
        "license filter; ND excluded. Anonymous access has a daily quota — "
        "set OPENVERSE_API_TOKEN for a registered (higher) quota. Throttling "
        "is waited out or reported, never bypassed."
    )

    def __init__(self, cache_dir, timeout: float = 30.0, delay: float = 1.0, queries: dict[str, tuple[str, ...]] | None = None, plate_types: tuple[str, ...] | None = None, token: str | None = None) -> None:
        super().__init__(cache_dir, timeout)
        self.delay = delay
        queries = queries or QUERIES
        if plate_types:
            queries = {k: v for k, v in queries.items() if k in plate_types}
        self.queries = queries
        self.token = token or os.environ.get("OPENVERSE_API_TOKEN", "")

    def _session(self):
        session = super()._session()
        if self.token:
            session.headers["Authorization"] = f"Bearer {self.token}"
        return session

    def check_available(self) -> tuple[bool, str]:
        try:
            session = self._session()
            r = session.get(API_URL, params={"q": "car", "page_size": 1}, timeout=self.timeout)
            if r.status_code == 200:
                return True, "ok" if not self.token else "ok:token"
            if r.status_code in (401, 429):
                hint = "anonymous_quota_exhausted" if not self.token else "token_rejected"
                return False, f"http_{r.status_code}:{hint}"
            return False, f"http_{r.status_code}"
        except Exception as exc:
            return False, f"unreachable:{type(exc).__name__}"

    def collect(self, limit: int) -> Iterator[RawItem]:
        session = self._session()
        per_type = max(1, limit // max(1, len(self.queries)))
        yielded = 0

        for plate_type, queries in self.queries.items():
            per_query = max(1, per_type // len(queries))
            for query in queries:
                if yielded >= limit:
                    return
                try:
                    r = request_with_retry(
                        session,
                        API_URL,
                        params={
                            "q": query,
                            "license": ",".join(ALLOWED_LICENSES),
                            "page_size": min(50, per_query * 2),
                            "mature": "false",
                        },
                        timeout=self.timeout,
                    )
                    results = r.json().get("results", [])
                except RateLimited as exc:
                    log.warning("openverse quota/throttle hit (%s); stopping this source", exc)
                    return
                except Exception as exc:
                    log.warning("openverse query %r failed: %s", query, exc)
                    continue

                for item in results:
                    if yielded >= limit:
                        return
                    url = item.get("url")
                    if not url:
                        continue
                    try:
                        resp = session.get(url, timeout=self.timeout)
                        resp.raise_for_status()
                    except Exception as exc:
                        log.warning("openverse download failed %s: %s", url, exc)
                        continue

                    ident = str(item.get("id", ""))[:40]
                    ext = ".png" if url.lower().endswith(".png") else ".jpg"
                    license_raw = item.get("license", "")
                    version = item.get("license_version") or ""
                    if version:
                        license_raw = f"cc-{license_raw}-{version}"
                    yield RawItem(
                        source=self.name,
                        source_url=item.get("foreign_landing_url") or url,
                        license_raw=license_raw,
                        filename=f"{ident}{ext}",
                        data=resp.content,
                        query=query,
                        suggested_type=plate_type,
                        attribution=str(item.get("creator") or "")[:200],
                    )
                    yielded += 1
                    time.sleep(self.delay)
