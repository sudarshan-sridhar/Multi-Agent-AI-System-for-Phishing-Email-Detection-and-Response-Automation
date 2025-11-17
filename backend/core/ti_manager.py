"""
Threat intelligence manager for phishing URL detection.

This module is responsible for:
  - Managing local caches of OpenPhish and URLHaus threat-intel feeds.
  - Persisting those feeds to disk so they survive process restarts.
  - Providing a simple runtime API (`check_url`) to query whether a URL
    or host appears in any of the known bad lists.

Design goals:
  - Be resilient to network failures (never hard-fail the main pipeline).
  - Avoid re-downloading feeds on every request by caching in memory and on disk.
  - Keep the public interface small and easy to integrate into scoring logic.
"""

import os
import time
from typing import Set, Tuple

import httpx

from .config import ti_config


class ThreatIntelManager:
    """
    Manages OpenPhish + URLHaus threat intel feeds.

    Responsibilities:
      • Maintain in-memory sets of:
          - full malicious URLs (OpenPhish)
          - known-bad hosts (URLHaus)
      • Persist these feeds to disk (for warm start on next process run).
      • Refresh data periodically or on demand.
      • Provide a fast, read-only interface for URL lookups.

    The class keeps stateful caches inside a single instance, which can be
    shared across the application (e.g., via module-level singleton).
    """

    def __init__(self):
        # In-memory threat intel sets for quick O(1) membership checks.
        self._openphish_urls: Set[str] = set()
        self._urlhaus_hosts: Set[str] = set()

        # Timestamp (epoch seconds) of the last successful load/refresh.
        # None indicates "never loaded".
        self._last_loaded: float | None = None

    # --------- Disk helpers ---------

    def _load_from_disk(self) -> None:
        """
        Populate in-memory caches from on-disk cache files, if present.

        This is used for a "warm start" so the system does not need to
        re-download feeds every time the service starts.
        """
        urls_path = ti_config.cache_path_openphish
        hosts_path = ti_config.cache_path_urlhaus

        if os.path.exists(urls_path):
            with open(urls_path, "r", encoding="utf-8") as f:
                self._openphish_urls = {line.strip() for line in f if line.strip()}

        if os.path.exists(hosts_path):
            with open(hosts_path, "r", encoding="utf-8") as f:
                self._urlhaus_hosts = {line.strip() for line in f if line.strip()}

        self._last_loaded = time.time()

    def _save_to_disk(self) -> None:
        """
        Persist in-memory threat intel sets to disk.

        This keeps cache files in sync with the latest successful download,
        allowing quick reuse after restarts.
        """
        with open(ti_config.cache_path_openphish, "w", encoding="utf-8") as f:
            for url in sorted(self._openphish_urls):
                f.write(url + "\n")

        with open(ti_config.cache_path_urlhaus, "w", encoding="utf-8") as f:
            for host in sorted(self._urlhaus_hosts):
                f.write(host + "\n")

    # --------- Public API ---------

    def ensure_loaded(self, max_age_hours: float = 24.0) -> None:
        """
        Ensure feeds are loaded into memory (from disk or network).

        Behavior:
          - If nothing has ever been loaded, try to read from disk.
          - If caches are empty after disk load, download fresh feeds.
          - If caches exist but are older than `max_age_hours`, refresh them.

        This method is safe to call from hot paths; it only downloads
        when required and otherwise returns quickly.
        """
        if self._last_loaded is None:
            # First attempt: try loading existing cache from disk.
            self._load_from_disk()

        # If still empty, fetch fresh data from the network.
        if not self._openphish_urls or not self._urlhaus_hosts:
            print("[TI] No cached feeds found; downloading fresh feeds...")
            self.update_feeds()
            return

        # Check age of cache (defensive try/except in case timestamp is corrupted).
        try:
            age_hours = (time.time() - self._last_loaded) / 3600.0
        except Exception:
            age_hours = max_age_hours + 1

        if age_hours > max_age_hours:
            print("[TI] Cache older than max_age_hours; refreshing in background...")
            # Note: currently refresh is synchronous. If needed, this could
            # be changed to a background thread in the future.
            self.update_feeds()

    def update_feeds(self) -> None:
        """
        Download OpenPhish + URLHaus feeds and write them to disk.

        This function:
          - Fetches raw lists from the configured TI endpoints.
          - Filters out comments/empty lines.
          - Updates in-memory sets.
          - Persists the updated sets to disk.

        Any network or parsing errors are logged but do not raise exceptions,
        to avoid impacting the main detection pipeline.
        """
        openphish_urls: Set[str] = set()
        urlhaus_hosts: Set[str] = set()

        try:
            with httpx.Client(timeout=60) as client:
                # OpenPhish: one URL per line.
                r1 = client.get(ti_config.openphish_url)
                r1.raise_for_status()
                for line in r1.text.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        openphish_urls.add(line)

                # URLHaus: hosts list (also one per line).
                r2 = client.get(ti_config.urlhaus_url)
                r2.raise_for_status()
                for line in r2.text.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        urlhaus_hosts.add(line)
        except Exception as e:
            print(f"[TI] Error while downloading feeds: {e}")
            # On error, retain whatever is already in memory / on disk.
            if self._last_loaded is None:
                # If we've never successfully loaded, we simply remain empty.
                pass
            return

        # Replace in-memory sets atomically and persist to disk.
        self._openphish_urls = openphish_urls
        self._urlhaus_hosts = urlhaus_hosts
        self._last_loaded = time.time()
        self._save_to_disk()
        print(f"[TI] Loaded {len(openphish_urls)} OpenPhish URLs and {len(urlhaus_hosts)} URLHaus hosts.")

    def check_url(self, url: str) -> Tuple[bool, bool]:
        """
        Check whether the given URL or its host appears in any threat intel list.

        Returns:
            (in_openphish, in_urlhaus)
              - in_openphish: True if the exact URL is present in OpenPhish.
              - in_urlhaus: True if any known-bad host from URLHaus is a substring
                            of the URL (simple heuristic host matching).

        This function is intentionally lightweight and can be called frequently
        during email analysis.
        """
        if not url:
            return False, False

        # Lazy initialization: ensure caches are populated before first lookup.
        if self._last_loaded is None:
            self.ensure_loaded()

        # Direct URL match for OpenPhish.
        in_openphish = url in self._openphish_urls

        # Host match for URLHaus: crude substring search for now.
        # This can be refined later to use proper host extraction.
        in_urlhaus = any(h in url for h in self._urlhaus_hosts)

        return in_openphish, in_urlhaus
