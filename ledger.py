"""Run-long ledger: what is truly completed, per-region counts, cached visibility windows, projected science."""

from __future__ import annotations

from datetime import datetime


def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


class Ledger:
    def __init__(self, init_publication):
        init_publication = init_publication or {}
        tiles = (init_publication.get("tile_catalog") or {}).get("tiles") or []
        self.region_of = {str(t["tile_id"]): str(t.get("region_id", "")) for t in tiles if "tile_id" in t}
        self.class_of = {str(t["tile_id"]): str(t.get("scheduling_class", "")) for t in tiles if "tile_id" in t}
        self.regions = sorted(set(self.region_of.values())) or [str(r) for r in (init_publication.get("tile_catalog") or {}).get("region_ids", [])]
        cal = init_publication.get("calendar") or {}
        self.night_count = int(cal.get("night_count") or 0)
        self.slots_per_night = (int(cal.get("slot_count") or 0) / self.night_count) if self.night_count else 0
        self.slot_in_night = 0
        self.completed: set[str] = set()
        self.counts = {r: 0 for r in self.regions}
        self.windows: dict[str, list[tuple[float, float]]] = {}
        self.first_ts = None
        self.science = 0.0
        self._pending = None  # (tile_id, est_science, was_completed_before)
        self.nights_seen: set[str] = set()
        self.last_weekly_ts = None
        self.now = None

    def update(self, snapshot):
        cursor = snapshot.get("cursor") or {}
        now = parse_ts(cursor.get("timestamp_utc"))
        self.now = now
        if self.first_ts is None and now is not None:
            self.first_ts = now
        night = cursor.get("night_id")
        try:
            self.slot_in_night = int(str(cursor.get("slot_id", "")).rsplit("-S", 1)[1]) - 1
        except (IndexError, ValueError):
            pass
        if night:
            self.nights_seen.add(str(night))
        progress = snapshot.get("progress") or {}
        done_ids = progress.get("completed_tile_ids")
        if done_ids is not None and len(done_ids) != len(self.completed):
            new = set(map(str, done_ids)) - self.completed
            for tile_id in new:
                region = self.region_of.get(tile_id)
                if region is not None:
                    self.counts[region] = self.counts.get(region, 0) + 1
            self.completed |= new
        if self._pending is not None:
            tile_id, est, before = self._pending
            if tile_id in self.completed and not before:
                self.science += est
            self._pending = None
        weekly = snapshot.get("weekly")
        night_start = snapshot.get("night_start")
        if isinstance(weekly, dict) and weekly.get("tile_windows"):
            self._load_windows(weekly["tile_windows"], replace=True)
        if isinstance(night_start, dict) and night_start.get("tile_windows") and not (isinstance(weekly, dict) and weekly.get("tile_windows")):
            self._load_windows(night_start["tile_windows"], replace=False)
        return now

    def _load_windows(self, windows, replace):
        if replace:
            self.windows = {}
        seen = {}
        for w in windows:
            tile_id = w.get("tile_id")
            s, e = parse_ts(w.get("window_start_utc")), parse_ts(w.get("window_end_utc"))
            if tile_id is None or s is None or e is None:
                continue
            seen.setdefault(str(tile_id), set()).add((s, e))
        for tile_id, items in seen.items():
            current = set(self.windows.get(tile_id, ())) if not replace else set()
            self.windows[tile_id] = sorted(current | items)

    def remaining_seconds(self, tile_id, now):
        spans = self.windows.get(tile_id)
        if spans is None or now is None:
            return None
        return sum(max(0.0, e - max(s, now)) for s, e in spans)

    def note_choice(self, tile_id, est_science):
        self._pending = (tile_id, float(est_science), tile_id in self.completed)

    def progress_fraction(self, now):
        if self.night_count <= 0 or self.slots_per_night <= 0:
            return None
        night_idx = max(0, len(self.nights_seen) - 1)
        within = min(1.0, self.slot_in_night / self.slots_per_night)
        return min(1.0, (night_idx + within) / self.night_count)

    def projected_science(self):
        f = self.progress_fraction(None)
        if f is None or f < 0.05:
            return self.science
        return max(self.science, self.science / f)
