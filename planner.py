"""Bounded LLM layer: a nightly region-weight plan (tie-break only) and a veto-only report review.

Every failure path returns None / the unchanged input, so the deterministic agent is always the fallback.
"""

from __future__ import annotations

import json
import os
import time

NIGHT_SYSTEM = (
    "You assist a telescope scheduler. Given per-region completion counts and remaining visibility supply, "
    "return one JSON object {\"region_weights\": {region_id: number}, \"note\": short sentence}. "
    "Weights express which regions deserve a slight preference when candidates are otherwise tied; "
    "use values between 0.9 and 1.1, higher for regions that are behind and still reachable."
)
REVIEW_SYSTEM = (
    "You audit a pending anomaly report for a telescope. The evidence is the ratio of realized to expected score "
    "for each exposure of one tile. A true NOVA tag reads about 1.5 and a true Reddening tag about 0.8 on almost "
    "every read; weather noise spreads reads between 0.9 and 1.0. Return one JSON object "
    "{\"confirm\": true|false, \"why\": short sentence}. Answer false when the evidence looks like noise."
)
CLAMP = (0.9, 1.1)


class Planner:
    def __init__(self, llm=None):
        self.llm = llm
        self.calls = 0
        self.vetoes = 0

    @property
    def enabled(self):
        return bool(self.llm is not None and getattr(self.llm, "enabled", False))

    def night_plan(self, ledger):
        if not self.enabled or not ledger.regions:
            return None
        supply = {}
        for tile_id, region in ledger.region_of.items():
            if tile_id not in ledger.completed and ledger.remaining_seconds(tile_id, ledger.now):
                supply[region] = supply.get(region, 0) + 1
        user = json.dumps({
            "night_index": len(ledger.nights_seen), "night_count": ledger.night_count,
            "completed_by_region": {r: ledger.counts[r] for r in ledger.regions},
            "reachable_unfinished_by_region": {r: supply.get(r, 0) for r in ledger.regions},
        }, separators=(",", ":"))
        self.calls += 1
        reply = self.llm.chat_json(NIGHT_SYSTEM, user, max_tokens=300, purpose="night_plan")
        if not isinstance(reply, dict) or not isinstance(reply.get("region_weights"), dict):
            return None
        weights = {}
        for region, value in reply["region_weights"].items():
            if region in ledger.counts:
                try:
                    weights[region] = min(CLAMP[1], max(CLAMP[0], float(value)))
                except (TypeError, ValueError):
                    continue
        return {"region_weights": weights, "note": str(reply.get("note", ""))[:200]} if weights else None

    def review_reports(self, reports, ratios_by_tile):
        """Return the reports to keep. Only tag reports can be vetoed; faults always pass."""
        if not self.enabled or not reports:
            return reports
        kept = []
        for report in reports:
            tile = report.get("tile_id")
            if report.get("kind") == "Instrument_Failure" or not tile:
                kept.append(report)
                continue
            ratios = [round(r, 3) for r in ratios_by_tile.get(tile, [])][-20:]
            self.calls += 1
            reply = self.llm.chat_json(
                REVIEW_SYSTEM, json.dumps({"tag": report["kind"], "ratios": ratios}, separators=(",", ":")),
                max_tokens=120, purpose="report_review")
            if isinstance(reply, dict) and reply.get("confirm") is False:
                self.vetoes += 1
                continue
            kept.append(report)
        return kept


def build_planner():
    """Planner backed by LLMClient when credentials exist; otherwise a disabled one."""
    try:
        from llm_client import LLMClient
        wall = float(os.environ.get("SAC_WALLCLOCK_SECONDS", 3600))
        return Planner(LLMClient(wallclock_seconds=wall, started_at=time.monotonic()))
    except Exception:  # noqa: BLE001
        return Planner(None)
