"""Deterministic scheduler: platform gain + coverage-evenness marginal + scarcity, ledger counts real completions."""

from __future__ import annotations

import os

from ledger import Ledger

W_DEFAULT = 0.35
SLOT = 900.0


def _env(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


EVEN_SCALE = _env("SAC_EVEN_SCALE", 0.0)
PRIOR = _env("SAC_EVEN_PRIOR", 2.0)
SCARCE_K = _env("SAC_SCARCE_K", 0.0)
SCARCE_S0 = _env("SAC_SCARCE_S0", 6.0)
REQ_LAST = _env("SAC_REQ_LAST", 0.0)
TIE_BAND = _env("SAC_TIE_BAND", 0.002)


def coverage_weight(snapshot):
    for key in ("score_config", "scoring", "competition"):
        block = snapshot.get(key)
        if isinstance(block, dict) and "coverage_bonus_weight" in block:
            try:
                return float(block["coverage_bonus_weight"])
            except (TypeError, ValueError):
                return 0.0
    contract = snapshot.get("scoring_contract") or {}
    block = contract.get("score_config") if isinstance(contract, dict) else None
    if isinstance(block, dict) and "coverage_bonus_weight" in block:
        try:
            return float(block["coverage_bonus_weight"])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def jain(counts):
    total = sum(counts)
    sq = sum(c * c for c in counts)
    return 0.0 if total <= 0 or sq <= 0 else total * total / (len(counts) * sq)


def even_marginals(counts_by_region, prior):
    """(E_after, dE) of adding one completed tile to each region, on prior-smoothed counts."""
    regions = list(counts_by_region)
    base = [counts_by_region[r] + prior for r in regions]
    n = len(base)
    total = sum(base)
    sq = sum(x * x for x in base)
    before = total * total / (n * sq)
    out = {}
    for r, x in zip(regions, base):
        after = (total + 1) ** 2 / (n * (sq + 2 * x + 1))
        out[r] = (after, after - before)
    return out


def get_ledger(memory):
    led = memory.get("_ledger")
    if led is None:
        led = Ledger(memory.get("_init"))
        memory["_ledger"] = led
    return led


def choose_action(candidates, snapshot, memory):
    led = get_ledger(memory)
    now = led.now
    if not candidates:
        return None
    weight = coverage_weight(snapshot)
    plan = memory.get("_plan") or {}
    region_w = plan.get("region_weights") or {}

    marg = even_marginals(led.counts, PRIOR) if led.regions and weight > 0 else {}
    s_proj = led.projected_science()
    cur_e = jain([led.counts[r] + 0.0 for r in led.regions]) if led.regions else 0.0

    # REQUIRED tiles with almost no windows left: a miss costs 1000, take them first.
    at_risk = []
    for rank, c in enumerate(candidates if REQ_LAST > 0 else ()):
        if (c.get("scheduling_class") or "").upper() != "REQUIRED" or c["tile_id"] in led.completed:
            continue
        rem = led.remaining_seconds(c["tile_id"], now)
        need = max(1.0, float(c.get("nominal_exptime_seconds") or SLOT))
        if rem is not None and rem <= (REQ_LAST * SLOT + need):
            at_risk.append((rem, rank, c))
    if at_risk:
        at_risk.sort(key=lambda item: (item[0], item[1]))
        chosen = at_risk[0][2]
        chosen["reason"] = "required tile about to leave its visibility windows"
        led.note_choice(chosen["tile_id"], chosen.get("estimated_science_score") or 0.0)
        return chosen

    best, best_val = None, float("-inf")
    scored = []
    for c in candidates:
        seconds = max(1.0, float(c.get("nominal_exptime_seconds") or SLOT))
        gain = float(c.get("estimated_total_gain") or 0.0)
        sci = float(c.get("estimated_science_score") or 0.0)
        tile_id = c["tile_id"]
        fresh = tile_id not in led.completed
        if fresh and marg:
            after, d_e = marg.get(c.get("region_id"), (cur_e, 0.0))
            gain += EVEN_SCALE * weight * (sci * after + s_proj * d_e)
        if fresh and SCARCE_K > 0:
            rem = led.remaining_seconds(tile_id, now)
            if rem is not None:
                slack = rem / seconds
                if slack < SCARCE_S0:
                    gain += SCARCE_K * sci * (SCARCE_S0 - slack) / SCARCE_S0
        val = gain / seconds
        scored.append((val, c))
        if val > best_val:
            best_val, best = val, c
    if region_w and best_val > 0:
        floor = best_val * (1.0 - TIE_BAND)
        tied = [(v * float(region_w.get(c.get("region_id"), 1.0)), c) for v, c in scored if v >= floor]
        pick = max(tied, key=lambda item: item[0])[1]
        if pick is not best:
            best = pick
            best["reason"] = "near-tie broken by the nightly region plan"
    best.setdefault("reason", "highest adjusted gain per second")
    led.note_choice(best["tile_id"], best.get("estimated_science_score") or 0.0)
    return best
