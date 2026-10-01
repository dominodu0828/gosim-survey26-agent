"""Standard-library OpenAI-compatible chat client with a wall-clock budget watchdog.

Fixed interface (docs/PLAN_v1.md section 4):

    client = LLMClient(wallclock_seconds=..., started_at=...)
    if client.enabled:
        payload = client.chat_json(system, user, max_tokens=600, purpose="night_plan")
    client.stats()  # {"calls", "failures", "seconds_used", "disabled_reason"}

Behaviour contract:
  * POST {base_url}/chat/completions (never doubled when base_url already ends in
    /chat/completions) with {"model", "messages": [system, user], "max_tokens", "temperature": 0}
    and an Authorization: Bearer header;
  * credentials come from the environment, platform variables first:
    OPENAI_BASE_URL + OPENAI_API_KEY + OPENAI_MODEL, then MODEL_BASE_URL + DEEPSEEK_API_KEY + MODEL_NAME;
  * MODEL_PROVIDER=deterministic (or missing key/base_url) disables the client without
    reading any key;
  * chat_json returns the first complete JSON object in the reply, or None on ANY failure
    (network, timeout, HTTP error, unparseable output) -- it never raises;
  * every call writes a single `LLMCALL {...}` line to stderr with purpose, ok, error_type,
    seconds and token usage. Keys, prompts and replies are never logged;
  * the client disables itself when the accumulated call time crosses
    min(budget_cap_seconds, budget_fraction * wallclock_seconds), when fewer than
    3 * timeout_seconds remain before started_at + wallclock_seconds, or after 5
    consecutive failures.

Standard library only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

CONSECUTIVE_FAILURE_LIMIT = 5
ENDGAME_MULTIPLIER = 3  # disable when less than 3 * timeout_seconds of the wall clock remain


def _first_json_object(text: str) -> dict:
    """Return the first complete, balanced JSON object found in `text`.

    Tolerates ```json fences and any prose before or after the object.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start:index + 1]
                    value = json.loads(candidate)  # may raise, caught by caller
                    if not isinstance(value, dict):
                        raise ValueError("model output must be a JSON object")
                    return value
        start = stripped.find("{", start + 1)
    raise ValueError("model output contains no JSON object")


class LLMClient:
    """Chat-completions client that can only get cheaper: every failure path returns None."""

    def __init__(self, *, wallclock_seconds: float, started_at: float,
                 budget_fraction: float = 0.25, budget_cap_seconds: float = 900.0,
                 timeout_seconds: float = 20.0) -> None:
        self.wallclock_seconds = float(wallclock_seconds)
        self.started_at = float(started_at)
        self.budget_fraction = float(budget_fraction)
        self.budget_cap_seconds = float(budget_cap_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.calls = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.seconds_used = 0.0
        self.disabled_reason: str | None = None
        # MODEL_PROVIDER=deterministic is decided once at start-up; the client then never reads a key.
        self.deterministic = os.environ.get("MODEL_PROVIDER", "").strip().lower() == "deterministic"
        if self.deterministic:
            self._disable("deterministic")
            self.base_url = self.api_key = self.model = None
        else:
            self.base_url, self.api_key, self.model = self._credentials()

    # -- configuration -----------------------------------------------------------

    @staticmethod
    def _credentials() -> tuple[str | None, str | None, str | None]:
        """Platform variables first (OPENAI_*), then the local .env style ones."""
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("MODEL_BASE_URL") or None
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or None
        model = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL_NAME") or None
        return base_url, api_key, model

    @property
    def budget_seconds(self) -> float:
        return min(self.budget_cap_seconds, self.budget_fraction * self.wallclock_seconds)

    @property
    def enabled(self) -> bool:
        if self.deterministic:
            return False
        if self.disabled_reason is not None:
            return False
        if not self.api_key or not self.base_url:
            return False
        if self.seconds_used >= self.budget_seconds:
            return False
        remaining = self.started_at + self.wallclock_seconds - time.monotonic()
        if remaining < ENDGAME_MULTIPLIER * self.timeout_seconds:
            return False
        return True

    def _disable(self, reason: str) -> None:
        if self.disabled_reason is None:
            self.disabled_reason = reason

    # -- requests ----------------------------------------------------------------

    def _endpoint(self) -> str:
        base = self.base_url or ""
        if base.endswith("/chat/completions"):
            return base
        return base.rstrip("/") + "/chat/completions"

    def _log(self, **fields) -> None:
        payload = {"purpose": fields.get("purpose", ""), "ok": fields.get("ok", False),
                   "error_type": fields.get("error_type"), "seconds": round(fields.get("seconds", 0.0), 3),
                   "prompt_tokens": fields.get("prompt_tokens"), "completion_tokens": fields.get("completion_tokens")}
        print("LLMCALL " + json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)

    def chat_json(self, system: str, user: str, *, max_tokens: int = 600, purpose: str = "") -> dict | None:
        if self.deterministic:
            return None
        if self.disabled_reason is not None:
            return None
        if not self.api_key or not self.base_url:
            self._disable("missing_credentials")
            return None
        if self.seconds_used >= self.budget_seconds:
            self._disable("budget_exhausted")
            return None
        if self.started_at + self.wallclock_seconds - time.monotonic() < ENDGAME_MULTIPLIER * self.timeout_seconds:
            self._disable("wallclock_endgame")
            return None

        body = json.dumps({
            "model": self.model or "",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": int(max_tokens),
            "temperature": 0,
        }).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(), data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        started = time.monotonic()
        error_type = None
        prompt_tokens = completion_tokens = None
        content = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            content = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            value = _first_json_object(content if isinstance(content, str) else json.dumps(content))
        except Exception as exc:  # noqa: BLE001 - any failure must degrade to None, never raise
            error_type = type(exc).__name__
        elapsed = time.monotonic() - started
        self.seconds_used += elapsed
        self.calls += 1
        if error_type is not None:
            self.failures += 1
            self.consecutive_failures += 1
            if self.consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                self._disable("consecutive_failures")
        else:
            self.consecutive_failures = 0
        self._log(purpose=purpose, ok=error_type is None, error_type=error_type,
                  seconds=elapsed, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if self.seconds_used >= self.budget_seconds:
            self._disable("budget_exhausted")
        if error_type is not None:
            return None
        return value

    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "seconds_used": round(self.seconds_used, 3),
            "disabled_reason": self.disabled_reason,
        }
