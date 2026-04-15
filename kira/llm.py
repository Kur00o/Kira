"""
kira/llm.py — LLM Interface (Gemini API + Ollama/Gemma)
=========================================================
Auto-selects backend from environment:
  - GEMINI_API_KEY set → uses Gemini REST API
  - OLLAMA_MODEL set (or GEMINI_API_KEY absent) → uses local Ollama

Gemini endpoint:
    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=KEY

Ollama endpoint:
    POST http://localhost:11434/api/generate
    {"model": "gemma3:4b", "prompt": "...", "stream": false}

Config (env vars / .env):
    GEMINI_API_KEY  — use Gemini if set
    GEMINI_MODEL    — default gemini-2.5-flash
    OLLAMA_HOST     — default http://localhost:11434
    OLLAMA_MODEL    — default gemma3:4b (used when no GEMINI_API_KEY)
"""

import json
import os
import time
import textwrap
from datetime import datetime, timezone
from typing import Optional

import requests


# ── Gemini configuration ──────────────────────────────────────────────────────

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_MODEL    = "gemini-2.5-flash"
GEMINI_API_KEY  = ""

# ── Ollama configuration ──────────────────────────────────────────────────────

OLLAMA_HOST  = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"

DEFAULT_TIMEOUT = 120
MAX_RETRIES     = 5
INITIAL_BACKOFF = 2

# Per-phase temperature — lower = more deterministic
PHASE_TEMPERATURE = {
    "RECON":        0.2,
    "ENUM":         0.2,
    "VULN_SCAN":    0.15,
    "EXPLOIT":      0.15,
    "POST_EXPLOIT": 0.15,
    "REPORT":       0.30,
}
DEFAULT_TEMPERATURE = 0.2


# ── Valid tools ───────────────────────────────────────────────────────────────

VALID_TOOLS = [
    "nmap_scan", "gobuster_dir", "searchsploit", "enum4linux",
    "curl_probe", "whatweb", "msf_search", "msf_exploit",
    "shell_cmd", "linpeas",
    "add_finding", "add_note", "advance_phase", "REPORT", "HALT",
]


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are Kira, an autonomous penetration testing AI agent.
Reply with ONLY a raw JSON object — no markdown, no prose, no code fences.

Required keys:
  "tool"      : tool name (string)
  "args"      : arguments (object, can be {})
  "reasoning" : one sentence (string)

RULES:
1. Never repeat a tool+args combo already in recent actions.
2. In RECON: if nmap returns 0 open ports, retry with flags="-Pn -sT -T4" (skips host discovery). Never HALT just because one scan found nothing.
3. In ENUM: run curl_probe → whatweb → gobuster_dir → searchsploit (short query e.g. "apache 2.4").
4. In EXPLOIT: call msf_search FIRST, then use a module name from its results.
5. Never invent Metasploit module names. Only use names returned by msf_search.
6. Always include the correct port in URLs (e.g. http://IP:8080/ not http://IP/).
7. If stuck with no valid action, emit HALT.

TOOLS:
  nmap_scan     : {"target":"IP","flags":"-sV -sC"} — omit "ports" to scan all 65535 ports
  gobuster_dir  : {"url":"http://IP:PORT/","wordlist":"/usr/share/wordlists/dirb/common.txt"}
  searchsploit  : {"query":"apache 2.4"}
  enum4linux    : {"target":"IP"}
  curl_probe    : {"url":"http://IP:PORT/","flags":"-sIL --max-time 10"}
  whatweb       : {"url":"http://IP:PORT/"}
  msf_search    : {"query":"apache"}
  msf_exploit   : {"module":"exploit/path/from/msf_search","options":{"RHOSTS":"IP","RPORT":8080}}
  shell_cmd     : {"cmd":"whoami","session_id":1}
  linpeas       : {"session_id":1}
  add_finding   : {"title":"...","severity":"critical|high|medium|low|info","port":8080,"cvss":7.5,"description":"...","remediation":"..."}
  add_note      : {"note":"..."}
  advance_phase : {}
  REPORT        : {}
  HALT          : {}

EXAMPLE:
{"tool":"nmap_scan","args":{"target":"10.10.10.5","flags":"-sV -sC"},"reasoning":"Start recon with a full port sweep."}
""").strip()


CORRECTION_PROMPT = textwrap.dedent("""
Your previous response was not valid JSON. Parse error: {error}

Reply with ONLY a raw JSON object with exactly these keys:
  "tool"      : string
  "args"      : object
  "reasoning" : string

No markdown. No prose. No code fences. Just the JSON object.
""").strip()


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Gemini API client for Kira.

    Parameters
    ----------
    api_key  : Gemini API key (default: GEMINI_API_KEY env var)
    model    : model name     (default: GEMINI_MODEL env var or gemini-2.5-flash)
    base_url : API base URL   (default: GEMINI_BASE_URL env var or generativelanguage.googleapis.com)
    timeout  : HTTP timeout in seconds
    verbose  : print call latency and token counts

    Swapping backends:
        Replace _call() and ping() — everything else is provider-agnostic.
    """

    def __init__(
        self,
        api_key:  str  = None,
        model:    str  = None,
        base_url: str  = None,
        timeout:  int  = DEFAULT_TIMEOUT,
        verbose:  bool = True,
        # Ollama kwargs
        host:     str  = None,
        # Legacy ignored kwargs
        provider: str  = None,
        project:  str  = None,
        location: str  = None,
    ):
        self.timeout  = timeout
        self.verbose  = verbose
        self._call_log: list[dict] = []

        # ── Auto-select backend ───────────────────────────────────────────
        # Use Gemini if GEMINI_API_KEY is provided, else fall back to Ollama
        resolved_key  = (api_key  or os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)).strip()
        ollama_model  = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
        ollama_host   = (host or os.getenv("OLLAMA_HOST", OLLAMA_HOST)).strip().rstrip("/")

        if resolved_key:
            self.provider = "gemini"
            self.api_key  = resolved_key
            self.model    = (model or os.getenv("GEMINI_MODEL", GEMINI_MODEL)).strip()
            self.base_url = (base_url or os.getenv("GEMINI_BASE_URL", GEMINI_BASE_URL)).rstrip("/")
            self.host     = None
        else:
            self.provider = "ollama"
            self.api_key  = None
            self.model    = model or ollama_model
            self.host     = ollama_host
            self.base_url = None

        if self.verbose:
            if self.provider == "gemini":
                print(f"[LLM] Gemini | model={self.model} | endpoint={self._endpoint()}")
            else:
                print(f"[LLM] Ollama  | host={self.host} | model={self.model}")

    # ── URL builder ───────────────────────────────────────────────────────────

    def _endpoint(self) -> str:
        """Full Gemini generateContent URL. Only valid for Gemini provider."""
        if self.provider != "gemini":
            return f"{self.host}/api/generate"
        return (
            f"{self.base_url}/v1beta/models/{self.model}"
            f":generateContent?key={self.api_key}"
        )

    # ── Public: structured action (JSON mode) ─────────────────────────────────

    def ask(
        self,
        user:        str,
        system:      str   = SYSTEM_PROMPT,
        temperature: float = 0.2,
    ) -> dict:
        """
        Send a prompt, return a validated action dict.
        Retries up to MAX_RETRIES on bad JSON.
        Returns a safe HALT dict after all retries are exhausted.
        """
        messages = [{"role": "user", "content": user}]

        for attempt in range(1, MAX_RETRIES + 1):
            raw, meta = self._call(system=system, messages=messages, temperature=temperature)

            if raw is None:
                return self._halt(f"LLM call failed: {meta.get('error')}", meta)

            parsed, parse_error = self._parse_json(raw)
            if parsed is not None:
                validated, val_error = self._validate_action(parsed)
                if validated is not None:
                    validated["_meta"] = meta
                    self._record(attempt, meta, ok=True)
                    if self.verbose:
                        self._print_ok(validated, meta)
                    return validated
                parse_error = val_error

            if self.verbose:
                self._print_retry(attempt, parse_error)

            # Feed the bad response back so the model can self-correct
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": CORRECTION_PROMPT.format(error=parse_error),
            })
            time.sleep(1)

        return self._halt(f"All {MAX_RETRIES} JSON parse attempts failed.", {})

    def next_action(self, context_summary: str, phase: str = "") -> dict:
        """Build user prompt from context + phase, return action dict."""
        phase_hint  = f"\nCurrent phase: {phase}" if phase else ""
        user_msg    = f"{context_summary}{phase_hint}\n\nWhat is your next action?"
        temperature = PHASE_TEMPERATURE.get(phase, DEFAULT_TEMPERATURE)
        return self.ask(user=user_msg, temperature=temperature)

    # ── Public: free-text generation (reporter mode) ──────────────────────────

    def generate_text(
        self,
        prompt:      str,
        temperature: float = 0.3,
        max_tokens:  int   = 500,
    ) -> str:
        """Free-text generation for ReportGenerator — no JSON enforcement."""
        if self.provider == "ollama":
            payload = {"model": self.model, "prompt": prompt, "stream": False,
                       "options": {"temperature": temperature, "num_predict": max_tokens}}
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
                    resp.raise_for_status()
                    return resp.json().get("response", "").strip()
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        time.sleep(INITIAL_BACKOFF)
                        continue
                    return f"(generate_text failed: {str(e)[:80]})"
            return "(Max retries exceeded)"

        # Gemini path
        payload = self._build_payload(
            system=None,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._post(payload)
                return self._extract_text(resp.json()).strip()
            except _RateLimitError as e:
                wait = INITIAL_BACKOFF ** attempt
                print(f"[LLM] ⚠ Rate limited (429). Waiting {wait}s (retry {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(INITIAL_BACKOFF)
                    continue
                return f"(generate_text failed: {str(e)[:80]})"
        return "(Max retries exceeded)"

    # ── Public: ping ──────────────────────────────────────────────────────────

    def ping(self) -> tuple[bool, str]:
        """Verify LLM is reachable. Returns (True, model) or (False, error)."""
        if self.provider == "ollama":
            return self._ping_ollama()
        return self._ping_gemini()

    def _ping_ollama(self) -> tuple[bool, str]:
        payload = {"model": self.model, "prompt": "Reply with the single word: ok",
                   "stream": False, "options": {"temperature": 0.1, "num_predict": 5}}
        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=15)
            if self.verbose:
                print(f"[LLM] Ping → HTTP {resp.status_code}")
            if resp.status_code == 404:
                return False, f"Model '{self.model}' not found. Run: ollama pull {self.model}"
            resp.raise_for_status()
            text = resp.json().get("response", "")
            if self.verbose:
                print(f"[LLM] Ping response: {text!r}")
            return True, self.model
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to Ollama at {self.host}. Run: ollama serve"
        except Exception as e:
            return False, str(e)

    def _ping_gemini(self) -> tuple[bool, str]:
        payload = self._build_payload(
            system=None,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0.1,
            max_tokens=10,
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self._endpoint(),
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=15,
                )

                if self.verbose:
                    print(f"[LLM] Ping → HTTP {resp.status_code}")

                if resp.status_code == 429:
                    wait = INITIAL_BACKOFF ** attempt
                    print(f"[LLM] ⚠ Rate limited (429). Waiting {wait}s (retry {attempt}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue

                if resp.status_code == 400:
                    body = resp.json()
                    msg  = body.get("error", {}).get("message", "Bad request")
                    return False, f"400 Bad Request — {msg}"

                if resp.status_code == 401:
                    return False, "401 Unauthorized — check GEMINI_API_KEY"

                if resp.status_code == 403:
                    body = resp.json()
                    msg  = body.get("error", {}).get("message", "Forbidden")
                    return False, f"403 Forbidden — {msg}"

                if resp.status_code == 404:
                    return False, (
                        f"404 Not Found — model '{self.model}' not found. "
                        f"Check model name at https://ai.google.dev/models"
                    )

                resp.raise_for_status()

                data = resp.json()
                text = self._extract_text(data)

                if self.verbose:
                    print(f"[LLM] Ping raw response: {text!r}")

                return True, self.model

            except requests.exceptions.ConnectionError:
                return False, (
                    f"Cannot connect to {self.base_url}. "
                    f"Check network connectivity."
                )
            except requests.exceptions.HTTPError as e:
                return False, f"HTTP {e.response.status_code}: {e}"
            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = INITIAL_BACKOFF ** attempt
                    time.sleep(wait)
                    continue
                return False, str(e)

        return False, "Max ping retries exceeded"

    # ── Internal: routing ─────────────────────────────────────────────────────

    def _call(self, system: str, messages: list, temperature: float) -> tuple:
        if self.provider == "ollama":
            return self._call_ollama(system, messages, temperature)
        return self._call_gemini(system, messages, temperature)

    # ── Internal: Gemini API call ─────────────────────────────────────────────

    def _call(self, system: str, messages: list, temperature: float) -> tuple:
        """
        POST to Gemini generateContent endpoint.
        Retries with exponential backoff on 429 rate limit errors.
        Returns (raw_text, meta) — raw_text is None on failure.
        """
        payload = self._build_payload(system, messages, temperature)
        start   = time.monotonic()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._post(payload)
                data     = resp.json()
                raw_text = self._extract_text(data)
                usage    = data.get("usageMetadata", {})
                meta = {
                    "latency_s":     round(time.monotonic() - start, 2),
                    "output_tokens": usage.get("candidatesTokenCount", 0),
                    "model":         self.model,
                    "provider":      "gemini",
                    "attempts":      attempt,
                }
                return raw_text, meta

            except _RateLimitError:
                wait = INITIAL_BACKOFF ** attempt
                print(f"[LLM] ⚠ Rate limited (429). Waiting {wait}s (retry {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = INITIAL_BACKOFF ** attempt
                    if self.verbose:
                        print(f"[LLM] Error: {str(e)[:80]}. Retry {attempt}/{MAX_RETRIES} in {wait}s...")
                    time.sleep(wait)
                    continue
                meta = {
                    "error":     str(e),
                    "latency_s": round(time.monotonic() - start, 2),
                    "provider":  "gemini",
                    "attempts":  attempt,
                }
                return None, meta

        meta = {
            "error":    "Max retries exceeded (rate limited)",
            "provider": "gemini",
            "attempts": MAX_RETRIES,
        }
        return None, meta

    # ── Internal: HTTP helper ─────────────────────────────────────────────────

    def _post(self, payload: dict) -> requests.Response:
        """
        POST payload to the Gemini endpoint.
        Raises _RateLimitError on 429, HTTPError on other 4xx/5xx.
        """
        resp = requests.post(
            self._endpoint(),
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code == 429:
            raise _RateLimitError()
        resp.raise_for_status()
        return resp

    # ── Internal: Ollama call ─────────────────────────────────────────────────

    def _call_ollama(self, system: str, messages: list, temperature: float) -> tuple:
        """POST to Ollama /api/generate. Returns (raw_text, meta)."""
        parts = []
        if system:
            parts.append(f"### SYSTEM\n{system}")
        for msg in messages:
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            parts.append(f"### {role}\n{msg['content']}")
        parts.append("### ASSISTANT")
        prompt = "\n\n".join(parts)

        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": temperature, "num_predict": 1024},
        }
        start = time.monotonic()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                meta = {
                    "latency_s":     round(time.monotonic() - start, 2),
                    "output_tokens": data.get("eval_count", 0),
                    "model":         self.model,
                    "provider":      "ollama",
                    "attempts":      attempt,
                }
                return data.get("response", ""), meta
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(INITIAL_BACKOFF)
                    continue
                meta = {"error": str(e), "latency_s": round(time.monotonic() - start, 2), "provider": "ollama", "attempts": attempt}
                return None, meta
        return None, {"error": "Max retries", "provider": "ollama", "attempts": MAX_RETRIES}

    # ── Internal: payload builder ─────────────────────────────────────────────

    def _build_payload(
        self,
        system:      Optional[str],
        messages:    list,
        temperature: float = 0.2,
        max_tokens:  int   = 1024,
    ) -> dict:
        """
        Build Gemini generateContent request body.

        Format:
            {"contents": [{"parts": [{"text": "..."}]}]}

        System prompt is prepended as the first user turn followed by a
        model acknowledgement (Gemini doesn't have a separate system field
        in v1beta).
        """
        contents = []

        if system:
            contents.append({"role": "user",  "parts": [{"text": system}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})

        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        return {
            "contents": contents,
            "generationConfig": {
                "temperature":     temperature,
                "maxOutputTokens": max_tokens,
            },
        }

    # ── Internal: response parser ─────────────────────────────────────────────

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Extract text from Gemini generateContent response."""
        try:
            return (
                data["candidates"][0]["content"]["parts"][0]["text"]
            )
        except (KeyError, IndexError, TypeError):
            return ""

    # ── Internal: JSON parsing + validation ───────────────────────────────────

    def _parse_json(self, raw: str) -> tuple[Optional[dict], Optional[str]]:
        text = raw.strip()
        # Strip markdown code fences if the model wraps output
        if text.startswith("```"):
            lines = text.splitlines()
            text  = "\n".join(lines[1:-1]).strip()
        # Some models prefix with "json\n{...}"
        if text.lower().startswith("json"):
            text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            return None, f"{e} — raw: {text[:120]!r}"
        # Unwrap single-key wrapper objects e.g. {"action": {...}}
        if isinstance(parsed, dict) and len(parsed) == 1:
            inner = next(iter(parsed.values()))
            if isinstance(inner, dict):
                parsed = inner
        return parsed, None

    def _validate_action(self, obj: dict) -> tuple[Optional[dict], Optional[str]]:
        missing = [k for k in ("tool", "args", "reasoning") if k not in obj]
        if missing:
            return None, f"Missing required keys: {missing}"
        tool = obj["tool"]
        if tool not in VALID_TOOLS:
            close = [t for t in VALID_TOOLS if tool.lower() in t.lower()]
            hint  = f" Did you mean: {close}?" if close else ""
            return None, f"Unknown tool '{tool}'.{hint} Valid: {VALID_TOOLS}"
        if not isinstance(obj["args"], dict):
            return None, f"'args' must be a JSON object, got {type(obj['args']).__name__}"
        if not isinstance(obj["reasoning"], str):
            return None, "'reasoning' must be a string"
        return {
            "tool":      tool,
            "args":      obj["args"],
            "reasoning": obj["reasoning"].strip(),
        }, None

    # ── Internal: helpers ─────────────────────────────────────────────────────

    def _halt(self, reason: str, meta: dict = None) -> dict:
        return {"tool": "HALT", "args": {}, "reasoning": reason, "_meta": meta or {}}

    def _record(self, attempts: int, meta: dict, ok: bool) -> None:
        self._call_log.append({
            "timestamp": _ts(),
            "attempts":  attempts,
            "ok":        ok,
            "latency_s": meta.get("latency_s"),
            "tokens":    meta.get("output_tokens"),
            "provider":  "gemini",
        })

    def _print_ok(self, action: dict, meta: dict) -> None:
        try:
            from rich.console import Console
            Console().print(
                f"[dim][LLM/gemini][/dim] "
                f"[green]{action['tool']}[/green] "
                f"[dim]({meta.get('latency_s','?')}s, "
                f"{meta.get('output_tokens','?')} tokens)[/dim]"
            )
        except ImportError:
            print(f"[LLM/gemini] {action['tool']} ({meta.get('latency_s')}s)")

    def _print_retry(self, attempt: int, error: str) -> None:
        try:
            from rich.console import Console
            Console().print(
                f"[dim][LLM][/dim] [yellow]Attempt {attempt} — bad JSON: {error[:80]}[/yellow]"
            )
        except ImportError:
            print(f"[LLM] Attempt {attempt} failed: {error[:80]}")


# ── Internal exceptions ───────────────────────────────────────────────────────

class _RateLimitError(Exception):
    """Raised internally when Gemini returns HTTP 429."""
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== llm.py smoke test (Gemini API) ===\n")

    # [1] Missing API key raises
    print("[1] Missing API key raises ValueError")
    _saved = os.environ.pop("GEMINI_API_KEY", None)
    try:
        LLMClient(verbose=False)
        print("    ERROR: should have raised")
        sys.exit(1)
    except ValueError as e:
        print(f"    OK — raised: {e}")
    finally:
        if _saved:
            os.environ["GEMINI_API_KEY"] = _saved

    # [2] Endpoint URL format
    print("\n[2] Endpoint URL format")
    c = LLMClient(api_key="test-key", verbose=False)
    url = c._endpoint()
    assert "generativelanguage.googleapis.com" in url, f"Wrong base: {url}"
    assert "v1beta/models/gemini-2.5-flash:generateContent" in url, f"Wrong path: {url}"
    assert "key=test-key" in url, f"Missing key param: {url}"
    print(f"    OK — {url}")

    # [3] Custom base URL
    print("\n[3] Custom base URL")
    c2 = LLMClient(api_key="k", base_url="https://custom.example.com", verbose=False)
    assert c2._endpoint().startswith("https://custom.example.com")
    print(f"    OK — {c2._endpoint()}")

    # [4] Payload format
    print("\n[4] Payload contents format")
    payload = c._build_payload(
        system="You are a tester.",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
    )
    assert payload["contents"][0] == {"role": "user",  "parts": [{"text": "You are a tester."}]}
    assert payload["contents"][1] == {"role": "model", "parts": [{"text": "Understood."}]}
    assert payload["contents"][2] == {"role": "user",  "parts": [{"text": "hello"}]}
    assert "generationConfig" in payload
    print(f"    OK — {len(payload['contents'])} content entries")

    # [5] Response text extraction
    print("\n[5] _extract_text")
    fake_resp = {"candidates": [{"content": {"parts": [{"text": "nmap_scan"}]}}]}
    assert LLMClient._extract_text(fake_resp) == "nmap_scan"
    assert LLMClient._extract_text({}) == ""
    print("    OK")

    # [6] JSON parse — clean
    print("\n[6] _parse_json — clean JSON")
    parsed, err = c._parse_json(
        '{"tool":"nmap_scan","args":{"target":"10.0.0.1"},"reasoning":"Start."}'
    )
    assert parsed and not err
    print(f"    OK — {parsed}")

    # [7] JSON parse — strips markdown fences
    print("\n[7] _parse_json — strips markdown fences")
    parsed2, err2 = c._parse_json(
        '```json\n{"tool":"HALT","args":{},"reasoning":"done"}\n```'
    )
    assert parsed2 and not err2 and parsed2["tool"] == "HALT"
    print(f"    OK — {parsed2}")

    # [8] Validate action
    print("\n[8] _validate_action")
    validated, verr = c._validate_action(
        {"tool": "searchsploit", "args": {"query": "apache"}, "reasoning": "check vulns"}
    )
    assert validated and not verr
    print(f"    OK — tool={validated['tool']}")

    # [9] Unknown tool rejected
    print("\n[9] Unknown tool rejected")
    _, verr2 = c._validate_action({"tool": "fake_tool", "args": {}, "reasoning": "test"})
    assert verr2 and "Unknown tool" in verr2
    print(f"    OK — {verr2[:60]}")

    # [10] msf_search in VALID_TOOLS
    print("\n[10] msf_search in VALID_TOOLS")
    assert "msf_search" in VALID_TOOLS
    print("    OK")

    # [11] Live ping (only if GEMINI_API_KEY is set)
    print("\n[11] Live Gemini API ping")
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        live = LLMClient(api_key=api_key, verbose=True)
        ok, msg = live.ping()
        if ok:
            print(f"    PASS — model: {msg}")
        else:
            print(f"    FAIL — {msg}")
    else:
        print("    Skipped (set GEMINI_API_KEY to test live)")

    print("\nAll offline tests passed.")
