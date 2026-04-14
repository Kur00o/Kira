"""
kira/llm.py — LLM Interface (Gemini via Vertex AI)
===================================================
All communication with the LLM flows through Google Gemini via Vertex AI.

Usage:
    llm = LLMClient()    # Pulls GOOGLE_API_KEY, GEMINI_PROJECT, GEMINI_LOCATION from .env

    # Planner action (JSON mode):
    action = llm.next_action(context_summary, phase="ENUM")
    # action["tool"], action["args"], action["reasoning"]

    # Reporter text (free-text mode):
    text = llm.generate_text("Write an exec summary for...", temperature=0.3)
"""

import json
import os
import time
import textwrap
from datetime import datetime, timezone
from typing import Optional

import requests


# ── Gemini configuration ──────────────────────────────────────────────────────

GEMINI_API_KEY  = ""                 # Loaded from GOOGLE_API_KEY env var
GEMINI_MODEL    = "gemini-2.5-flash"
GEMINI_HOST     = "https://us-central1-aiplatform.googleapis.com"  # Vertex AI endpoint
GEMINI_PROJECT  = ""                 # GCP project ID — set via env var GEMINI_PROJECT
GEMINI_LOCATION = "us-central1"      # GCP region

DEFAULT_TIMEOUT = 120
MAX_RETRIES     = 5  # Increased from 3 to handle rate limits
RETRY_DELAY     = 1.5
RATE_LIMIT_BACKOFF = True  # Enable exponential backoff for 429 errors
INITIAL_BACKOFF = 1.0  # Start with 1 second backoff

# Per-phase temperature: lower for deterministic exploitation decisions.
PHASE_TEMPERATURE = {
    "RECON":        0.2,
    "ENUM":         0.2,
    "VULN_SCAN":    0.15,
    "EXPLOIT":      0.15,
    "POST_EXPLOIT": 0.15,
    "REPORT":       0.30,
}
DEFAULT_TEMPERATURE = 0.2


# ── Valid tools (unchanged from Day 3) ────────────────────────────────────────

VALID_TOOLS = [
    "nmap_scan", "gobuster_dir", "searchsploit", "enum4linux",
    "curl_probe", "whatweb", "msf_exploit", "shell_cmd", "linpeas",
    "add_finding", "add_note", "advance_phase", "REPORT", "HALT",
]


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are Kira, an autonomous penetration testing AI agent.
Reply with ONLY a raw JSON object — no markdown, no prose, no code fences.

Required keys:
  "tool"      : tool name (string)
  "args"      : arguments (object, can be {})
  "reasoning" : one sentence (string)

RULES:
1. Never repeat a tool+args combo already in recent actions.
2. In ENUM: run curl_probe → whatweb → gobuster_dir → searchsploit (use short query e.g. "apache 2.4").
3. In EXPLOIT: call msf_search FIRST (e.g. {"query":"apache"}), then use a module name from its results.
4. Never invent Metasploit module names. Only use names returned by msf_search.
5. Always use the correct port in URLs (e.g. http://IP:8080/ not http://IP/).
6. If stuck with no valid action, emit HALT.

TOOLS:
  nmap_scan     : {"target":"IP","flags":"-sV -sC"} — omit "ports" to scan all 65535 ports automatically
  gobuster_dir  : {"url":"http://IP:PORT/","wordlist":"/usr/share/wordlists/dirb/common.txt"}
  searchsploit  : {"query":"apache 2.4"}
  enum4linux    : {"target":"IP"}
  curl_probe    : {"url":"http://IP:PORT/","flags":"-sI"}
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
{"tool":"msf_search","args":{"query":"apache"},"reasoning":"Find real Metasploit module names for Apache before exploiting."}
""").strip()


CORRECTION_PROMPT = textwrap.dedent("""
Your previous response was not valid JSON. Parse error: {error}

You MUST reply with ONLY a raw JSON object using exactly these keys:
  "tool"      : string
  "args"      : object
  "reasoning" : string

No markdown. No prose. No code fences. Just the JSON object.
""").strip()


# ── LLMClient ──────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Google Gemini LLM client for Kira autonomous penetration testing.

    Parameters
    ----------
    model   : override model tag (default: GEMINI_MODEL)
    api_key : API key; falls back to GOOGLE_API_KEY env var
    timeout : HTTP timeout in seconds
    verbose : print each call's latency and token count
    """

    def __init__(
        self,
        model:   str  = None,
        api_key: str  = None,
        timeout: int  = DEFAULT_TIMEOUT,
        verbose: bool = True,
    ):
        self.provider = "gemini"
        self.host    = GEMINI_HOST
        self.model   = model or GEMINI_MODEL
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", GEMINI_API_KEY)
        self.project = os.getenv("GEMINI_PROJECT", GEMINI_PROJECT)
        self.location = os.getenv("GEMINI_LOCATION", GEMINI_LOCATION)
        self.timeout = timeout
        self.verbose = verbose
        self._call_log: list[dict] = []

        if not self.api_key:
            raise ValueError(
                "Gemini requires an API key. "
                "Pass api_key= or set GOOGLE_API_KEY environment variable."
            )

    # ── Public: structured action (JSON mode) ─────────────────────────────────

    def ask(
        self,
        user:        str,
        system:      str   = SYSTEM_PROMPT,
        temperature: float = 0.2,
    ) -> dict:
        """
        Send a prompt and always return a parsed action dict.
        Retries up to MAX_RETRIES times on bad JSON.
        Returns a safe HALT dict after all retries are exhausted.
        """
        messages   = [{"role": "user", "content": user}]
        last_error = None

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

            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": CORRECTION_PROMPT.format(error=parse_error),
            })
            time.sleep(RETRY_DELAY)

        return self._halt(f"All {MAX_RETRIES} JSON parse attempts failed.", {})

    def next_action(self, context_summary: str, phase: str = "") -> dict:
        """
        Convenience wrapper — builds the user prompt from context + phase,
        calls ask(), returns the action dict.
        """
        phase_hint = f"\nCurrent phase: {phase}" if phase else ""
        user_msg   = f"{context_summary}{phase_hint}\n\nWhat is your next action?"
        temperature = PHASE_TEMPERATURE.get(phase, DEFAULT_TEMPERATURE)
        return self.ask(user=user_msg, temperature=temperature)

    # ── Public: free-text generation (reporter mode) ──────────────────────────

    def generate_text(
        self,
        prompt:      str,
        temperature: float = 0.3,
        max_tokens:  int   = 500,
    ) -> str:
        """
        Free-text generation — does NOT enforce JSON output.
        Used by ReportGenerator for exec summary and finding writeups.
        Includes exponential backoff for rate limits (429).

        Parameters
        ----------
        prompt      : user prompt
        temperature : 0.0–1.0
        max_tokens  : approximate token cap

        Returns
        -------
        str : raw model response, stripped of leading/trailing whitespace
        """
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature":  temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Vertex AI endpoint with Bearer token auth
                headers = {"Authorization": f"Bearer {self.api_key}"}
                url = f"{self.host}/v1/projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}:generateContent"
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                
                if resp.status_code == 429:
                    # Rate limited
                    if attempt < MAX_RETRIES:
                        wait_time = backoff * (2 ** (attempt - 1))
                        if self.verbose:
                            print(f"[LLM] Rate limited (429). Waiting {wait_time:.1f}s for text generation...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "(Rate limited - try again later)"
                
                resp.raise_for_status()
                data = resp.json()
                # Extract text from Gemini response
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return text.strip()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited
                    if attempt < MAX_RETRIES:
                        wait_time = backoff * (2 ** (attempt - 1))
                        if self.verbose:
                            print(f"[LLM] Rate limited (429). Waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "(Rate limited - try again later)"
                return f"(Generation error: {e})"
            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait_time = backoff * (2 ** (attempt - 1))
                    if self.verbose:
                        print(f"[LLM] Error in text generation: {str(e)[:60]}. Retry {attempt}/{MAX_RETRIES}...")
                    time.sleep(wait_time)
                    continue
                return f"(Generation failed: {str(e)[:50]})"
        
        return "(Max retries exceeded)"

    # ── Ping ──────────────────────────────────────────────────────────────────

    def ping(self) -> tuple[bool, str]:
        """
        Quick connectivity check for Gemini API.
        Makes a minimal generateContent request to verify API key and model work.
        Includes exponential backoff for rate limits (429).
        Returns (True, model_name) or (False, error_message).
        """
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "ok"}],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 10,
            },
        }
        
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Vertex AI endpoint with Bearer token auth
                headers = {"Authorization": f"Bearer {self.api_key}"}
                url = f"{self.host}/v1/projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}:generateContent"
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=10,
                )
                
                if resp.status_code == 429:
                    # Rate limited on ping, retry with exponential backoff
                    if attempt < MAX_RETRIES:
                        wait_time = backoff * (2 ** (attempt - 1))
                        if self.verbose:
                            print(f"[LLM] Ping rate limited (429). Waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return False, "API rate limit exceeded (429). Try again later."
                
                if resp.status_code == 404:
                    # Try listing available models to give better error
                    try:
                        list_resp = requests.get(
                            f"{self.host}/v1beta/models",
                            params={"key": self.api_key},
                            timeout=10,
                        )
                        if list_resp.status_code == 200:
                            models = list_resp.json().get("models", [])
                            available = [m.get("displayName", m.get("name", "?")) for m in models[:3]]
                            return False, f"Model '{self.model}' not found. Available: {available}. Check https://aistudio.google.com/apikey"
                    except:
                        pass
                    return False, f"Model '{self.model}' not found (404). Check API key and model name at https://aistudio.google.com/apikey"
                
                resp.raise_for_status()
                return True, self.model
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    if attempt < MAX_RETRIES:
                        wait_time = backoff * (2 ** (attempt - 1))
                        if self.verbose:
                            print(f"[LLM] Ping rate limited (429). Waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return False, "API rate limit exceeded (429). Try again later."
                error_msg = str(e)
                if "401" in error_msg or "Unauthorized" in error_msg:
                    return False, "API key invalid or revoked. Check https://aistudio.google.com/apikey"
                return False, error_msg
            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait_time = backoff * (2 ** (attempt - 1))
                    if self.verbose:
                        print(f"[LLM] Ping error: {str(e)[:60]}. Retry {attempt}/{MAX_RETRIES}...")
                    time.sleep(wait_time)
                    continue
                return False, str(e)
        
        return False, "Max ping retries exceeded"

    # ── Internal: routing ─────────────────────────────────────────────────────

    def _call_with_backoff(self, func, *args, **kwargs):
        """
        Execute func with exponential backoff retry on 429 (rate limit) errors.
        Returns (success, result) tuple.
        """
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = func(*args, **kwargs)
                if isinstance(result, requests.Response):
                    if result.status_code == 429:
                        # Rate limited, apply exponential backoff
                        if attempt < MAX_RETRIES:
                            wait_time = backoff * (2 ** (attempt - 1))
                            if self.verbose:
                                print(f"[LLM] Rate limited (429). Waiting {wait_time:.1f}s before retry {attempt}/{MAX_RETRIES}...")
                            time.sleep(wait_time)
                            continue
                        else:
                            return False, result
                    else:
                        result.raise_for_status()
                return True, result
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited, apply exponential backoff
                    if attempt < MAX_RETRIES:
                        wait_time = backoff * (2 ** (attempt - 1))
                        if self.verbose:
                            print(f"[LLM] Rate limited (429). Waiting {wait_time:.1f}s before retry {attempt}/{MAX_RETRIES}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return False, e
                return False, e
            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait_time = backoff * (2 ** (attempt - 1))
                    if self.verbose:
                        print(f"[LLM] Error: {str(e)[:80]}. Retry {attempt}/{MAX_RETRIES} in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                return False, e
        return False, Exception("Max retries exceeded")

    def _call(self, system: str, messages: list, temperature: float) -> tuple:
        """Call Gemini API with structured prompt."""
        return self._call_gemini(system, messages, temperature)

    def _call_gemini(self, system: str, messages: list, temperature: float) -> tuple:
        """
        Call Gemini API with exponential backoff on rate limits.
        Returns (raw_text, meta) tuple.
        """
        start = time.monotonic()
        
        # Convert messages to Gemini format and prepend system message
        contents = []
        
        # Add system message as the first user message
        if system:
            contents.append({
                "role": "user",
                "parts": [{"text": system}],
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}],
            })
        
        # Add conversation messages
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}],
            })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 1024,
            },
        }
        
        # Retry loop with exponential backoff
        backoff = INITIAL_BACKOFF
        last_error = None
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Vertex AI endpoint with Bearer token auth
                headers = {"Authorization": f"Bearer {self.api_key}"}
                url = f"{self.host}/v1/projects/{self.project}/locations/{self.location}/publishers/google/models/{self.model}:generateContent"
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                
                if resp.status_code == 429:
                    # Rate limited
                    if attempt < MAX_RETRIES:
                        wait_time = backoff * (2 ** (attempt - 1))
                        if self.verbose:
                            print(f"[LLM] Rate limited (429). Waiting {wait_time:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        last_error = "Rate limit exceeded after max retries"
                        break
                
                resp.raise_for_status()
                data = resp.json()
                raw_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                usage = data.get("usageMetadata", {})
                meta = {
                    "latency_s":     round(time.monotonic() - start, 2),
                    "output_tokens": usage.get("outputTokenCount", 0),
                    "model":         self.model,
                    "provider":      "gemini",
                    "attempts":      attempt,
                }
                return raw_text, meta
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited
                    if attempt < MAX_RETRIES:
                        wait_time = backoff * (2 ** (attempt - 1))
                        if self.verbose:
                            print(f"[LLM] Rate limited (429). Waiting {wait_time:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        last_error = str(e)
                        break
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    wait_time = backoff * (2 ** (attempt - 1))
                    if self.verbose:
                        print(f"[LLM] HTTP Error: {e}. Retry {attempt}/{MAX_RETRIES} in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                break
                
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    wait_time = backoff * (2 ** (attempt - 1))
                    if self.verbose:
                        print(f"[LLM] Error: {str(e)[:80]}. Retry {attempt}/{MAX_RETRIES} in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                break
        
        meta = {
            "error":     last_error or "Unknown error",
            "latency_s": round(time.monotonic() - start, 2),
            "provider":  "gemini",
            "attempts":  MAX_RETRIES,
        }
        return None, meta

    # ── Internal: parsing + validation ────────────────────────────────────────

    def _parse_json(self, raw: str) -> tuple[Optional[dict], Optional[str]]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text  = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            return None, f"{e} — raw: {text[:120]!r}"
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

    # ── Internal: helpers ──────────────────────────────────────────────────────

    def _halt(self, reason: str, meta: dict = None) -> dict:
        return {"tool": "HALT", "args": {}, "reasoning": reason, "_meta": meta or {}}

    def _record(self, attempts: int, meta: dict, ok: bool) -> None:
        self._call_log.append({
            "timestamp": _ts(),
            "attempts":  attempts,
            "ok":        ok,
            "latency_s": meta.get("latency_s"),
            "tokens":    meta.get("output_tokens"),
            "provider":  meta.get("provider", self.provider),
        })

    def _print_ok(self, action: dict, meta: dict) -> None:
        try:
            from rich.console import Console
            Console().print(
                f"[dim][LLM/{self.provider}][/dim] "
                f"[green]{action['tool']}[/green] "
                f"[dim]({meta.get('latency_s','?')}s, "
                f"{meta.get('output_tokens','?')} tokens)[/dim]"
            )
            Console().print(f"  [dim italic]{action['reasoning']}[/dim italic]")
        except ImportError:
            print(f"[LLM/{self.provider}] {action['tool']} ({meta.get('latency_s')}s) — {action['reasoning']}")

    def _print_retry(self, attempt: int, error: str) -> None:
        try:
            from rich.console import Console
            Console().print(
                f"[dim][LLM][/dim] [yellow]Attempt {attempt} — bad JSON: {error[:80]}[/yellow]"
            )
        except ImportError:
            print(f"[LLM] Attempt {attempt} failed: {error[:80]}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== llm.py smoke test (Gemini-only) ===\n")

    # [1] Basic initialization — Gemini
    print("[1] LLMClient(api_key='test-key')")
    try:
        llm = LLMClient(api_key="test-key", verbose=False)
        assert llm.provider == "gemini"
        assert llm.model == GEMINI_MODEL
        print(f"    provider={llm.provider}  model={llm.model}  host={llm.host}  OK\n")
    except Exception as e:
        print(f"    ERROR: {e}\n")
        sys.exit(1)

    # [2] Gemini requires key
    print("[2] Gemini without key raises ValueError")
    os.environ.pop("GOOGLE_API_KEY", None)
    try:
        LLMClient(verbose=False)
        assert False, "Should raise"
    except ValueError as e:
        print(f"    Correctly raised: {e}\n")

    # [3] generate_text exists
    print("[3] generate_text() method exists")
    client = LLMClient(api_key="test-key", verbose=False)
    assert hasattr(client, "generate_text"), "generate_text missing"
    print(f"    generate_text present  OK\n")

    # [4] JSON parse + validation (provider-independent)
    print("[4] _parse_json + _validate_action")
    llm2 = LLMClient(api_key="test-key", verbose=False)
    parsed, err = llm2._parse_json(
        '{"tool": "nmap_scan", "args": {"target": "10.10.10.5"}, "reasoning": "Start."}'
    )
    assert parsed is not None and err is None
    validated, verr = llm2._validate_action(parsed)
    assert validated is not None and verr is None
    print(f"    parse+validate OK: tool={validated['tool']}\n")

    # [5] Live Gemini ping (only if API key is available)
    print("[5] Pinging Gemini API...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        llm3 = LLMClient(api_key=api_key, verbose=True)
        ok, msg = llm3.ping()
        if ok:
            print(f"    Reachable — model: {msg}")
        else:
            print(f"    Unreachable: {msg}")
    else:
        print(f"    Skipped (set GOOGLE_API_KEY to test live)")

    print("\nAll offline tests passed.")