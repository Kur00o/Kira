"""
kira/llm.py — LLM Interface  (Day 4 update)
=============================================
All communication with the LLM flows here.

Day 4 additions:
  - Multi-provider support: Ollama | Anthropic | OpenAI
  - generate_text() for free-text reporter prompts (non-JSON mode)
  - Switching providers is a one-line config change (PROVIDER = "...")

Usage:
    # Local Ollama (default)
    llm = LLMClient()

    # Anthropic Claude
    llm = LLMClient(provider="anthropic", api_key="sk-ant-...")
    # or: export ANTHROPIC_API_KEY=sk-ant-... then LLMClient(provider="anthropic")

    # OpenAI
    llm = LLMClient(provider="openai", api_key="sk-...")

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


# ── Provider configuration ────────────────────────────────────────────────────
# Switch LLM backend by changing PROVIDER.
# Supported: "ollama" | "anthropic" | "openai"

PROVIDER        = "ollama"           # ← change this to switch backends

# Ollama (local)
OLLAMA_HOST     = "http://localhost:11434"
OLLAMA_MODEL    = "gemma3:4b"

# Anthropic Claude (cloud)
ANTHROPIC_KEY   = ""                 # or: export ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# OpenAI (cloud)
OPENAI_KEY      = ""                 # or: export OPENAI_API_KEY=sk-...
OPENAI_MODEL    = "gpt-4o-mini"

DEFAULT_TIMEOUT = 120
MAX_RETRIES     = 3
RETRY_DELAY     = 1.5

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
    Unified LLM wrapper. Supports Ollama, Anthropic, and OpenAI.

    Parameters
    ----------
    provider : "ollama" | "anthropic" | "openai"  (default: PROVIDER constant)
    host     : Ollama server URL (Ollama only)
    model    : override model tag
    api_key  : API key (Anthropic / OpenAI); falls back to env var
    timeout  : HTTP timeout in seconds
    verbose  : print each call's latency and token count
    """

    def __init__(
        self,
        host:     str  = None,
        model:    str  = None,
        provider: str  = None,
        api_key:  str  = None,
        timeout:  int  = DEFAULT_TIMEOUT,
        verbose:  bool = True,
    ):
        self.provider = (provider or PROVIDER).lower()
        self.timeout  = timeout
        self.verbose  = verbose
        self._call_log: list[dict] = []

        if self.provider == "ollama":
            self.host    = (host or OLLAMA_HOST).rstrip("/")
            self.model   = model or OLLAMA_MODEL
            self.api_key = None

        elif self.provider == "anthropic":
            self.host    = "https://api.anthropic.com"
            self.model   = model or ANTHROPIC_MODEL
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_KEY)
            if not self.api_key:
                raise ValueError(
                    "Anthropic provider requires an API key. "
                    "Pass api_key= or set ANTHROPIC_API_KEY environment variable."
                )

        elif self.provider == "openai":
            self.host    = "https://api.openai.com"
            self.model   = model or OPENAI_MODEL
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", OPENAI_KEY)
            if not self.api_key:
                raise ValueError(
                    "OpenAI provider requires an API key. "
                    "Pass api_key= or set OPENAI_API_KEY environment variable."
                )

        else:
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                "Supported: ollama | anthropic | openai"
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

        Parameters
        ----------
        prompt      : user prompt
        temperature : 0.0–1.0
        max_tokens  : approximate token cap

        Returns
        -------
        str : raw model response, stripped of leading/trailing whitespace
        """
        if self.provider == "ollama":
            return self._generate_text_ollama(prompt, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._generate_text_anthropic(prompt, temperature, max_tokens)
        elif self.provider == "openai":
            return self._generate_text_openai(prompt, temperature, max_tokens)
        return ""

    def _generate_text_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model":    self.model,
            "stream":   False,
            "options":  {"temperature": temperature, "num_predict": max_tokens},
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
        except Exception:
            return ""

    def _generate_text_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> str:
        headers = {
            "x-api-key":         self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }
        payload = {
            "model":       self.model,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "messages":    [{"role": "user", "content": prompt}],
        }
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"].strip()
        except Exception:
            return ""

    def _generate_text_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       self.model,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "messages":    [{"role": "user", "content": prompt}],
        }
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    # ── Ping ──────────────────────────────────────────────────────────────────

    def ping(self) -> tuple[bool, str]:
        """
        Quick connectivity check.
        Returns (True, model_name) or (False, error_message).
        """
        if self.provider == "ollama":
            try:
                resp = requests.get(f"{self.host}/api/tags", timeout=5)
                resp.raise_for_status()
                models = resp.json().get("models", [])
                names  = [m.get("name", "") for m in models]
                if self.model not in names:
                    return False, (
                        f"Model '{self.model}' not pulled. "
                        f"Run: ollama pull {self.model}. "
                        f"Available: {names}"
                    )
                return True, self.model
            except requests.exceptions.ConnectionError:
                return False, (
                    f"Cannot connect to Ollama at {self.host}. "
                    "Is Ollama running? Run: ollama serve"
                )
            except Exception as e:
                return False, str(e)

        elif self.provider == "anthropic":
            try:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key":         self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type":      "application/json",
                    },
                    json={"model": self.model, "max_tokens": 5,
                          "messages": [{"role": "user", "content": "ping"}]},
                    timeout=10,
                )
                resp.raise_for_status()
                return True, self.model
            except Exception as e:
                return False, str(e)

        elif self.provider == "openai":
            try:
                resp = requests.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10,
                )
                resp.raise_for_status()
                return True, self.model
            except Exception as e:
                return False, str(e)

        return False, "Unknown provider"

    # ── Internal: routing ─────────────────────────────────────────────────────

    def _call(self, system: str, messages: list, temperature: float) -> tuple:
        """Route to the correct provider's structured call implementation."""
        if self.provider == "anthropic":
            return self._call_anthropic(system, messages, temperature)
        elif self.provider == "openai":
            return self._call_openai(system, messages, temperature)
        else:
            return self._call_ollama_native(system, messages, temperature)

    # Keep _call_ollama as alias so existing planner code that calls it still works
    def _call_ollama(self, system, messages, temperature):
        return self._call(system, messages, temperature)

    def _call_ollama_native(self, system: str, messages: list, temperature: float) -> tuple:
        start  = time.monotonic()
        payload = {
            "model":    self.model,
            "stream":   False,
            "options":  {"temperature": temperature},
            "messages": [{"role": "system", "content": system}] + messages,
        }
        meta = {}
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data     = resp.json()
            raw_text = data.get("message", {}).get("content", "")
            meta = {
                "latency_s":     round(time.monotonic() - start, 2),
                "output_tokens": data.get("eval_count", 0),
                "model":         data.get("model", self.model),
                "provider":      "ollama",
            }
            return raw_text, meta
        except Exception as e:
            meta = {
                "error":     str(e),
                "latency_s": round(time.monotonic() - start, 2),
                "provider":  "ollama",
            }
            return None, meta

    def _call_anthropic(self, system: str, messages: list, temperature: float) -> tuple:
        start = time.monotonic()
        headers = {
            "x-api-key":         self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }
        payload = {
            "model":       self.model,
            "max_tokens":  1024,
            "temperature": temperature,
            "system":      system,
            "messages":    messages,
        }
        meta = {}
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data     = resp.json()
            raw_text = data["content"][0]["text"]
            usage    = data.get("usage", {})
            meta = {
                "latency_s":     round(time.monotonic() - start, 2),
                "output_tokens": usage.get("output_tokens", 0),
                "model":         data.get("model", self.model),
                "provider":      "anthropic",
            }
            return raw_text, meta
        except Exception as e:
            meta = {
                "error":     str(e),
                "latency_s": round(time.monotonic() - start, 2),
                "provider":  "anthropic",
            }
            return None, meta

    def _call_openai(self, system: str, messages: list, temperature: float) -> tuple:
        start = time.monotonic()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       self.model,
            "temperature": temperature,
            "messages":    [{"role": "system", "content": system}] + messages,
        }
        meta = {}
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data     = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            usage    = data.get("usage", {})
            meta = {
                "latency_s":     round(time.monotonic() - start, 2),
                "output_tokens": usage.get("completion_tokens", 0),
                "model":         data.get("model", self.model),
                "provider":      "openai",
            }
            return raw_text, meta
        except Exception as e:
            meta = {
                "error":     str(e),
                "latency_s": round(time.monotonic() - start, 2),
                "provider":  "openai",
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

    print("=== llm.py smoke test (Day 4 — provider abstraction) ===\n")

    # [1] Provider routing — Ollama (default)
    print("[1] LLMClient(provider='ollama')")
    llm = LLMClient(provider="ollama", verbose=False)
    assert llm.provider == "ollama"
    assert llm.model == OLLAMA_MODEL
    print(f"    provider={llm.provider}  model={llm.model}  host={llm.host}  OK\n")

    # [2] Provider routing — Anthropic
    print("[2] LLMClient(provider='anthropic', api_key='sk-test')")
    llm_ant = LLMClient(provider="anthropic", api_key="sk-ant-test", verbose=False)
    assert llm_ant.provider == "anthropic"
    assert llm_ant.model == ANTHROPIC_MODEL
    print(f"    provider={llm_ant.provider}  model={llm_ant.model}  OK\n")

    # [3] Provider routing — OpenAI
    print("[3] LLMClient(provider='openai', api_key='sk-test')")
    llm_oai = LLMClient(provider="openai", api_key="sk-test", verbose=False)
    assert llm_oai.provider == "openai"
    print(f"    provider={llm_oai.provider}  model={llm_oai.model}  OK\n")

    # [4] Unknown provider raises
    print("[4] Unknown provider raises ValueError")
    try:
        LLMClient(provider="groq", verbose=False)
        assert False, "Should raise"
    except ValueError as e:
        print(f"    Correctly raised: {e}\n")

    # [5] Anthropic requires key
    print("[5] Anthropic without key raises ValueError")
    os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        LLMClient(provider="anthropic", verbose=False)
        assert False, "Should raise"
    except ValueError as e:
        print(f"    Correctly raised: {e}\n")

    # [6] generate_text exists and routes correctly
    print("[6] generate_text() method exists on all providers")
    for p, key in [("ollama", None), ("anthropic", "sk-test"), ("openai", "sk-test")]:
        client = LLMClient(provider=p, api_key=key, verbose=False)
        assert hasattr(client, "generate_text"), f"generate_text missing for {p}"
        print(f"    {p}: generate_text present  OK")

    # [7] JSON parse + validation (provider-independent)
    print("\n[7] _parse_json + _validate_action (provider-independent)")
    llm2 = LLMClient(provider="ollama", verbose=False)
    parsed, err = llm2._parse_json(
        '{"tool": "nmap_scan", "args": {"target": "10.10.10.5"}, "reasoning": "Start."}'
    )
    assert parsed is not None and err is None
    validated, verr = llm2._validate_action(parsed)
    assert validated is not None and verr is None
    print(f"    parse+validate OK: tool={validated['tool']}\n")

    # [8] Live Ollama ping (only if running)
    host = sys.argv[1] if len(sys.argv) > 1 else OLLAMA_HOST
    llm3 = LLMClient(provider="ollama", host=host, verbose=True)
    print(f"[8] Pinging Ollama at {host}...")
    ok, msg = llm3.ping()
    if ok:
        print(f"    Reachable — model: {msg}")
        print("\n[9] Live generate_text() call...")
        result = llm3.generate_text("Say hello in one sentence.", temperature=0.3, max_tokens=50)
        print(f"    Response: {result[:100]}")
    else:
        print(f"    Unreachable: {msg}")
        print("    (Skipping live generation test)")

    print("\nAll offline tests passed.")