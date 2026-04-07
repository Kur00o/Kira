"""
kira/llm.py — LLM Interface
============================
All communication with the local Gemma 3 4B model (via Ollama) flows here.
No other module calls the Ollama API directly.

Responsibilities:
  - Send prompts to Gemma and always return parsed JSON
  - Retry on malformed JSON with escalating correction prompts
  - Validate the returned action against a known tool schema
  - Track token usage and response latency per call
  - Fail loudly with actionable messages (wrong host, model not pulled, etc.)

Usage:
    llm = LLMClient(host="http://192.168.1.42:11434")

    action = llm.ask(
        system="You are Kira, a penetration testing AI...",
        user="Given these open ports [22, 80], what should I do next?",
    )
    # action is always a dict — never raises on bad JSON, retries instead
    print(action["tool"], action["args"], action["reasoning"])

    # Convenience wrapper used by the planner every loop tick:
    action = llm.next_action(context_summary, phase="ENUM")
"""

import json
import time
import textwrap
from datetime import datetime, timezone
from typing import Optional

import requests


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "gemma3:4b"
DEFAULT_HOST    = "http://localhost:11434"
DEFAULT_TIMEOUT = 120          # seconds for a single HTTP request
MAX_RETRIES     = 3            # JSON parse retries before giving up
RETRY_DELAY     = 1.5          # seconds between retries

# Every action the planner is allowed to emit.
# The LLM is shown this list so it knows its option space.
VALID_TOOLS = [
    "nmap_scan",
    "gobuster_dir",
    "searchsploit",
    "enum4linux",
    "curl_probe",
    "whatweb",
    "msf_exploit",
    "shell_cmd",
    "linpeas",
    "add_finding",
    "add_note",
    "advance_phase",
    "REPORT",
    "HALT",
]

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are Kira, an autonomous penetration testing AI agent operating inside an
authorized security lab environment.

Your job is to analyse the current session state and decide the single best
next action to advance the pentest toward root access and a complete report.

RULES:
1. You MUST reply with ONLY a valid JSON object — no markdown, no prose, no
   code fences. Raw JSON only.
2. Every response MUST contain exactly these three keys:
     "tool"      : one of the valid tool names listed below
     "args"      : a JSON object of arguments for that tool (can be {})
     "reasoning" : one sentence explaining why you chose this action
3. Choose the most targeted, efficient action given the current phase and
   findings. Do not repeat an action that was already taken with the same args.
4. If you have no useful next action, emit HALT with a reason in "reasoning".
5. Only emit REPORT when you have at least one confirmed vulnerability finding.

VALID TOOLS AND THEIR ARGS:
  nmap_scan        : {"target": "IP", "flags": "-sV -sC", "ports": "22,80"}
  gobuster_dir     : {"url": "http://IP", "wordlist": "/path/to/list.txt"}
  searchsploit     : {"query": "service version string"}
  enum4linux       : {"target": "IP"}
  curl_probe       : {"url": "http://IP/path", "flags": "-sI"}
  whatweb          : {"url": "http://IP"}
  msf_exploit      : {"module": "exploit/...", "options": {"RHOSTS": "IP"}}
  shell_cmd        : {"cmd": "whoami", "session_id": 1}
  linpeas          : {"session_id": 1}
  add_finding      : {"title": "...", "severity": "critical|high|medium|low|info",
                      "port": 80, "cvss": 9.8, "description": "...",
                      "remediation": "..."}
  add_note         : {"note": "free text observation"}
  advance_phase    : {}
  REPORT           : {}
  HALT             : {}

EXAMPLE (valid response):
{"tool": "gobuster_dir", "args": {"url": "http://10.10.10.5", "wordlist": "/usr/share/wordlists/dirb/common.txt"}, "reasoning": "Port 80 is running Apache; directory enumeration may reveal admin panels or config files."}
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
    Thin, robust wrapper around the Ollama /api/chat endpoint.

    Parameters
    ----------
    host    : Ollama server URL, e.g. "http://192.168.1.42:11434"
    model   : model tag pulled via `ollama pull`, default "gemma3:4b"
    timeout : per-request HTTP timeout in seconds
    verbose : if True, print each call's latency and token count
    """

    def __init__(
        self,
        host:    str = DEFAULT_HOST,
        model:   str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        verbose: bool = True,
    ):
        self.host    = host.rstrip("/")
        self.model   = model
        self.timeout = timeout
        self.verbose = verbose

        self._call_log: list[dict] = []   # in-memory log of every call

    # ── Public API ─────────────────────────────────────────────────────────────

    def ask(
        self,
        user:   str,
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.2,    # low = more deterministic / less hallucination
    ) -> dict:
        """
        Send a prompt and always return a parsed dict.

        Retries up to MAX_RETRIES times with a correction prompt if the
        model returns malformed JSON. After all retries are exhausted,
        returns a safe HALT action instead of raising.

        Parameters
        ----------
        user        : the user-turn content (context summary + question)
        system      : system prompt (defaults to Kira's pentest persona)
        temperature : 0.0–1.0; keep low for reliable JSON output

        Returns
        -------
        dict with keys: tool, args, reasoning
             (plus _meta with latency / token info)
        """
        messages = [{"role": "user", "content": user}]
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            raw, meta = self._call_ollama(
                system=system,
                messages=messages,
                temperature=temperature,
            )

            if raw is None:
                # Network / server error — already logged in meta
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
                # Valid JSON but wrong schema — treat as parse error
                parse_error = val_error

            # Failed — append correction turn and retry
            if self.verbose:
                self._print_retry(attempt, parse_error)

            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role":    "user",
                "content": CORRECTION_PROMPT.format(error=parse_error),
            })
            last_error = parse_error

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        # All retries exhausted
        self._record(MAX_RETRIES, meta, ok=False)
        return self._halt(
            f"Could not get valid JSON after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}",
        )

    def next_action(self, context_summary: str, phase: str = "") -> dict:
        """
        Convenience method used by the planner on every loop tick.
        Wraps the context summary in a standard user prompt.

        Parameters
        ----------
        context_summary : output of StateManager.get_context_summary()
        phase           : current phase name (added to prompt for clarity)
        """
        phase_line = f"Current phase: {phase}\n\n" if phase else ""
        user = (
            f"{phase_line}"
            f"{context_summary}\n\n"
            "Based on the session state above, what is the single best next action?"
        )
        return self.ask(user=user)

    def ping(self) -> tuple[bool, str]:
        """
        Check that the Ollama server is reachable and the model is loaded.

        Returns (True, model_name) on success, (False, error_message) on failure.
        """
        try:
            r = requests.get(
                f"{self.host}/api/tags",
                timeout=10,
            )
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]

            # Accept both "gemma3:4b" and "gemma3:4b-instruct-q4_K_M" etc.
            matched = [m for m in models if m.startswith(self.model.split(":")[0])]
            if not matched:
                return False, (
                    f"Model '{self.model}' not found on {self.host}.\n"
                    f"Available: {models or ['(none)']}\n"
                    f"Fix: ollama pull {self.model}"
                )
            return True, matched[0]

        except requests.exceptions.ConnectionError:
            return False, (
                f"Cannot reach Ollama at {self.host}.\n"
                "Is it running?  →  ollama serve\n"
                "Is it exposed?  →  OLLAMA_HOST=0.0.0.0 ollama serve"
            )
        except requests.exceptions.Timeout:
            return False, f"Ollama at {self.host} timed out during ping."
        except Exception as e:
            return False, f"Unexpected error pinging Ollama: {e}"

    def call_log(self) -> list[dict]:
        """Return a copy of the in-memory call log."""
        return list(self._call_log)

    # ── Internal: HTTP call ────────────────────────────────────────────────────

    def _call_ollama(
        self,
        system:      str,
        messages:    list,
        temperature: float,
    ) -> tuple[Optional[str], dict]:
        """
        POST to /api/chat. Returns (raw_text, meta_dict).
        raw_text is None on any network or HTTP error.
        """
        payload = {
            "model":   self.model,
            "stream":  False,
            "format":  "json",          # Ollama JSON mode — forces valid JSON output
            "options": {
                "temperature": temperature,
                "num_predict": 512,     # cap output tokens — actions are short
            },
            "messages": [
                {"role": "system", "content": system},
                *messages,
            ],
        }

        t0 = time.monotonic()
        meta: dict = {"timestamp": _ts()}

        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()

        except requests.exceptions.ConnectionError:
            meta["error"] = (
                f"Cannot reach Ollama at {self.host}. "
                "Run: OLLAMA_HOST=0.0.0.0 ollama serve"
            )
            return None, meta

        except requests.exceptions.Timeout:
            meta["error"] = (
                f"Ollama request timed out after {self.timeout}s. "
                "The model may be loading — try again in a moment."
            )
            return None, meta

        except requests.exceptions.HTTPError as e:
            meta["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return None, meta

        except Exception as e:
            meta["error"] = f"Unexpected HTTP error: {e}"
            return None, meta

        elapsed = time.monotonic() - t0
        body = resp.json()

        raw_text = body.get("message", {}).get("content", "")
        meta.update({
            "latency_s":      round(elapsed, 2),
            "prompt_tokens":  body.get("prompt_eval_count", 0),
            "output_tokens":  body.get("eval_count", 0),
            "model":          body.get("model", self.model),
        })

        return raw_text, meta

    # ── Internal: parsing + validation ────────────────────────────────────────

    def _parse_json(self, raw: str) -> tuple[Optional[dict], Optional[str]]:
        """
        Attempt to parse raw string as JSON.
        Strips markdown fences if the model ignored the format directive.
        Returns (dict, None) on success, (None, error_string) on failure.
        """
        text = raw.strip()

        # Strip common markdown wrappings the model might add anyway
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first line (```json or ```) and last (```)
            text = "\n".join(lines[1:-1]).strip()

        # Some models wrap with a single outer key like {"response": {...}}
        # Try to unwrap one level if top-level has a single key
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            return None, f"{e} — raw: {text[:120]!r}"

        # Unwrap single-key envelope if needed
        if isinstance(parsed, dict) and len(parsed) == 1:
            inner = next(iter(parsed.values()))
            if isinstance(inner, dict):
                parsed = inner

        return parsed, None

    def _validate_action(self, obj: dict) -> tuple[Optional[dict], Optional[str]]:
        """
        Check that the parsed dict has the required keys and a valid tool name.
        Returns (validated_dict, None) on success, (None, error_string) on failure.
        """
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

        # Normalise — strip any extra keys the model may have added
        return {
            "tool":      tool,
            "args":      obj["args"],
            "reasoning": obj["reasoning"].strip(),
        }, None

    # ── Internal: helpers ──────────────────────────────────────────────────────

    def _halt(self, reason: str, meta: dict = None) -> dict:
        return {
            "tool":      "HALT",
            "args":      {},
            "reasoning": reason,
            "_meta":     meta or {},
        }

    def _record(self, attempts: int, meta: dict, ok: bool) -> None:
        self._call_log.append({
            "timestamp": _ts(),
            "attempts":  attempts,
            "ok":        ok,
            "latency_s": meta.get("latency_s"),
            "tokens":    meta.get("output_tokens"),
        })

    def _print_ok(self, action: dict, meta: dict) -> None:
        try:
            from rich.console import Console
            c = Console()
            c.print(
                f"[dim][LLM][/dim] "
                f"[green]{action['tool']}[/green] "
                f"[dim]({meta.get('latency_s', '?')}s, "
                f"{meta.get('output_tokens', '?')} tokens)[/dim]"
            )
            c.print(f"  [dim italic]{action['reasoning']}[/dim italic]")
        except ImportError:
            print(f"[LLM] {action['tool']} ({meta.get('latency_s')}s) — {action['reasoning']}")

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

    host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    print(f"=== llm.py smoke test  (host: {host}) ===\n")

    llm = LLMClient(host=host, verbose=True)

    # ── 1. Ping ────────────────────────────────────────────────────────────────
    print("[1] Pinging Ollama...")
    ok, msg = llm.ping()
    if ok:
        print(f"    Server reachable — model: {msg}\n")
    else:
        print(f"    UNREACHABLE: {msg}")
        print("\n[OFFLINE] Running JSON-parse + validation tests only.\n")

    # ── 2. JSON parse: valid ───────────────────────────────────────────────────
    print("[2] _parse_json — valid input")
    parsed, err = llm._parse_json(
        '{"tool": "nmap_scan", "args": {"target": "10.10.10.5"}, "reasoning": "Start with recon."}'
    )
    assert parsed is not None and err is None, f"Expected success, got: {err}"
    print(f"    parsed OK: tool={parsed['tool']}\n")

    # ── 3. JSON parse: markdown fences ────────────────────────────────────────
    print("[3] _parse_json — strips markdown fences")
    fenced = '```json\n{"tool": "gobuster_dir", "args": {}, "reasoning": "Enumerate web."}\n```'
    parsed, err = llm._parse_json(fenced)
    assert parsed is not None, f"Fence strip failed: {err}"
    print(f"    stripped OK: tool={parsed['tool']}\n")

    # ── 4. JSON parse: invalid ─────────────────────────────────────────────────
    print("[4] _parse_json — invalid JSON")
    parsed, err = llm._parse_json("This is not JSON at all.")
    assert parsed is None and err is not None
    print(f"    correctly rejected: {err[:60]}\n")

    # ── 5. Validation: missing keys ────────────────────────────────────────────
    print("[5] _validate_action — missing keys")
    _, err = llm._validate_action({"tool": "nmap_scan"})
    assert err is not None and "Missing" in err
    print(f"    correctly rejected: {err}\n")

    # ── 6. Validation: unknown tool ────────────────────────────────────────────
    print("[6] _validate_action — unknown tool")
    _, err = llm._validate_action(
        {"tool": "rm_rf", "args": {}, "reasoning": "chaos"}
    )
    assert err is not None and "Unknown tool" in err
    print(f"    correctly rejected: {err[:80]}\n")

    # ── 7. Validation: valid action ────────────────────────────────────────────
    print("[7] _validate_action — valid action")
    validated, err = llm._validate_action(
        {"tool": "searchsploit", "args": {"query": "Apache 2.4.49"},
         "reasoning": "Check for known CVEs.", "extra_key": "ignored"}
    )
    assert validated is not None and err is None
    assert "extra_key" not in validated, "Extra keys should be stripped"
    print(f"    validated OK: {validated}\n")

    # ── 8. Live LLM call (only if server reachable) ────────────────────────────
    if ok:
        print("[8] Live next_action call...")
        fake_context = textwrap.dedent("""
            === KIRA SESSION CONTEXT ===
            Target     : 10.10.10.5
            Phase      : RECON — Port scanning and service fingerprinting
            User       : none | Root: False

            Open ports : 22, 80
              22/tcp  OpenSSH 7.9
              80/tcp  Apache httpd 2.4.49

            Recent actions:
              [2026-04-06T10:00:00Z] nmap_scan → Found 2 open ports: 22, 80
            === END CONTEXT ===
        """).strip()

        action = llm.next_action(fake_context, phase="RECON")
        print(f"    tool      : {action['tool']}")
        print(f"    args      : {action['args']}")
        print(f"    reasoning : {action['reasoning']}")
        meta = action.get("_meta", {})
        print(f"    latency   : {meta.get('latency_s')}s")
        print(f"    tokens    : {meta.get('output_tokens')}")
        assert action["tool"] in VALID_TOOLS, "Returned invalid tool"
        print()

    print("All tests passed.")