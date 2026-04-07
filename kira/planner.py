"""
llm.py — Kira Autonomous Penetration Testing Agent
LLM interface for Gemma 3 4B via Ollama.

Handles:
- Structured JSON action decisions (the planner)
- Free-text reasoning queries (for report generation, vuln analysis)
- Connection health checks
- Prompt templating and token-budget enforcement

Usage (standalone test):
    python llm.py --host http://192.168.1.x:11434
    python llm.py --host http://localhost:11434 --prompt "What should I do after finding port 80 open?"
"""

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DEFAULT_HOST  = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"
MAX_RETRIES   = 3
RETRY_DELAY   = 2   # seconds between retries
REQUEST_TIMEOUT = 90  # seconds — Gemma 4B can be slow on first token


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

PLANNER_SYSTEM = """You are Kira, a senior penetration tester operating inside an authorized lab environment.
Your job is to decide the single best next action to progress a penetration test.

RULES:
1. Only ever target the IP address given in the state. Never target anything else.
2. Reply with ONLY a single JSON object — no prose, no markdown fences, nothing else.
3. JSON format: {"tool": "<name>", "args": {<key>: <value>}, "reasoning": "<one sentence>"}
4. Be methodical and follow the pentest lifecycle: recon → enumeration → exploitation → privesc → report.
5. Do not repeat an action already listed in "Last 3 actions" unless you have new information that justifies it.
6. When you have enough data to move forward, call advance_phase.
7. If the target appears fully compromised or no further actions are possible, call done.

AVAILABLE TOOLS:
  nmap_initial   Run a fast -sV -sC scan.                       Args: {}
  nmap_full      Run a full -p- port scan.                      Args: {}
  gobuster       Brute-force web directories.                   Args: {"url": "http://<ip>", "wordlist": "<path> (optional)"}
  metasploit     Execute an MSF module.                         Args: {"module": "<path>", "options": {"RHOSTS": "<ip>", ...}}
  run_shell      Run an arbitrary shell command.                Args: {"command": "<cmd>"}
  advance_phase  Move to the next pentest phase.                Args: {}
  generate_report  Write the final pentest report.             Args: {}
  done           Signal that testing is complete.               Args: {}

EXAMPLE DECISIONS:
State: phase=recon, no ports found yet
Output: {"tool": "nmap_initial", "args": {}, "reasoning": "No scan data exists — start with a version and script scan."}

State: phase=recon, ports 22/ssh and 80/http found, last action=nmap_initial
Output: {"tool": "advance_phase", "args": {}, "reasoning": "Initial recon complete with open ports identified — move to enumeration."}

State: phase=enumeration, port 80/http open, no directories found yet
Output: {"tool": "gobuster", "args": {"url": "http://10.10.10.3"}, "reasoning": "HTTP service found — brute-force directories to discover attack surface."}
"""

ANALYST_SYSTEM = """You are Kira, a senior penetration tester writing a professional security report.
Provide clear, accurate, concise analysis. Use markdown formatting.
Focus on actionable findings and remediation steps.
"""


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class Action:
    tool: str
    args: dict = field(default_factory=dict)
    reasoning: str = ""

    def is_valid(self) -> bool:
        return bool(self.tool)

    def __str__(self):
        return f"{self.tool}({self.args}) — {self.reasoning}"


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0


# ─────────────────────────────────────────────
# Core client
# ─────────────────────────────────────────────

class OllamaClient:
    """
    Thin HTTP client for Ollama's /api/chat endpoint.
    No external dependencies — uses stdlib urllib only.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        logger: Optional[logging.Logger] = None,
    ):
        self.host  = host.rstrip("/")
        self.model = model
        self.log   = logger or logging.getLogger("kira.llm")

    # ── Public ────────────────────────────────

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model is available."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
                models = [m["name"] for m in body.get("models", [])]
                if self.model not in models:
                    self.log.warning(
                        f"[LLM] Model '{self.model}' not found on host. "
                        f"Available: {models}"
                    )
                    return False
                self.log.info(f"[LLM] Connected — model '{self.model}' ready")
                return True
        except Exception as e:
            self.log.error(f"[LLM] Health check failed: {e}")
            return False

    def decide(self, context_summary: str) -> Action:
        """
        Ask the planner to choose the next pentest action.
        Returns a validated Action dataclass.
        Falls back to a safe default if the LLM fails.
        """
        prompt = (
            f"Current pentest state:\n"
            f"{context_summary}\n\n"
            f"What is the single best next action? Reply with JSON only."
        )
        response = self._chat(
            system=PLANNER_SYSTEM,
            user=prompt,
            json_mode=True,
        )
        return self._parse_action(response.content)

    def analyse(self, prompt: str) -> str:
        """
        Free-text reasoning call — used for report generation,
        vulnerability descriptions, remediation suggestions.
        Returns raw markdown string.
        """
        response = self._chat(
            system=ANALYST_SYSTEM,
            user=prompt,
            json_mode=False,
        )
        return response.content

    # ── Private ───────────────────────────────

    def _chat(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        POST to /api/chat with retry logic.
        json_mode=True sets format="json" — critical for the planner.
        """
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,      # low temp = consistent structured output
                "num_predict": 512,      # enough for an action + report sections
                "stop": ["\n\n\n"],      # prevent runaway generation
            },
        }
        if json_mode:
            payload["format"] = "json"

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.time()
                data = json.dumps(payload).encode()
                req  = urllib.request.Request(
                    f"{self.host}/api/chat",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    body = json.loads(resp.read())

                elapsed_ms = int((time.time() - t0) * 1000)
                content    = body["message"]["content"]
                usage      = body.get("usage", {})

                self.log.debug(
                    f"[LLM] {elapsed_ms}ms | "
                    f"in={usage.get('prompt_tokens','?')} "
                    f"out={usage.get('completion_tokens','?')} tokens"
                )
                return LLMResponse(
                    content=content,
                    model=body.get("model", self.model),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    duration_ms=elapsed_ms,
                )

            except urllib.error.URLError as e:
                last_error = e
                self.log.warning(f"[LLM] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

            except json.JSONDecodeError as e:
                last_error = e
                self.log.error(f"[LLM] Bad JSON from Ollama: {e}")
                break  # retrying won't fix a malformed response

        self.log.error(f"[LLM] All retries exhausted. Last error: {last_error}")
        raise ConnectionError(f"Ollama unreachable at {self.host}: {last_error}")

    def _parse_action(self, raw: str) -> Action:
        """
        Parse the LLM's JSON output into an Action.
        Handles common failure modes:
          - Extra text before/after the JSON object
          - Missing keys
          - Nested args wrapped as a string
        Falls back to a safe nmap_initial action if parsing fails.
        """
        try:
            # Strip any accidental prose wrapping the JSON
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")

            cleaned = raw[start:end]
            parsed  = json.loads(cleaned)

            tool      = parsed.get("tool", "").strip()
            args      = parsed.get("args", {})
            reasoning = parsed.get("reasoning", "")

            # Sometimes the model wraps args as a JSON string instead of object
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if not tool:
                raise ValueError("Empty tool field")

            action = Action(tool=tool, args=args, reasoning=reasoning)
            self.log.info(f"[LLM] → {action}")
            return action

        except Exception as e:
            self.log.error(f"[LLM] Failed to parse action: {e} | raw={raw[:200]}")
            return Action(
                tool="nmap_initial",
                args={},
                reasoning=f"Parse error fallback — raw: {raw[:80]}",
            )


# ─────────────────────────────────────────────
# Convenience wrapper (used by main.py)
# ─────────────────────────────────────────────

class LLMPlanner:
    """
    Thin facade so main.py only needs to import LLMPlanner from llm.
    Identical interface to the inline class in main.py — replace that
    class with: from llm import LLMPlanner
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        logger: Optional[logging.Logger] = None,
    ):
        self.client = OllamaClient(host=host, model=model, logger=logger)
        self.log    = logger or logging.getLogger("kira.planner")

    def health_check(self) -> bool:
        return self.client.health_check()

    def decide(self, context: str) -> dict:
        """Returns a plain dict so main.py dispatch logic is unchanged."""
        action = self.client.decide(context)
        return {"tool": action.tool, "args": action.args, "reasoning": action.reasoning}

    def analyse(self, prompt: str) -> str:
        return self.client.analyse(prompt)


# ─────────────────────────────────────────────
# Standalone smoke test
# ─────────────────────────────────────────────

def _run_smoke_test(host: str, prompt: Optional[str]):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    client = OllamaClient(host=host)

    print(f"\n[1] Health check → {host}")
    ok = client.health_check()
    if not ok:
        print(f"    FAIL — is Ollama running? Try: ollama serve")
        print(f"           Is the model pulled? Try: ollama pull {DEFAULT_MODEL}")
        return

    print("    PASS\n")

    # Test 1 — planner decision
    print("[2] Planner decision test")
    mock_context = """Target: 10.10.10.3
Current phase: recon
Completed phases: none
Last action: none

Open ports (0):

Last 3 actions: none"""

    action = client.decide(mock_context)
    print(f"    tool:      {action.tool}")
    print(f"    args:      {action.args}")
    print(f"    reasoning: {action.reasoning}")
    print(f"    valid:     {action.is_valid()}\n")

    # Test 2 — analyst free text
    print("[3] Analyst reasoning test")
    analyst_prompt = prompt or (
        "Port 21 (FTP) is open running vsftpd 2.3.4. "
        "What vulnerability should I check for, and what Metasploit module would exploit it?"
    )
    print(f"    Prompt: {analyst_prompt[:80]}...")
    result = client.analyse(analyst_prompt)
    print(f"    Response:\n{result}\n")

    print("[✓] Smoke test complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="llm",
        description="Kira LLM module smoke test",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Ollama host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom analyst prompt to test (optional)",
    )
    args = parser.parse_args()
    _run_smoke_test(args.host, args.prompt)