"""
kira/chat.py — KiraChat
========================
Conversational REPL shell that wraps the existing Planner.

Three modes:
  CHAT   — user asks security questions; LLM answers in plain English
  SCAN   — user triggers autonomous scan; Planner.run() is called unchanged
  EXIT   — user types exit/quit/bye; start() returns

Rules strictly followed:
  - Does NOT modify any existing module.
  - Reuses the SAME LLMClient instance passed in from main.py.
  - Calls Planner.run() as a black box — never reaches inside it.
  - llm.ask() always returns a dict; _handle_chat() extracts the
    'reasoning' field as the plain-text response.

Usage (from main.py):
    from kira.chat import KiraChat
    KiraChat(planner=planner, state=state, llm=llm,
             max_iter=args.max_iter, verbose=args.verbose).start()
"""

import re
import time
from pathlib import Path

# ── Chat system prompt ─────────────────────────────────────────────────────────
# This prompt is used for conversational (non-scan) turns.
# It overrides Kira's JSON-only persona to get a plain-English response.
# We use add_note as the JSON vehicle and tell the model to put its full
# answer inside the "reasoning" field — that field is then extracted and
# printed as plain text.

CHAT_SYSTEM_PROMPT = """You are Kira, an autonomous penetration testing agent and security expert.
You help users understand penetration testing concepts, explain vulnerabilities,
and discuss findings from scan sessions.

CURRENT SESSION STATE is provided at the start of each user message.
Use it to give contextual, specific answers about THIS target and its findings.
If no scan has been run yet, answer from general security knowledge.

RULES:
- Answer in plain conversational English. Never output JSON prose to the user.
- You MUST reply in valid JSON with exactly these keys:
    {"tool": "add_note", "args": {}, "reasoning": "<YOUR FULL ANSWER HERE>"}
  Put your complete answer — including all explanation — inside the "reasoning"
  field. The "tool" must always be "add_note" and "args" must always be {}.
- Keep answers under 200 words unless the user explicitly asks for more detail.
- Only reference findings, ports, or vulnerabilities that appear in the
  session state. Do not invent findings that are not there.
- If the user asks you to start a scan, tell them to say
  'start scan on TARGET_IP' or just provide the IP with a trigger word.
- You may explain CVEs, attack techniques, remediation steps, and security
  concepts freely from your training knowledge.
- Be direct and technically precise. Assume the user is a security professional.
- Do NOT start your reasoning with "reasoning:" or any JSON key name."""


# ── Scan trigger words ─────────────────────────────────────────────────────────

_TRIGGER_WORDS = {
    "start", "scan", "hack", "test", "pentest",
    "find", "attack", "enumerate", "exploit", "recon",
    "run", "begin", "check", "target",
}

_TRIGGER_PHRASES = {
    "start scan", "begin scan", "run scan", "start pentest",
    "find vulnerabilities", "find vulns", "start hacking",
    "run kira", "start kira", "begin pentest",
}

_EXIT_WORDS = {"exit", "quit", "bye", "done", "q", ":q"}

_IPV4_RE = re.compile(
    r"\b((?:25[0-5]|2[0-4]\d|[01]?\d\d?)\."
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\."
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\."
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?))\b"
)


# ── KiraChat ───────────────────────────────────────────────────────────────────

class KiraChat:
    """
    Conversational REPL shell layered on top of the existing Planner.

    Parameters
    ----------
    planner  : existing Planner instance from main.py
    state    : existing StateManager instance
    llm      : existing LLMClient instance — REUSED, not re-created
    max_iter : passed from args.max_iter
    verbose  : passed from args.verbose
    """

    def __init__(self, planner, state, llm, max_iter: int = 50, verbose: bool = True):
        self._planner  = planner
        self._state    = state
        self._llm      = llm
        self._max_iter = max_iter
        self._verbose  = verbose

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Main REPL loop. Runs until user exits.
        Prints welcome banner, then loops:
          1. Print "kira> " prompt
          2. Read user input
          3. Route to _handle_chat(), _handle_scan_trigger(), or exit
        """
        self._print_banner()

        while True:
            try:
                user_input = input("\nkira> ").strip()
            except (EOFError, KeyboardInterrupt):
                # Ctrl+D or Ctrl+C at the prompt — clean exit
                self._print_goodbye()
                return

            if not user_input:
                continue

            lower = user_input.lower().strip()

            # ── Exit ──────────────────────────────────────────────────────────
            if lower in _EXIT_WORDS:
                self._print_goodbye()
                return

            # ── Scan trigger ──────────────────────────────────────────────────
            if self._is_scan_trigger(user_input):
                self._handle_scan_trigger(user_input)
                continue

            # ── Chat ──────────────────────────────────────────────────────────
            self._handle_chat(user_input)

    # ── Trigger detection ─────────────────────────────────────────────────────

    def _is_scan_trigger(self, message: str) -> bool:
        """
        Returns True if message should kick off planner.run().

        Matches either:
          (a) an IPv4 address + at least one trigger word, OR
          (b) an exact trigger phrase anywhere in the message
        """
        lower = message.lower()

        # Check exact trigger phrases first
        for phrase in _TRIGGER_PHRASES:
            if phrase in lower:
                return True

        # Check IP + trigger word combination
        if _IPV4_RE.search(message):
            words = set(re.findall(r"[a-z]+", lower))
            if words & _TRIGGER_WORDS:
                return True

        return False

    def _extract_ip(self, message: str) -> str | None:
        """Extracts the first IPv4 address from message, or None."""
        match = _IPV4_RE.search(message)
        return match.group(1) if match else None

    # ── Chat handler ──────────────────────────────────────────────────────────

    def _handle_chat(self, message: str) -> None:
        """
        Call llm.ask() with CHAT_SYSTEM_PROMPT and the current context.
        Extract the 'reasoning' field from the dict response and print
        it as plain text. Never crashes on unexpected response shapes.
        """
        user_turn = self._build_chat_prompt(message)

        try:
            action = self._llm.ask(
                user=user_turn,
                system=CHAT_SYSTEM_PROMPT,
                temperature=0.4,   # slightly higher than planner for natural prose
            )
        except Exception as exc:
            self._print_kira(f"Sorry, I couldn't reach the LLM right now: {exc}")
            return

        # action is always a dict — extract plain-text answer from 'reasoning'
        if isinstance(action, dict):
            response = (
                action.get("reasoning")
                or action.get("response")
                or action.get("content")
                or str(action)
            )
        else:
            response = str(action)

        # Strip any JSON-like wrapper the model might have leaked
        response = response.strip()
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]

        self._print_kira(response)

    # ── Scan handler ──────────────────────────────────────────────────────────

    def _handle_scan_trigger(self, message: str) -> None:
        """
        Extracts IP if present, updates state target if needed,
        calls planner.run(), prints result summary, then returns
        to chat mode — does NOT exit.
        """
        extracted_ip = self._extract_ip(message)

        # If a new IP was mentioned, update state target
        if extracted_ip:
            current_target = self._state.get("target")
            if current_target != extracted_ip:
                try:
                    self._state.update(target=extracted_ip)
                except Exception:
                    # If state doesn't support updating target mid-session,
                    # log and continue — Planner will use what's in state
                    pass

        target = self._state.get("target") or extracted_ip or "unknown"

        print(f"\n[KIRA] Understood. Starting autonomous scan on {target}...")
        print(f"[KIRA] You can interrupt with Ctrl+C at any time.\n")

        t_start = time.monotonic()
        outcome = "ERROR"

        try:
            outcome = self._planner.run(max_iterations=self._max_iter)
        except KeyboardInterrupt:
            outcome = "INTERRUPTED"
            print(f"\n[KIRA] Scan interrupted by user.")
        except Exception as exc:
            outcome = "ERROR"
            print(f"\n[KIRA] Scan error: {exc}")
            if self._verbose:
                import traceback
                traceback.print_exc()

        elapsed = time.monotonic() - t_start

        # Print outcome
        outcome_labels = {
            "DONE":        "Scan complete.",
            "ROOT":        "Root obtained — scan complete.",
            "HALTED":      "Agent halted (see reasoning above).",
            "MAX_ITER":    f"Max iterations reached.",
            "INTERRUPTED": "Scan interrupted.",
            "ERROR":       "Scan encountered an error.",
        }
        label = outcome_labels.get(outcome, outcome)
        print(f"\n[KIRA] Session ended: {outcome} — {label}")

        # Print session summary using the shared helper from main
        try:
            from main import _print_session_summary
            session_dir = Path(self._state.session_dir)
            _print_session_summary(self._state, session_dir, elapsed)
        except Exception:
            # Fallback if import fails — print a minimal summary
            findings = self._state.get("findings") or []
            ports    = self._state.get("open_ports") or []
            print(f"  Open ports : {len(ports)}   Findings: {len(findings)}   Elapsed: {elapsed:.0f}s")

        # Prompt the user to ask questions
        print(f"\n[KIRA] You can now ask me questions about the findings.")
        print(f"       Try: 'explain the findings' or 'what should I fix first?'")

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_chat_prompt(self, user_message: str) -> str:
        """
        Returns the full user-turn content:
          <context summary>
          ---
          User question: <user_message>
        """
        context = self._state.get_context_summary()
        return (
            f"CURRENT SESSION STATE:\n"
            f"{context}\n"
            f"---\n"
            f"User question: {user_message}"
        )

    # ── Display helpers ───────────────────────────────────────────────────────

    def _print_kira(self, message: str) -> None:
        """Print a Kira response with the [KIRA] prefix."""
        print(f"[KIRA] {message}")

    def _print_banner(self) -> None:
        """Print the chat welcome banner."""
        try:
            from rich.console import Console
            from rich.panel   import Panel
            c = Console()
            c.print(Panel(
                "[bold]Kira v0.3.0[/bold] — Autonomous Pentest Agent\n"
                "[dim]Type a target IP + trigger word to start scanning.[/dim]\n"
                "[dim]Ask me anything about pentesting. Type 'exit' to quit.[/dim]",
                border_style="dim",
                expand=False,
            ))
        except ImportError:
            print("┌─────────────────────────────────────────────────────────┐")
            print("│  Kira v0.3.0 — Autonomous Pentest Agent                 │")
            print("│  Type a target IP + trigger word to start scanning.     │")
            print("│  Ask me anything about pentesting. Type 'exit' to quit. │")
            print("└─────────────────────────────────────────────────────────┘")

    def _print_goodbye(self) -> None:
        """Print the exit message."""
        print("\n[KIRA] Session ended. Stay authorized.")