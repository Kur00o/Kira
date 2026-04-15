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
    Conversational REPL shell for Kira.
    Handles chat, target discovery from user input, and scan triggering.
    Lazily initializes state and planner when target is provided.

    Parameters
    ----------
    runner   : ToolRunner instance
    llm      : LLMClient instance
    msf      : MSF client or None
    kb       : KnowledgeBase or None
    session_dir : Path to session directory
    log      : KiraLogger instance
    authorized_by : Authorization string
    max_iter : max iterations per scan
    verbose  : verbose output
    initial_target : optional target from --target CLI arg
    no_report : skip report generation
    """

    def __init__(
        self,
        runner,
        llm,
        msf,
        kb,
        session_dir,
        log,
        authorized_by,
        max_iter: int = 50,
        verbose: bool = True,
        initial_target: str = None,
        no_report: bool = False,
    ):
        self._runner = runner
        self._llm = llm
        self._msf = msf
        self._kb = kb
        self._session_dir = session_dir
        self._log = log
        self._authorized_by = authorized_by
        self._max_iter = max_iter
        self._verbose = verbose
        self._no_report = no_report
        
        # State and planner initialized lazily when target is discovered
        self._state = None
        self._planner = None
        self._guard = None
        
        # If target provided via CLI, initialize immediately
        if initial_target:
            self._init_for_target(initial_target)

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_for_target(self, target: str) -> None:
        """
        Initialize state, planner, and guard for a given target.
        Called when target is discovered (either from CLI or chat).
        """
        from kira.state import StateManager
        from kira.planner import Planner
        from kira.guardrails import ScopeGuard
        
        # Initialize state for this target
        self._state = StateManager(session_dir=str(self._session_dir))
        self._state.init(target=target, authorized_by=self._authorized_by)
        self._log.info(f"Target set: {target}")
        
        # Scope guard
        self._guard = ScopeGuard(authorized_target=target, authorized_by=self._authorized_by)
        self._guard.validate_startup(self._log)
        
        # Phase transition logger hook
        _orig_advance = self._state.advance_phase
        def _logged_advance():
            old = self._state.phase
            new = _orig_advance()
            if new != old:
                self._log.phase(old, new)
            return new
        self._state.advance_phase = _logged_advance
        
        # Create planner
        self._planner = Planner(
            state=self._state,
            runner=self._runner,
            llm=self._llm,
            msf=self._msf,
            kb=self._kb,
            verbose=True,
            logger=self._log,
            guard=self._guard,
        )

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

    def _extract_iterations(self, message: str) -> int | None:
        """
        Extracts iteration count from message.
        Looks for patterns like:
          - "scan 10.10.10.5 for 15 iterations"
          - "scan 10.10.10.5 with 20 iterations"
          - "scan 10.10.10.5 in 30 steps"
          - "scan 10.10.10.5 25"
        Returns the number, or None if not found.
        """
        lower = message.lower()
        
        # Pattern: "for/with/in NUMBER (iterations/steps/rounds/times)"
        patterns = [
            r"(?:for|with|in)\s+(\d+)\s*(?:iterations?|steps?|rounds?|times?)?",
            r"(?:iterations?|steps?|rounds?|times?)\s+(\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    pass
        
        # Also check for a bare number at the end (e.g., "scan 10.10.10.5 25")
        # But only if there's an IP in the message (to avoid false positives)
        if _IPV4_RE.search(message):
            numbers = re.findall(r"\b(\d+)\b", message)
            # Filter out IP octets (numbers that appear in the IP)
            ip_octets = set()
            for ip_match in _IPV4_RE.finditer(message):
                ip = ip_match.group(1)
                ip_octets.update(ip.split("."))
            
            for num_str in reversed(numbers):  # Check from end of message
                num = int(num_str)
                if num_str not in ip_octets and 1 <= num <= 1000:
                    return num
        
        return None

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
        Extracts IP and iteration count from message, initializes target if new,
        calls planner.run(), prints result summary, returns to chat mode.
        """
        extracted_ip = self._extract_ip(message)

        # Determine which target to use
        if extracted_ip:
            target = extracted_ip
        elif self._state:
            target = self._state.target
        else:
            self._print_kira("No target IP found in your message. Please specify one (e.g., '10.10.10.5 for 15 iterations').")
            return

        # Extract iteration count from message, or ask user
        max_iterations = self._extract_iterations(message)
        
        if max_iterations is None:
            # Prompt user for iteration count
            try:
                print()  # Blank line for readability
                iter_input = input(f"[KIRA] How many iterations? (default {self._max_iter}): ").strip()
                if iter_input:
                    try:
                        max_iterations = int(iter_input)
                        if max_iterations < 1:
                            max_iterations = self._max_iter
                            self._print_kira(f"Invalid input. Using default {self._max_iter} iterations.")
                    except ValueError:
                        max_iterations = self._max_iter
                        self._print_kira(f"Invalid number. Using default {self._max_iter} iterations.")
                else:
                    max_iterations = self._max_iter
            except (EOFError, KeyboardInterrupt):
                self._print_kira("Scan cancelled.")
                return

        # Initialize target if not already done
        if not self._state or self._state.target != target:
            try:
                self._init_for_target(target)
            except Exception as e:
                self._print_kira(f"Failed to initialize target {target}: {e}")
                return

        print(f"\n[KIRA] Understood. Starting autonomous scan on {target}...")
        print(f"[KIRA] Max iterations: {max_iterations}")
        print(f"[KIRA] You can interrupt with Ctrl+C at any time.\n")

        t_start = time.monotonic()
        outcome = "ERROR"

        try:
            outcome = self._planner.run(max_iterations=max_iterations)
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
            _print_session_summary(self._state, self._session_dir, elapsed)
        except Exception:
            # Fallback if import fails — print a minimal summary
            findings = self._state.get("findings") or []
            ports    = self._state.get("open_ports") or []
            print(f"  Open ports : {len(ports)}   Findings: {len(findings)}   Elapsed: {elapsed:.0f}s")

        # Offer report generation
        if not self._no_report:
            from main import run_report
            finding_count = len(self._state.get("findings", []))
            if outcome in ("DONE", "ROOT") or (outcome == "MAX_ITER" and finding_count > 0):
                try:
                    run_report(self._session_dir, self._llm, self._log, outcome, finding_count)
                except Exception as e:
                    print(f"[KIRA] Report generation failed: {e}")

        # Prompt the user to ask questions
        print(f"\n[KIRA] You can now ask me questions about the findings.")
        print(f"       Try: 'explain the findings' or 'what should I fix first?'")

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_chat_prompt(self, user_message: str) -> str:
        """
        Returns the full user-turn content with optional context.
        If no scan has been run yet, uses general knowledge mode.
        """
        if self._state:
            context = self._state.get_context_summary()
        else:
            context = "No active target or scan yet."

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