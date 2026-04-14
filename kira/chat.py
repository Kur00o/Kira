"""
kira/chat.py — KiraChat Conversational Shell
=============================================
A conversational REPL layer wrapping the Planner, allowing users to:
  - Ask questions about pentesting concepts and current findings
  - Trigger autonomous scans by providing a target IP + trigger word
  - Exit cleanly

Three modes:
  1. CHAT        — User asks questions; LLM responds conversationally
  2. SCAN_TRIGGER — User specifies target IP; Planner.run() executes autonomously
  3. EXIT         — User types exit/quit; session ends cleanly
"""

import re
from typing import Optional

from kira.state import StateManager
from kira.llm import LLMClient
from kira.planner import Planner


# ── Chat system prompt ────────────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are Kira, an autonomous penetration testing agent and security expert.
You help users understand penetration testing concepts, explain vulnerabilities,
and discuss findings from scan sessions.

CURRENT SESSION STATE is provided at the start of each user message.
Use it to give contextual, specific answers about THIS target and its findings.
If no scan has been run yet, answer from general security knowledge.

RULES:
- Answer in plain conversational English. Never output JSON.
- Keep answers under 200 words unless the user asks for detail.
- Only reference findings, ports, or vulnerabilities that appear in the
  session state. Do not invent findings that are not there.
- If the user asks you to start a scan, tell them to say
  'start scan on TARGET_IP' or just provide the IP with a trigger word.
- You may explain CVEs, attack techniques, remediation steps, and security
  concepts freely from your training knowledge.
- Be direct and technically precise. Assume the user is a security professional."""


# ── ANSI colours ──────────────────────────────────────────────────────────────
# Minimal colour support for KiraChat

class C:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# ── KiraChat ───────────────────────────────────────────────────────────────────

class KiraChat:
    """
    Conversational shell wrapping Planner.
    Supports chat mode (question answering), scan triggers, and exit.
    """

    EXIT_KEYWORDS = {"exit", "quit", "bye", "done", "q", ":q"}

    def __init__(
        self,
        planner: Planner,
        state: StateManager,
        llm: LLMClient,
        max_iter: int = 50,
        verbose: bool = True,
    ):
        """
        Initialize KiraChat.

        Parameters
        ----------
        planner : Planner
            Existing Planner instance from main.py
        state : StateManager
            Existing StateManager instance
        llm : LLMClient
            Existing LLMClient instance (reused — no new connection)
        max_iter : int
            Maximum iterations to pass to planner.run()
        verbose : bool
            Enable verbose output
        """
        self.planner = planner
        self.state = state
        self.llm = llm
        self.max_iter = max_iter
        self.verbose = verbose

    def start(self) -> None:
        """
        Main REPL loop. Runs until user exits.
        Three modes: CHAT, SCAN_TRIGGER, EXIT.
        """
        self._print_welcome()

        while True:
            try:
                user_input = input(f"{C.CYAN}kira> {C.RESET}").strip()
            except EOFError:
                # Ctrl+D or end of input stream
                break
            except KeyboardInterrupt:
                # Ctrl+C
                print()
                self._print_goodbye()
                break

            if not user_input:
                continue

            # Route based on input type
            if user_input.lower() in self.EXIT_KEYWORDS:
                self._print_goodbye()
                break
            elif self._is_scan_trigger(user_input):
                self._handle_scan_trigger(user_input)
            else:
                self._handle_chat(user_input)

    def _is_scan_trigger(self, message: str) -> bool:
        """
        Detect if message should trigger a scan.

        Returns True if:
          - Message contains an IPv4 address AND a trigger word, OR
          - Message contains an exact trigger phrase
        """
        msg_lower = message.lower()

        # Exact phrase triggers
        exact_phrases = [
            "start scan",
            "begin scan",
            "run scan",
            "start pentest",
            "find vulnerabilities",
            "find vulns",
            "start hacking",
            "run kira",
            "start kira",
            "begin pentest",
        ]
        for phrase in exact_phrases:
            if phrase in msg_lower:
                return True

        # IPv4 + trigger word pattern
        ip_pattern = r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        trigger_words = [
            "start",
            "scan",
            "hack",
            "test",
            "pentest",
            "find",
            "attack",
            "enumerate",
            "exploit",
            "recon",
            "run",
            "begin",
            "check",
            "target",
        ]

        ip_match = re.search(ip_pattern, message)
        if ip_match:
            has_trigger = any(word in msg_lower for word in trigger_words)
            if has_trigger:
                return True

        return False

    def _extract_ip(self, message: str) -> Optional[str]:
        """
        Extract first IPv4 address from message.
        Returns None if no IP found.
        """
        ip_pattern = r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
        match = re.search(ip_pattern, message)
        return match.group(0) if match else None

    def _build_chat_prompt(self, user_message: str) -> str:
        """
        Build the full user-turn content for a chat call.
        Includes session context summary + user message.
        """
        context = self.state.get_context_summary()
        separator = "\n" + "=" * 50 + "\nUSER QUESTION:\n"
        return context + separator + user_message

    def _handle_chat(self, message: str) -> None:
        """
        Chat mode: Call LLM with user question + state context.
        Print response as plain text.
        """
        full_prompt = self._build_chat_prompt(message)

        try:
            # Use generate_text for free-form mode instead of ask() (which is JSON mode)
            response = self.llm.generate_text(
                prompt=full_prompt,
                temperature=0.3,
                max_tokens=300,
            )

            if response:
                print(f"{C.GREEN}[KIRA]{C.RESET} {response}\n")
            else:
                print(f"{C.YELLOW}[KIRA]{C.RESET} I couldn't generate a response. Try again.\n")

        except Exception as e:
            print(f"{C.YELLOW}[KIRA]{C.RESET} Error: {e}\n")

    def _handle_scan_trigger(self, message: str) -> None:
        """
        Scan mode: Extract IP, update target if needed, run planner, return to chat.
        """
        extracted_ip = self._extract_ip(message)

        if extracted_ip:
            current_target = self.state.target
            if current_target != extracted_ip:
                # Re-init state for new target
                print(
                    f"{C.YELLOW}[KIRA]{C.RESET} Target changed from "
                    f"{current_target} to {extracted_ip}. Resetting state.\n"
                )
                self.state.init(target=extracted_ip, authorized_by=self.state.get("authorized_by"))
            target = extracted_ip
        else:
            # Use current state target
            target = self.state.target
            if not target:
                print(
                    f"{C.YELLOW}[KIRA]{C.RESET} No IP found in message. "
                    f"Please provide a target IP (e.g., 'start scan on 10.10.10.5').\n"
                )
                return

        # Confirmation
        print(f"{C.GREEN}[KIRA]{C.RESET} Understood. Starting autonomous scan on {target}...")
        print(f"{C.DIM}       You can interrupt with Ctrl+C at any time.{C.RESET}\n")

        # Run the planner synchronously
        try:
            outcome = self.planner.run(max_iterations=self.max_iter)
        except KeyboardInterrupt:
            outcome = "INTERRUPTED"
            print(f"\n{C.YELLOW}[KIRA]{C.RESET} Interrupted by user.\n")

        # Print session summary
        print(f"\n{C.GREEN}[KIRA]{C.RESET} Session ended: {outcome}\n")
        self._print_session_summary()

        # Return to chat mode
        print()

    def _print_session_summary(self) -> None:
        """Print a compact summary of the session after scan completes."""
        findings = self.state.get("findings") or []
        ports = self.state.get("open_ports") or []
        sessions = self.state.get("sessions") or []
        is_root = self.state.get("is_root", False)
        phase = self.state.get("phase", "?")

        sev_counts: dict[str, int] = {}
        for f in findings:
            sev = f.get("severity", "info")
            sev_counts[sev] = sev_counts.get(sev, 0) + 1

        print(f"  Phase          {phase}")
        print(f"  Root obtained  {'YES ✓' if is_root else 'no'}")
        print(f"  Sessions open  {len(sessions)}")
        print(f"  Open ports     {len(ports)}")
        print(f"  Findings       {len(findings)}  " + "  ".join(
            f"{k}:{v}" for k, v in sev_counts.items()
        ))

        if findings:
            print(f"\n  {C.BOLD}Top findings:{C.RESET}")
            for f in sorted(findings, key=lambda x: x.get("cvss", 0), reverse=True)[:5]:
                sev = f.get("severity", "info").upper()
                print(
                    f"    [{sev:8s}] CVSS {f.get('cvss', '?'):<4}  "
                    f"port {f.get('port', '?'):<5}  {f.get('title', 'untitled')}"
                )

    def _print_welcome(self) -> None:
        """Print welcome banner."""
        target = self.state.target or "(none)"
        print()
        print(f"{C.BOLD}┌{'─' * 57}┐{C.RESET}")
        print(f"{C.BOLD}│  Kira v0.3.0 — Autonomous Pentest Agent{' ' * 16}│{C.RESET}")
        print(f"{C.BOLD}│  Type a target IP + trigger word to start scanning.{' ' * 3}│{C.RESET}")
        print(f"{C.BOLD}│  Ask me anything about pentesting. Type 'exit' to quit. │{C.RESET}")
        print(f"{C.BOLD}└{'─' * 57}┘{C.RESET}")
        print()

    def _print_goodbye(self) -> None:
        """Print goodbye message."""
        print(f"{C.GREEN}[KIRA]{C.RESET} Session ended. Stay authorized.\n")
