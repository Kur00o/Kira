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
        session_dir=None,
        log=None,
        no_report: bool = False,
    ):
        self.planner    = planner
        self.state      = state
        self.llm        = llm
        self.max_iter   = max_iter
        self.verbose    = verbose
        self.session_dir = session_dir
        self.log        = log
        self.no_report  = no_report
        self._last_report_path: str = None

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
            elif self._is_report_request(user_input):
                self._handle_report_request()
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

    def _extract_iterations(self, message: str) -> Optional[int]:
        """
        Extract iteration count from message.
        Handles: 'iterations = 50', 'iterations=50', '50 iterations', 'iter 50'
        """
        import re
        patterns = [
            r'iterations?\s*[=:]\s*(\d+)',
            r'(\d+)\s*iterations?',
            r'iter\s*[=:]\s*(\d+)',
            r'(\d+)\s*iter\b',
            r'max[_\s]iter\s*[=:]\s*(\d+)',
        ]
        for pat in patterns:
            m = re.search(pat, message, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 1 <= val <= 500:
                    return val
        return None

    def _handle_scan_trigger(self, message: str) -> None:
        """
        Scan mode: Extract IP, update target if needed, run planner, return to chat.
        """
        extracted_ip = self._extract_ip(message)

        if extracted_ip:
            current_target = self.state.target
            if current_target != extracted_ip:
                print(
                    f"{C.YELLOW}[KIRA]{C.RESET} Target changed from "
                    f"{current_target} to {extracted_ip}. Resetting state.\n"
                )
                self.state.init(target=extracted_ip, authorized_by=self.state.get("authorized_by"))
            target = extracted_ip
        else:
            target = self.state.target
            if not target or target == "pending":
                print(
                    f"{C.YELLOW}[KIRA]{C.RESET} No IP found in message. "
                    f"Please provide a target IP (e.g., 'start scan on 10.10.10.5').\n"
                )
                return

        # Install a real ScopeGuard now that we have a valid target
        from kira.guardrails import ScopeGuard
        guard = ScopeGuard(
            authorized_target=target,
            authorized_by=self.state.get("authorized_by") or "authorized",
        )
        self.planner._guard = guard

        # Confirmation + old-style header to match original output
        # Check if user specified a custom iteration count in the message
        custom_iter = self._extract_iterations(message)
        max_iter    = custom_iter if custom_iter else self.max_iter
        if custom_iter:
            print(f"{C.GREEN}[KIRA]{C.RESET} Using {custom_iter} iterations as requested.\n")
        print(f"\n{'─' * 50}")
        print(f"  STARTING AGENT LOOP")
        print(f"{'─' * 50}")
        print(f"\n\033[1m\033[92m[KIRA] Agent loop starting...\033[0m")
        print(f"\033[2m       Target: {target}  |  Max iterations: {max_iter}\033[0m\n")

        # Run the planner synchronously
        try:
            outcome = self.planner.run(max_iterations=max_iter)
        except KeyboardInterrupt:
            outcome = "INTERRUPTED"
            print(f"\n\033[93m[KIRA] Interrupted by user.\033[0m\n")

        # Old-style session complete section
        print(f"\n{'─' * 50}")
        print(f"  SESSION COMPLETE")
        print(f"{'─' * 50}")
        outcome_label = {
            "HALTED":      "HALTED",
            "MAX_ITER":    "MAX_ITER",
            "DONE":        "DONE",
            "INTERRUPTED": "INTERRUPTED",
        }.get(outcome, outcome)
        print(f"Outcome: {outcome_label}")
        print(f"{'─' * 50}")
        self._print_session_summary()
        if outcome in ("HALTED", "MAX_ITER"):
            print(f"  Tip: review {self.state.session_dir}/actions.jsonl to see why the agent stopped.")
        print()

        # Auto-generate report
        if not self.no_report:
            self._generate_report(outcome)

        # Return to chat mode
        print()

    def _generate_report(self, outcome: str = "DONE") -> None:
        """Generate HTML + Markdown report and store the path."""
        findings = self.state.get("findings") or []
        if not findings:
            print(f"{C.YELLOW}[KIRA]{C.RESET} No findings — skipping report generation.\n")
            return

        session_dir = self.session_dir or self.state.session_dir
        print(f"\n{'─' * 50}")
        print(f"  GENERATING REPORT")
        print(f"{'─' * 50}")
        try:
            from kira.reporter import ReportGenerator
            reporter = ReportGenerator(session_dir=str(session_dir), llm=self.llm)
            paths    = reporter.generate()
            self._last_report_path = paths.html
            print(f"  ✓ Markdown : {paths.markdown}")
            print(f"  ✓ HTML     : {paths.html}")
            print(f"\n  Open with: xdg-open {paths.html}\n")
            if self.log:
                self.log.info(f"Report generated: {paths.html}")
        except Exception as e:
            print(f"  ✗ Report generation failed: {e}")
            import traceback; traceback.print_exc()

    def _is_report_request(self, message: str) -> bool:
        """Detect if user is asking to see/open/generate the report."""
        msg = message.lower()
        return any(kw in msg for kw in [
            "report", "html", "open report", "show report",
            "generate report", "view report", "see report",
        ])

    def _handle_report_request(self) -> None:
        """Handle a chat request to open or generate the report."""
        session_dir = self.session_dir or self.state.session_dir
        html_path   = self._last_report_path

        # Check if report already exists on disk
        if not html_path and session_dir:
            candidate = session_dir / "report.html"
            if candidate.exists():
                html_path = str(candidate)

        if html_path:
            import webbrowser
            from pathlib import Path as _Path
            abs_path = _Path(html_path).resolve()
            print(f"{C.GREEN}[KIRA]{C.RESET} Opening report: {abs_path}")
            try:
                webbrowser.open(f"file://{abs_path}")
            except Exception:
                print(f"{C.GREEN}[KIRA]{C.RESET} Run: xdg-open {abs_path}\n")
        else:
            findings = self.state.get("findings") or []
            if findings:
                print(f"{C.GREEN}[KIRA]{C.RESET} Generating report now...")
                self._generate_report()
            else:
                print(f"{C.YELLOW}[KIRA]{C.RESET} No scan has been run yet — no report to show.\n")

    def _print_session_summary(self) -> None:
        """Print session summary matching the original Kira output format exactly."""
        import time
        from kira.tool_runner import ToolRunner as TR

        findings = self.state.get("findings") or []
        ports    = self.state.get("open_ports") or []
        sessions = self.state.get("sessions") or []
        is_root  = self.state.get("is_root", False)
        phase    = self.state.get("phase", "?")

        # Get action log stats
        session_dir = self.state.session_dir
        log_path    = str(session_dir / "actions.jsonl")
        stats       = TR.summarise_action_log(log_path)

        # Elapsed from session start
        started = self.state.get("started_at")
        elapsed = ""
        if started:
            try:
                from datetime import datetime, timezone
                start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                delta    = datetime.now(timezone.utc) - start_dt
                elapsed  = f"{int(delta.total_seconds())}s"
            except Exception:
                elapsed = "?"

        sev_counts: dict[str, int] = {}
        for f in findings:
            sev = f.get("severity", "info")
            sev_counts[sev] = sev_counts.get(sev, 0) + 1

        sev_str = "  ".join(f"{k}:{v}" for k, v in sev_counts.items())

        print(f"{'─' * 50}")
        print(f"  Kira Session Summary")
        print(f"{'─' * 50}")
        print(f"  Target          {self.state.target}")
        print(f"  Final phase     {phase}")
        print(f"  Root obtained   {'YES ✓' if is_root else 'no'}")
        print(f"  Sessions open   {len(sessions)}")
        print(f"  Open ports      {len(ports)}")
        print(f"  Findings        {len(findings)}  {sev_str}")
        print(f"  Actions run     {stats['total_actions']}  "
              f"(ok={stats['successful']}  failed={stats['failed']})")
        print(f"  Tools used      {', '.join(stats['tools_used']) or 'none'}")
        if elapsed:
            print(f"  Elapsed         {elapsed}")
        print(f"  Session dir     {session_dir}")
        print(f"{'─' * 50}")

        if findings:
            print(f"\n  Top findings:")
            for f in sorted(findings, key=lambda x: x.get("cvss", 0), reverse=True)[:5]:
                sev = f.get("severity", "info").upper()
                print(
                    f"  [{sev:8s}] CVSS {f.get('cvss', '?'):<4}  "
                    f"port {f.get('port', '?'):<5}  {f.get('title', 'untitled')}"
                )
        print()

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
