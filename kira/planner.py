"""
kira/planner.py — Planner
==========================
The brain of Kira. Runs the observe → think → act loop, dispatching
each LLM-chosen action to the correct executor, updating state, and
auto-advancing phases when completion criteria are met.
"""

from __future__ import annotations

import json
import time
from typing import Optional


# ── Phase completion criteria ─────────────────────────────────────────────────

PHASE_COMPLETION = {
    "RECON":        ["open_ports"],
    "ENUM":         ["open_ports"],   # ENUM completes when services enumerated; findings not required
    "VULN_SCAN":    ["findings"],
    "EXPLOIT":      ["sessions"],
    "POST_EXPLOIT": ["current_user"],
    "REPORT":       [],   # terminal — always complete
    "DONE":         [],
}

# Minimum findings required before REPORT is valid
MIN_FINDINGS_FOR_REPORT = 0


class PhaseController:
    """
    Tracks per-phase context focus and completion logic.
    Injected into the LLM prompt alongside the state summary.
    """

    PHASE_FOCUS = {
        "RECON": (
            "Run nmap_scan with no ports argument to trigger a full 65535-port sweep "
            "followed by a targeted version scan on discovered ports. "
            "Goal: find every open port and its service version."
        ),
        "ENUM": (
            "Enumerate each discovered service. For HTTP services, run tools in this order: "
            "curl_probe (get banner), whatweb (fingerprint), gobuster_dir (find paths), "
            "then searchsploit for each service version found. "
            "IMPORTANT: Always include the correct port in URLs (e.g. http://IP:8080/). "
            "After enumeration, use add_finding to record any vulnerabilities, "
            "then advance_phase to move to VULN_SCAN."
        ),
        "VULN_SCAN": (
            "Cross-reference every service version with searchsploit. "
            "For EACH service in the services list, call searchsploit with the exact "
            "version string (e.g. 'Apache httpd 2.4.25', 'OpenSSH 7.4'). "
            "After each searchsploit call, findings are auto-added. "
            "Once all services are checked, call advance_phase. "
            "Goal: build a prioritised list of exploitable vulnerabilities."
        ),
        "EXPLOIT": (
            "Attempt exploitation. FIRST call msf_search with the service name "
            "(e.g. 'apache') to get real Metasploit module names. "
            "Then call msf_exploit with a module from those results — NEVER invent module names. "
            "Set RPORT to the correct port (e.g. 8080). "
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
        required = PHASE_COMPLETION.get(self._state.phase, [])
        for key in required:
            if not self._state.get(key):
                return False
        # For ENUM: also require that at least one enumeration tool has run
        # AND searchsploit has been attempted for service version info
        if self._state.phase == "ENUM":
            enum_tools = {"searchsploit", "curl_probe", "whatweb", "enum4linux", "gobuster_dir"}
            actions = self._state.get("actions_taken") or []
            ran = {a.get("tool") for a in actions}
            if not ran.intersection(enum_tools):
                return False
            # Don't advance until searchsploit has been tried
            if "searchsploit" not in ran:
                return False
        return True

    def context_with_focus(self) -> str:
        summary = self._state.get_context_summary()
        focus = self.focus()
        return f"{summary}\n\nCURRENT FOCUS: {focus}" if focus else summary


class Planner:
    """
    Observe → Think → Act agent loop for Kira.
    """

    MAX_SAME_ACTION = 3

    def __init__(
        self,
        state,
        runner,
        llm,
        msf=None,
        kb=None,
        verbose: bool = True,
        logger=None,
        guard=None,
    ):
        self._state = state
        self._runner = runner
        self._llm = llm
        self._msf = msf
        self._kb = kb
        self._verbose = verbose
        self._logger = logger
        self._guard = guard
        self._phase_ctrl = PhaseController(state)
        self._action_history: list[str] = []

    # Ordered steps the planner enforces in ENUM regardless of LLM choice
    ENUM_SEQUENCE = ["curl_probe", "whatweb", "searchsploit", "gobuster_dir"]

    def run(self, max_iterations: int = 10) -> str:
        if self._verbose:
            self._print_banner()

        for iteration in range(1, max_iterations + 1):
            if self._verbose:
                self._print_iter_header(iteration, max_iterations)

            context = self._phase_ctrl.context_with_focus()
            action = self._llm.next_action(context_summary=context, phase=self._state.phase)

            tool = action.get("tool", "HALT")
            args = action.get("args", {}) or {}
            reasoning = action.get("reasoning", "")

            # ── Programmatic ENUM sequencer ───────────────────────────────
            # Override LLM if it picks a tool already done or loops on curl_probe
            if self._state.phase == "ENUM":
                forced = self._next_enum_step()
                if forced and forced != tool:
                    if self._verbose:
                        self._print_info(f"ENUM sequencer: overriding '{tool}' → '{forced}'")
                    tool = forced
                    args = self._default_enum_args(forced)
                    action = {"tool": tool, "args": args, "reasoning": f"ENUM sequencer: running {forced}"}

            if self._verbose:
                self._print_action(tool, args, reasoning)

            if self._anti_loop_check(tool, args):
                msg = (
                    f"Anti-loop guard triggered: '{tool}' with identical args "
                    f"seen {self.MAX_SAME_ACTION} times consecutively."
                )
                self._state.log_action("ANTI_LOOP", {}, msg)
                if self._verbose:
                    self._print_warn(msg)
                # Instead of halting, inject a recovery note and continue
                self._state.add_note(
                    f"LOOP DETECTED: Do NOT call '{tool}' again with the same args. "
                    f"Try a different tool or advance_phase."
                )
                self._action_history.clear()
                continue

            if tool == "HALT":
                self._state.log_action("HALT", args, reasoning)
                if self._verbose:
                    self._print_info(f"Agent halted: {reasoning}")
                return "HALTED"

            if tool == "REPORT":
                findings = self._state.get("findings", [])
                if len(findings) < MIN_FINDINGS_FOR_REPORT:
                    self._state.log_action(
                        "REPORT_REJECTED",
                        {},
                        "REPORT requested but no findings in state yet. Continue enumeration.",
                    )
                    if self._verbose:
                        self._print_warn("REPORT rejected — no findings yet. Continuing...")
                    continue
                self._state.log_action("REPORT", {}, "Generating final report.")
                self._state.update(phase="REPORT")
                if self._verbose:
                    self._print_info("REPORT action received — exiting loop.")
                return "DONE"

            result_summary = self._dispatch(action)

            # Auto-inject msf_search at start of EXPLOIT if LLM skips it
            if (self._state.phase == "EXPLOIT"
                    and tool == "msf_exploit"
                    and self._msf is not None):
                actions = self._state.get("actions_taken") or []
                has_searched = any(a.get("tool") == "msf_search" for a in actions)
                if not has_searched:
                    # Run msf_search automatically before the exploit attempt
                    services = dict(self._state.get("services") or {})
                    query = list(services.values())[0].split()[0].lower() if services else "apache"
                    search_result = self._do_msf_search({"query": query})
                    self._state.log_action("msf_search", {"query": query}, search_result)
                    if self._verbose:
                        self._print_info(f"Auto msf_search: {search_result}")
                    # Now re-validate the module the LLM picked
                    result_summary = self._dispatch(action)

            if self._logger:
                self._logger.action(
                    tool=tool,
                    args=args,
                    result={"ok": not result_summary.startswith(("BLOCKED", "FAILED")), "summary": result_summary},
                    elapsed_s=0.0,
                )

            self._state.log_action(tool, args, result_summary)
            self._sync_kb_to_state()

            if self._verbose:
                self._print_result(result_summary)

            exit_reason = self._check_phase_gate()
            if exit_reason:
                return exit_reason

            time.sleep(0.5)

        self._state.log_action("MAX_ITER", {}, f"Loop terminated after {max_iterations} iterations.")
        if self._verbose:
            self._print_warn(f"Max iterations ({max_iterations}) reached.")
        return "MAX_ITER"

    def _dispatch(self, action: dict) -> str:
        if self._guard is not None:
            allowed, reason = self._guard.check_action(action)
            if not allowed:
                self._state.log_error(action.get("tool", "unknown"), reason)
                return f"BLOCKED by guardrail: {reason[:120]}"

        tool = action.get("tool", "")
        args = action.get("args", {}) or {}
        target = self._state.target or ""

        try:
            if tool == "nmap_scan":
                return self._do_nmap(args, target)
            if tool == "gobuster_dir":
                return self._do_gobuster(args)
            if tool == "searchsploit":
                return self._do_searchsploit(args)
            if tool == "enum4linux":
                return self._do_enum4linux(args, target)
            if tool == "curl_probe":
                return self._do_curl(args)
            if tool == "whatweb":
                return self._do_whatweb(args)
            if tool == "msf_search":
                return self._do_msf_search(args)
            if tool == "msf_exploit":
                return self._do_msf_exploit(args)
            if tool == "shell_cmd":
                return self._do_shell_cmd(args)
            if tool == "linpeas":
                return self._do_linpeas(args)
            if tool == "add_finding":
                return self._do_add_finding(args)
            if tool == "add_note":
                return self._do_add_note(args)
            if tool == "advance_phase":
                old = self._state.phase
                new = self._state.advance_phase()
                return f"Phase advanced: {old} → {new}"
            return f"Unknown tool dispatched: '{tool}'"
        except Exception as exc:
            msg = f"Dispatch error for '{tool}': {exc}"
            self._state.log_error(tool, msg)
            return msg

    # ── Tool implementations ─────────────────────────────────────────────────

    def _do_nmap(self, args: dict, target: str) -> str:
        tgt = args.get("target", target)
        flags = args.get("flags", "-sV -sC")
        ports_arg = _normalize_ports_arg(args.get("ports"))

        # ── Two-stage RECON ───────────────────────────────────────────────
        # Stage 1: fast SYN scan across all 65535 ports (no version detection)
        # Stage 2: targeted -sV -sC only on the open ports found
        # Skip stage 1 if caller explicitly passed a port list or we already
        # have open ports in state (e.g. re-running nmap in a later phase).
        existing_ports = list(self._state.get("open_ports") or [])
        run_full_scan = (
            not ports_arg
            and not existing_ports
            and self._state.phase == "RECON"
        )

        if run_full_scan:
            if self._verbose:
                self._print_info("Stage 1: full port sweep (all 65535 ports, no version)...")
            sweep = self._runner.nmap(
                target=tgt,
                flags="-sS -T4 --min-rate 5000 --open",
                ports=NMAP_FULL_PORTS,
                timeout=300,
            )
            discovered = []
            if sweep.ok and sweep.artifact_path:
                try:
                    from kira.parsers.nmap_parser import parse_nmap_xml, extract_state_fields
                    parsed = parse_nmap_xml(sweep.artifact_path)
                    fields = extract_state_fields(parsed)
                    discovered = fields.get("open_ports", [])
                except Exception:
                    pass

            if not discovered:
                # Fallback: parse from stdout if XML failed
                import re
                for line in (sweep.stdout or "").splitlines():
                    m = re.match(r"(\d+)/tcp\s+open", line)
                    if m:
                        discovered.append(int(m.group(1)))

            if not discovered:
                return f"Stage 1 sweep found no open ports on {tgt}. rc={sweep.returncode}"

            ports_arg = ",".join(str(p) for p in sorted(discovered))
            if self._verbose:
                self._print_info(f"Stage 1 found {len(discovered)} open ports: {discovered}")
                self._print_info(f"Stage 2: version scan on ports {ports_arg}...")

        # Stage 2 (or single-stage if ports were specified)
        use_ports = ports_arg or NMAP_HEAVY_PORTS
        result = self._runner.nmap(target=tgt, flags=flags, ports=use_ports)
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
                if self._kb:
                    for f in get_notable_script_findings(parsed):
                        self._kb.add_from_dict(f)
                return (
                    f"Found {len(fields.get('open_ports', []))} open ports: "
                    f"{fields.get('open_ports', [])}. "
                    f"Services: {list(fields.get('services', {}).values())[:5]}"
                )
            except Exception as e:
                return f"nmap OK but parse failed: {e}. raw: {result.summary}"
        return result.summary

    def _do_gobuster(self, args: dict) -> str:
        raw_url = args.get("url") or _default_http_url(self._state)
        url = _normalize_http_tool_url(raw_url, self._state)
        wordlist = args.get("wordlist", "/usr/share/wordlists/dirb/common.txt")
        # Try a different wordlist if the default one already returned 0 results
        actions = self._state.get("actions_taken") or []
        prior_gobuster = [
            a for a in actions
            if a.get("tool") == "gobuster_dir"
            and "Found 0 paths" in (a.get("result_summary") or "")
        ]
        if prior_gobuster and wordlist == "/usr/share/wordlists/dirb/common.txt":
            wordlist = "/usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt"
        result = self._runner.gobuster(url=url, wordlist=wordlist)
        if result.ok:
            try:
                from kira.parsers.gobuster_parser import GobusterParser
                parsed = GobusterParser(raw=result.stdout, base_url=url, port=_url_to_port(url) or 80).parse()
                paths = list(parsed.all_paths or [])
                self._state.update(web_paths=paths)
                if self._kb:
                    # gobuster_parser may return Finding objects; store as dicts when possible
                    for f in (parsed.findings or []):
                        self._kb.add_from_dict(getattr(f, "to_dict", lambda: f)())
                juicy = list(parsed.juicy_paths or [])
                return f"Found {len(paths)} paths. Juicy: {juicy[:5] or 'none'}"
            except Exception as e:
                return f"gobuster OK but parse failed: {e}"
        return result.summary

    def _do_searchsploit(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return "searchsploit skipped — no query provided"

        # Normalize query: "Apache httpd 2.4.25" → try exact, then fallback to shorter
        queries_to_try = [query]
        parts = query.split()
        if len(parts) >= 2:
            # e.g. "Apache httpd 2.4.25" → "Apache 2.4.25", "apache 2.4"
            queries_to_try.append(f"{parts[0]} {parts[-1]}")
            version = parts[-1]
            if "." in version:
                short_ver = ".".join(version.split(".")[:2])
                queries_to_try.append(f"{parts[0]} {short_ver}")

        raw_results = []
        used_query = query
        for q in queries_to_try:
            result = self._runner.searchsploit(q)
            if result.ok and result.stdout:
                try:
                    from kira.parsers.vuln_scanner import parse_searchsploit_json
                    raw_results = parse_searchsploit_json(result.stdout)
                    if raw_results:
                        used_query = q
                        break
                except Exception as e:
                    return f"searchsploit parse failed: {e}"

        if not raw_results:
            return f"searchsploit '{query}': 0 results"

        # Determine port from state services matching the query
        port = 0
        services = dict(self._state.get("services") or {})
        for p_str, svc in services.items():
            if any(word.lower() in str(svc).lower() for word in query.split()):
                try:
                    port = int(p_str)
                except ValueError:
                    pass
                break

        type_cvss = {"remote": 9.0, "webapps": 8.5, "local": 7.0,
                     "dos": 5.0, "shellcode": 8.0, "papers": 0.0}

        added = 0
        for r in raw_results[:10]:
            edb_type = r.get("type", "").lower()
            cvss = type_cvss.get(edb_type, 5.0)
            severity = (
                "critical" if cvss >= 9.0 else
                "high"     if cvss >= 7.0 else
                "medium"   if cvss >= 4.0 else
                "low"
            )
            finding = {
                "title": r.get("title", query),
                "severity": severity,
                "port": port,
                "service": query.split()[0] if query else "",
                "cvss": cvss,
                "cve": r.get("cve") or "",
                "exploit_available": edb_type in ("remote", "webapps", "shellcode"),
                "description": f"EDB-ID {r.get('edb_id','?')}: {r.get('title','')}",
                "remediation": "Apply vendor patch or upgrade to latest version.",
            }
            if self._kb:
                self._kb.add_from_dict(finding)
            else:
                self._state.add_finding(finding)
            added += 1

        self._sync_kb_to_state()
        return f"searchsploit '{used_query}': {len(raw_results)} results, {added} findings added"

    def _do_enum4linux(self, args: dict, target: str) -> str:
        tgt = args.get("target", target)
        flags = args.get("flags", "-a")
        result = self._runner.enum4linux(target=tgt, flags=flags)
        return result.summary

    def _do_curl(self, args: dict) -> str:
        url = _normalize_http_tool_url(args.get("url") or _default_http_url(self._state), self._state)
        flags = args.get("flags", "-sIL --max-time 10")
        # Always add --max-time if not present to avoid rc=28 timeouts
        if "--max-time" not in flags:
            flags = flags + " --max-time 10"
        result = self._runner.curl(url=url, flags=flags)
        if result.ok:
            for line in result.stdout.splitlines():
                if line.lower().startswith("server:"):
                    server = line.split(":", 1)[1].strip()
                    port = _url_to_port(url)
                    svcs = dict(self._state.get("services") or {})
                    if port and str(port) not in svcs:
                        svcs[str(port)] = server
                        self._state.update(services=svcs)
                    break
        return result.summary

    def _do_whatweb(self, args: dict) -> str:
        url = _normalize_http_tool_url(args.get("url") or _default_http_url(self._state), self._state)
        result = self._runner.whatweb(url=url, timeout=60)
        return result.summary

    def _do_msf_search(self, args: dict) -> str:
        """Search Metasploit for real module names matching a query."""
        query = args.get("query", "")
        if not query:
            return "msf_search: no query provided"
        results = self._msf_search(query)
        if not results:
            return f"msf_search '{query}': no modules found"
        names = [r["module"] for r in results[:8]]
        # Store in notes so LLM can see them
        self._state.add_note(f"MSF modules for '{query}': {', '.join(names)}")
        return f"msf_search '{query}': found {len(results)} modules — {names[:5]}"

    def _do_msf_exploit(self, args: dict) -> str:
        if self._msf is None:
            return "Metasploit RPC not connected."

        module = args.get("module", "")
        options = args.get("options", {}) or {}
        target = self._state.target or ""

        if not options.get("RHOSTS"):
            options["RHOSTS"] = target

        # If no module specified, search for one based on known services
        if not module:
            services = dict(self._state.get("services") or {})
            for svc in services.values():
                query = svc.split()[0].lower() if svc else ""
                if query:
                    break
            else:
                return "msf_exploit: no module specified and no services to search"
            results = self._msf_search(query)
            if not results:
                return f"msf_exploit: no modules found for '{query}'"
            module = results[0]["module"]

        # Validate module exists via MSF search before running
        mod_short = module.split("/")[-1] if "/" in module else module
        found = self._msf_search(mod_short)
        valid_modules = [r["module"] for r in found]

        # Check if the exact module path exists
        full_path = module.lstrip("exploit/").lstrip("exploits/")
        module_exists = any(
            full_path in r["module"] or module in r["module"]
            for r in found
        )

        if not module_exists and found:
            # Use the best real match instead of the hallucinated one
            module = found[0]["module"]
            if not module.startswith("exploit"):
                module = f"exploit/{module}"
            self._state.add_note(f"Module substituted: using {module} (original not found)")

        elif not module_exists:
            return (
                f"msf_exploit: module '{module}' not found in Metasploit. "
                f"Use msf_search first to find a valid module name."
            )

        try:
            # Use the kira MSFClient wrapper if available (has run_module)
            if hasattr(self._msf, 'run_module'):
                result = self._msf.run_module(module=module, options=options)
                if result.get("success") and result.get("session_id"):
                    sid = result["session_id"]
                    sessions = list(self._state.get("sessions") or [])
                    sessions.append({"id": sid, "type": "shell", "via": module})
                    self._state.update(sessions=sessions)
                    return f"Session {sid} opened via {module}"
                return f"msf_exploit ran '{module}': {result.get('output') or result.get('error') or 'no session'}"
            else:
                # Raw pymetasploit3 client
                if hasattr(self._runner, "msf_exploit"):
                    return self._runner.msf_exploit(args).summary
                return "msf_exploit: runner has no msf_exploit method"
        except Exception as e:
            return f"msf_exploit error: {e}"

    def _msf_search(self, query: str) -> list[dict]:
        """Search MSF for modules matching query. Returns list of dicts."""
        if self._msf is None:
            return []
        try:
            if hasattr(self._msf, 'search'):
                return self._msf.search(query) or []
            # Raw pymetasploit3 — search via modules list
            results = []
            for mtype in ["exploits", "auxiliary"]:
                try:
                    mods = self._msf.modules.list(mtype)
                    for name in mods:
                        if query.lower() in name.lower():
                            results.append({"module": f"{mtype.rstrip('s')}/{name}", "type": mtype})
                except Exception:
                    continue
            return results[:10]
        except Exception:
            return []

    def _do_shell_cmd(self, args: dict) -> str:
        if self._msf is None:
            return "shell_cmd: Metasploit RPC not connected."
        cmd = args.get("cmd", "id")
        session_id = int(args.get("session_id", 1))
        result = self._runner.shell_cmd(cmd=cmd, session_id=session_id)
        return result.summary

    def _do_linpeas(self, args: dict) -> str:
        if self._msf is None:
            return "linpeas: Metasploit RPC not connected."
        session_id = int(args.get("session_id", 1))
        result = self._runner.linpeas(session_id=session_id)
        return result.summary

    def _do_add_finding(self, args: dict) -> str:
        required = ("title", "severity", "description")
        missing = [k for k in required if not args.get(k)]
        if missing:
            return f"add_finding skipped — missing fields: {missing}"
        finding = {
            "title": args.get("title", ""),
            "severity": args.get("severity", "info"),
            "port": int(args.get("port", 0)),
            "service": args.get("service", ""),
            "cvss": float(args.get("cvss", 0.0)),
            "cve": args.get("cve", ""),
            "exploit_available": bool(args.get("exploit_available", False)),
            "description": args.get("description", ""),
            "remediation": args.get("remediation", ""),
        }
        if self._kb:
            self._kb.add_from_dict(finding)
        else:
            self._state.add_finding(finding)
        return f"Finding added: [{finding['severity'].upper()}] {finding['title']}"

    def _do_add_note(self, args: dict) -> str:
        note = args.get("note", "")
        if not note:
            return "add_note skipped — no note text"
        self._state.add_note(note)
        return f"Note saved: {note[:100]}"

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _next_enum_step(self) -> Optional[str]:
        """
        Return the next ENUM tool that hasn't been successfully run yet.
        Returns None if all steps are done (let LLM decide).
        """
        actions = self._state.get("actions_taken") or []
        ran_ok = {
            a.get("tool") for a in actions
            if not (a.get("result_summary") or "").startswith(("FAILED", "TIMEOUT", "BLOCKED"))
        }
        for step in self.ENUM_SEQUENCE:
            if step not in ran_ok:
                return step
        return None  # all steps done

    def _default_enum_args(self, tool: str) -> dict:
        """Return sensible default args for each ENUM tool."""
        target = self._state.target or ""
        open_ports = list(self._state.get("open_ports") or [])
        # Pick the first HTTP port
        http_port = next((p for p in (8080, 80, 8443, 443, 8000, 8090) if p in open_ports), 80)
        scheme = "https" if http_port in (443, 8443) else "http"
        base_url = f"{scheme}://{target}:{http_port}/" if http_port not in (80, 443) else f"{scheme}://{target}/"

        services = dict(self._state.get("services") or {})
        # Build searchsploit query from first service version
        ss_query = "apache"
        for svc in services.values():
            if svc:
                parts = svc.split()
                # "Apache httpd 2.4.25" → "apache 2.4"
                name = parts[0].lower()
                ver = parts[-1] if len(parts) > 1 else ""
                short_ver = ".".join(ver.split(".")[:2]) if "." in ver else ver
                ss_query = f"{name} {short_ver}".strip()
                break

        defaults = {
            "curl_probe":   {"url": base_url, "flags": "-sIL --max-time 10"},
            "whatweb":      {"url": base_url},
            "searchsploit": {"query": ss_query},
            "gobuster_dir": {"url": base_url, "wordlist": "/usr/share/wordlists/dirb/common.txt"},
        }
        return defaults.get(tool, {})

    def _anti_loop_check(self, tool: str, args: dict) -> bool:
        # Normalize URLs in args to avoid false negatives (trailing slash differences)
        normalized = {}
        for k, v in args.items():
            if isinstance(v, str) and v.startswith("http"):
                v = v.rstrip("/") + "/"
            normalized[k] = v
        key = f"{tool}:{json.dumps(normalized, sort_keys=True)}"
        self._action_history.append(key)
        tail = self._action_history[-self.MAX_SAME_ACTION:]
        return len(tail) == self.MAX_SAME_ACTION and len(set(tail)) == 1

    def _sync_kb_to_state(self) -> None:
        if self._kb is None:
            return
        try:
            dicts = self._kb.to_state_dicts()
            if dicts:
                self._state.update(findings=dicts)
        except Exception as e:
            # Fallback: write findings directly without KB
            pass

    def _check_phase_gate(self) -> Optional[str]:
        if not self._phase_ctrl.is_phase_complete():
            # Safety: if VULN_SCAN has run searchsploit on all services but found
            # nothing exploitable, still advance rather than loop forever
            if self._state.phase == "VULN_SCAN":
                actions = self._state.get("actions_taken") or []
                ss_runs = [a for a in actions if a.get("tool") == "searchsploit"]
                services = dict(self._state.get("services") or {})
                if len(ss_runs) >= max(len(services), 1):
                    old = self._state.phase
                    new = self._state.advance_phase()
                    if self._verbose:
                        self._print_info(f"Phase gate (no findings): {old} → {new}")
                    return None
            return None
        current = self._state.phase
        # VULN_SCAN: only advance if we actually ran searchsploit in THIS phase
        if current == "VULN_SCAN":
            actions = self._state.get("actions_taken") or []
            vuln_scan_entry = next(
                (i for i, a in enumerate(actions) if a.get("tool") == "advance_phase"
                 or (a.get("result_summary", "").startswith("Phase advanced") and "ENUM" in a.get("result_summary", ""))),
                0
            )
            ss_in_vuln = [
                a for a in actions[vuln_scan_entry:]
                if a.get("tool") == "searchsploit"
            ]
            if not ss_in_vuln:
                return None  # don't advance yet, let VULN_SCAN do its work
        if current == "POST_EXPLOIT":
            self._state.update(phase="REPORT")
            return "DONE"
        if current in ("REPORT", "DONE"):
            return "DONE"
        old = current
        new = self._state.advance_phase()
        if self._verbose:
            self._print_info(f"Phase gate: {old} → {new}")
        return None

    # ── Console output ───────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        try:
            from rich.console import Console
            from rich.panel import Panel
            Console().print(
                Panel(
                    f"[bold green]Kira[/bold green] — target: [cyan]{self._state.target}[/cyan]  "
                    f"phase: [yellow]{self._state.phase}[/yellow]",
                    title="Agent loop started",
                    border_style="dim",
                )
            )
        except Exception:
            print(f"\n[KIRA] target={self._state.target}  phase={self._state.phase}")

    def _print_iter_header(self, i: int, total: int) -> None:
        try:
            from rich.console import Console
            Console().print(f"\n[dim]─── iteration {i}/{total} │ phase: {self._state.phase} ───[/dim]")
        except Exception:
            print(f"\n--- iter {i}/{total}  phase={self._state.phase} ---")

    def _print_action(self, tool: str, args: dict, reasoning: str) -> None:
        try:
            from rich.console import Console
            c = Console()
            c.print(f"  [bold cyan]THINK[/bold cyan]  tool=[green]{tool}[/green]  args={args}")
            c.print(f"  [dim]reason: {reasoning[:120]}[/dim]")
        except Exception:
            print(f"  THINK  tool={tool}  args={args}\n  reason: {reasoning[:120]}")

    def _print_result(self, summary: str) -> None:
        try:
            from rich.console import Console
            Console().print(f"  [bold]RESULT[/bold]  {summary[:160]}")
        except Exception:
            print(f"  RESULT  {summary[:160]}")

    def _print_info(self, msg: str) -> None:
        try:
            from rich.console import Console
            Console().print(f"  [bold green]INFO[/bold green]   {msg}")
        except Exception:
            print(f"  INFO   {msg}")

    def _print_warn(self, msg: str) -> None:
        try:
            from rich.console import Console
            Console().print(f"  [bold yellow]WARN[/bold yellow]   {msg}")
        except Exception:
            print(f"  WARN   {msg}")


# ── URL helpers ───────────────────────────────────────────────────────────────

def _url_to_port(url: str) -> Optional[int]:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.port:
            return parsed.port
        return 443 if parsed.scheme == "https" else 80
    except Exception:
        return None


def _normalize_http_tool_url(url: str, state) -> str:
    from urllib.parse import urlparse, urlunparse

    raw = (url or "").strip()
    if not raw:
        return raw
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return raw

    auth = ""
    hostpart = parsed.netloc
    if "@" in hostpart:
        auth, hostpart = hostpart.split("@", 1)
        auth = auth + "@"
    host = hostpart.split(":")[0]
    if not host:
        return raw

    open_ports = set(state.get("open_ports") or [])

    # Fix http://host/8080(/path) → http://host:8080(/path)
    if parsed.port is None and parsed.path not in ("", "/"):
        rest = parsed.path.lstrip("/")
        if rest:
            first, _, remainder = rest.partition("/")
            if first.isdigit():
                port = int(first)
                if 1 <= port <= 65535:
                    new_path = "/" + remainder if remainder else "/"
                    netloc = f"{auth}{host}:{port}"
                    raw = urlunparse((parsed.scheme, netloc, new_path, "", parsed.query, parsed.fragment))
                    parsed = urlparse(raw)

    cur = _url_to_port(raw)
    if not open_ports or cur in open_ports:
        return raw

    preference = (80, 443, 8080, 8443, 8000, 8888, 8090)
    pick = next((hp for hp in preference if hp in open_ports), None)
    if pick is None:
        services = dict(state.get("services") or {})
        for p_str, svc in services.items():
            if not any(k in str(svc).lower() for k in ("http", "web", "apache", "nginx", "iis", "httpd")):
                continue
            try:
                pi = int(p_str)
            except ValueError:
                continue
            if pi in open_ports:
                pick = pi
                break
    if pick is None:
        return raw

    scheme = "https" if pick in (443, 8443) else "http"
    netloc = f"{auth}{host}" if pick in (80, 443) else f"{auth}{host}:{pick}"
    pth = parsed.path if parsed.path else "/"
    return urlunparse((scheme, netloc, pth, "", parsed.query, parsed.fragment))


NMAP_HEAVY_PORTS = "22,25,53,80,443,3128,4443,4444,8090,8443"
NMAP_FULL_PORTS  = "-"   # all 65535 ports


def _normalize_ports_arg(ports: Optional[str]) -> Optional[str]:
    if not ports:
        return None
    p = str(ports).strip()
    if p.startswith("-p "):
        return p[3:].strip()
    if p.startswith("-p"):
        return p[2:].strip()
    return p


def _default_http_url(state) -> str:
    target = state.target or "127.0.0.1"
    open_ports = list(state.get("open_ports") or [])
    for p in (80, 443, 8080, 8443, 8000, 8888, 55553):
        if p in open_ports:
            scheme = "https" if p == 443 else "http"
            return f"{scheme}://{target}" if p in (80, 443) else f"{scheme}://{target}:{p}"

    services = dict(state.get("services") or {})
    for p_str, svc in services.items():
        svc_l = str(svc).lower()
        if "http" in svc_l or "web" in svc_l:
            try:
                p = int(p_str)
            except Exception:
                continue
            scheme = "https" if p == 443 else "http"
            return f"{scheme}://{target}" if p in (80, 443) else f"{scheme}://{target}:{p}"

    return f"http://{target}"