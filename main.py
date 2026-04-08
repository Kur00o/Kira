"""
main.py — Kira Autonomous Penetration Testing Agent
CLI entry point and agent orchestration loop.

Usage:
    python main.py --target 10.10.10.3
    python main.py --target 10.10.10.3 --phase recon
    python main.py --target 10.10.10.3 --output ./reports --verbose
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import shutil
import shlex
from datetime import datetime
from pathlib import Path

from kira.parsers.nmap_parser import NmapParser, NmapResult
from kira.parsers.vuln_scanner import (   # ← Phase 5
    scan_services,
    ToolRunner as VulnToolRunner,
    KnowledgeBase,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

VERSION = "0.1.0"

PHASES = ["recon", "enumeration", "exploitation", "privesc", "report"]

# Add this near the top of main.py, after imports

class C:
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

BANNER = f"""
{C.BLUE}{C.BOLD}  _  ___           
 | |/ (_)_ _ __ _ 
 | ' <| | '_/ _` |
 |_|\_\_|_| \__,_|{C.RESET}
{C.DIM}  Autonomous Pentest Agent v{VERSION}{C.RESET}
{C.YELLOW}  [!] Authorized environments only{C.RESET}
"""
# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────

def setup_logging(output_dir: Path, verbose: bool) -> logging.Logger:
    log_path = output_dir / "kira.log"
    handlers = [logging.FileHandler(log_path, encoding='utf-8')]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))

    # Ensure stdout can handle UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    logging.basicConfig(
        force=True,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("kira")


# ─────────────────────────────────────────────
# State Manager
# ─────────────────────────────────────────────

class StateManager:
    """Persists agent state to disk as JSON between loop iterations."""

    def __init__(self, state_path: Path):
        self.path = state_path
        self.state = self._default_state()
        self._dirty = False
        if state_path.exists():
            self._load()

    def _default_state(self) -> dict:
        return {
            "target": "",
            "phase": "recon",
            "completed_phases": [],
            "findings": {
                "open_ports": [],
                "services": {},          # ← Phase 5: port → version_string map
                "directories": [],
                "vulnerabilities": [],   # ← Phase 5: Finding dicts stored here
                "credentials": [],
                "exploits_attempted": [],
                "flags": [],
            },
            "actions_taken": [],
            "last_action": None,
            "last_action_time": None,
            "scan_files": {},
        }

    def _load(self):
        with open(self.path) as f:
            self.state = json.load(f)
        # Back-compat: ensure services dict exists for older state files
        self.state["findings"].setdefault("services", {})

    def save(self, force: bool = False):
        self._dirty = True
        if force:
            with open(self.path, "w") as f:
                json.dump(self.state, f, indent=2)
            self._dirty = False

    def flush(self, force: bool = False):
        if self._dirty or force:
            with open(self.path, "w") as f:
                json.dump(self.state, f, indent=2)
            self._dirty = False

    def set(self, key: str, value):
        self.state[key] = value
        self.save()

    def get(self, key: str, default=None):
        return self.state.get(key, default)

    def add_finding(self, category: str, finding):
        container = self.state["findings"].get(category)

        if isinstance(container, list):
            container.append(finding)
        elif isinstance(container, dict):
            raise TypeError(f"Cannot append to dict-based category: {category}")

        self.save()

    def set_service(self, port: str, version_string: str):
        """Register a port → version string for vuln_scanner consumption."""
        self.state["findings"]["services"][port] = version_string
        self.save()

    def log_action(self, tool: str, args: dict, result: str, reasoning: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool,
            "args": args,
            "result_summary": result[:300],
            "reasoning": reasoning,
        }
        self.state["actions_taken"].append(entry)
        self.state["last_action"] = tool
        self.state["last_action_time"] = entry["timestamp"]
        self.save()

    def get_context_summary(self) -> str:
        """
        Compact summary injected into every LLM prompt — keep under ~400 tokens.
        """
        s = self.state
        findings = s["findings"]

        lines = [
            f"Target: {s['target']}",
            f"Current phase: {s['phase']}",
            f"Completed phases: {', '.join(s['completed_phases']) or 'none'}",
            f"Last action: {s['last_action'] or 'none'}",
            "",
            f"Open ports ({len(findings['open_ports'])}):",
        ]
        for p in findings["open_ports"][:10]:
            lines.append(f"  {p.get('port')}/{p.get('protocol')} {p.get('service')} {p.get('product','')} {p.get('version','')}".rstrip())

        if findings["directories"]:
            lines.append(f"\nDiscovered paths ({len(findings['directories'])}):")
            for d in findings["directories"][:8]:
                lines.append(f"  {d}")

        if findings["vulnerabilities"]:
            lines.append(f"\nVulnerabilities ({len(findings['vulnerabilities'])}):")
            for v in findings["vulnerabilities"][:5]:
                # Support both old-style dicts and new Finding dicts
                title = v.get("title") or v.get("version_string", "?")
                sev   = v.get("severity") or (
                    f"CVSS {v['cvss_estimate']}" if v.get("cvss_estimate") else "?"
                )
                cve   = f" [{v['cve']}]" if v.get("cve") else ""
                msf   = " [MSF]" if v.get("exploit_available") else ""
                lines.append(f"  [{sev}]{cve}{msf} {title}")

        if s["actions_taken"]:
            last3 = s["actions_taken"][-3:]
            lines.append("\nLast 3 actions:")
            for a in last3:
                lines.append(f"  {a['tool']} — {a['result_summary'][:80]}")

        return "\n".join(lines)

    def advance_phase(self):
        current = self.state["phase"]
        if current not in self.state["completed_phases"]:
            self.state["completed_phases"].append(current)
        idx = PHASES.index(current)
        if idx + 1 < len(PHASES):
            self.state["phase"] = PHASES[idx + 1]
        self.save()


# ─────────────────────────────────────────────
# Tool Executor
# ─────────────────────────────────────────────

class ToolExecutor:
    def __init__(self, output_dir: Path, logger: logging.Logger, timeout: int = 300):
        self.output_dir = output_dir
        self.logger = logger
        self.timeout = timeout

    def run(self, command: list[str], label: str) -> tuple[str, str, int]:
        self.logger.info(f"[EXEC] {' '.join(command)}")
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(timeout=self.timeout)
            if proc.returncode != 0:
                self.logger.warning(f"[{label}] exited {proc.returncode}: {stderr[:200]}")
            return stdout, stderr, proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            self.logger.error(f"[{label}] timed out after {self.timeout}s")
            return "", "timeout", -1
        except FileNotFoundError:
            self.logger.error(f"[{label}] tool not found: {command[0]}")
            return "", f"tool not found: {command[0]}", -1

    def nmap_scan(self, target: str) -> Path:
        xml_out = self.output_dir / "nmap_initial.xml"
        command = [
            "nmap", "-sV", "-sC",
            "-oX", str(xml_out),
            target,
        ]
        self.run(command, "nmap")
        return xml_out

    def nmap_full(self, target: str) -> Path:
        xml_out = self.output_dir / "nmap_full.xml"
        command = [
            "nmap", "-sV", "-p-",
            "--min-rate", "1000",
            "-oX", str(xml_out),
            target,
        ]
        self.run(command, "nmap-full")
        return xml_out

    def gobuster(self, target_url: str, wordlist: str = "/usr/share/wordlists/dirb/common.txt") -> str:
        out_file = self.output_dir / "gobuster.txt"
        command = [
            "gobuster", "dir",
            "-u", target_url,
            "-w", wordlist,
            "-o", str(out_file),
            "-q",
        ]
        stdout, _, _ = self.run(command, "gobuster")
        return str(out_file) if out_file.exists() else stdout


# ─────────────────────────────────────────────
# LLM Planner
# ─────────────────────────────────────────────

class LLMPlanner:
    """
    Calls Gemma 3 4B via Ollama to decide the next action.
    Returns a structured action dict: {tool, args, reasoning}
    """

    TOOL_SCHEMA = """
Available tools:
  nmap_initial   — Fast version/script scan. Args: {target: str}
  nmap_full      — Full port scan (-p-). Args: {target: str}
  gobuster       — Directory brute-force. Args: {url: str, wordlist: str (optional)}
  vuln_scan      — CVE cross-reference via searchsploit. Args: {} (reads services from state)
  metasploit     — Run an MSF module. Args: {module: str, options: dict}
  run_shell      — Run arbitrary shell command. Args: {command: str}
  advance_phase  — Move to next pentest phase. Args: {}
  generate_report — Write final report. Args: {}
  done           — Target fully compromised or no further actions possible. Args: {}
"""

    SYSTEM_PROMPT = """You are Kira, a senior penetration tester AI agent.
You receive the current state of a pentest and decide the single best next action.

Rules:
- Only target the provided IP. Never attack anything else.
- Always reply with ONLY valid JSON — no prose, no markdown fences.
- Format: {"tool": "<name>", "args": {<key>: <value>}, "reasoning": "<1 sentence>"}
- Be methodical: recon → enumeration → exploitation → privesc → report.
- After nmap scans populate services, run vuln_scan before attempting exploitation.
- Do not repeat an action already taken unless you have new information.
- If you have enough recon data, call advance_phase to move forward.
""" + TOOL_SCHEMA

    def __init__(self, host: str = "http://localhost:11434", model: str = "gemma3:4b", logger: logging.Logger = None):
        self.host = host.rstrip("/")
        self.model = model
        self.logger = logger or logging.getLogger("kira.planner")
        self._health_checked = False

    def health_check(self) -> bool:
        if self._health_checked:
            return True
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
                models = [m["name"] for m in body.get("models", [])]
                if self.model not in models:
                    self.logger.warning(f"[LLM] Model '{self.model}' not found. Available: {models}")
                    return False
                self._health_checked = True
                self.logger.info(f"[LLM] Connected — model '{self.model}' ready")
                return True
        except Exception as e:
            self.logger.error(f"[LLM] Health check failed (Ollama not running?): {e}")
            return False

    def decide(self, context: str) -> dict:
        if not self._health_checked and not self.health_check():
            self.logger.warning("[LLM] Ollama unavailable — using fallback")
            return self._fallback_action()

        prompt = f"Current pentest state:\n{context}\n\nWhat is the single best next action?"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
        }

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                import urllib.request
                data = json.dumps(payload).encode()
                req = urllib.request.Request(
                    f"{self.host}/api/chat",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                timeout = 35 if attempt == 0 else 15
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = json.loads(resp.read())
                    content = body["message"]["content"]
                    try:
                        action = json.loads(content)
                    except Exception:
                        self.logger.error(f"[LLM RAW OUTPUT] {content}")
                        return self._fallback_action()
                    self.logger.debug(f"[LLM] decided: {action}")
                    return action
            except json.JSONDecodeError:
                if attempt < max_retries:
                    self.logger.warning(f"[LLM] JSON parse error (attempt {attempt+1}), retrying...")
                    time.sleep(0.1)
                    continue
                self.logger.error(f"[LLM] Failed to parse JSON after {max_retries+1} attempts")
                return self._fallback_action()
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(f"[LLM] Request failed (attempt {attempt+1}): {e}")
                    time.sleep(0.1)
                    continue
                self.logger.error(f"[LLM] failed: {e}")
                return self._fallback_action()

        return self._fallback_action()

    def _fallback_action(self) -> dict:
        return {
            "tool": "nmap_initial",
            "args": {},
            "reasoning": "LLM unavailable — falling back to initial scan",
        }


# ─────────────────────────────────────────────
# Agent Loop
# ─────────────────────────────────────────────

class KiraAgent:
    def __init__(
        self,
        target: str,
        output_dir: Path,
        ollama_host: str,
        start_phase: str,
        verbose: bool,
        max_iterations: int = 5,
    ):
        self.target = target
        self.output_dir = output_dir
        self.max_iterations = max_iterations

        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(output_dir, verbose)
        self.state = StateManager(output_dir / "state.json")
        self.executor = ToolExecutor(output_dir, self.logger)
        self.planner = LLMPlanner(host=ollama_host, logger=self.logger)

        # Phase 5: shared vuln scanner helpers
        self._vuln_runner = VulnToolRunner()
        self._vuln_kb     = KnowledgeBase()

        self.state.set("target", target)
        if self.state.get("phase") == "recon" and start_phase:
            self.state.set("phase", start_phase)

    # ── Action dispatch ───────────────────────

    def _dispatch(self, action: dict) -> str:
        tool = action.get("tool", "")
        args = action.get("args", {})
        reasoning = action.get("reasoning", "")

        self.logger.info(f"[ACTION] {tool} | {reasoning}")

        if tool == "nmap_initial":
            return self._do_nmap_initial()

        elif tool == "nmap_full":
            return self._do_nmap_full()

        elif tool == "gobuster":
            url = args.get("url", f"http://{self.target}")
            wordlist = args.get("wordlist", "/usr/share/wordlists/dirb/common.txt")
            return self._do_gobuster(url, wordlist)

        elif tool == "vuln_scan":                          # ← Phase 5
            return self._do_vuln_scan()

        elif tool == "metasploit":
            module = args.get("module", "")
            options = args.get("options", {})
            return self._do_metasploit(module, options)

        elif tool == "run_shell":
            command = args.get("command", "")
            if not command:
                return "no command provided"
            stdout, stderr, rc = self.executor.run(shlex.split(command), "shell")
            return stdout or stderr

        elif tool == "advance_phase":
            old = self.state.get("phase")
            self.state.advance_phase()
            new = self.state.get("phase")
            return f"Advanced from {old} → {new}"

        elif tool == "generate_report":
            return self._do_generate_report()

        elif tool == "done":
            return "DONE"

        else:
            return f"Unknown tool: {tool}"

    # ── Phase actions ─────────────────────────

    def _do_nmap_initial(self) -> str:
        xml_path = self.executor.nmap_scan(self.target)
        if not xml_path.exists():
            return "nmap scan failed — XML not found"
        result = NmapParser(str(xml_path)).parse()
        self._ingest_nmap(result)
        self.state.state["scan_files"]["nmap_initial"] = str(xml_path)
        self.state.save()
        return f"Found {len(result.open_ports())} open ports\n{result.summary()}"

    def _do_nmap_full(self) -> str:
        xml_path = self.executor.nmap_full(self.target)
        if not xml_path.exists():
            return "full nmap scan failed"
        result = NmapParser(str(xml_path)).parse()
        self._ingest_nmap(result)
        self.state.state["scan_files"]["nmap_full"] = str(xml_path)
        self.state.save()
        return f"Full scan: {len(result.open_ports())} open ports"

    def _ingest_nmap(self, result: NmapResult):
        """Store open ports AND build the services dict for vuln_scanner."""
        for p in result.open_ports():
            # Deduplicate open_ports list by port number
            existing = [x["port"] for x in self.state.state["findings"]["open_ports"]]
            if p["port"] not in existing:
                self.state.add_finding("open_ports", p)

            # Build version string for vuln_scanner: "Product Version"
            product = p.get("product", "").strip()
            version = p.get("version", "").strip()
            if product or version:
                version_string = f"{product} {version}".strip()
                port_key = f"{p['port']}/{p.get('protocol', 'tcp')}"
                self.state.set_service(port_key, version_string)

    # ── Phase 5: CVE cross-reference ─────────

    def _do_vuln_scan(self) -> str:
        """
        Run searchsploit against every service version in state,
        store results as vulnerability Findings, return a summary.
        """

        if shutil.which("searchsploit") is None:
            return "searchsploit not available on this system (skipping vuln_scan)"

        services: dict = self.state.state["findings"].get("services", {})
        if not services:
            return "No services in state yet — run an nmap scan first"

        self.logger.info(f"[VULN] Scanning {len(services)} service(s) via searchsploit")
        print(f"\n  {C.CYAN}[vuln_scan]{C.RESET} Checking {len(services)} service version(s)...")

        findings = scan_services(services, self._vuln_runner, self._vuln_kb)

        if not findings:
            return "searchsploit returned no results for discovered services"

        stored = 0
        for f in findings:
            fd = f.to_dict()
            # Avoid duplicates: check by port
            existing_ports = [
                v.get("port")
                for v in self.state.state["findings"]["vulnerabilities"]
                if isinstance(v, dict)
            ]
            if fd["port"] not in existing_ports:
                self.state.add_finding("vulnerabilities", fd)
                stored += 1

        # Human-readable summary
        lines = [f"vuln_scan: {stored} new finding(s) from {len(findings)} service(s)"]
        for f in findings:
            msf = " [MSF AVAILABLE]" if f.exploit_available else ""
            cve = f" {f.cve}" if f.cve else ""
            lines.append(
                f"  {f.port} | {f.version_string}{cve}{msf} | "
                f"CVSS~{f.cvss_estimate} | {len(f.edb_ids)} exploit(s)"
            )

        summary = "\n".join(lines)
        self.logger.info(f"[VULN] {summary}")
        return summary

    def _do_gobuster(self, url: str, wordlist: str) -> str:
        out = self.executor.gobuster(url, wordlist)
        out_path = self.output_dir / "gobuster.txt"
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                dirs = [
                    line.split()[0].strip()
                    for line in f
                    if line.strip() and not line.startswith("Error")
                ]
            for d in dirs:
                self.state.add_finding("directories", d)
            return f"Found {len(dirs)} paths: {dirs[:5]}"
        return "gobuster produced no output"

    def _do_metasploit(self, module: str, options: dict) -> str:
        if not module:
            return "no module specified"
        rc_path = self.output_dir / "exploit.rc"
        out_path = self.output_dir / "msf_output.txt"
        lines = [f"use {module}"]
        for k, v in options.items():
            lines.append(f"set {k} {v}")
        lines += ["run", "exit -y"]
        rc_path.write_text("\n".join(lines))

        stdout, stderr, rc = self.executor.run(
            ["msfconsole", "-q", "-r", str(rc_path), "-o", str(out_path)],
            "metasploit",
        )
        result_text = out_path.read_text() if out_path.exists() else stdout
        success = any(kw in result_text for kw in ["Meterpreter session", "Command shell session", "root@"])
        status = "SUCCESS" if success else "no session"
        self.state.add_finding("exploits_attempted", {
            "module": module,
            "options": options,
            "status": status,
        })
        return f"MSF {module}: {status}\n{result_text[:500]}"

    def _do_generate_report(self) -> str:
        from kira.reporter import generate_report
        report_path = generate_report(self.state.state, self.output_dir)
        return f"Report written to {report_path}"

    # ── Main loop ─────────────────────────────

    def run(self):
        self.logger.info(f"Kira starting — target: {self.target}")
        print(f"\n[*] Target: {self.target}")
        print(f"[*] Output: {self.output_dir}")
        print(f"[*] Phase:  {self.state.get('phase')}\n")

        for i in range(1, self.max_iterations + 1):
            phase = self.state.get("phase")
            self.logger.info(f"── Iteration {i}/{self.max_iterations} | phase={phase} ──")
            print(f"[{i:02d}] Phase: {phase}", end="  ", flush=True)

            context = self.state.get_context_summary()
            action = self.planner.decide(context)
            tool = action.get("tool", "unknown")
            print(f"→ {tool}")

            result = self._dispatch(action)
            print(result)
            self.state.log_action(tool, action.get("args", {}), result, action.get("reasoning", ""))

            self.state.flush()

            if result == "DONE":
                self.logger.info("Agent signalled DONE")
                print("\n[✓] Agent completed — target fully processed")
                break

            time.sleep(0.1)

        else:
            self.logger.warning("Max iterations reached")
            print(f"\n[!] Max iterations ({self.max_iterations}) reached")

        self.state.flush(force=True)
        print("\n── Final Findings ──────────────────────")
        print(self.state.get_context_summary())
        print(f"\n[*] Full log: {self.output_dir / 'kira.log'}")
        print(f"[*] State:    {self.output_dir / 'state.json'}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="kira",
        description="Kira — Autonomous Penetration Testing Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="  Example: python main.py --target 10.10.10.3 --verbose\n"
               "  [!] Only use against authorized targets.",
    )
    parser.add_argument("--target", required=True, help="Target IP or hostname")
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="recon",
        help="Start from a specific phase (default: recon)",
    )
    parser.add_argument(
        "--output",
        default="./kira_output",
        help="Directory for logs, scan files, and reports (default: ./kira_output)",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama API host (default: http://localhost:11434, or $OLLAMA_HOST env var)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum agent loop iterations (default: 5)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print logs to stdout")
    parser.add_argument("--version", action="version", version=f"Kira {VERSION}")
    return parser.parse_args()


def main():
    print(BANNER.format(version=VERSION))
    print(f"{C.DIM}{'─' * 45}{C.RESET}")
    print(f"  {C.YELLOW}WARNING:{C.RESET} For authorized environments only.")
    print(f"  Unauthorized use is illegal.")
    print(f"{C.DIM}{'─' * 45}{C.RESET}")
    confirm = input(f"\n  Confirm target is authorized {C.GREEN}[y/N]{C.RESET}: ").strip().lower()

    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    args = parse_args()
    output_dir = Path(args.output) / args.target.replace(".", "_")

    agent = KiraAgent(
        target=args.target,
        output_dir=output_dir,
        ollama_host=args.ollama_host,
        start_phase=args.phase,
        verbose=args.verbose,
        max_iterations=args.max_iter,
    )

    print(f"\n{C.BLUE}[*] Checking Ollama connectivity...{C.RESET}")
    if not agent.planner.health_check():
        print(f"{C.RED}[!] ERROR: Ollama is not responding!{C.RESET}")
        print(f"    Start Ollama with: {C.CYAN}ollama serve{C.RESET}")
        print(f"    Pull the model with: {C.CYAN}ollama pull gemma3:4b{C.RESET}")
        sys.exit(1) 

    agent.run()


if __name__ == "__main__":
    main()