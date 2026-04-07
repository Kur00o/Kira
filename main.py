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
from datetime import datetime
from pathlib import Path

from kira.parsers.nmap_parser import NmapParser, NmapResult

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

VERSION = "0.1.0"

PHASES = ["recon", "enumeration", "exploitation", "privesc", "report"]

BANNER = r"""
  _  ___           
 | |/ (_)_ _ __ _ 
 | ' <| | '_/ _` |
 |_|\_\_|_| \__,_|
  Autonomous Pentest Agent v{version}
  [!] Authorized environments only
"""

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────

def setup_logging(output_dir: Path, verbose: bool) -> logging.Logger:
    log_path = output_dir / "kira.log"
    handlers = [logging.FileHandler(log_path)]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
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
        if state_path.exists():
            self._load()

    def _default_state(self) -> dict:
        return {
            "target": "",
            "phase": "recon",
            "completed_phases": [],
            "findings": {
                "open_ports": [],
                "services": [],
                "directories": [],
                "vulnerabilities": [],
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

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2)

    def set(self, key: str, value):
        self.state[key] = value
        self.save()

    def get(self, key: str, default=None):
        return self.state.get(key, default)

    def add_finding(self, category: str, finding: dict):
        self.state["findings"][category].append(finding)
        self.save()

    def log_action(self, tool: str, args: dict, result: str, reasoning: str = ""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool,
            "args": args,
            "result_summary": result[:300],  # keep state file lean
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
        for p in findings["open_ports"][:10]:  # cap at 10
            lines.append(f"  {p.get('port')}/{p.get('protocol')} {p.get('service')} {p.get('product','')} {p.get('version','')}".rstrip())

        if findings["directories"]:
            lines.append(f"\nDiscovered paths ({len(findings['directories'])}):")
            for d in findings["directories"][:8]:
                lines.append(f"  {d}")

        if findings["vulnerabilities"]:
            lines.append(f"\nVulnerabilities ({len(findings['vulnerabilities'])}):")
            for v in findings["vulnerabilities"][:5]:
                lines.append(f"  [{v.get('severity','?')}] {v.get('title','?')}")

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
        """
        Execute a shell command, stream stdout to log, return (stdout, stderr, returncode).
        """
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
            "nmap", "-sV", "-sC", "-p-",
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
- Do not repeat an action already taken unless you have new information.
- If you have enough recon data, call advance_phase to move forward.
""" + TOOL_SCHEMA

    def __init__(self, host: str = "http://localhost:11434", model: str = "gemma3:4b", logger: logging.Logger = None):
        self.host = host.rstrip("/")
        self.model = model
        self.logger = logger or logging.getLogger("kira.planner")

    def decide(self, context: str) -> dict:
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

        try:
            import urllib.request
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
                content = body["message"]["content"]
                action = json.loads(content)
                self.logger.debug(f"[LLM] decided: {action}")
                return action
        except Exception as e:
            self.logger.error(f"[LLM] failed: {e}")
            # Safe fallback: continue with nmap if planner is down
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
        max_iterations: int = 30,
    ):
        self.target = target
        self.output_dir = output_dir
        self.max_iterations = max_iterations

        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(output_dir, verbose)
        self.state = StateManager(output_dir / "state.json")
        self.executor = ToolExecutor(output_dir, self.logger)
        self.planner = LLMPlanner(host=ollama_host, logger=self.logger)

        # Initialise state
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

        elif tool == "metasploit":
            module = args.get("module", "")
            options = args.get("options", {})
            return self._do_metasploit(module, options)

        elif tool == "run_shell":
            command = args.get("command", "")
            if not command:
                return "no command provided"
            stdout, stderr, rc = self.executor.run(command.split(), "shell")
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
        for p in result.open_ports():
            # Deduplicate by port number
            existing = [x["port"] for x in self.state.state["findings"]["open_ports"]]
            if p["port"] not in existing:
                self.state.add_finding("open_ports", p)

    def _do_gobuster(self, url: str, wordlist: str) -> str:
        out = self.executor.gobuster(url, wordlist)
        # Parse output file line by line
        out_path = self.output_dir / "gobuster.txt"
        if out_path.exists():
            with open(out_path) as f:
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
        # Build resource script
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
        # Basic success detection
        success = any(kw in result_text for kw in ["Meterpreter session", "Command shell session", "root@"])
        status = "SUCCESS" if success else "no session"
        self.state.add_finding("exploits_attempted", {
            "module": module,
            "options": options,
            "status": status,
        })
        return f"MSF {module}: {status}\n{result_text[:500]}"

    def _do_generate_report(self) -> str:
        from kira.reporter import generate_report  # imported lazily (Day 4 module)
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

            # Get LLM decision
            context = self.state.get_context_summary()
            action = self.planner.decide(context)
            tool = action.get("tool", "unknown")
            print(f"→ {tool}")

            # Execute
            result = self._dispatch(action)
            self.state.log_action(tool, action.get("args", {}), result, action.get("reasoning", ""))

            if result == "DONE":
                self.logger.info("Agent signalled DONE")
                print("\n[✓] Agent completed — target fully processed")
                break

            # Small delay to avoid hammering the LLM
            time.sleep(1)

        else:
            self.logger.warning("Max iterations reached")
            print(f"\n[!] Max iterations ({self.max_iterations}) reached")

        # Always print final state summary
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
        default=30,
        help="Maximum agent loop iterations (default: 30)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print logs to stdout")
    parser.add_argument("--version", action="version", version=f"Kira {VERSION}")
    return parser.parse_args()


def main():
    print(BANNER.format(version=VERSION))

    # Ethical usage gate
    print("─" * 45)
    print("  WARNING: For authorized environments only.")
    print("  Unauthorized use is illegal.")
    print("─" * 45)
    confirm = input("\n  Confirm target is authorized [y/N]: ").strip().lower()
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
    agent.run()


if __name__ == "__main__":
    main()