"""
main.py — Kira Autonomous Penetration Testing Agent
=====================================================
CLI entry point. Wires all Day 1-3 modules together and launches
the Planner loop autonomously — no human input after startup.

Usage:
    python main.py --target 10.10.10.5 --authorized-by "Lab VM"
    python main.py --target 10.10.10.5 --authorized-by "HTB" --ollama-host http://192.168.1.42:11434
    python main.py --target 10.10.10.5 --authorized-by "test" --no-msf --verbose

Merge resolution: kept development branch entirely.
All inline classes (StateManager, ToolExecutor, LLMPlanner, KiraAgent)
from main branch discarded — replaced by kira.* module imports.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Kira modules ───────────────────────────────────────────────────────────────

def _require_module(name: str, path: str):
    try:
        return __import__(name)
    except ImportError:
        print(f"[KIRA] ERROR: cannot import '{name}'. Is {path} in your project?")
        sys.exit(1)

from kira.state       import StateManager
from kira.tool_runner import ToolRunner
from kira.llm         import LLMClient
from kira.planner     import Planner

try:
    from kira.findings import KnowledgeBase
    _KB_AVAILABLE = True
except ImportError:
    _KB_AVAILABLE = False

try:
    from pymetasploit3.msfrpc import MsfRpcClient as _MsfRpcClient
    _MSF_LIB_AVAILABLE = True
except ImportError:
    _MSF_LIB_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────────────────
# CONFLICT RESOLUTION: kept development values (VERSION=0.3.0, max_iter=50)
# Discarded: main branch's VERSION string and max_iter default of 5.

VERSION = "0.3.0"

PHASES = ["RECON", "ENUM", "VULN_SCAN", "EXPLOIT", "POST_EXPLOIT", "REPORT", "DONE"]

MSF_DEFAULT_HOST = "127.0.0.1"
MSF_DEFAULT_PORT = 55553
MSF_DEFAULT_PASS = "kirapass"


# ── ANSI colours ───────────────────────────────────────────────────────────────
# CONFLICT RESOLUTION: kept development's C class (defined before BANNER).
# Discarded: main branch's BANNER that referenced C before C was defined.

class C:
    BLUE_BRIGHT = "\033[94m"
    BLUE_DIM    = "\033[34m"
    CYAN        = "\033[96m"
    CYAN_DIM    = "\033[36m"
    WHITE       = "\033[97m"
    YELLOW      = "\033[93m"
    GREEN       = "\033[92m"
    RED         = "\033[91m"
    BLUE        = "\033[94m"
    DIM         = "\033[2m"
    BOLD        = "\033[1m"
    RESET       = "\033[0m"


BANNER = f"""
{C.BLUE_BRIGHT}{C.BOLD}  _  ___
 | |/ (_)_ _ __ _
 | ' <| | '_/ _` |
 |_|\\_\\_|_| \\__,_|{C.RESET}
{C.DIM}{C.CYAN}  AUTONOMOUS PENTEST AGENT v{VERSION}{C.RESET}
{C.YELLOW}  \u26a0  Authorized environments only{C.RESET}
"""


# ── MSF connection helper ──────────────────────────────────────────────────────
# CONFLICT RESOLUTION: kept development's MSFClient class.
# Discarded: main branch's inline KiraAgent._do_metasploit() + vuln_scan logic
#            (those belong in planner.py and parsers/vuln_scanner.py).

class MSFClient:
    """
    Thin wrapper around pymetasploit3.MsfRpcClient.
    Returns (client, ok, message) — never raises.
    """

    @classmethod
    def connect(
        cls,
        host:     str  = MSF_DEFAULT_HOST,
        port:     int  = MSF_DEFAULT_PORT,
        password: str  = MSF_DEFAULT_PASS,
        ssl:      bool = True,
    ):
        if not _MSF_LIB_AVAILABLE:
            return None, False, (
                "pymetasploit3 not installed. Run: pip install pymetasploit3"
            )
        try:
            client = _MsfRpcClient(password, server=host, port=port, ssl=ssl)
            return client, True, f"connected to {host}:{port}"
        except ConnectionRefusedError:
            return None, False, (
                f"msfrpcd not running on {host}:{port}. "
                f"Start it: msfrpcd -P {password} -p {port} -a {host}"
            )
        except Exception as exc:
            return None, False, f"MSF connection error: {exc}"


# ── Session summary ────────────────────────────────────────────────────────────

def _print_session_summary(state: StateManager, session_dir: Path, elapsed_s: float):
    """Print a compact summary table after the planner loop exits."""
    findings = state.get("findings") or []
    ports    = state.get("open_ports") or []
    sessions = state.get("sessions") or []
    is_root  = state.get("is_root", False)
    phase    = state.get("phase", "?")

    log_path = session_dir / "actions.jsonl"
    stats    = ToolRunner.summarise_action_log(str(log_path))

    sev_counts: dict[str, int] = {}
    for f in findings:
        sev = f.get("severity", "info")
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

    print(f"\n{C.BOLD}{'─' * 50}{C.RESET}")
    print(f"{C.BOLD}  Kira Session Summary{C.RESET}")
    print(f"{'─' * 50}")
    print(f"  Target          {state.target}")
    print(f"  Final phase     {phase}")
    print(f"  Root obtained   {'YES \u2713' if is_root else 'no'}")
    print(f"  Sessions open   {len(sessions)}")
    print(f"  Open ports      {len(ports)}")
    print(f"  Findings        {len(findings)}  "
          + "  ".join(
              f"{C.RED if k == 'critical' else C.YELLOW}{k}:{v}{C.RESET}"
              for k, v in sev_counts.items()
          ))
    print(f"  Actions run     {stats['total_actions']}  "
          f"(ok={stats['successful']}  failed={stats['failed']})")
    print(f"  Tools used      {', '.join(stats['tools_used']) or 'none'}")
    print(f"  Elapsed         {elapsed_s:.0f}s")
    print(f"  Session dir     {session_dir}")
    print(f"{'─' * 50}")

    if findings:
        print(f"\n{C.BOLD}  Top findings:{C.RESET}")
        for f in sorted(findings, key=lambda x: x.get("cvss", 0), reverse=True)[:5]:
            sev   = f.get("severity", "info").upper()
            color = {"CRITICAL": C.RED, "HIGH": C.RED,
                     "MEDIUM": C.YELLOW, "LOW": C.BLUE}.get(sev, C.DIM)
            print(
                f"  {color}[{sev:8s}]{C.RESET} CVSS {f.get('cvss','?'):<4}  "
                f"port {f.get('port','?'):<5}  {f.get('title','untitled')}"
            )
    print()


# ── CLI argument parser ────────────────────────────────────────────────────────
# CONFLICT RESOLUTION: kept development's --max-iter default of 50.
# Discarded: main branch's default of 5 (too low for a real pentest).

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kira",
        description="Kira \u2014 Autonomous Penetration Testing Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --target 10.10.10.5 --authorized-by 'HTB Lab'\n"
            "  python main.py --target 10.10.10.5 --authorized-by 'test' --no-msf\n"
            "\n"
            "  [!] Only use against targets you are explicitly authorized to test."
        ),
    )

    parser.add_argument("--target", "-t", required=True,
                        help="Target IP address or hostname")
    parser.add_argument("--authorized-by", required=True, metavar="AUTHORIZATION",
                        help="Required: written confirmation of authorization")
    parser.add_argument("--ollama-host",
                        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                        metavar="URL",
                        help="Ollama API URL (default: http://localhost:11434)")
    parser.add_argument("--output", "-o", default="./kira_sessions", metavar="DIR",
                        help="Root directory for session output")
    parser.add_argument("--max-iter", type=int, default=50, metavar="N",
                        help="Maximum planner loop iterations (default: 50)")

    msf = parser.add_argument_group("Metasploit RPC (optional)")
    msf.add_argument("--no-msf", action="store_true",
                     help="Disable Metasploit (RECON\u2192VULN_SCAN only)")
    msf.add_argument("--msf-host",   default=MSF_DEFAULT_HOST)
    msf.add_argument("--msf-port",   type=int, default=MSF_DEFAULT_PORT)
    msf.add_argument("--msf-pass",   default=MSF_DEFAULT_PASS)
    msf.add_argument("--msf-no-ssl", action="store_true",
                     help="Disable SSL for msfrpcd connection")

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Stream tool output to terminal")
    parser.add_argument("--version", action="version", version=f"Kira {VERSION}")

    return parser


# ── Main ───────────────────────────────────────────────────────────────────────
# CONFLICT RESOLUTION: kept development's main() entirely.
# Discarded: main branch's inline agent.run() call and health_check() block
#            (health check is now llm.ping() in the LLMClient module).

def main():
    print(BANNER)
    print(f"{C.DIM}{'─' * 50}{C.RESET}")
    print(f"  {C.YELLOW}WARNING:{C.RESET} For authorized environments only.")
    print(f"  Unauthorized use is illegal and unethical.")
    print(f"{C.DIM}{'─' * 50}{C.RESET}\n")

    parser = _build_parser()
    args   = parser.parse_args()

    # ── 1. Startup info ───────────────────────────────────────────────────────
    print(f"  Target        : {C.CYAN}{args.target}{C.RESET}")
    print(f"  Authorized by : {C.GREEN}{args.authorized_by}{C.RESET}")
    print(f"  Ollama host   : {args.ollama_host}")
    print(f"  MSF           : {'disabled (--no-msf)' if args.no_msf else f'{args.msf_host}:{args.msf_port}'}")
    print()

    # ── 2. Session directory ──────────────────────────────────────────────────
    safe_target = args.target.replace(".", "_").replace("/", "_")
    timestamp   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output) / f"{safe_target}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Session dir   : {session_dir}\n")

    # ── 3. StateManager ───────────────────────────────────────────────────────
    state = StateManager(session_dir=str(session_dir))
    state.init(target=args.target, authorized_by=args.authorized_by)

    # ── 4. LLM client + ping ──────────────────────────────────────────────────
    print(f"{C.DIM}[INIT] Connecting to LLM at {args.ollama_host}...{C.RESET}")
    llm = LLMClient(host=args.ollama_host, verbose=args.verbose)
    llm_ok, llm_msg = llm.ping()
    if llm_ok:
        print(f"{C.GREEN}[OK  ] LLM: {llm_msg}{C.RESET}")
    else:
        print(f"{C.RED}[FAIL] LLM: {llm_msg}{C.RESET}")
        print(f"\n  Kira cannot run without a working LLM connection.")
        print(f"  Fix: ollama pull gemma3:4b && OLLAMA_HOST=0.0.0.0 ollama serve")
        sys.exit(1)

    # ── 5. MSF client — graceful degradation ─────────────────────────────────
    msf_client = None

    if args.no_msf:
        print(f"{C.YELLOW}[SKIP] MSF disabled via --no-msf{C.RESET}")
        print(f"       Kira will run RECON \u2192 ENUM \u2192 VULN_SCAN then stop.")
    else:
        print(f"{C.DIM}[INIT] Connecting to msfrpcd at {args.msf_host}:{args.msf_port}...{C.RESET}")
        msf_client, msf_ok, msf_msg = MSFClient.connect(
            host=args.msf_host, port=args.msf_port,
            password=args.msf_pass, ssl=not args.msf_no_ssl,
        )
        if msf_ok:
            print(f"{C.GREEN}[OK  ] MSF: {msf_msg}{C.RESET}")
        else:
            print(f"{C.YELLOW}[WARN] MSF unavailable: {msf_msg}{C.RESET}")
            print(f"       Start msfrpcd: msfrpcd -P {args.msf_pass} "
                  f"-p {args.msf_port} -a {args.msf_host}")
            msf_client = None

    # ── 6. ToolRunner ─────────────────────────────────────────────────────────
    runner = ToolRunner(session_dir=str(session_dir),
                        verbose=args.verbose, msf=msf_client)
    avail   = runner.check_tools()
    present = [t for t, ok in avail.items() if ok]
    missing = [t for t, ok in avail.items() if not ok]
    if present:
        print(f"{C.DIM}[TOOLS] Available: {', '.join(present)}{C.RESET}")
    if missing:
        print(f"{C.YELLOW}[WARN ] Missing:   {', '.join(missing)}{C.RESET}")

    # ── 7. KnowledgeBase ──────────────────────────────────────────────────────
    kb = KnowledgeBase() if _KB_AVAILABLE else None
    if not _KB_AVAILABLE:
        print(f"{C.DIM}[INFO] findings.py not built yet \u2014 kb=None{C.RESET}")

    # ── 8. Planner ────────────────────────────────────────────────────────────
    planner = Planner(state=state, runner=runner, llm=llm,
                      msf=msf_client, kb=kb, verbose=True)

    # ── 9. Run ────────────────────────────────────────────────────────────────
    print(f"\n{C.GREEN}{C.BOLD}[KIRA] Agent loop starting...{C.RESET}")
    print(f"{C.DIM}       Target: {args.target}  |  Max iterations: {args.max_iter}{C.RESET}\n")

    t_start = time.monotonic()

    try:
        outcome = planner.run(max_iterations=args.max_iter)
    except KeyboardInterrupt:
        outcome = "INTERRUPTED"
        print(f"\n{C.YELLOW}[KIRA] Interrupted by user.{C.RESET}")
    except Exception as exc:
        outcome = "ERROR"
        print(f"\n{C.RED}[KIRA] Unhandled error: {exc}{C.RESET}")
        import traceback; traceback.print_exc()

    elapsed = time.monotonic() - t_start

    # ── 10. Exit ──────────────────────────────────────────────────────────────
    outcome_colors = {
        "DONE": C.GREEN, "ROOT": C.GREEN,
        "HALTED": C.YELLOW, "MAX_ITER": C.YELLOW,
        "INTERRUPTED": C.YELLOW, "ERROR": C.RED,
    }
    print(f"\n{outcome_colors.get(outcome, C.DIM)}{C.BOLD}[KIRA] Session ended: {outcome}{C.RESET}")

    _print_session_summary(state, session_dir, elapsed)

    if outcome == "DONE":
        print(f"  {C.DIM}Next: run reporter.py to generate the full pentest report.{C.RESET}")
    elif outcome in ("HALTED", "MAX_ITER"):
        print(f"  {C.DIM}Tip: review {session_dir}/actions.jsonl to see why the agent stopped.{C.RESET}")
    if not msf_client and not args.no_msf:
        print(f"  {C.DIM}Tip: start msfrpcd to enable the EXPLOIT phase next run.{C.RESET}")

    print()
    return 0 if outcome in ("DONE", "ROOT") else 1


if __name__ == "__main__":
    sys.exit(main())