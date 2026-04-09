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


# ── Kira package root ─────────────────────────────────────────────────────────
# Allow running as `python main.py` from the kira/ directory or project root.
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from state       import StateManager
    from llm         import LLMClient
    from tool_runner import ToolRunner
    from findings    import KnowledgeBase
    from planner     import Planner
    from logger      import KiraLogger
    from reporter    import ReportGenerator
except ImportError as e:
    print(f"[ERROR] Failed to import Kira module: {e}")
    print("        Make sure you are running from the kira/ directory.")
    sys.exit(1)

_KB_AVAILABLE = True

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


# ── Session summary ────────────────────────────────────────────────────────

def _make_session_dir(target: str, custom: str = None) -> Path:
    """Create session directory with auto-generated name if needed."""
    if custom:
        d = Path(custom)
    else:
        safe   = target.replace(".", "_").replace("/", "_")
        stamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        d      = Path("sessions") / f"{safe}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── LLM factory ───────────────────────────────────────────────────────────────

def build_llm(args, verbose: bool) -> LLMClient:
    """Construct LLMClient from CLI args. Prints clear error on failure."""
    provider = args.provider or "ollama"
    kwargs   = {
        "provider": provider,
        "verbose":  verbose,
    }
    if args.model:
        kwargs["model"] = args.model
    if provider == "ollama":
        kwargs["host"] = args.ollama_host
    elif provider in ("anthropic", "openai"):
        key = args.api_key or os.getenv(
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY", ""
        )
        if not key:
            _die(
                f"Provider '{provider}' requires an API key.\n"
                f"Pass --api-key sk-... or set the environment variable:\n"
                f"  export {'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY'}=sk-..."
            )
        kwargs["api_key"] = key

    try:
        return LLMClient(**kwargs)
    except ValueError as e:
        _die(str(e))


# ── MSF factory ───────────────────────────────────────────────────────────────

def build_msf(no_msf: bool, args):
    """Try to attach MSFClient. Returns None if --no-msf or unavailable."""
    if no_msf:
        _print_warn("--no-msf flag set — Metasploit integration disabled")
        return None

    try:
        msf_client, msf_ok, msf_msg = MSFClient.connect(
            host=args.msf_host, port=args.msf_port,
            password=args.msf_pass, ssl=not args.msf_no_ssl,
        )
        if msf_ok:
            _print_ok(f"Metasploit RPC connected: {msf_msg}")
            return msf_client
        else:
            _print_warn(f"MSF unavailable: {msf_msg}")
            return None
    except Exception as e:
        _print_warn(f"MSF init error: {e}")
        return None


# ── Report generation ─────────────────────────────────────────────────────────

def run_report(
    session_dir: Path,
    llm:         LLMClient,
    log:         KiraLogger,
    outcome:     str,
    finding_count: int,
) -> None:
    """
    Invoke ReportGenerator. Called on DONE, ROOT, or MAX_ITER (if findings exist).
    """
    if finding_count == 0:
        _print_warn("No findings recorded — skipping report generation.")
        log.info("Report skipped: no findings")
        return

    log.info(f"Generating report (outcome={outcome}, findings={finding_count})")
    _print_section("GENERATING REPORT")

    try:
        reporter = ReportGenerator(session_dir=str(session_dir), llm=llm)
        paths    = reporter.generate()

        _print_ok(f"Markdown report : {paths.markdown}")
        _print_ok(f"HTML report     : {paths.html}")
        log.info(f"Report complete: {paths.html}")

        # Try to open in browser
        _try_open_browser(paths.html)

    except Exception as e:
        _print_err(f"Report generation failed: {e}")
        log.error("reporter", str(e))


def _try_open_browser(html_path: str) -> None:
    """Attempt to open the HTML report in the default browser."""
    import webbrowser
    try:
        webbrowser.open(f"file://{Path(html_path).resolve()}")
        print("[REPORT] Opening in browser...")
    except Exception:
        pass


# ── Session summary ────────────────────────────────────────────────────────────────

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



# ── Terminal helpers ──────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    try:
        from rich.console import Console
        Console().print(f"\n[bold cyan]{'─'*50}[/bold cyan]")
        Console().print(f"[bold cyan]  {title}[/bold cyan]")
        Console().print(f"[bold cyan]{'─'*50}[/bold cyan]")
    except ImportError:
        print(f"\n{'─'*50}")
        print(f"  {title}")
        print(f"{'─'*50}")


def _print_ok(msg: str) -> None:
    try:
        from rich.console import Console
        Console().print(f"[green]  ✓[/green] {msg}")
    except ImportError:
        print(f"  ✓ {msg}")


def _print_warn(msg: str) -> None:
    try:
        from rich.console import Console
        Console().print(f"[yellow]  ⚠[/yellow] {msg}")
    except ImportError:
        print(f"  ⚠ {msg}")


def _print_err(msg: str) -> None:
    try:
        from rich.console import Console
        Console().print(f"[red]  ✗[/red] {msg}")
    except ImportError:
        print(f"  ✗ {msg}")


def _die(msg: str) -> None:
    _print_err(msg)
    sys.exit(1)


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
            "  python main.py --target 10.10.10.5 --authorized-by 'test' --provider anthropic\n"
            "\n"
            "  [!] Only use against targets you are explicitly authorized to test."
        ),
    )

    # Required
    parser.add_argument("--target", "-t", required=True,
                        help="Target IP address or hostname")
    parser.add_argument("--authorized-by", required=True, metavar="AUTHORIZATION",
                        help="Required: written confirmation of authorization")

    # LLM
    parser.add_argument("--provider", default=None,
                        choices=["ollama", "anthropic", "openai"],
                        help="LLM provider (default: ollama)")
    parser.add_argument("--ollama-host",
                        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                        metavar="URL",
                        help="Ollama API URL (default: http://localhost:11434)")
    parser.add_argument("--model", default=None,
                        help="Override model name")
    parser.add_argument("--api-key", default=None,
                        help="API key for Anthropic or OpenAI")

    # Session
    parser.add_argument("--session-dir", default=None,
                        help="Custom session directory (auto-generated if not set)")
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

    # Flags
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Stream tool output to terminal")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip automatic report generation at session end")
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

    verbose = args.verbose

    # ── Session directory ──────────────────────────────────────────────────────
    session_dir = _make_session_dir(args.target, args.session_dir)
    _print_ok(f"Session dir: {session_dir}")

    # ── KiraLogger ─────────────────────────────────────────────────────────────
    log = KiraLogger(session_dir=str(session_dir), verbose=verbose)
    log.info(f"Kira session started — target={args.target}")

    # ── 1. Startup info ───────────────────────────────────────────────────────
    print(f"  Target        : {C.CYAN}{args.target}{C.RESET}")
    print(f"  Authorized by : {C.GREEN}{args.authorized_by}{C.RESET}")
    print(f"  Ollama host   : {args.ollama_host}")
    print(f"  MSF           : {'disabled (--no-msf)' if args.no_msf else f'{args.msf_host}:{args.msf_port}'}")
    print()

    # ── StateManager ──────────────────────────────────────────────────────────
    state = StateManager(session_dir=str(session_dir))
    state.init(target=args.target, authorized_by=args.authorized_by)
    _print_ok(f"Target: {args.target}  |  Authorized by: {args.authorized_by}")

    # ── LLM client + ping ──────────────────────────────────────────────────────
    _print_section("CONNECTING TO LLM")
    llm = build_llm(args, verbose)

    ok, msg = llm.ping()
    if ok:
        _print_ok(f"LLM ready: [{llm.provider}] {llm.model}")
        log.info(f"LLM connected: provider={llm.provider} model={llm.model}")
    else:
        _print_warn(f"LLM ping failed: {msg}")
        _print_warn("Continuing — agent will HALT if LLM is unreachable")
        log.error("llm", f"Ping failed: {msg}")

    # ── MSF client — graceful degradation ─────────────────────────────────────
    _print_section("METASPLOIT")
    msf_client = build_msf(args.no_msf, args)

    # ── ToolRunner + KnowledgeBase ────────────────────────────────────────────
    runner = ToolRunner(session_dir=str(session_dir),
                        verbose=verbose, msf=msf_client)
    kb = KnowledgeBase() if _KB_AVAILABLE else None

    avail   = runner.check_tools()
    present = [t for t, ok in avail.items() if ok]
    missing = [t for t, ok in avail.items() if not ok]
    if present:
        print(f"{C.DIM}[TOOLS] Available: {', '.join(present)}{C.RESET}")
    if missing:
        print(f"{C.YELLOW}[WARN ] Missing:   {', '.join(missing)}{C.RESET}")

    # ── Phase transition logger hook ──────────────────────────────────────────
    # Monkey-patch StateManager.advance_phase to emit log.phase() events
    _orig_advance = state.advance_phase
    def _logged_advance():
        old = state.phase
        new = _orig_advance()
        if new != old:
            log.phase(old, new)
            _print_section(f"PHASE: {old} → {new}")
        return new
    state.advance_phase = _logged_advance

    # ── Planner ────────────────────────────────────────────────────────────────
    _print_section("STARTING AGENT LOOP")
    log.info(f"Agent loop starting: max_iter={args.max_iter}")

    planner = Planner(state=state, runner=runner, llm=llm,
                      msf=msf_client, kb=kb, verbose=True)

    # ── Run ────────────────────────────────────────────────────────────────────
    print(f"{C.GREEN}{C.BOLD}[KIRA] Agent loop starting...{C.RESET}")
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

    # ── Session summary ────────────────────────────────────────────────────────
    _print_section("SESSION COMPLETE")
    finding_count = len(state.get("findings", []))
    
    outcome_colors = {
        "DONE": C.GREEN, "ROOT": C.GREEN,
        "HALTED": C.YELLOW, "MAX_ITER": C.YELLOW,
        "INTERRUPTED": C.YELLOW, "ERROR": C.RED,
    }
    print(f"{outcome_colors.get(outcome, C.DIM)}{C.BOLD}Outcome: {outcome}{C.RESET}")
    
    _print_session_summary(state, session_dir, elapsed)
    
    log.info(f"Session ended: outcome={outcome} findings={finding_count}")

    # ── Auto-report ────────────────────────────────────────────────────────────
    if not args.no_report:
        if outcome in ("DONE", "ROOT") or (outcome == "MAX_ITER" and finding_count > 0):
            run_report(session_dir, llm, log, outcome, finding_count)
        else:
            _print_warn(f"No report generated (outcome={outcome}, findings={finding_count})")
    else:
        log.info("Report skipped: --no-report flag set")

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