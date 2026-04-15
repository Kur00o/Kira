"""
main.py — Kira Autonomous Penetration Testing Agent
=====================================================
CLI entry point. Wires all modules together and launches the Planner loop.

Usage:
    python main.py
    python main.py --target 10.10.10.5 --authorized-by "Lab VM"
    python main.py --target 10.10.10.5 --authorized-by "HTB" --api-key AIzaSy...
    python main.py --target 10.10.10.5 --authorized-by "test" --no-msf --verbose

Requires GEMINI_API_KEY set in .env or environment.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required if env vars are already set


# ── Kira package root ─────────────────────────────────────────────────────────
# Allow running as `python main.py` from the kira/ directory or project root.
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from kira.state       import StateManager
    from kira.llm         import LLMClient
    from kira.tool_runner import ToolRunner
    from kira.findings    import KnowledgeBase
    from kira.planner     import Planner
    from kira.chat        import KiraChat
    from kira.logger      import KiraLogger
    from kira.reporter    import ReportGenerator
    from kira.guardrails  import ScopeGuard
except ImportError as e:
    # Fallback for legacy execution contexts where modules are imported
    # from a flat path.
    try:
        from kira.state       import StateManager
        from kira.llm         import LLMClient
        from kira.tool_runner import ToolRunner
        from kira.findings    import KnowledgeBase
        from kira.planner     import Planner
        from kira.logger      import KiraLogger
        from kira.reporter    import ReportGenerator
        from kira.guardrails  import ScopeGuard
    except ImportError:
        print(f"[ERROR] Failed to import Kira module: {e}")
        print("        Run from the repository root with: python main.py ...")
        sys.exit(1)

_KB_AVAILABLE = True

try:
    from pymetasploit3.msfrpc import MsfRpcClient as _MsfRpcClient
    _MSF_LIB_AVAILABLE = True
except ImportError:
    _MSF_LIB_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────────────────
# CONFLICT RESOLUTION: kept development values (VERSION=0.3.0).
# Default max iterations set to 10 for faster runs.

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
    """Construct Gemini LLMClient from env vars and CLI args."""
    api_key = (getattr(args, "api_key", None) or os.getenv("GEMINI_API_KEY", "")).strip()
    model   = getattr(args, "model", None) or os.getenv("GEMINI_MODEL", "gemini-2.5-flash") or None

    if not api_key:
        _die(
            "No Gemini API key found.\n"
            "Set GEMINI_API_KEY in your .env file:\n"
            "  GEMINI_API_KEY=AIzaSy..."
        )

    try:
        return LLMClient(api_key=api_key, model=model, verbose=verbose)
    except ValueError as e:
        _die(str(e))


# ── MSF factory ───────────────────────────────────────────────────────────────

def build_msf(no_msf: bool, args):
    """
    Try to attach Metasploit RPC client.
    Strategy:
      1) connect to existing msfrpcd (current behavior)
      2) if unavailable and SSL mode is enabled, try auto-start wrapper client
    Returns None if unavailable.
    """
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
        _print_warn(f"MSF unavailable: {msf_msg}")

        # Auto-start fallback (SSL mode only).
        if args.msf_no_ssl:
            return None
        try:
            from kira.msf_client import MSFClient as AutoMSFClient
            auto = AutoMSFClient()
            if auto.auto_start(
                host=args.msf_host,
                port=args.msf_port,
                password=args.msf_pass,
                ssl=not args.msf_no_ssl,
            ):
                _print_ok(
                    f"Metasploit RPC auto-started and connected: "
                    f"{args.msf_host}:{args.msf_port}"
                )
                return auto
            _print_warn("MSF auto-start failed — continuing without Metasploit")
            return None
        except Exception as auto_exc:
            _print_warn(f"MSF auto-start error: {auto_exc}")
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
# Default max iterations set to 10.

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kira",
        description="Kira \u2014 Autonomous Pentesting Agent (Conversational)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --authorized-by 'HTB Lab' --no-msf --verbose\n"
            "  python main.py --session-dir ./my_session\n"
            "\n"
            "  [!] Chat with Kira to set targets and trigger scans.\n"
            "  [!] Set GEMINI_API_KEY in .env before running."
        ),
    )

    # Optional: can be provided but target will be discovered through chat
    parser.add_argument("--target", "-t", default=None,
                        help="(Optional) Target IP address or hostname — can be set via chat")
    parser.add_argument("--authorized-by", default="Lab VM", metavar="AUTHORIZATION",
                        help="Authorization identifier (default: 'Lab VM')")

    # LLM (Gemini)
    parser.add_argument("--api-key", default=None,
                        help="Gemini API key (overrides GEMINI_API_KEY env var)")
    parser.add_argument("--model", default=None,
                        help="Override Gemini model (default: gemini-2.5-flash)")

    # Session
    parser.add_argument("--session-dir", default=None,
                        help="Custom session directory (auto-generated if not set)")
    parser.add_argument("--max-iter", type=int, default=20, metavar="N",
                        help="Maximum planner loop iterations per scan (default: 20)")

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

    # ── Session directory — generic (no target in name yet) ──────────────────────
    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_dir = Path("sessions") / f"kira_{stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    _print_ok(f"Session dir: {session_dir}")

    # ── KiraLogger ─────────────────────────────────────────────────────────────
    log = KiraLogger(session_dir=str(session_dir), verbose=verbose)
    log.info("Kira session started (conversational mode)")

    # ── Startup info ───────────────────────────────────────────────────────────
    print(f"  Authorized by : {C.GREEN}{args.authorized_by}{C.RESET}")
    print(f"  Gemini model  : {args.model or os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}")
    print(f"  MSF           : {'disabled (--no-msf)' if args.no_msf else f'{args.msf_host}:{args.msf_port}'}")
    print()

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
                        verbose=False, msf=msf_client)
    kb = KnowledgeBase() if _KB_AVAILABLE else None

    avail   = runner.check_tools()
    present = [t for t, ok in avail.items() if ok]
    missing = [t for t, ok in avail.items() if not ok]
    if present:
        print(f"{C.DIM}[TOOLS] Available: {', '.join(present)}{C.RESET}")
    if missing:
        print(f"{C.YELLOW}[WARN ] Missing:   {', '.join(missing)}{C.RESET}")

    # ── StateManager ──────────────────────────────────────────────────────────
    # Use a placeholder target until the user provides one via chat.
    # KiraChat.re-inits state when a real target is given.
    initial_target = args.target or ""
    state = StateManager(session_dir=str(session_dir))
    state.init(
        target=initial_target or "pending",
        authorized_by=args.authorized_by,
    )

    # ── Scope guardrails ──────────────────────────────────────────────────────
    # Only validate target if one was provided on the CLI.
    # If no target yet, create a permissive guard that validates per-action.
    if initial_target:
        guard = ScopeGuard(authorized_target=initial_target, authorized_by=args.authorized_by)
        guard.validate_startup(log)
    else:
        guard = None  # KiraChat will create a real guard when target is set

    # ── Phase transition logger hook ──────────────────────────────────────────
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
    planner = Planner(
        state=state,
        runner=runner,
        llm=llm,
        msf=msf_client,
        kb=kb,
        verbose=True,   # always show step-by-step output
        logger=log,
        guard=guard,
    )

    # ── Start KiraChat ─────────────────────────────────────────────────────────
    _print_section("KIRA CHAT")

    try:
        chat = KiraChat(
            planner=planner,
            state=state,
            llm=llm,
            max_iter=args.max_iter,
            verbose=args.verbose,
            session_dir=session_dir,
            log=log,
            no_report=args.no_report,
        )
        chat.start()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}[KIRA] Session interrupted by user.{C.RESET}")
    except Exception as exc:
        print(f"\n{C.RED}[KIRA] Unhandled error: {exc}{C.RESET}")
        import traceback; traceback.print_exc()

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())