"""
kira/logger.py — KiraLogger
============================
Structured JSONL event logger for a Kira session.
Every planner action, finding, phase transition, and error
gets written to session_dir/kira.log as a typed JSON line.

The reporter reads this file to build the attack timeline.

Note: kira.log ≠ actions.jsonl
  actions.jsonl = raw tool output for replay (written by ToolRunner)
  kira.log      = typed agent events for the report timeline (written here)

Usage:
    from logger import KiraLogger

    log = KiraLogger(session_dir="./sessions/my_scan", verbose=True)
    log.phase("RECON", "ENUM")
    log.action("nmap_scan", {"target": "10.10.10.5"}, {"ok": True, "summary": "3 ports"}, 12.4)
    log.finding({"title": "Apache CVE-2021-41773", "severity": "critical", "cvss": 9.8})
    log.error("gobuster", "wordlist not found")
    log.info("Starting phase 2 enumeration")

    entries = KiraLogger.load_log("./sessions/my_scan/kira.log")
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any


# ── Entry types ────────────────────────────────────────────────────────────────

ENTRY_TYPES = ("phase", "action", "finding", "error", "info")

# Rich colour map per entry type (used in verbose terminal output)
_TYPE_STYLE = {
    "phase":   "bold cyan",
    "action":  "green",
    "finding": "bold yellow",
    "error":   "bold red",
    "info":    "dim",
}

_TYPE_ICON = {
    "phase":   "⟶",
    "action":  "▶",
    "finding": "⚑",
    "error":   "✗",
    "info":    "·",
}


# ── KiraLogger ─────────────────────────────────────────────────────────────────

class KiraLogger:
    """
    Thread-safe JSONL structured logger.

    Each line in kira.log is:
        {"ts": "2026-04-09T12:00:00Z", "type": "action", "data": {...}}

    Parameters
    ----------
    session_dir : path to the session directory (kira.log is written here)
    verbose     : if True, pretty-print events to terminal via rich (or plain print)
    """

    LOG_FILENAME = "kira.log"

    def __init__(self, session_dir: str, verbose: bool = True):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.log_path    = self.session_dir / self.LOG_FILENAME
        self.verbose     = verbose
        self._lock       = threading.Lock()

    # ── Five typed event methods ───────────────────────────────────────────────

    def phase(self, old_phase: str, new_phase: str) -> None:
        """
        Log a phase transition event.

        Args:
            old_phase : the phase being left   (e.g. "RECON")
            new_phase : the phase being entered (e.g. "ENUM")
        """
        self._write("phase", {
            "from": old_phase,
            "to":   new_phase,
        })
        if self.verbose:
            self._print("phase", f"{old_phase} → {new_phase}")

    def action(
        self,
        tool:      str,
        args:      dict,
        result:    dict,
        elapsed_s: float = 0.0,
    ) -> None:
        """
        Log a planner tool action and its result.

        Args:
            tool      : tool name (e.g. "nmap_scan")
            args      : arguments passed to the tool
            result    : result dict with at minimum {"ok": bool, "summary": str}
            elapsed_s : wall-clock execution time in seconds
        """
        ok      = result.get("ok", False)
        summary = result.get("summary", "")
        self._write("action", {
            "tool":      tool,
            "args":      args,
            "ok":        ok,
            "summary":   summary,
            "elapsed_s": round(elapsed_s, 2),
        })
        if self.verbose:
            status = "OK" if ok else "FAIL"
            self._print("action", f"[{status}] {tool}({_args_preview(args)}) → {summary[:80]}")

    def finding(self, finding: dict) -> None:
        """
        Log a new vulnerability finding.

        Args:
            finding : Finding-compatible dict with title, severity, cvss, etc.
        """
        title    = finding.get("title", "Untitled")
        severity = finding.get("severity", "info").upper()
        cvss     = finding.get("cvss", 0.0)
        self._write("finding", {
            "title":    title,
            "severity": severity,
            "cvss":     cvss,
            "port":     finding.get("port", 0),
            "service":  finding.get("service", ""),
            "cve":      finding.get("cve", ""),
        })
        if self.verbose:
            self._print("finding", f"[{severity}] CVSS {cvss} — {title}")

    def error(self, tool: str, message: str) -> None:
        """
        Log a tool or LLM error.

        Args:
            tool    : which tool failed (e.g. "gobuster", "llm")
            message : human-readable error description
        """
        self._write("error", {
            "tool":    tool,
            "message": message[:300],
        })
        if self.verbose:
            self._print("error", f"[{tool}] {message[:120]}")

    def info(self, message: str) -> None:
        """
        Log a free-form informational message.

        Args:
            message : any text observation from the planner or system
        """
        self._write("info", {"message": message})
        if self.verbose:
            self._print("info", message)

    # ── Static: read log ──────────────────────────────────────────────────────

    @staticmethod
    def load_log(path: str) -> list[dict]:
        """
        Load all entries from a kira.log file.
        Returns list of typed entry dicts.
        Malformed lines are silently skipped.

        Each entry has: {"ts": str, "type": str, "data": dict}
        """
        p = Path(path)
        if not p.exists():
            return []

        entries = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

        return entries

    @staticmethod
    def summarise_log(path: str) -> dict:
        """
        Produce summary statistics from a kira.log file.

        Returns:
            {
                "total_events":    int,
                "phase_transitions": list[{"from", "to", "ts"}],
                "actions":         list[{"tool", "ok", "elapsed_s", "ts"}],
                "findings_logged": int,
                "errors":          int,
            }
        """
        entries = KiraLogger.load_log(path)
        result: dict[str, Any] = {
            "total_events":       len(entries),
            "phase_transitions":  [],
            "actions":            [],
            "findings_logged":    0,
            "errors":             0,
        }
        for e in entries:
            t = e.get("type")
            d = e.get("data", {})
            ts = e.get("ts", "")
            if t == "phase":
                result["phase_transitions"].append({
                    "from": d.get("from"), "to": d.get("to"), "ts": ts
                })
            elif t == "action":
                result["actions"].append({
                    "tool":      d.get("tool"),
                    "ok":        d.get("ok"),
                    "elapsed_s": d.get("elapsed_s"),
                    "ts":        ts,
                })
            elif t == "finding":
                result["findings_logged"] += 1
            elif t == "error":
                result["errors"] += 1
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write(self, entry_type: str, data: dict) -> None:
        """Serialize and append one entry to kira.log (thread-safe)."""
        entry = {
            "ts":   _ts(),
            "type": entry_type,
            "data": data,
        }
        line = json.dumps(entry, default=str)
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _print(self, entry_type: str, message: str) -> None:
        """Pretty-print an event to the terminal using rich if available."""
        icon  = _TYPE_ICON.get(entry_type, "·")
        style = _TYPE_STYLE.get(entry_type, "")
        ts    = datetime.now().strftime("%H:%M:%S")

        try:
            from rich.console import Console
            Console().print(
                f"[dim]{ts}[/dim] [{style}]{icon} {message}[/{style}]"
            )
        except ImportError:
            print(f"{ts} {icon} [{entry_type.upper()}] {message}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _args_preview(args: dict, max_len: int = 60) -> str:
    """Compact one-liner representation of args dict for terminal output."""
    if not args:
        return ""
    parts = [f"{k}={str(v)[:20]!r}" for k, v in list(args.items())[:3]]
    preview = ", ".join(parts)
    return preview[:max_len]


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=== logger.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        log = KiraLogger(session_dir=tmp, verbose=True)

        print("[1] Logging phase transition:")
        log.phase("RECON", "ENUM")

        print("\n[2] Logging action (success):")
        log.action(
            "nmap_scan",
            {"target": "10.10.10.5", "flags": "-sV"},
            {"ok": True, "summary": "Found 3 open ports: 22, 80, 3306"},
            elapsed_s=14.2,
        )

        print("\n[3] Logging action (failure):")
        log.action(
            "gobuster_dir",
            {"url": "http://10.10.10.5"},
            {"ok": False, "summary": "FAILED: wordlist not found"},
            elapsed_s=0.1,
        )

        print("\n[4] Logging finding:")
        log.finding({
            "title":    "Apache 2.4.49 Path Traversal (CVE-2021-41773)",
            "severity": "critical",
            "cvss":     9.8,
            "port":     80,
            "service":  "http",
            "cve":      "CVE-2021-41773",
        })

        print("\n[5] Logging error:")
        log.error("msf_exploit", "Module exploit/multi/handler timed out after 120s")

        print("\n[6] Logging info:")
        log.info("Session established — moving to POST_EXPLOIT phase")

        print("\n[7] load_log():")
        entries = KiraLogger.load_log(str(log.log_path))
        assert len(entries) == 6, f"Expected 6 entries, got {len(entries)}"
        for e in entries:
            assert "ts" in e and "type" in e and "data" in e
            print(f"    [{e['ts']}] {e['type']:<10} {str(e['data'])[:60]}")

        print("\n[8] summarise_log():")
        summary = KiraLogger.summarise_log(str(log.log_path))
        print(f"    total_events      = {summary['total_events']}")
        print(f"    phase_transitions = {summary['phase_transitions']}")
        print(f"    actions           = {len(summary['actions'])}")
        print(f"    findings_logged   = {summary['findings_logged']}")
        print(f"    errors            = {summary['errors']}")
        assert summary["total_events"] == 6
        assert len(summary["phase_transitions"]) == 1
        assert summary["findings_logged"] == 1
        assert summary["errors"] == 1

        print("\n[9] load_log on non-existent file returns []:")
        result = KiraLogger.load_log("/nonexistent/kira.log")
        assert result == []
        print("    correctly returned []")

        print("\n[10] Thread-safety: 10 concurrent writes:")
        import threading as _threading
        log2 = KiraLogger(session_dir=tmp + "/t", verbose=False)
        threads = [
            _threading.Thread(target=log2.info, args=(f"msg {i}",))
            for i in range(10)
        ]
        for t in threads: t.start()
        for t in threads: t.join()
        entries2 = KiraLogger.load_log(str(log2.log_path))
        assert len(entries2) == 10, f"Expected 10, got {len(entries2)}"
        print(f"    all 10 entries written safely")

    print("\nAll tests passed.")