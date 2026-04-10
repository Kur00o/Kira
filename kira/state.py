"""
kira/state.py — StateManager
Single source of truth for the entire Kira agent session.
Every module reads from and writes to this; nothing else touches state.json directly.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Phase ordering ────────────────────────────────────────────────────────────

PHASES = ["RECON", "ENUM", "VULN_SCAN", "EXPLOIT", "POST_EXPLOIT", "REPORT", "DONE"]

PHASE_DESCRIPTIONS = {
    "RECON":        "Port scanning and service fingerprinting",
    "ENUM":         "Deep service enumeration and directory brute-force",
    "VULN_SCAN":    "CVE lookup and vulnerability identification",
    "EXPLOIT":      "Active exploitation of discovered vulnerabilities",
    "POST_EXPLOIT": "Post-exploitation enumeration and privilege escalation",
    "REPORT":       "Generating final pentest report",
    "DONE":         "Session complete",
}


# ── Default schema ─────────────────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        # Identity
        "session_id":     None,
        "target":         None,
        "authorized_by":  None,
        "started_at":     None,
        "updated_at":     None,

        # Phase tracking
        "phase":          "RECON",
        "phase_history":  [],          # list of {"phase": ..., "entered_at": ...}

        # Recon results
        "open_ports":     [],          # [22, 80, 443]
        "services":       {},          # {"80": "Apache httpd 2.4.49"}
        "os_guess":       None,
        "hostnames":      [],

        # Enumeration results
        "web_paths":      [],          # ["/admin", "/.git"]
        "smb_shares":     [],
        "usernames":      [],
        "credentials":    [],          # [{"user": "admin", "pass": "...", "service": "ssh"}]

        # Exploitation
        "sessions":       [],          # [{"id": 1, "type": "meterpreter", "opened_at": ...}]
        "current_user":   None,
        "is_root":        False,
        "shell_history":  [],          # commands run on sessions

        # Findings — list of Finding-like dicts
        "findings":       [],

        # Agent memory
        "actions_taken":  [],          # last N actions for LLM context
        "notes":          [],          # free-form agent observations
        "errors":         [],          # tool errors, for self-correction
    }


# ── StateManager ───────────────────────────────────────────────────────────────

class StateManager:
    """
    Thread-safe, file-backed state store for a Kira session.

    Usage:
        sm = StateManager(session_dir="./sessions/my_scan")
        sm.init(target="10.10.10.5", authorized_by="Lab VM")
        sm.update(open_ports=[22, 80], phase="ENUM")
        summary = sm.get_context_summary()
    """

    MAX_ACTIONS_IN_CONTEXT = 3   # last N actions fed to LLM
    MAX_FINDINGS_IN_CONTEXT = 3  # top N scored findings fed to LLM

    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.session_dir / "state.json"
        self._lock = threading.Lock()
        self._state: dict = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def init(self, target: str, authorized_by: str) -> None:
        """
        Create a fresh session. Overwrites any existing state file.
        Always call this when starting a new scan.
        """
        now = _ts()
        self._state = _default_state()
        self._state.update({
            "session_id":    _session_id(target),
            "target":        target,
            "authorized_by": authorized_by,
            "started_at":    now,
            "updated_at":    now,
            "phase_history": [{"phase": "RECON", "entered_at": now}],
        })
        self._save_locked()

    def load(self) -> "StateManager":
        """
        Load state from disk. Returns self for chaining.
        Raises FileNotFoundError if no session exists yet.
        """
        if not self.state_file.exists():
            raise FileNotFoundError(
                f"No session found at {self.state_file}. "
                "Run sm.init() to start a new session."
            )
        with self._lock:
            with open(self.state_file, "r") as f:
                self._state = json.load(f)
        return self

    def save(self) -> None:
        """Force a save to disk (thread-safe)."""
        with self._lock:
            self._save_locked()

    # ── Read ───────────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def get_all(self) -> dict:
        """Return a deep copy of the full state (safe to mutate)."""
        return json.loads(json.dumps(self._state))

    @property
    def target(self) -> str | None:
        return self._state.get("target")

    @property
    def phase(self) -> str:
        return self._state.get("phase", "RECON")

    @property
    def is_root(self) -> bool:
        return self._state.get("is_root", False)

    # ── Write ──────────────────────────────────────────────────────────────────

    def update(self, **kwargs) -> None:
        """
        Merge keyword arguments into state and save.

        Special handling:
          - Lists (open_ports, findings, etc.) are replaced, not appended.
            Use add_finding(), log_action() etc. to append safely.
          - phase: if changed, records entry in phase_history automatically.
        """
        with self._lock:
            new_phase = kwargs.get("phase")
            if new_phase and new_phase != self._state.get("phase"):
                if new_phase not in PHASES:
                    raise ValueError(
                        f"Unknown phase '{new_phase}'. Valid: {PHASES}"
                    )
                self._state["phase_history"].append({
                    "phase":      new_phase,
                    "entered_at": _ts(),
                })

            self._state.update(kwargs)
            self._state["updated_at"] = _ts()
            self._save_locked()

    def advance_phase(self) -> str:
        """Move to the next phase. Returns the new phase name."""
        current = self._state.get("phase", "RECON")
        idx = PHASES.index(current)
        if idx < len(PHASES) - 1:
            next_phase = PHASES[idx + 1]
            self.update(phase=next_phase)
            return next_phase
        return current

    # ── Append helpers (thread-safe) ──────────────────────────────────────────

    def add_finding(self, finding: dict) -> None:
        """
        Append a finding dict to state. Deduplicates by (title, port).
        Expected keys: title, severity, port, service, description,
                       cvss, cve, remediation, exploit_available.
        """
        with self._lock:
            existing_keys = {
                (f.get("title"), f.get("port"))
                for f in self._state["findings"]
            }
            key = (finding.get("title"), finding.get("port"))
            if key not in existing_keys:
                finding.setdefault("discovered_at", _ts())
                finding.setdefault("severity", "info")
                finding.setdefault("cvss", 0.0)
                self._state["findings"].append(finding)
                self._state["updated_at"] = _ts()
                self._save_locked()

    def log_action(self, tool: str, args: dict, result_summary: str) -> None:
        """
        Append an action to the action log (in-state + actions.jsonl).
        Keeps only the last 50 in state to avoid bloating the file.
        """
        action = {
            "tool":           tool,
            "args":           args,
            "result_summary": result_summary[:300],   # truncate long outputs
            "timestamp":      _ts(),
        }
        with self._lock:
            self._state["actions_taken"].append(action)
            self._state["actions_taken"] = self._state["actions_taken"][-50:]
            self._state["updated_at"] = _ts()
            self._save_locked()

        # Also write to the append-only JSONL log
        jsonl_path = self.session_dir / "actions.jsonl"
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(action) + "\n")

    def log_error(self, tool: str, message: str) -> None:
        """Record a tool error so the LLM planner can self-correct."""
        with self._lock:
            self._state["errors"].append({
                "tool":      tool,
                "message":   message[:200],
                "timestamp": _ts(),
            })
            self._state["errors"] = self._state["errors"][-20:]
            self._state["updated_at"] = _ts()
            self._save_locked()

    def add_note(self, note: str) -> None:
        """Planner can write free-form observations here."""
        with self._lock:
            self._state["notes"].append({"note": note, "timestamp": _ts()})
            self._save_locked()

    # ── LLM context ───────────────────────────────────────────────────────────

    def get_context_summary(self) -> str:
        """
        Produce a compact, token-efficient summary of current state
        to inject into the LLM planner prompt.

        Targets ~300-400 tokens — enough context, not enough to confuse a 4B model.
        """
        s = self._state

        lines = [
            "=== KIRA SESSION CONTEXT ===",
            f"Target     : {s.get('target')}",
            f"Phase      : {s.get('phase')} — {PHASE_DESCRIPTIONS.get(s.get('phase', ''), '')}",
            f"User       : {s.get('current_user') or 'none'} | Root: {s.get('is_root', False)}",
            "",
        ]

        # Open ports + services
        ports = s.get("open_ports", [])
        services = s.get("services", {})
        if ports:
            lines.append(f"Open ports : {', '.join(str(p) for p in sorted(ports))}")
            for port, svc in list(services.items())[:8]:
                lines.append(f"  {port}/tcp  {svc}")
        else:
            lines.append("Open ports : none discovered yet")

        # OS guess
        if s.get("os_guess"):
            lines.append(f"OS guess   : {s['os_guess']}")

        # Web paths (top 5)
        web_paths = s.get("web_paths", [])
        if web_paths:
            lines.append(f"Web paths  : {', '.join(web_paths[:5])}")
            if len(web_paths) > 5:
                lines[-1] += f" (+{len(web_paths)-5} more)"

        # Active sessions
        sessions = s.get("sessions", [])
        if sessions:
            lines.append(f"Sessions   : {len(sessions)} active")
            for sess in sessions[:3]:
                lines.append(f"  id={sess.get('id')} type={sess.get('type')}")

        # Top findings by CVSS
        findings = sorted(
            s.get("findings", []),
            key=lambda f: f.get("cvss", 0),
            reverse=True
        )[:self.MAX_FINDINGS_IN_CONTEXT]
        if findings:
            lines.append("")
            lines.append(f"Top findings ({len(findings)}):")
            for f in findings:
                lines.append(
                    f"  [{f.get('severity','?').upper():8s}] "
                    f"CVSS {f.get('cvss', '?'):<4} | "
                    f"port {f.get('port','?')} | "
                    f"{f.get('title','untitled')}"
                )

        # Last N actions
        actions = s.get("actions_taken", [])[-self.MAX_ACTIONS_IN_CONTEXT:]
        if actions:
            lines.append("")
            lines.append("Recent actions:")
            for a in actions:
                lines.append(f"  [{a.get('timestamp','')[:19]}] {a.get('tool')} → {a.get('result_summary','')[:80]}")

        # Last error (for self-correction)
        errors = s.get("errors", [])
        if errors:
            last_err = errors[-1]
            lines.append("")
            lines.append(f"Last error : [{last_err.get('tool')}] {last_err.get('message','')[:100]}")

        lines.append("=== END CONTEXT ===")
        return "\n".join(lines)

    # ── Reporting helpers ─────────────────────────────────────────────────────

    def get_findings_by_severity(self) -> dict:
        """
        Returns findings grouped by severity level.
        Useful for report generation.
        """
        grouped = {"critical": [], "high": [], "medium": [], "low": [], "info": []}
        for f in self._state.get("findings", []):
            sev = f.get("severity", "info").lower()
            grouped.setdefault(sev, []).append(f)
        return grouped

    def session_duration(self) -> str:
        """Human-readable duration since session start."""
        started = self._state.get("started_at")
        if not started:
            return "unknown"
        try:
            start = datetime.fromisoformat(started)
            delta = datetime.now(timezone.utc) - start.replace(tzinfo=timezone.utc)
            mins = int(delta.total_seconds() // 60)
            secs = int(delta.total_seconds() % 60)
            return f"{mins}m {secs}s"
        except Exception:
            return "unknown"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_locked(self) -> None:
        """Write state to disk. Must be called while holding self._lock."""
        tmp = self.state_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._state, f, indent=2, default=str)
        tmp.replace(self.state_file)   # atomic rename

    def __repr__(self) -> str:
        return (
            f"<StateManager target={self._state.get('target')} "
            f"phase={self._state.get('phase')} "
            f"findings={len(self._state.get('findings', []))}>"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _session_id(target: str) -> str:
    safe = target.replace(".", "_").replace("/", "_")
    return f"{safe}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        sm = StateManager(session_dir=tmp)

        # Init
        sm.init(target="10.10.10.5", authorized_by="Lab VM — authorized")
        print("[1] init OK:", sm)

        # Update
        sm.update(
            open_ports=[22, 80, 3306],
            services={"22": "OpenSSH 7.9", "80": "Apache httpd 2.4.49", "3306": "MySQL 5.7"},
            os_guess="Linux 4.x",
            phase="ENUM",
        )
        print("[2] update OK: phase =", sm.phase)

        # Add finding
        sm.add_finding({
            "title":             "Apache 2.4.49 Path Traversal (CVE-2021-41773)",
            "severity":          "critical",
            "cvss":              9.8,
            "cve":               "CVE-2021-41773",
            "port":              80,
            "service":           "http",
            "description":       "Path traversal and RCE via mod_cgi.",
            "exploit_available": True,
            "remediation":       "Update to Apache 2.4.51 or later.",
        })

        # Dedup check
        sm.add_finding({
            "title": "Apache 2.4.49 Path Traversal (CVE-2021-41773)",
            "port":  80,
        })
        assert len(sm.get("findings")) == 1, "Dedup failed"
        print("[3] add_finding + dedup OK")

        # Log action
        sm.log_action("nmap_scan", {"target": "10.10.10.5"}, "Found 3 open ports: 22, 80, 3306")
        print("[4] log_action OK")

        # Advance phase
        sm.advance_phase()
        print("[5] advance_phase OK: phase =", sm.phase)

        # Reload from disk
        sm2 = StateManager(session_dir=tmp).load()
        assert sm2.phase == "VULN_SCAN"
        assert len(sm2.get("findings")) == 1
        print("[6] reload from disk OK:", sm2)

        # Context summary
        print("\n" + sm2.get_context_summary())

    print("\nAll tests passed.")