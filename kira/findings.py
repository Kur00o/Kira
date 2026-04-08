"""
kira/findings.py — Finding dataclass + KnowledgeBase
=====================================================
Typed container for every vulnerability discovered during a Kira session.
KnowledgeBase deduplicates by (title, port), scores by CVSS, and feeds
both the planner context and the report generator.

Usage:
    from findings import Finding, KnowledgeBase

    kb = KnowledgeBase()
    kb.add(Finding(
        title="Apache 2.4.49 Path Traversal",
        severity="critical",
        port=80,
        service="http",
        cvss=9.8,
        cve="CVE-2021-41773",
        description="Path traversal and RCE via mod_cgi.",
        exploit_available=True,
        remediation="Upgrade to Apache 2.4.51+",
    ))
    print(kb.top(3))
    print(kb.by_severity())
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
import json


# ── Constants ──────────────────────────────────────────────────────────────────

VALID_SEVERITIES = ("critical", "high", "medium", "low", "info")

# CVSS floor per severity band — used for validation + auto-scoring
SEVERITY_CVSS_FLOOR = {
    "critical": 9.0,
    "high":     7.0,
    "medium":   4.0,
    "low":      0.1,
    "info":     0.0,
}


# ── Finding ────────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    """
    A single, typed vulnerability or observation.

    Required:
        title    : short descriptive name
        severity : critical / high / medium / low / info
        port     : affected port number (0 for host-level findings)

    Optional but highly recommended:
        service         : service name ("http", "ssh", …)
        description     : what was found and why it matters
        cvss            : CVSS 3.x base score (0.0 – 10.0)
        cve             : CVE identifier ("CVE-2021-41773")
        exploit_available : True if a working exploit exists
        remediation     : how to fix it
    """
    title:             str
    severity:          str
    port:              int

    service:           str  = ""
    description:       str  = ""
    cvss:              float = 0.0
    cve:               str  = ""
    exploit_available: bool  = False
    remediation:       str  = ""
    discovered_at:     str  = field(default_factory=lambda: _ts())

    def __post_init__(self):
        # Normalise severity
        sev = self.severity.strip().lower()
        if sev not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{self.severity}'. "
                f"Must be one of: {VALID_SEVERITIES}"
            )
        self.severity = sev

        # Validate CVSS range
        if not (0.0 <= self.cvss <= 10.0):
            raise ValueError(
                f"CVSS score must be between 0.0 and 10.0, got {self.cvss}"
            )

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def dedup_key(self) -> tuple:
        """Deduplication key: (title, port)."""
        return (self.title.strip().lower(), self.port)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def one_liner(self) -> str:
        """Compact string for LLM context injection."""
        exploit_flag = " [EXPLOIT AVAILABLE]" if self.exploit_available else ""
        cve_str = f" {self.cve}" if self.cve else ""
        return (
            f"[{self.severity.upper():8s}] CVSS {self.cvss:<4.1f} | "
            f"port {self.port} {self.service:<8} | "
            f"{self.title}{cve_str}{exploit_flag}"
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        """Reconstruct from a plain dict (e.g. loaded from state.json)."""
        return cls(
            title=d.get("title", "Untitled"),
            severity=d.get("severity", "info"),
            port=int(d.get("port", 0)),
            service=d.get("service", ""),
            description=d.get("description", ""),
            cvss=float(d.get("cvss", 0.0)),
            cve=d.get("cve", ""),
            exploit_available=bool(d.get("exploit_available", False)),
            remediation=d.get("remediation", ""),
            discovered_at=d.get("discovered_at", _ts()),
        )


# ── KnowledgeBase ──────────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    In-memory store for all Findings in a session.

    Guarantees:
      - No duplicate (title, port) pairs
      - Sorted access by CVSS score
      - Grouped views by severity for report generation
      - Serialisable to/from plain dicts for StateManager integration
    """

    def __init__(self):
        self._findings: list[Finding] = []
        self._keys: set[tuple] = set()   # dedup tracker

    # ── Add / remove ──────────────────────────────────────────────────────────

    def add(self, finding: Finding) -> bool:
        """
        Add a Finding. Silently ignores duplicates.

        Returns True if the finding was added, False if it was a duplicate.
        """
        key = finding.dedup_key
        if key in self._keys:
            return False
        self._keys.add(key)
        self._findings.append(finding)
        return True

    def add_from_dict(self, d: dict) -> bool:
        """Convenience: construct Finding from dict and add."""
        return self.add(Finding.from_dict(d))

    def remove(self, title: str, port: int) -> bool:
        """Remove a finding by (title, port). Returns True if found."""
        key = (title.strip().lower(), port)
        if key not in self._keys:
            return False
        self._findings = [f for f in self._findings if f.dedup_key != key]
        self._keys.discard(key)
        return True

    def clear(self):
        self._findings.clear()
        self._keys.clear()

    # ── Queries ───────────────────────────────────────────────────────────────

    def top(self, n: int = 5) -> list[Finding]:
        """Return top-N findings sorted by CVSS score descending."""
        return sorted(self._findings, key=lambda f: f.cvss, reverse=True)[:n]

    def by_severity(self) -> dict[str, list[Finding]]:
        """
        Return all findings grouped by severity level.
        Keys are always present (empty list if no findings at that level).

        Returns:
            {
                "critical": [...],
                "high":     [...],
                "medium":   [...],
                "low":      [...],
                "info":     [...],
            }
        """
        grouped: dict[str, list[Finding]] = {s: [] for s in VALID_SEVERITIES}
        for f in self._findings:
            grouped[f.severity].append(f)
        # Sort each group by CVSS descending
        for sev in grouped:
            grouped[sev].sort(key=lambda f: f.cvss, reverse=True)
        return grouped

    def by_port(self, port: int) -> list[Finding]:
        """All findings for a specific port."""
        return [f for f in self._findings if f.port == port]

    def exploitable(self) -> list[Finding]:
        """All findings where exploit_available is True, sorted by CVSS."""
        return sorted(
            [f for f in self._findings if f.exploit_available],
            key=lambda f: f.cvss,
            reverse=True,
        )

    def count(self) -> int:
        return len(self._findings)

    def all(self) -> list[Finding]:
        """All findings sorted by CVSS descending."""
        return sorted(self._findings, key=lambda f: f.cvss, reverse=True)

    # ── StateManager integration ──────────────────────────────────────────────

    def to_state_dicts(self) -> list[dict]:
        """
        Convert all findings to plain dicts for StateManager.update(findings=...).
        Round-trips cleanly through Finding.from_dict().
        """
        return [f.to_dict() for f in self.all()]

    @classmethod
    def from_state_dicts(cls, dicts: list[dict]) -> "KnowledgeBase":
        """Rebuild a KnowledgeBase from StateManager's findings list."""
        kb = cls()
        for d in dicts:
            try:
                kb.add(Finding.from_dict(d))
            except (ValueError, KeyError):
                pass   # skip malformed entries gracefully
        return kb

    # ── LLM context ───────────────────────────────────────────────────────────

    def context_block(self, n: int = 5) -> str:
        """
        Compact multi-line string for injection into LLM planner prompts.
        Shows top-N findings by CVSS.
        """
        top = self.top(n)
        if not top:
            return "Findings: none yet"
        lines = [f"Findings ({self.count()} total, top {len(top)} shown):"]
        for f in top:
            lines.append(f"  {f.one_liner()}")
        return "\n".join(lines)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._findings)

    def __repr__(self) -> str:
        counts = {s: 0 for s in VALID_SEVERITIES}
        for f in self._findings:
            counts[f.severity] += 1
        summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
        return f"<KnowledgeBase [{summary}] total={len(self._findings)}>"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== findings.py smoke test ===\n")

    # [1] Valid Finding construction
    print("[1] Construct valid Finding")
    f1 = Finding(
        title="Apache 2.4.49 Path Traversal",
        severity="critical",
        port=80,
        service="http",
        cvss=9.8,
        cve="CVE-2021-41773",
        description="Path traversal and RCE via mod_cgi on affected Apache versions.",
        exploit_available=True,
        remediation="Upgrade to Apache 2.4.51 or later.",
    )
    print(f"   one_liner: {f1.one_liner()}")
    assert f1.severity == "critical"
    assert f1.dedup_key == ("apache 2.4.49 path traversal", 80)

    # [2] Severity validator
    print("\n[2] Severity validator rejects bad values")
    try:
        Finding(title="Bad", severity="EXTREME", port=0)
        assert False, "Should have raised"
    except ValueError as e:
        print(f"   Correctly rejected: {e}")

    # [3] CVSS range validator
    print("\n[3] CVSS range validator")
    try:
        Finding(title="Bad", severity="high", port=80, cvss=11.0)
        assert False, "Should have raised"
    except ValueError as e:
        print(f"   Correctly rejected: {e}")

    # [4] KnowledgeBase dedup
    print("\n[4] KnowledgeBase dedup")
    kb = KnowledgeBase()
    added = kb.add(f1)
    assert added is True
    added_again = kb.add(f1)
    assert added_again is False
    assert len(kb) == 1
    print(f"   Dedup OK — count={len(kb)}")

    # [5] Add more findings
    print("\n[5] Adding mixed-severity findings")
    kb.add(Finding(title="SSH Weak Ciphers", severity="medium", port=22, service="ssh", cvss=5.3))
    kb.add(Finding(title="FTP Anonymous Login", severity="high", port=21, service="ftp", cvss=7.5, exploit_available=True))
    kb.add(Finding(title="Open Redirect", severity="low", port=80, service="http", cvss=3.1))
    kb.add(Finding(title="SSL Certificate Expired", severity="info", port=443, service="https", cvss=0.0))
    print(f"   {kb}")

    # [6] top()
    print("\n[6] top(3) by CVSS")
    for f in kb.top(3):
        print(f"   {f.one_liner()}")

    # [7] by_severity()
    print("\n[7] by_severity()")
    grouped = kb.by_severity()
    for sev, findings in grouped.items():
        if findings:
            print(f"   {sev}: {[f.title for f in findings]}")

    # [8] exploitable()
    print("\n[8] exploitable()")
    for f in kb.exploitable():
        print(f"   {f.one_liner()}")

    # [9] State round-trip
    print("\n[9] to_state_dicts / from_state_dicts round-trip")
    dicts = kb.to_state_dicts()
    kb2 = KnowledgeBase.from_state_dicts(dicts)
    assert len(kb2) == len(kb)
    assert kb2.top(1)[0].title == kb.top(1)[0].title
    print(f"   Round-trip OK — {kb2}")

    # [10] context_block for LLM
    print("\n[10] context_block()")
    print(kb.context_block(3))

    print("\nAll tests passed.")