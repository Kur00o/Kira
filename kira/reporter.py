"""
kira/reporter.py — ReportGenerator
=====================================
Reads state.json + actions.jsonl + kira.log from the session directory.
Uses the LLM to write professional narrative text.
Renders a clean pentest report in both Markdown and HTML.

Report structure:
  1. Cover          — target, date, authorized_by, severity counts
  2. Exec Summary   ← LLM-written (1 call, ~200 words)
  3. Timeline       — built from actions.jsonl chronologically
  4. Risk Table     — findings sorted by CVSS with severity badges
  5. Finding detail ← LLM-written per finding (1 call each, capped at 10)
  6. Appendix       — raw nmap/gobuster output excerpts

Usage:
    from reporter import ReportGenerator
    reporter = ReportGenerator(session_dir="./sessions/scan1", llm=llm_client)
    paths = reporter.generate()
    print(paths.markdown)
    print(paths.html)
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── ReportPaths ────────────────────────────────────────────────────────────────

@dataclass
class ReportPaths:
    markdown: str
    html:     str


# ── Severity ordering ──────────────────────────────────────────────────────────

_SEV_ORDER = ["critical", "high", "medium", "low", "info"]

_SEV_EMOJI = {
    "critical": "🔴",
    "high":     "🟠",
    "medium":   "🟡",
    "low":      "🟢",
    "info":     "🔵",
}

# Max LLM calls for per-finding writeups
MAX_FINDING_LLM_CALLS = 10


# ── ReportGenerator ────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Reads session artefacts, calls the LLM for narrative sections,
    and renders report.md + report.html.

    Parameters
    ----------
    session_dir : path to the session directory
                  (must contain state.json, optionally actions.jsonl + kira.log)
    llm         : LLMClient instance — used for exec summary + finding writeups
                  Pass None to skip all LLM calls (raw data only)
    """

    TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html.j2"

    def __init__(self, session_dir: str, llm=None):
        self.session_dir = Path(session_dir)
        self.llm         = llm

        self._state:   dict       = {}
        self._actions: list[dict] = []
        self._log:     list[dict] = []

    # ── Public ─────────────────────────────────────────────────────────────────

    def generate(self) -> ReportPaths:
        """
        Main entry point. Loads data, calls LLM, writes files.

        Returns ReportPaths with absolute paths to both output files.
        """
        self._load_data()

        findings     = self._sorted_findings()
        timeline     = self._build_timeline()
        sev_counts   = self._severity_counts(findings)
        appendix     = self._build_appendix()

        exec_summary = self._write_exec_summary(findings, sev_counts)
        findings     = self._enrich_findings_with_writeups(findings)

        md_path   = self._render_markdown(findings, timeline, sev_counts, exec_summary, appendix)
        html_path = self._render_html(findings, timeline, sev_counts, exec_summary, appendix)

        return ReportPaths(markdown=str(md_path), html=str(html_path))

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Load state.json, actions.jsonl, kira.log from session_dir."""
        state_file = self.session_dir / "state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                self._state = json.load(f)

        actions_file = self.session_dir / "actions.jsonl"
        if actions_file.exists():
            for line in actions_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._actions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        log_file = self.session_dir / "kira.log"
        if log_file.exists():
            for line in log_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self._log.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # ── Data helpers ───────────────────────────────────────────────────────────

    def _sorted_findings(self) -> list[dict]:
        """Return findings sorted: severity order first, then CVSS descending."""
        findings = list(self._state.get("findings", []))
        def sort_key(f):
            sev_rank = _SEV_ORDER.index(f.get("severity", "info").lower()) \
                       if f.get("severity", "info").lower() in _SEV_ORDER else 99
            return (sev_rank, -float(f.get("cvss", 0.0)))
        findings.sort(key=sort_key)
        return findings

    def _severity_counts(self, findings: list[dict]) -> dict[str, int]:
        """Count findings per severity level."""
        counts = {s: 0 for s in _SEV_ORDER}
        for f in findings:
            sev = f.get("severity", "info").lower()
            counts[sev] = counts.get(sev, 0) + 1
        return counts

    def _build_timeline(self) -> list[dict]:
        """
        Build chronological timeline from kira.log action entries.
        Falls back to actions.jsonl if kira.log has no action entries.
        """
        # Prefer kira.log typed action entries
        timeline = []
        for entry in self._log:
            if entry.get("type") == "action":
                d = entry.get("data", {})
                timeline.append({
                    "ts":        entry.get("ts", ""),
                    "tool":      d.get("tool", ""),
                    "ok":        d.get("ok", False),
                    "summary":   d.get("summary", ""),
                    "elapsed_s": d.get("elapsed_s", 0.0),
                })

        # Fall back to actions.jsonl
        if not timeline and self._actions:
            for entry in self._actions:
                timeline.append({
                    "ts":        entry.get("timestamp", ""),
                    "tool":      entry.get("tool", ""),
                    "ok":        entry.get("ok", False),
                    "summary":   entry.get("summary", ""),
                    "elapsed_s": entry.get("elapsed_s", 0.0),
                })

        return timeline

    def _build_appendix(self) -> list[dict]:
        """
        Collect raw tool output files from session_dir/raw/.
        Returns list of {"label": str, "content": str} dicts.
        Truncates each file to 4000 chars to keep the report readable.
        """
        raw_dir = self.session_dir / "raw"
        sections = []
        if not raw_dir.exists():
            return sections

        # Prioritize nmap XML/txt and gobuster files
        priority = ["nmap", "gobuster", "enum4linux", "whatweb"]
        files = sorted(raw_dir.iterdir(), key=lambda p: p.stat().st_mtime)

        # Sort priority files first
        def file_priority(p):
            name = p.name.lower()
            for i, pref in enumerate(priority):
                if pref in name:
                    return i
            return len(priority)

        files.sort(key=file_priority)

        for f in files[:6]:       # max 6 files in appendix
            if not f.is_file():
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                if len(content) > 4000:
                    content = content[:4000] + "\n\n... [truncated]"
                sections.append({
                    "label":   f.name,
                    "content": content,
                })
            except Exception:
                pass

        return sections

    # ── LLM narrative generation ───────────────────────────────────────────────

    def _write_exec_summary(
        self,
        findings:   list[dict],
        sev_counts: dict[str, int],
    ) -> str:
        """
        Generate 2–3 paragraph executive summary via LLM.
        Falls back to a data-driven template if LLM is unavailable.
        """
        if self.llm is None:
            return self._fallback_exec_summary(findings, sev_counts)

        target       = self._state.get("target", "the target")
        authorized   = self._state.get("authorized_by", "authorized party")
        total        = len(findings)
        critical_n   = sev_counts.get("critical", 0)
        high_n       = sev_counts.get("high", 0)
        exploitable  = [f for f in findings if f.get("exploit_available")]
        top_findings = "\n".join(
            f"- {f.get('title')} (CVSS {f.get('cvss')}, {f.get('severity')})"
            for f in findings[:5]
        )

        prompt = textwrap.dedent(f"""
            Write a professional penetration test executive summary for a security report.

            Target: {target}
            Authorized by: {authorized}
            Total findings: {total}
            Critical: {critical_n}, High: {high_n}
            Exploitable vulnerabilities: {len(exploitable)}
            Top findings:
            {top_findings or 'None recorded.'}

            Write exactly 2-3 paragraphs of professional security assessment prose.
            Tone: formal, direct, security-professional.
            Do NOT use bullet points. Do NOT use markdown headers.
            Do NOT start with "I" or reference yourself.
            Around 150-200 words total.
        """).strip()

        try:
            raw = self.llm.generate_text(prompt, temperature=0.3, max_tokens=400)
            return raw.strip() if raw else self._fallback_exec_summary(findings, sev_counts)
        except Exception:
            return self._fallback_exec_summary(findings, sev_counts)

    def _fallback_exec_summary(
        self,
        findings:   list[dict],
        sev_counts: dict[str, int],
    ) -> str:
        """Data-driven executive summary when LLM is unavailable."""
        target     = self._state.get("target", "the target system")
        total      = len(findings)
        critical_n = sev_counts.get("critical", 0)
        high_n     = sev_counts.get("high", 0)
        exploitable = sum(1 for f in findings if f.get("exploit_available"))

        severity_str = []
        for sev in _SEV_ORDER:
            n = sev_counts.get(sev, 0)
            if n > 0:
                severity_str.append(f"{n} {sev}")
        sev_summary = ", ".join(severity_str) if severity_str else "no vulnerabilities"

        para1 = (
            f"An authorized penetration test was conducted against {target}. "
            f"The assessment identified {total} security finding{'s' if total != 1 else ''} "
            f"across the target environment: {sev_summary}."
        )
        para2 = (
            f"A total of {exploitable} finding{'s' if exploitable != 1 else ''} "
            f"{'have' if exploitable != 1 else 'has'} confirmed exploit availability, "
            f"representing immediate risk. "
            + (
                f"The {critical_n} critical and {high_n} high severity findings should be "
                f"remediated as a matter of priority before the system is placed into production."
                if critical_n + high_n > 0
                else "No critical or high severity findings were identified."
            )
        )
        para3 = (
            "Detailed technical findings, reproduction steps, and remediation guidance "
            "are provided in the sections below. The remediation recommendations should be "
            "reviewed and implemented in order of CVSS score severity."
        )
        return f"{para1}\n\n{para2}\n\n{para3}"

    def _enrich_findings_with_writeups(self, findings: list[dict]) -> list[dict]:
        """
        Add LLM-written 'writeup' and 'impact' fields to each finding.
        Caps at MAX_FINDING_LLM_CALLS to keep runtime reasonable.
        Returns the enriched list (safe mutation of copies).
        """
        if self.llm is None:
            return findings

        enriched = []
        for i, f in enumerate(findings):
            f = dict(f)     # copy — don't mutate state
            if i < MAX_FINDING_LLM_CALLS:
                f["writeup"], f["impact"] = self._llm_finding_writeup(f)
            else:
                f.setdefault("writeup", "")
                f.setdefault("impact",  "")
            enriched.append(f)
        return enriched

    def _llm_finding_writeup(self, finding: dict) -> tuple[str, str]:
        """
        Generate 'analysis' text and 'impact' text for a single finding.
        Returns (writeup, impact) strings. Falls back to empty strings on error.
        """
        prompt = textwrap.dedent(f"""
            Write a brief technical analysis for the following security finding.
            Respond with ONLY a JSON object with exactly two keys:
            "writeup": one concise paragraph describing the technical finding (2-3 sentences)
            "impact":  one sentence describing the business/security impact

            Finding:
            Title:       {finding.get('title', '')}
            Severity:    {finding.get('severity', '')}
            CVSS:        {finding.get('cvss', '')}
            Service:     {finding.get('service', '')} on port {finding.get('port', '')}
            CVE:         {finding.get('cve', 'N/A')}
            Description: {finding.get('description', 'N/A')}

            Respond ONLY with raw JSON. No markdown. No prose outside the JSON.
            Example: {{"writeup": "...", "impact": "..."}}
        """).strip()

        try:
            raw = self.llm.generate_text(prompt, temperature=0.3, max_tokens=300)
            if not raw:
                return "", ""
            # Strip markdown fences if model ignores instructions
            raw = raw.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1]).strip()
            parsed = json.loads(raw)
            return (
                str(parsed.get("writeup", "")).strip(),
                str(parsed.get("impact",  "")).strip(),
            )
        except Exception:
            return "", ""

    # ── Markdown renderer ──────────────────────────────────────────────────────

    def _render_markdown(
        self,
        findings:     list[dict],
        timeline:     list[dict],
        sev_counts:   dict[str, int],
        exec_summary: str,
        appendix:     list[dict],
    ) -> Path:
        """Write report.md and return its path."""
        lines: list[str] = []
        state  = self._state
        target = state.get("target", "Unknown")
        date   = _fmt_date(state.get("started_at"))
        auth   = state.get("authorized_by", "Unknown")

        # Cover
        lines += [
            f"# Penetration Test Report",
            f"",
            f"**Target:** `{target}`  ",
            f"**Date:** {date}  ",
            f"**Authorized by:** {auth}  ",
            f"**Duration:** {self._duration()}  ",
            f"**Total findings:** {len(findings)}  ",
            f"",
        ]

        # Severity summary
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for sev in _SEV_ORDER:
            n = sev_counts.get(sev, 0)
            lines.append(f"| {_SEV_EMOJI.get(sev,'')} {sev.capitalize()} | {n} |")
        lines += ["", "---", ""]

        # Executive Summary
        lines += [
            "## Executive Summary",
            "",
        ]
        for para in exec_summary.split("\n\n"):
            lines.append(para.strip())
            lines.append("")
        lines += ["---", ""]

        # Timeline
        lines += [
            "## Attack Timeline",
            "",
            "| Timestamp | Tool | Status | Summary | Duration |",
            "|-----------|------|--------|---------|----------|",
        ]
        for e in timeline:
            status = "✅" if e.get("ok") else "❌"
            ts     = str(e.get("ts", ""))[:19]
            tool   = e.get("tool", "")
            summ   = str(e.get("summary", ""))[:80].replace("|", "\\|")
            elapsed = e.get("elapsed_s", "")
            lines.append(f"| `{ts}` | `{tool}` | {status} | {summ} | {elapsed}s |")
        lines += ["", "---", ""]

        # Risk Table
        lines += [
            "## Risk Table",
            "",
            "| # | Vulnerability | Severity | CVSS | Port | CVE | Exploit |",
            "|---|---------------|----------|------|------|-----|---------|",
        ]
        for i, f in enumerate(findings, 1):
            sev  = f.get("severity", "info")
            emoji = _SEV_EMOJI.get(sev.lower(), "")
            title = f.get("title", "")[:60].replace("|", "\\|")
            cvss  = f.get("cvss", 0.0)
            port  = f.get("port", "")
            cve   = f.get("cve", "—") or "—"
            expl  = "✅" if f.get("exploit_available") else "—"
            lines.append(f"| {i} | {title} | {emoji} {sev.capitalize()} | {cvss} | {port} | {cve} | {expl} |")
        lines += ["", "---", ""]

        # Finding details
        lines += ["## Finding Detail", ""]
        for i, f in enumerate(findings, 1):
            sev   = f.get("severity", "info")
            emoji = _SEV_EMOJI.get(sev.lower(), "")
            lines += [
                f"### {i}. {f.get('title', 'Untitled')}",
                f"",
                f"**Severity:** {emoji} {sev.capitalize()}  |  "
                f"**CVSS:** {f.get('cvss', 0.0)}  |  "
                f"**Port:** {f.get('port', '?')}/{f.get('service', '?')}",
            ]
            if f.get("cve"):
                lines.append(f"**CVE:** {f['cve']}")
            if f.get("exploit_available"):
                lines.append("**⚠️ Exploit Available**")
            lines += [""]
            if f.get("description"):
                lines += ["**Description**", "", f.get("description", ""), ""]
            if f.get("writeup"):
                lines += ["**Analysis**", "", f["writeup"], ""]
            if f.get("impact"):
                lines += ["**Impact**", "", f["impact"], ""]
            if f.get("remediation"):
                lines += ["**Remediation**", "", f.get("remediation", ""), ""]
            lines += ["---", ""]

        # Appendix
        lines += ["## Appendix — Raw Tool Output", ""]
        for app in appendix:
            lines += [
                f"### {app['label']}",
                "",
                "```",
                app["content"],
                "```",
                "",
            ]

        content = "\n".join(lines)
        out_path = self.session_dir / "report.md"
        out_path.write_text(content, encoding="utf-8")
        return out_path

    # ── HTML renderer ──────────────────────────────────────────────────────────

    def _render_html(
        self,
        findings:     list[dict],
        timeline:     list[dict],
        sev_counts:   dict[str, int],
        exec_summary: str,
        appendix:     list[dict],
    ) -> Path:
        """
        Render report.html using Jinja2 if available,
        falling back to an inline template if Jinja2 is not installed.
        """
        state  = self._state
        target = state.get("target", "Unknown")
        date   = _fmt_date(state.get("started_at"))
        auth   = state.get("authorized_by", "Unknown")

        # Enrich findings with defaults for template
        for f in findings:
            f.setdefault("writeup", "")
            f.setdefault("impact",  "")
            f.setdefault("cve",     "")
            f.setdefault("service", "")

        exec_paragraphs = [p.strip() for p in exec_summary.split("\n\n") if p.strip()]

        template_vars = {
            "target":                   target,
            "date":                     date,
            "authorized_by":            auth,
            "duration":                 self._duration(),
            "total_findings":           len(findings),
            "severity_counts":          sev_counts,
            "exec_summary_paragraphs":  exec_paragraphs,
            "timeline_entries":         timeline,
            "findings":                 findings,
            "appendix_sections":        appendix,
        }

        out_path = self.session_dir / "report.html"

        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
            template_dir = self.TEMPLATE_PATH.parent
            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(["html"]),
            )
            template = env.get_template(self.TEMPLATE_PATH.name)
            html = template.render(**template_vars)
        except ImportError:
            # Jinja2 not installed — use fallback inline renderer
            html = self._render_html_fallback(template_vars)
        except Exception as e:
            html = self._render_html_fallback(template_vars)

        out_path.write_text(html, encoding="utf-8")
        return out_path

    def _render_html_fallback(self, v: dict) -> str:
        """
        Minimal but functional HTML report when Jinja2 is not available.
        Reads and inlines the template manually using simple string replacement.
        """
        # Try to load the template file for string-based rendering
        if self.TEMPLATE_PATH.exists():
            try:
                tmpl = self.TEMPLATE_PATH.read_text(encoding="utf-8")
                # Simple variable substitution for non-loop parts
                tmpl = tmpl.replace("{{ target }}", _esc(v["target"]))
                tmpl = tmpl.replace("{{ date }}", _esc(v["date"]))
                tmpl = tmpl.replace("{{ authorized_by }}", _esc(v["authorized_by"]))
                tmpl = tmpl.replace("{{ duration }}", _esc(v["duration"]))
                tmpl = tmpl.replace("{{ total_findings }}", str(v["total_findings"]))
                # For loops/complex blocks: replace with pre-rendered HTML chunks
                tmpl = _replace_jinja_blocks(tmpl, v)
                return tmpl
            except Exception:
                pass

        # Absolute fallback — pure Python HTML
        return _minimal_html_report(v)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _duration(self) -> str:
        """Human-readable session duration from state."""
        started = self._state.get("started_at")
        updated = self._state.get("updated_at")
        if not started:
            return "unknown"
        try:
            start = datetime.fromisoformat(started.replace("Z", "+00:00"))
            end   = datetime.fromisoformat(updated.replace("Z", "+00:00")) \
                    if updated else datetime.now(timezone.utc)
            delta = end - start
            mins  = int(delta.total_seconds() // 60)
            secs  = int(delta.total_seconds() % 60)
            return f"{mins}m {secs}s"
        except Exception:
            return "unknown"


# ── LLMClient extension: generate_text ────────────────────────────────────────
# Patch generate_text onto whatever llm object is passed in if it doesn't exist.
# This avoids modifying llm.py directly but gracefully adds the method.

def _patch_llm_generate_text(llm) -> None:
    """
    If the LLMClient doesn't have generate_text(), add it dynamically.
    generate_text() is the non-JSON mode used by the reporter.
    """
    if llm is None or hasattr(llm, "generate_text"):
        return

    import types

    def generate_text(self, prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
        """
        Send a free-text prompt and return the raw string response.
        Unlike ask(), does NOT enforce JSON output.
        """
        import requests as _req
        payload = {
            "model":  self.model,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            resp = _req.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            return ""

    llm.generate_text = types.MethodType(generate_text, llm)


# ── HTML fallback helpers ──────────────────────────────────────────────────────

_SEV_COLORS = {
    "critical": "#e84545",
    "high":     "#f5a623",
    "medium":   "#f5e642",
    "low":      "#4caf86",
    "info":     "#5c8dde",
}

def _replace_jinja_blocks(tmpl: str, v: dict) -> str:
    """
    Replace Jinja2 loop blocks in the template with pre-rendered HTML.
    Handles {% for %} / {% endfor %} and {% if %} / {% endif %} blocks.
    This is a best-effort approach for the fallback path.
    """
    import re

    # severity_counts loop
    sev_pills_html = ""
    for sev, count in v["severity_counts"].items():
        color = _SEV_COLORS.get(sev, "#888")
        sev_pills_html += (
            f'<div class="sev-pill">'
            f'<div class="sev-dot" style="background:{color};"></div>'
            f'<div><div class="sev-count">{count}</div>'
            f'<div class="sev-label">{sev}</div></div></div>'
        )
    tmpl = re.sub(
        r'\{%.*?for sev.*?%\}.*?\{%.*?endfor.*?%\}',
        sev_pills_html,
        tmpl, flags=re.DOTALL
    )

    # exec_summary loop
    summary_html = "".join(
        f"<p>{_esc(p)}</p>" for p in v["exec_summary_paragraphs"]
    )
    tmpl = re.sub(
        r'\{%.*?for paragraph.*?%\}.*?\{%.*?endfor.*?%\}',
        summary_html,
        tmpl, flags=re.DOTALL
    )

    # Strip all remaining Jinja blocks
    tmpl = re.sub(r'\{%.*?%\}', '', tmpl, flags=re.DOTALL)
    tmpl = re.sub(r'\{\{.*?\}\}', '', tmpl, flags=re.DOTALL)

    return tmpl


def _minimal_html_report(v: dict) -> str:
    """Absolute fallback: minimal but complete HTML report."""
    findings_html = ""
    for i, f in enumerate(v["findings"], 1):
        sev   = f.get("severity", "info").lower()
        color = _SEV_COLORS.get(sev, "#888")
        findings_html += f"""
        <div style="border:1px solid #2a2f42;border-radius:6px;margin-bottom:1rem;overflow:hidden;">
          <div style="background:#1c2030;padding:0.75rem 1rem;display:flex;gap:0.75rem;align-items:center;">
            <span style="background:{color}22;color:{color};border:1px solid {color}44;
                  font-family:monospace;font-size:0.7rem;padding:0.15rem 0.5rem;border-radius:3px;
                  text-transform:uppercase;">{_esc(sev)}</span>
            <strong style="color:#fff;">{_esc(f.get('title',''))}</strong>
            <span style="color:#6b7289;font-size:0.8rem;margin-left:auto;font-family:monospace;">
              CVSS {f.get('cvss',0)} · port {f.get('port','')}
            </span>
          </div>
          <div style="padding:1rem;">
            <p style="color:#d4d8e8;">{_esc(f.get('description',''))}</p>
            {f'<p style="color:#d4d8e8;margin-top:0.5rem;"><strong>Remediation:</strong> {_esc(f.get("remediation",""))}</p>' if f.get('remediation') else ''}
          </div>
        </div>"""

    timeline_rows = ""
    for e in v["timeline_entries"]:
        status_color = "#4caf86" if e.get("ok") else "#e84545"
        status_text  = "OK" if e.get("ok") else "FAIL"
        timeline_rows += f"""
        <tr>
          <td style="font-family:monospace;font-size:0.78rem;color:#6b7289;">{_esc(str(e.get('ts',''))[:19])}</td>
          <td style="font-family:monospace;color:#ff7b54;">{_esc(e.get('tool',''))}</td>
          <td><span style="color:{status_color};font-family:monospace;font-size:0.75rem;">{status_text}</span></td>
          <td style="color:#d4d8e8;">{_esc(str(e.get('summary',''))[:80])}</td>
          <td style="font-family:monospace;font-size:0.78rem;color:#6b7289;">{e.get('elapsed_s','')}s</td>
        </tr>"""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"/>
<title>Kira Report — {_esc(v['target'])}</title>
<style>
  body{{font-family:sans-serif;background:#0d0f14;color:#d4d8e8;margin:0;padding:2rem 4rem;}}
  h1{{color:#fff;font-size:2rem;margin-bottom:0.5rem;}}
  h2{{color:#fff;font-size:1.3rem;margin:2rem 0 1rem;border-bottom:1px solid #2a2f42;padding-bottom:0.5rem;}}
  table{{width:100%;border-collapse:collapse;margin-bottom:1rem;}}
  th{{text-align:left;color:#6b7289;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;
      padding:0.5rem;border-bottom:1px solid #2a2f42;}}
  td{{padding:0.6rem 0.5rem;border-bottom:1px solid #1c2030;}}
  pre{{background:#141720;border:1px solid #2a2f42;padding:1rem;overflow-x:auto;
       font-size:0.78rem;border-radius:4px;max-height:300px;}}
</style></head><body>
<h1>Penetration Test Report</h1>
<p><strong>Target:</strong> {_esc(v['target'])} &nbsp;·&nbsp;
   <strong>Date:</strong> {_esc(v['date'])} &nbsp;·&nbsp;
   <strong>Authorized by:</strong> {_esc(v['authorized_by'])}</p>
<h2>Executive Summary</h2>
{''.join(f'<p style="margin-bottom:1rem;">{_esc(p)}</p>' for p in v['exec_summary_paragraphs'])}
<h2>Attack Timeline</h2>
<table><thead><tr><th>Timestamp</th><th>Tool</th><th>Status</th><th>Summary</th><th>Duration</th></tr></thead>
<tbody>{timeline_rows}</tbody></table>
<h2>Findings ({len(v['findings'])})</h2>
{findings_html}
{''.join(f'<h2>Appendix — {_esc(a["label"])}</h2><pre>{_esc(a["content"])}</pre>' for a in v['appendix_sections'])}
<p style="color:#6b7289;font-size:0.78rem;margin-top:3rem;">Generated by Kira · {_esc(v['date'])}</p>
</body></html>"""


# ── Tiny helpers ───────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    """HTML-escape a string."""
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _fmt_date(ts: Optional[str]) -> str:
    if not ts:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(ts)[:10]


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil

    print("=== reporter.py smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        session_dir = Path(tmp)

        # [1] Write fake state.json
        print("[1] Writing fake session data...")
        state = {
            "target":        "10.10.10.5",
            "authorized_by": "Lab VM",
            "started_at":    "2026-04-09T10:00:00Z",
            "updated_at":    "2026-04-09T10:42:00Z",
            "phase":         "REPORT",
            "open_ports":    [22, 80, 445],
            "services":      {"22": "OpenSSH 7.9", "80": "Apache 2.4.49", "445": "Samba 4.x"},
            "findings": [
                {
                    "title":             "Apache 2.4.49 Path Traversal (CVE-2021-41773)",
                    "severity":          "critical",
                    "cvss":              9.8,
                    "cve":               "CVE-2021-41773",
                    "port":              80,
                    "service":           "http",
                    "description":       "Path traversal and RCE via mod_cgi on unpatched Apache.",
                    "exploit_available": True,
                    "remediation":       "Upgrade Apache to 2.4.51 or later.",
                    "discovered_at":     "2026-04-09T10:12:00Z",
                },
                {
                    "title":             "SSH Weak Cipher Suites",
                    "severity":          "medium",
                    "cvss":              5.3,
                    "cve":               "",
                    "port":              22,
                    "service":           "ssh",
                    "description":       "Server accepts deprecated CBC cipher modes.",
                    "exploit_available": False,
                    "remediation":       "Disable CBC ciphers in sshd_config.",
                    "discovered_at":     "2026-04-09T10:08:00Z",
                },
                {
                    "title":             "SMB Null Session",
                    "severity":          "high",
                    "cvss":              7.5,
                    "cve":               "",
                    "port":              445,
                    "service":           "smb",
                    "description":       "Null session allowed — share enumeration without credentials.",
                    "exploit_available": True,
                    "remediation":       "Disable null sessions via registry or Samba config.",
                    "discovered_at":     "2026-04-09T10:20:00Z",
                },
            ],
        }
        (session_dir / "state.json").write_text(json.dumps(state, indent=2))

        # [2] Write fake actions.jsonl
        actions = [
            {"timestamp": "2026-04-09T10:02:00Z", "tool": "nmap_scan",    "ok": True,  "elapsed_s": 14.2, "summary": "Found 3 open ports: 22, 80, 445"},
            {"timestamp": "2026-04-09T10:06:00Z", "tool": "gobuster_dir", "ok": True,  "elapsed_s": 22.1, "summary": "Found /admin, /backup, /.git"},
            {"timestamp": "2026-04-09T10:11:00Z", "tool": "searchsploit", "ok": True,  "elapsed_s":  1.3, "summary": "CVE-2021-41773 found for Apache 2.4.49"},
            {"timestamp": "2026-04-09T10:15:00Z", "tool": "msf_exploit",  "ok": True,  "elapsed_s": 18.7, "summary": "Session 1 opened (meterpreter)"},
            {"timestamp": "2026-04-09T10:30:00Z", "tool": "linpeas",      "ok": True,  "elapsed_s": 45.0, "summary": "SUID pkexec found, sudo NOPASSWD detected"},
        ]
        with open(session_dir / "actions.jsonl", "w") as f:
            for a in actions:
                f.write(json.dumps(a) + "\n")

        # [3] Write fake raw nmap output
        raw_dir = session_dir / "raw"
        raw_dir.mkdir()
        (raw_dir / "nmap_20260409_100200.txt").write_text(
            "PORT   STATE SERVICE VERSION\n"
            "22/tcp open  ssh     OpenSSH 7.9 (protocol 2.0)\n"
            "80/tcp open  http    Apache httpd 2.4.49 ((Unix))\n"
            "445/tcp open  microsoft-ds Samba 4.x\n"
        )

        # [4] Generate report (no LLM — fallback mode)
        print("[2] Generating report (no LLM)...")
        reporter = ReportGenerator(session_dir=str(session_dir), llm=None)
        paths = reporter.generate()

        assert Path(paths.markdown).exists(), "report.md not created"
        assert Path(paths.html).exists(),     "report.html not created"

        md_content   = Path(paths.markdown).read_text()
        html_content = Path(paths.html).read_text()

        # Spot-check markdown
        assert "10.10.10.5" in md_content
        assert "CVE-2021-41773" in md_content
        assert "SMB Null Session" in md_content
        assert "Attack Timeline" in md_content
        assert "Appendix" in md_content
        print(f"   report.md   : {len(md_content):,} chars  ✓")

        # Spot-check HTML
        assert "10.10.10.5" in html_content
        assert "CVE-2021-41773" in html_content
        print(f"   report.html : {len(html_content):,} chars  ✓")

        print(f"\n   Markdown: {paths.markdown}")
        print(f"   HTML:     {paths.html}")

    print("\nAll tests passed.")