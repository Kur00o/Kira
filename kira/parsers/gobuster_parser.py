"""
kira/parsers/gobuster_parser.py — Gobuster / ffuf output parser
===============================================================
Parses raw stdout from gobuster (dir mode) and ffuf (JSON mode) into
clean, structured path data. Auto-flags high-value paths so the LLM
planner knows where to dig deeper. Auto-creates critical Findings for
particularly dangerous discoveries (/.git, /.env, etc.).

Usage:
    from gobuster_parser import GobusterParser

    # From a raw stdout string
    parser = GobusterParser(raw=result.stdout)
    result = parser.parse()

    # From a saved output file
    parser = GobusterParser(file_path="raw/gobuster_20240101_120000.txt")
    result = parser.parse()

    print(result.all_paths)     # ["/index.html", "/admin", "/.git"]
    print(result.juicy_paths)   # ["/.git", "/admin"]
    print(result.status_map)    # {"/admin": 200, "/.git": 301}
    print(result.findings)      # [Finding(...)] — auto-created for critical paths
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Import Finding for auto-generated critical findings
try:
    from findings import Finding
    _FINDINGS_AVAILABLE = True
except ImportError:
    _FINDINGS_AVAILABLE = False


# ── High-value path constants ──────────────────────────────────────────────────

JUICY_PATHS = [
    "/.git",
    "/.env",
    "/admin",
    "/backup",
    "/config",
    "/api",
    "/phpinfo.php",
    "/.htaccess",
    "/.htpasswd",
    "/wp-admin",
    "/wp-login.php",
    "/manager",          # Tomcat
    "/phpmyadmin",
    "/console",          # JBoss / Wildfly
    "/actuator",         # Spring Boot
    "/swagger",
    "/swagger-ui",
    "/.svn",
    "/secret",
    "/private",
    "/uploads",
    "/shell",
    "/cmd",
    "/debug",
    "/test",
    "/install",
]

# Subset that warrant a critical/high Finding automatically
CRITICAL_PATHS = {
    "/.git":        ("Git Repository Exposed", "critical", 9.3,
                     "A .git directory is publicly accessible. Attackers can reconstruct "
                     "the full source code, credentials, and history.",
                     "Block access to /.git at the web server level immediately."),
    "/.env":        ("Environment File Exposed", "critical", 9.8,
                     "The .env file is publicly readable. It likely contains database "
                     "credentials, API keys, and secret tokens.",
                     "Block access to .env files via web server config."),
    "/.htpasswd":   ("htpasswd File Exposed", "high", 7.5,
                     "An htpasswd file is accessible. It may contain hashed credentials.",
                     "Block access to .htpasswd files in web server configuration."),
    "/phpinfo.php": ("PHPInfo Page Exposed", "medium", 5.3,
                     "phpinfo() reveals server configuration, PHP version, extensions, "
                     "and potential attack surface.",
                     "Remove phpinfo.php from production servers."),
    "/.svn":        ("SVN Repository Exposed", "high", 7.5,
                     "An .svn directory is publicly accessible, exposing source code.",
                     "Block access to /.svn at the web server level."),
}


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class GobusterResult:
    """
    Structured output from a gobuster / ffuf run.

    Attributes:
        all_paths   : every discovered path (sorted)
        juicy_paths : subset matching JUICY_PATHS patterns
        status_map  : {"/path": http_status_code}
        size_map    : {"/path": response_size_bytes}  (when available)
        findings    : auto-created Finding objects for critical paths
        source      : "gobuster" | "ffuf" | "unknown"
        raw_lines   : original parsed lines (for debugging)
    """
    all_paths:   list[str]            = field(default_factory=list)
    juicy_paths: list[str]            = field(default_factory=list)
    status_map:  dict[str, int]       = field(default_factory=dict)
    size_map:    dict[str, int]       = field(default_factory=dict)
    findings:    list                 = field(default_factory=list)  # list[Finding]
    source:      str                  = "unknown"
    raw_lines:   int                  = 0

    def summary(self) -> str:
        """Compact summary for LLM context injection."""
        lines = [
            f"Web paths discovered: {len(self.all_paths)} total, "
            f"{len(self.juicy_paths)} high-value"
        ]
        if self.juicy_paths:
            lines.append(f"  HIGH-VALUE: {', '.join(self.juicy_paths[:8])}")
        regular = [p for p in self.all_paths if p not in self.juicy_paths]
        if regular:
            lines.append(f"  Other paths: {', '.join(regular[:10])}")
            if len(regular) > 10:
                lines[-1] += f" (+{len(regular) - 10} more)"
        if self.findings:
            lines.append(f"  Auto-findings: {len(self.findings)} critical/high issues flagged")
        return "\n".join(lines)


# ── Parser ─────────────────────────────────────────────────────────────────────

class GobusterParser:
    """
    Parses gobuster dir-mode stdout or ffuf JSON output.

    Accepts either a raw string or a file path — not both.
    Autodetects format (gobuster text vs ffuf JSON).

    Parameters
    ----------
    raw       : raw stdout/file content as a string
    file_path : path to a saved gobuster/ffuf output file
    base_url  : optional — used to generate Finding URLs in descriptions
    port      : target port number — attached to any auto-generated Findings
    """

    def __init__(
        self,
        raw:       Optional[str]  = None,
        file_path: Optional[str]  = None,
        base_url:  Optional[str]  = None,
        port:      int            = 80,
    ):
        if raw is None and file_path is None:
            raise ValueError("Provide either 'raw' or 'file_path'")
        if raw is not None and file_path is not None:
            raise ValueError("Provide only one of 'raw' or 'file_path'")

        self._raw      = raw
        self._file_path = file_path
        self.base_url  = base_url
        self.port      = port

    def parse(self) -> GobusterResult:
        """
        Parse the input and return a GobusterResult.
        Autodetects gobuster text format vs ffuf JSON format.
        """
        content = self._load()
        stripped = content.strip()

        if self._looks_like_ffuf_json(stripped):
            return self._parse_ffuf(stripped)
        else:
            return self._parse_gobuster(stripped)

    # ── Private: loading ──────────────────────────────────────────────────────

    def _load(self) -> str:
        if self._raw is not None:
            return self._raw
        path = Path(self._file_path)
        if not path.exists():
            raise FileNotFoundError(f"Output file not found: {self._file_path}")
        return path.read_text(encoding="utf-8", errors="replace")

    # ── Private: format detection ─────────────────────────────────────────────

    @staticmethod
    def _looks_like_ffuf_json(content: str) -> bool:
        """Heuristic: ffuf JSON output starts with { and has a 'results' key."""
        return content.startswith("{") and '"results"' in content[:200]

    # ── Private: gobuster text parser ─────────────────────────────────────────

    def _parse_gobuster(self, content: str) -> GobusterResult:
        """
        Parse gobuster dir-mode output.

        Expected line formats:
            /admin                (Status: 200) [Size: 1234]
            /backup               (Status: 301) [Size: 0] [--> /backup/]
            /.git/HEAD            (Status: 200) [Size: 23]
        """
        paths: dict[str, int]  = {}   # path → status
        sizes: dict[str, int]  = {}   # path → size
        raw_lines = 0

        # Regex: capture path, status code, and optional size
        line_re = re.compile(
            r"^(?P<path>/\S*)"          # path starting with /
            r".*?\(Status:\s*(?P<status>\d{3})\)"  # (Status: NNN)
            r"(?:.*?\[Size:\s*(?P<size>\d+)\])?",  # optional [Size: NNN]
            re.IGNORECASE,
        )

        for line in content.splitlines():
            raw_lines += 1
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                continue

            m = line_re.match(line)
            if not m:
                continue

            path   = m.group("path").rstrip("/") or "/"
            status = int(m.group("status"))
            size   = int(m.group("size")) if m.group("size") else 0

            # Skip common false positives
            if status in (404, 400):
                continue

            paths[path] = status
            if size:
                sizes[path] = size

        return self._build_result(paths, sizes, raw_lines, source="gobuster")

    # ── Private: ffuf JSON parser ─────────────────────────────────────────────

    def _parse_ffuf(self, content: str) -> GobusterResult:
        """
        Parse ffuf JSON output (produced with -of json).

        Expected structure:
            {"results": [{"url": "...", "status": 200, "length": 1234, "input": {"FUZZ": "admin"}}, ...]}
        """
        paths: dict[str, int] = {}
        sizes: dict[str, int] = {}
        raw_lines = 0

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"ffuf JSON parse error: {e}")

        results = data.get("results", [])
        raw_lines = len(results)

        for entry in results:
            status = int(entry.get("status", 0))
            if status in (404, 400, 0):
                continue

            # Prefer the FUZZ input value to reconstruct the path
            fuzz_input = entry.get("input", {}).get("FUZZ", "")
            url        = entry.get("url", "")

            if fuzz_input:
                path = "/" + fuzz_input.strip("/")
            elif url:
                # Strip base URL to get relative path
                path = "/" + url.split("/", 3)[-1].lstrip("/") if "/" in url else url
            else:
                continue

            paths[path]  = status
            length = entry.get("length", 0)
            if length:
                sizes[path] = int(length)

        return self._build_result(paths, sizes, raw_lines, source="ffuf")

    # ── Private: shared result builder ────────────────────────────────────────

    def _build_result(
        self,
        paths:     dict[str, int],
        sizes:     dict[str, int],
        raw_lines: int,
        source:    str,
    ) -> GobusterResult:
        """Classify paths, flag juicy ones, auto-create Findings."""
        all_paths   = sorted(paths.keys())
        juicy_paths = self._flag_juicy(all_paths)
        findings    = self._auto_findings(juicy_paths, paths) if _FINDINGS_AVAILABLE else []

        return GobusterResult(
            all_paths=all_paths,
            juicy_paths=juicy_paths,
            status_map=dict(paths),
            size_map=sizes,
            findings=findings,
            source=source,
            raw_lines=raw_lines,
        )

    def _flag_juicy(self, paths: list[str]) -> list[str]:
        """Return paths that match JUICY_PATHS prefixes."""
        juicy = []
        for path in paths:
            path_lower = path.lower()
            if any(path_lower == j or path_lower.startswith(j + "/")
                   for j in JUICY_PATHS):
                juicy.append(path)
        return juicy

    def _auto_findings(self, juicy_paths: list[str], status_map: dict[str, int]) -> list:
        """
        For paths matching CRITICAL_PATHS, auto-generate a Finding.
        Only creates findings for paths that returned a success-ish status code.
        """
        findings = []
        for path in juicy_paths:
            path_lower = path.lower().rstrip("/")
            if path_lower not in CRITICAL_PATHS:
                continue

            status = status_map.get(path, 0)
            # Only flag if the path is actually reachable (not redirected away to 403/404)
            if status not in (200, 301, 302, 403):
                continue

            title, severity, cvss, description, remediation = CRITICAL_PATHS[path_lower]

            url_note = ""
            if self.base_url:
                url_note = f" URL: {self.base_url.rstrip('/')}{path} (HTTP {status})"

            try:
                findings.append(Finding(
                    title=title,
                    severity=severity,
                    port=self.port,
                    service="http",
                    cvss=cvss,
                    description=description + url_note,
                    exploit_available=(severity == "critical"),
                    remediation=remediation,
                ))
            except Exception:
                pass  # Never crash the parser

        return findings


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== gobuster_parser.py smoke test ===\n")

    # ── Sample gobuster output ─────────────────────────────────────────────────
    SAMPLE_GOBUSTER = """\
===============================================================
Gobuster v3.6
===============================================================
/index.html           (Status: 200) [Size: 10918]
/admin                (Status: 200) [Size: 2048]
/.git                 (Status: 301) [Size: 0] [--> /.git/]
/.git/HEAD            (Status: 200) [Size: 23]
/.env                 (Status: 200) [Size: 512]
/backup               (Status: 403) [Size: 0]
/phpinfo.php          (Status: 200) [Size: 75421]
/images               (Status: 301) [Size: 0] [--> /images/]
/login                (Status: 200) [Size: 1024]
/notfound             (Status: 404) [Size: 196]
===============================================================
"""

    print("[1] Parse gobuster text output")
    parser = GobusterParser(raw=SAMPLE_GOBUSTER, base_url="http://10.10.10.5", port=80)
    result = parser.parse()

    print(f"   source     : {result.source}")
    print(f"   all_paths  : {result.all_paths}")
    print(f"   juicy_paths: {result.juicy_paths}")
    print(f"   status_map : {result.status_map}")
    print(f"   raw_lines  : {result.raw_lines}")

    assert "/.git" in result.juicy_paths
    assert "/.env" in result.juicy_paths
    assert "/admin" in result.juicy_paths
    assert "/notfound" not in result.all_paths   # 404s excluded
    assert result.status_map.get("/.git") == 301
    print("   Assertions passed\n")

    # ── Auto-findings ──────────────────────────────────────────────────────────
    print("[2] Auto-generated Findings")
    for f in result.findings:
        print(f"   {f.one_liner()}")
    assert any("Git" in f.title for f in result.findings)
    assert any(".env" in f.title or "Environment" in f.title for f in result.findings)
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("[3] Summary block")
    print(result.summary())
    print()

    # ── ffuf JSON ─────────────────────────────────────────────────────────────
    SAMPLE_FFUF = json.dumps({
        "results": [
            {"url": "http://10.10.10.5/admin",  "status": 200, "length": 2048,
             "input": {"FUZZ": "admin"}},
            {"url": "http://10.10.10.5/.git",   "status": 301, "length": 0,
             "input": {"FUZZ": ".git"}},
            {"url": "http://10.10.10.5/login",  "status": 200, "length": 1024,
             "input": {"FUZZ": "login"}},
            {"url": "http://10.10.10.5/404page","status": 404, "length": 196,
             "input": {"FUZZ": "404page"}},
        ]
    })

    print("[4] Parse ffuf JSON output")
    parser2 = GobusterParser(raw=SAMPLE_FFUF, base_url="http://10.10.10.5", port=80)
    result2  = parser2.parse()

    print(f"   source     : {result2.source}")
    print(f"   all_paths  : {result2.all_paths}")
    print(f"   juicy_paths: {result2.juicy_paths}")

    assert result2.source == "ffuf"
    assert "/.git" in result2.juicy_paths
    assert "/404page" not in result2.all_paths
    print("   Assertions passed\n")

    # ── File not found ─────────────────────────────────────────────────────────
    print("[5] File not found error")
    try:
        GobusterParser(file_path="/nonexistent/path.txt").parse()
        assert False, "Should have raised"
    except FileNotFoundError as e:
        print(f"   Correctly raised: {e}")

    print("\nAll tests passed.")