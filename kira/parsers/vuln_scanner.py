"""
parsers/vuln_scanner.py
Phase 5 — CVE Cross-Reference via searchsploit
Reads the services dict from state, runs searchsploit on every version string,
and turns results into tagged Findings with CVE IDs and exploit availability flags.
"""

import json
import subprocess
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    port: str
    service: str
    version_string: str
    cve: Optional[str]
    exploit_available: bool
    cvss_estimate: Optional[float]
    edb_ids: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    raw_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "port": self.port,
            "service": self.service,
            "version_string": self.version_string,
            "cve": self.cve,
            "exploit_available": self.exploit_available,
            "cvss_estimate": self.cvss_estimate,
            "edb_ids": self.edb_ids,
            "titles": self.titles,
        }


# ---------------------------------------------------------------------------
# Tool runner (thin wrapper so tests can mock it)
# ---------------------------------------------------------------------------

class ToolRunner:
    def searchsploit(self, version_string: str) -> str:
        """
        Run: searchsploit --json <version_string>
        Returns raw stdout as a string (may be empty / invalid JSON on no hits).
        """
        try:
            result = subprocess.run(
                ["searchsploit", "--json", version_string],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip()
        except FileNotFoundError:
            raise RuntimeError(
                "searchsploit not found. Install exploit-db: "
                "sudo apt install exploitdb"
            )
        except subprocess.TimeoutExpired:
            return ""


# ---------------------------------------------------------------------------
# Knowledge base (severity hints keyed on EDB type strings)
# ---------------------------------------------------------------------------

class KnowledgeBase:
    # Maps EDB "type" field → rough CVSS estimate
    TYPE_CVSS: dict[str, float] = {
        "remote":   9.0,
        "webapps":  8.5,
        "local":    7.0,
        "dos":      5.0,
        "shellcode": 8.0,
        "papers":   0.0,
        "hardware": 6.0,
    }

    def cvss_from_type(self, edb_type: str) -> float:
        return self.TYPE_CVSS.get(edb_type.lower().strip(), 5.0)


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def parse_searchsploit_json(raw: str) -> list[dict]:
    """
    Parse searchsploit --json output into a list of exploit dicts.
    Each dict has keys: title, edb_id, type, platform, path, cve (may be None).

    - Sets exploit_available=True if type contains "Metasploit"  (checked later)
    - Maps EDB severity hint to cvss estimate via KnowledgeBase
    """
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    exploits = data.get("RESULTS_EXPLOIT", [])
    shellcodes = data.get("RESULTS_SHELLCODE", [])
    all_results = exploits + shellcodes

    parsed = []
    for entry in all_results:
        title = entry.get("Title", "")
        edb_id = str(entry.get("EDB-ID", ""))
        edb_type = entry.get("Type", "")
        platform = entry.get("Platform", "")
        path = entry.get("Path", "")

        # Extract first CVE from title if present
        cve_match = _CVE_RE.search(title)
        cve = cve_match.group(0).upper() if cve_match else None

        parsed.append({
            "title": title,
            "edb_id": edb_id,
            "type": edb_type,
            "platform": platform,
            "path": path,
            "cve": cve,
        })

    return parsed


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

def scan_services(
    services: dict,
    runner: ToolRunner,
    kb: KnowledgeBase,
) -> list[Finding]:
    """
    Iterate every port → version_string in `services`, run searchsploit,
    and return a list of Finding objects.

    `services` format (as stored in state):
        {
            "80/tcp":  "Apache httpd 2.4.49",
            "22/tcp":  "OpenSSH 7.4",
            "3306/tcp": "MySQL 5.7.34",
            ...
        }

    Findings are created per-port. If searchsploit returns nothing, the port
    is skipped cleanly (no Finding emitted).
    """
    findings: list[Finding] = []

    for port, version_string in services.items():
        if not version_string:
            continue

        raw_json = runner.searchsploit(version_string)
        results = parse_searchsploit_json(raw_json)

        if not results:
            # Skip services with no searchsploit results cleanly
            continue

        # Aggregate across all results for this service
        exploit_available = any(
            "metasploit" in r["type"].lower() or
            "metasploit" in r["title"].lower()
            for r in results
        )

        # Pick highest CVSS estimate across all result types
        cvss_scores = [kb.cvss_from_type(r["type"]) for r in results]
        cvss_estimate = max(cvss_scores) if cvss_scores else None

        # Collect unique CVEs and EDB IDs
        cves = list({r["cve"] for r in results if r["cve"]})
        primary_cve = cves[0] if cves else None
        edb_ids = [r["edb_id"] for r in results if r["edb_id"]]
        titles = [r["title"] for r in results]

        # Derive service name from version string (first word)
        service_name = version_string.split()[0] if version_string else "unknown"

        finding = Finding(
            port=port,
            service=service_name,
            version_string=version_string,
            cve=primary_cve,
            exploit_available=exploit_available,
            cvss_estimate=cvss_estimate,
            edb_ids=edb_ids,
            titles=titles,
            raw_results=results,
        )
        findings.append(finding)

    return findings


# ---------------------------------------------------------------------------
# Convenience: load from state dict and return serialisable output
# ---------------------------------------------------------------------------

def run(state: dict) -> list[dict]:
    """
    Entry point called by the planner on Day 3.
    Reads state["services"], returns list of Finding dicts.
    """
    services: dict = state.get("services", {})
    runner = ToolRunner()
    kb = KnowledgeBase()
    findings = scan_services(services, runner, kb)
    return [f.to_dict() for f in findings]


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    demo_state = {
        "services": {
            "80/tcp":   "Apache httpd 2.4.49",
            "22/tcp":   "OpenSSH 7.4",
            "443/tcp":  "nginx 1.14.0",
            "3306/tcp": "MySQL 5.7.34",
        }
    }

    print("[*] Running vuln_scanner against demo services …\n")
    results = run(demo_state)

    if not results:
        print("[-] No findings (searchsploit returned no results or is not installed).")
        sys.exit(0)

    for f in results:
        print(f"[+] {f['port']} | {f['version_string']}")
        print(f"    CVE            : {f['cve'] or 'N/A'}")
        print(f"    Exploit avail  : {f['exploit_available']}")
        print(f"    CVSS estimate  : {f['cvss_estimate']}")
        print(f"    EDB IDs        : {', '.join(f['edb_ids'][:5]) or 'none'}")
        print()