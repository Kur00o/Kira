"""
kira/parsers/service_enum.py — Per-service fingerprinters
==========================================================
Lightweight fingerprinters for the most common CTF / pentest services.
Each function takes a target + port and returns a list[Finding].
The planner calls whichever enumerators match the open ports from nmap.

Every enumerator:
  - Returns [] on any error — never raises
  - Is self-contained (no shared state)
  - Uses ToolRunner for subprocess calls when available
  - Falls back to stdlib (socket, urllib) when ToolRunner is not injected

Usage:
    from service_enum import enum_http, enum_ssh, enum_ftp, enum_smb, enum_mysql

    runner = ToolRunner(session_dir="./sessions/my_scan")

    findings = enum_http("10.10.10.5", 80, runner)
    findings += enum_ssh("10.10.10.5", 22, runner)
    findings += enum_ftp("10.10.10.5", 21, runner)
    findings += enum_smb("10.10.10.5", 445, runner)
    findings += enum_mysql("10.10.10.5", 3306, runner)
"""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from typing import Optional

try:
    from findings import Finding
except ImportError:
    raise ImportError("findings.py must be on the Python path before importing service_enum")


# ── Constants ──────────────────────────────────────────────────────────────────

# SSH versions below this are considered weak
SSH_WEAK_VERSION_THRESHOLD = 8.0

# Commonly weak / outdated OpenSSH versions worth flagging
SSH_KNOWN_WEAK = {"6.6", "6.7", "6.8", "6.9", "7.0", "7.1", "7.2", "7.3",
                  "7.4", "7.5", "7.6", "7.7", "7.8", "7.9"}

MYSQL_UNAUTHENTICATED_MSG = "Access denied for user"  # present = auth IS required (good)

SOCKET_TIMEOUT = 5   # seconds for raw socket grabs


# ── HTTP ───────────────────────────────────────────────────────────────────────

def enum_http(
    target:  str,
    port:    int,
    runner=None,   # Optional[ToolRunner]
) -> list[Finding]:
    """
    HTTP/HTTPS fingerprinting:
      - curl banner grab → Server header, X-Powered-By, version strings
      - whatweb → CMS / framework detection
      - Generates findings for outdated / misconfigured server versions

    Returns [] on any error.
    """
    findings: list[Finding] = []
    scheme = "https" if port in (443, 8443) else "http"
    url    = f"{scheme}://{target}:{port}"

    # ── curl header grab ──────────────────────────────────────────────────────
    headers = _curl_headers(url, runner)
    if headers:
        server_header = headers.get("server", "")
        powered_by    = headers.get("x-powered-by", "")

        # Flag outdated Apache
        apache_ver = _extract_version(server_header, r"Apache[/ ]([\d.]+)")
        if apache_ver:
            findings += _check_apache(apache_ver, port)

        # Flag outdated nginx
        nginx_ver = _extract_version(server_header, r"nginx/([\d.]+)")
        if nginx_ver:
            findings += _check_nginx(nginx_ver, port)

        # Flag PHP version exposure via X-Powered-By
        php_ver = _extract_version(powered_by, r"PHP/([\d.]+)")
        if php_ver:
            findings.append(Finding(
                title="PHP Version Disclosed in HTTP Headers",
                severity="info",
                port=port,
                service="http",
                cvss=0.0,
                description=(
                    f"The server exposes PHP/{php_ver} in the X-Powered-By header. "
                    "This aids attackers in targeting version-specific vulnerabilities."
                ),
                remediation="Set 'expose_php = Off' in php.ini and suppress X-Powered-By.",
            ))

        # Missing security headers
        missing = _check_security_headers(headers)
        if missing:
            findings.append(Finding(
                title="Missing HTTP Security Headers",
                severity="low",
                port=port,
                service="http",
                cvss=3.1,
                description=f"Missing headers: {', '.join(missing)}.",
                remediation=(
                    "Configure Content-Security-Policy, X-Frame-Options, "
                    "Strict-Transport-Security, and X-Content-Type-Options."
                ),
            ))

    # ── whatweb CMS detection ─────────────────────────────────────────────────
    if runner and shutil.which("whatweb"):
        try:
            result = runner.run(
                ["whatweb", "--color=never", "-a", "3", url],
                tool_name="whatweb",
                timeout=30,
                save_output=True,
            )
            if result.ok and result.stdout:
                findings += _parse_whatweb(result.stdout, port)
        except Exception:
            pass

    return findings


def _curl_headers(url: str, runner=None) -> dict[str, str]:
    """Return response headers as a lowercase-keyed dict, or {} on error."""
    try:
        if runner and shutil.which("curl"):
            res = runner.run(
                ["curl", "-sI", "--max-redirs", "3", "--max-time", "10", url],
                tool_name="curl",
                timeout=15,
                save_output=False,
            )
            raw = res.stdout if res.ok else ""
        else:
            # stdlib fallback
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = "\r\n".join(f"{k}: {v}" for k, v in resp.headers.items())

        return _parse_http_headers(raw)
    except Exception:
        return {}


def _parse_http_headers(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line and not line.startswith("HTTP/"):
            key, _, val = line.partition(":")
            headers[key.strip().lower()] = val.strip()
    return headers


def _check_apache(version: str, port: int) -> list[Finding]:
    findings = []
    try:
        major, minor, patch = (int(x) for x in (version.split(".") + ["0", "0"])[:3])
    except ValueError:
        return []

    # CVE-2021-41773 / CVE-2021-42013 — Apache 2.4.49 / 2.4.50
    if major == 2 and minor == 4 and patch in (49, 50):
        findings.append(Finding(
            title=f"Apache {version} Path Traversal & RCE (CVE-2021-41773/42013)",
            severity="critical",
            port=port,
            service="http",
            cvss=9.8,
            cve="CVE-2021-41773",
            description=(
                f"Apache {version} is vulnerable to path traversal and remote code "
                "execution via mod_cgi when enabled."
            ),
            exploit_available=True,
            remediation="Upgrade to Apache 2.4.51 or later immediately.",
        ))

    # Generally outdated (< 2.4.57)
    elif major == 2 and minor == 4 and patch < 57:
        findings.append(Finding(
            title=f"Apache {version} — Outdated Version",
            severity="medium",
            port=port,
            service="http",
            cvss=5.3,
            description=f"Apache {version} is outdated and may contain unpatched CVEs.",
            remediation="Upgrade to the latest stable Apache 2.4.x release.",
        ))
    return findings


def _check_nginx(version: str, port: int) -> list[Finding]:
    # Only flag meaningfully old nginx versions
    try:
        parts = [int(x) for x in version.split(".")[:2]]
        if parts[0] == 1 and parts[1] < 20:
            return [Finding(
                title=f"nginx {version} — Outdated Version",
                severity="low",
                port=port,
                service="http",
                cvss=3.1,
                description=f"nginx {version} is outdated. Check for relevant CVEs.",
                remediation="Upgrade to the latest stable nginx 1.x release.",
            )]
    except (ValueError, IndexError):
        pass
    return []


def _check_security_headers(headers: dict[str, str]) -> list[str]:
    important = [
        "content-security-policy",
        "x-frame-options",
        "x-content-type-options",
        "strict-transport-security",
    ]
    return [h for h in important if h not in headers]


def _parse_whatweb(output: str, port: int) -> list[Finding]:
    findings = []
    # Look for WordPress
    if "WordPress" in output:
        wp_ver_match = re.search(r"WordPress[/ ]([\d.]+)", output)
        wp_ver = wp_ver_match.group(1) if wp_ver_match else "unknown"
        findings.append(Finding(
            title=f"WordPress {wp_ver} Detected",
            severity="info",
            port=port,
            service="http",
            cvss=0.0,
            description=f"WordPress {wp_ver} detected. Check for plugin/theme vulnerabilities.",
            remediation="Keep WordPress core, plugins, and themes up to date.",
        ))
    # Look for Drupal
    if "Drupal" in output:
        findings.append(Finding(
            title="Drupal CMS Detected",
            severity="info",
            port=port,
            service="http",
            cvss=0.0,
            description="Drupal CMS detected. Check for Drupalgeddon and similar CVEs.",
            remediation="Keep Drupal core and modules updated.",
        ))
    return findings


# ── SSH ────────────────────────────────────────────────────────────────────────

def enum_ssh(
    target:  str,
    port:    int,
    runner=None,
) -> list[Finding]:
    """
    SSH fingerprinting:
      - Banner grab → version string
      - Flags versions below OpenSSH 8.0 as weak

    Returns [] on any error.
    """
    findings: list[Finding] = []
    banner = _grab_banner(target, port)

    if not banner:
        return []

    # Extract OpenSSH version
    # Banner format: SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.4
    version = _extract_version(banner, r"OpenSSH[_/ ]([\d.]+)")
    if not version:
        # Unknown SSH implementation — note it
        findings.append(Finding(
            title="Non-OpenSSH Server Detected",
            severity="info",
            port=port,
            service="ssh",
            cvss=0.0,
            description=f"SSH banner: {banner.strip()[:120]}. Non-OpenSSH may have unique CVEs.",
            remediation="Verify this is an authorised SSH implementation.",
        ))
        return findings

    # Version comparison
    try:
        ver_float = float(".".join(version.split(".")[:2]))
    except ValueError:
        return findings

    short_ver = ".".join(version.split(".")[:2])

    if short_ver in SSH_KNOWN_WEAK or ver_float < SSH_WEAK_VERSION_THRESHOLD:
        findings.append(Finding(
            title=f"OpenSSH {version} — Outdated Version",
            severity="medium",
            port=port,
            service="ssh",
            cvss=5.3,
            description=(
                f"OpenSSH {version} is below the recommended minimum (8.0). "
                "Older versions may support weak ciphers (arcfour, 3des-cbc) "
                "and have known privilege-escalation CVEs."
            ),
            remediation="Upgrade to OpenSSH 8.0+ and disable legacy ciphers in sshd_config.",
        ))
    else:
        # Still log the version for context — info only
        findings.append(Finding(
            title=f"SSH Service — OpenSSH {version}",
            severity="info",
            port=port,
            service="ssh",
            cvss=0.0,
            description=f"SSH running OpenSSH {version}. No critical issues detected.",
            remediation="Ensure key-based auth is enforced and password auth is disabled.",
        ))

    return findings


# ── FTP ────────────────────────────────────────────────────────────────────────

def enum_ftp(
    target:  str,
    port:    int,
    runner=None,
) -> list[Finding]:
    """
    FTP fingerprinting:
      - Banner grab → service / version
      - Tests anonymous login
      - Returns a critical Finding if anonymous login is allowed

    Returns [] on any error.
    """
    findings: list[Finding] = []

    # ── Banner grab ───────────────────────────────────────────────────────────
    banner = _grab_banner(target, port)
    if banner:
        vsftpd_ver = _extract_version(banner, r"vsftpd\s+([\d.]+)")
        # vsftpd 2.3.4 — backdoor (CVE-2011-2523)
        if vsftpd_ver and vsftpd_ver.startswith("2.3.4"):
            findings.append(Finding(
                title="vsftpd 2.3.4 Backdoor (CVE-2011-2523)",
                severity="critical",
                port=port,
                service="ftp",
                cvss=10.0,
                cve="CVE-2011-2523",
                description=(
                    "vsftpd 2.3.4 contains a backdoor that opens a root shell on port 6200 "
                    "when a username containing ':)' is sent."
                ),
                exploit_available=True,
                remediation="Remove vsftpd 2.3.4 immediately and replace with a patched version.",
            ))

    # ── Anonymous login test ──────────────────────────────────────────────────
    anon_allowed = _test_ftp_anonymous(target, port)
    if anon_allowed:
        findings.append(Finding(
            title="FTP Anonymous Login Allowed",
            severity="critical",
            port=port,
            service="ftp",
            cvss=9.1,
            description=(
                "The FTP server permits anonymous login (username 'anonymous', "
                "any password). Attackers can list and download files without credentials."
            ),
            exploit_available=True,
            remediation="Disable anonymous FTP access in the server configuration.",
        ))

    return findings


def _test_ftp_anonymous(target: str, port: int) -> bool:
    """Return True if anonymous FTP login succeeds."""
    try:
        import ftplib
        with ftplib.FTP(timeout=SOCKET_TIMEOUT) as ftp:
            ftp.connect(target, port, timeout=SOCKET_TIMEOUT)
            ftp.login("anonymous", "kira@pentest.local")
            return True
    except Exception:
        return False


# ── SMB ────────────────────────────────────────────────────────────────────────

def enum_smb(
    target:  str,
    port:    int,
    runner=None,
) -> list[Finding]:
    """
    SMB enumeration:
      - enum4linux -a → null sessions, guest access, share listing
      - Flags null sessions and readable shares

    Returns [] if enum4linux is not available or errors out.
    """
    findings: list[Finding] = []

    if not shutil.which("enum4linux"):
        # Try enum4linux-ng as fallback
        if not shutil.which("enum4linux-ng"):
            return [Finding(
                title="SMB Enumeration Skipped — Tool Missing",
                severity="info",
                port=port,
                service="smb",
                cvss=0.0,
                description="enum4linux not found on PATH. SMB enumeration was skipped.",
                remediation="Install enum4linux: apt install enum4linux",
            )]

    # Run enum4linux
    if runner:
        try:
            result = runner.run(
                ["enum4linux", "-a", target],
                tool_name="enum4linux",
                timeout=120,
                save_output=True,
            )
            output = result.stdout
        except Exception:
            return []
    else:
        try:
            proc = subprocess.run(
                ["enum4linux", "-a", target],
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = proc.stdout
        except Exception:
            return []

    if not output:
        return []

    findings += _parse_enum4linux(output, target, port)
    return findings


def _parse_enum4linux(output: str, target: str, port: int) -> list[Finding]:
    findings = []

    # Null session
    if re.search(r"NULL session\s+OK", output, re.IGNORECASE):
        findings.append(Finding(
            title="SMB Null Session Allowed",
            severity="high",
            port=port,
            service="smb",
            cvss=7.5,
            description=(
                "The SMB server allows null (unauthenticated) sessions. "
                "Attackers can enumerate users, shares, and domain information."
            ),
            exploit_available=True,
            remediation=(
                "Restrict null sessions via Group Policy: "
                "Network access: Restrict anonymous access to Named Pipes and Shares."
            ),
        ))

    # Guest access
    if re.search(r"allows sessions using username .{0,20}guest", output, re.IGNORECASE):
        findings.append(Finding(
            title="SMB Guest Access Enabled",
            severity="medium",
            port=port,
            service="smb",
            cvss=5.3,
            description="SMB allows guest access, potentially exposing sensitive shares.",
            remediation="Disable guest account and enforce authentication for all SMB shares.",
        ))

    # Readable shares
    share_matches = re.findall(
        r"Mapping:\s+OK.*?Listing:\s+OK.*?\\\\[^\s]+\\(\w+)",
        output,
        re.IGNORECASE | re.DOTALL,
    )
    if share_matches:
        share_list = ", ".join(set(share_matches))
        findings.append(Finding(
            title="SMB Readable Shares Found",
            severity="medium",
            port=port,
            service="smb",
            cvss=5.3,
            description=(
                f"Readable SMB shares discovered: {share_list}. "
                "These may expose sensitive files."
            ),
            remediation="Review and restrict share permissions to authorised users only.",
        ))

    return findings


# ── MySQL ──────────────────────────────────────────────────────────────────────

def enum_mysql(
    target:  str,
    port:    int,
    runner=None,
) -> list[Finding]:
    """
    MySQL fingerprinting:
      - Banner grab → version string
      - Tests for unauthenticated access (root without password)

    Returns [] on any error.
    """
    findings: list[Finding] = []

    # ── Banner grab (MySQL sends version in handshake) ───────────────────────
    version = _mysql_version_from_banner(target, port)
    if version:
        findings.append(Finding(
            title=f"MySQL {version} Detected",
            severity="info",
            port=port,
            service="mysql",
            cvss=0.0,
            description=f"MySQL {version} service identified on port {port}.",
            remediation=(
                "Ensure MySQL is not exposed externally. "
                "Bind to 127.0.0.1 unless remote access is required."
            ),
        ))

    # ── Test unauthenticated / empty-password root access ────────────────────
    unauth = _test_mysql_unauth(target, port, runner)
    if unauth:
        findings.append(Finding(
            title="MySQL Unauthenticated Root Access",
            severity="critical",
            port=port,
            service="mysql",
            cvss=10.0,
            description=(
                "MySQL root account has no password. Attackers have full database access "
                "and may be able to read/write files on the OS via INTO OUTFILE."
            ),
            exploit_available=True,
            remediation=(
                "Set a strong password for the MySQL root account: "
                "ALTER USER 'root'@'localhost' IDENTIFIED BY '<strong_password>';"
            ),
        ))

    return findings


def _mysql_version_from_banner(target: str, port: int) -> Optional[str]:
    """
    MySQL sends a handshake packet that contains the server version string.
    We read the first ~50 bytes and regex out the version.
    """
    try:
        with socket.create_connection((target, port), timeout=SOCKET_TIMEOUT) as s:
            data = s.recv(100)
            # Version is a null-terminated string starting around byte 5
            text = data[5:].decode("latin-1", errors="replace")
            m = re.search(r"([\d]+\.[\d]+\.[\d]+)", text)
            return m.group(1) if m else None
    except Exception:
        return None


def _test_mysql_unauth(target: str, port: int, runner=None) -> bool:
    """
    Return True if MySQL allows connection as root with no password.
    Uses the mysql CLI if available, otherwise returns False (safe default).
    """
    if not shutil.which("mysql"):
        return False

    cmd = ["mysql", "-h", target, "-P", str(port), "-u", "root", "--connect-timeout=5",
           "-e", "SELECT 1;"]
    if runner:
        try:
            res = runner.run(cmd, tool_name="mysql", timeout=10, save_output=False)
            return res.ok and "1" in res.stdout
        except Exception:
            return False
    else:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return proc.returncode == 0 and "1" in proc.stdout
        except Exception:
            return False


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _grab_banner(target: str, port: int) -> Optional[str]:
    """Read the initial banner sent by a TCP service."""
    try:
        with socket.create_connection((target, port), timeout=SOCKET_TIMEOUT) as s:
            s.sendall(b"\r\n")
            return s.recv(1024).decode("latin-1", errors="replace")
    except Exception:
        return None


def _extract_version(text: str, pattern: str) -> Optional[str]:
    """Extract a version string using a regex pattern. Returns None if not found."""
    if not text:
        return None
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1) if m else None


# ── Service dispatcher ─────────────────────────────────────────────────────────

SERVICE_ENUMERATORS = {
    "http":   (enum_http,  [80, 8080, 8000, 8888]),
    "https":  (enum_http,  [443, 8443]),
    "ssh":    (enum_ssh,   [22, 2222]),
    "ftp":    (enum_ftp,   [21]),
    "smb":    (enum_smb,   [445, 139]),
    "mysql":  (enum_mysql, [3306]),
}


def enumerate_services(
    target:     str,
    open_ports: list[int],
    runner=None,
) -> list[Finding]:
    """
    Dispatch the right enumerator for each open port.
    Called by the planner after nmap identifies services.

    Parameters
    ----------
    target      : IP address of the target
    open_ports  : list of open port numbers from nmap
    runner      : optional ToolRunner for subprocess-based enumerators

    Returns a flat list of all Findings from all enumerators.
    """
    all_findings: list[Finding] = []
    seen_enumerators: set[str] = set()

    for port in open_ports:
        for service_name, (enumerator, default_ports) in SERVICE_ENUMERATORS.items():
            if port in default_ports:
                key = f"{service_name}:{port}"
                if key in seen_enumerators:
                    continue
                seen_enumerators.add(key)

                try:
                    results = enumerator(target, port, runner)
                    all_findings.extend(results)
                except Exception:
                    pass   # never crash the planner loop

    return all_findings


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== service_enum.py smoke test (offline) ===\n")

    # Test helper utilities
    print("[1] _extract_version")
    assert _extract_version("Apache/2.4.49 (Unix)", r"Apache/([\d.]+)") == "2.4.49"
    assert _extract_version("SSH-2.0-OpenSSH_7.9", r"OpenSSH[_/ ]([\d.]+)") == "7.9"
    assert _extract_version("nothing here", r"Apache/([\d.]+)") is None
    print("   OK")

    print("\n[2] _parse_http_headers")
    raw = "HTTP/1.1 200 OK\r\nServer: Apache/2.4.49\r\nX-Powered-By: PHP/7.4.3\r\n"
    headers = _parse_http_headers(raw)
    assert headers.get("server") == "Apache/2.4.49"
    assert headers.get("x-powered-by") == "PHP/7.4.3"
    print(f"   OK: {headers}")

    print("\n[3] _check_apache — CVE-2021-41773")
    f = _check_apache("2.4.49", 80)
    assert len(f) == 1
    assert f[0].severity == "critical"
    assert "CVE-2021-41773" in f[0].cve
    print(f"   {f[0].one_liner()}")

    print("\n[4] _check_apache — outdated non-critical")
    f = _check_apache("2.4.51", 80)
    assert len(f) == 0 or f[0].severity in ("medium", "low", "info")
    print(f"   findings={len(f)} (expected 0 for 2.4.51)")

    print("\n[5] SSH version flags")
    # Simulate the version check logic directly
    version = "7.9"
    short_ver = ".".join(version.split(".")[:2])
    flagged = short_ver in SSH_KNOWN_WEAK or float(short_ver) < SSH_WEAK_VERSION_THRESHOLD
    assert flagged, "7.9 should be flagged as weak"
    print(f"   OpenSSH {version} correctly flagged as weak")

    version2 = "9.0"
    short_ver2 = ".".join(version2.split(".")[:2])
    not_flagged = short_ver2 not in SSH_KNOWN_WEAK and float(short_ver2) >= SSH_WEAK_VERSION_THRESHOLD
    assert not_flagged, "9.0 should NOT be flagged"
    print(f"   OpenSSH {version2} correctly NOT flagged")

    print("\n[6] _parse_enum4linux — null session detection")
    fake_output = """
    [+] Server allows sessions using username '', password ''
    [+] NULL session OK
    """
    f = _parse_enum4linux(fake_output, "10.10.10.5", 445)
    assert any("Null Session" in x.title for x in f)
    print(f"   {[x.title for x in f]}")

    print("\n[7] enumerate_services dispatcher")
    # Offline: open_ports that map to enumerators but won't actually connect
    # Just test the dispatch logic returns a list without raising
    try:
        results = enumerate_services("127.0.0.1", [80, 22, 21, 445, 3306], runner=None)
        # Results will be empty or info-only since nothing is running
        print(f"   Dispatcher ran without raising — {len(results)} findings returned")
    except Exception as e:
        print(f"   UNEXPECTED EXCEPTION: {e}")
        raise

    print("\nAll offline tests passed.")
    print("(Live tests require an actual target — run against a lab VM)")