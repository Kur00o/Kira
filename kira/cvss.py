"""
kira/cvss.py — CVSS v3 Base Score Calculator
=============================================
Implements the NVD CVSS v3.1 base score formula.
Used by the reporter for risk table, finding sort order,
and remediation priority.

Usage:
    from cvss import calculate_cvss3, severity_label, estimate_cvss_from_finding

    score = calculate_cvss3(
        attack_vector="N", attack_complexity="L",
        privileges_required="N", user_interaction="N",
        scope="U", confidentiality="H",
        integrity="H", availability="H"
    )
    # → 9.8

    label = severity_label(score)  # → "Critical"

    # Auto-estimate from a finding dict (heuristic):
    score = estimate_cvss_from_finding({
        "service": "http", "port": 80,
        "exploit_available": True, "cve": "CVE-2021-41773"
    })
"""

from __future__ import annotations

# ── CVSS v3.1 Metric Value Tables ─────────────────────────────────────────────
# Source: https://www.first.org/cvss/v3.1/specification-document

_AV = {
    "N": 0.85,   # Network
    "A": 0.62,   # Adjacent
    "L": 0.55,   # Local
    "P": 0.20,   # Physical
}

_AC = {
    "L": 0.77,   # Low
    "H": 0.44,   # High
}

_PR_UNCHANGED = {
    "N": 0.85,   # None
    "L": 0.62,   # Low
    "H": 0.27,   # High
}

_PR_CHANGED = {
    "N": 0.85,   # None
    "L": 0.68,   # Low
    "H": 0.50,   # High
}

_UI = {
    "N": 0.85,   # None
    "R": 0.62,   # Required
}

_C_I_A = {
    "N": 0.00,   # None
    "L": 0.22,   # Low
    "H": 0.56,   # High
}


# ── Core formula ──────────────────────────────────────────────────────────────

def calculate_cvss3(
    attack_vector:        str,   # N / A / L / P
    attack_complexity:    str,   # L / H
    privileges_required:  str,   # N / L / H
    user_interaction:     str,   # N / R
    scope:                str,   # U / C  (Unchanged / Changed)
    confidentiality:      str,   # N / L / H
    integrity:            str,   # N / L / H
    availability:         str,   # N / L / H
) -> float:
    """
    Calculate NVD CVSS v3.1 Base Score.

    All parameters use single-letter abbreviations as per the spec.
    Returns a float in [0.0, 10.0], rounded to 1 decimal place.

    Example:
        calculate_cvss3("N","L","N","N","U","H","H","H")  → 9.8
        calculate_cvss3("N","L","L","N","U","L","N","N")  → 5.3
    """
    # Validate and convert to uppercase
    av = attack_vector.upper().strip()
    ac = attack_complexity.upper().strip()
    pr = privileges_required.upper().strip()
    ui = user_interaction.upper().strip()
    s  = scope.upper().strip()
    c  = confidentiality.upper().strip()
    i  = integrity.upper().strip()
    a  = availability.upper().strip()

    # Look up numeric values
    av_val = _AV.get(av)
    ac_val = _AC.get(ac)
    pr_val = (_PR_CHANGED if s == "C" else _PR_UNCHANGED).get(pr)
    ui_val = _UI.get(ui)
    c_val  = _C_I_A.get(c)
    i_val  = _C_I_A.get(i)
    a_val  = _C_I_A.get(a)

    for name, val in [("AV", av_val), ("AC", ac_val), ("PR", pr_val),
                      ("UI", ui_val), ("C", c_val), ("I", i_val), ("A", a_val)]:
        if val is None:
            raise ValueError(f"Invalid CVSS metric value for {name}")

    # Exploitability sub-score
    exploitability = 8.22 * av_val * ac_val * pr_val * ui_val

    # Impact sub-score
    iss = 1.0 - (1.0 - c_val) * (1.0 - i_val) * (1.0 - a_val)

    if s == "U":  # Unchanged scope
        impact = 6.42 * iss
    else:          # Changed scope
        impact = 7.52 * (iss - 0.029) - 3.25 * ((iss - 0.02) ** 15)

    # Base score
    if impact <= 0:
        base = 0.0
    elif s == "U":
        base = _roundup(min(impact + exploitability, 10.0))
    else:
        base = _roundup(min(1.08 * (impact + exploitability), 10.0))

    return round(base, 1)


def severity_label(score: float) -> str:
    """
    Map a CVSS v3 base score to its severity label.

    Returns: "None" | "Low" | "Medium" | "High" | "Critical"
    Per NVD thresholds: https://nvd.nist.gov/vuln-metrics/cvss
    """
    if score == 0.0:    return "None"
    if score < 4.0:     return "Low"
    if score < 7.0:     return "Medium"
    if score < 9.0:     return "High"
    return "Critical"


# ── Heuristic estimator ───────────────────────────────────────────────────────

# Well-known CVE → exact CVSS score lookup (curated subset)
_KNOWN_CVE_SCORES: dict[str, float] = {
    "CVE-2021-41773": 9.8,   # Apache path traversal / RCE
    "CVE-2021-42013": 9.8,   # Apache mod_cgi RCE follow-on
    "CVE-2021-4034":  7.8,   # pkexec LPE (PwnKit)
    "CVE-2021-3156":  7.8,   # sudo heap overflow (Baron Samedit)
    "CVE-2019-0708":  9.8,   # BlueKeep RDP
    "CVE-2017-0144":  9.3,   # EternalBlue SMBv1 (MS17-010)
    "CVE-2016-5195":  7.8,   # Dirty COW
    "CVE-2014-6271":  9.8,   # Shellshock
    "CVE-2014-0160":  7.5,   # Heartbleed
    "CVE-2021-22986": 9.8,   # F5 BIG-IP RCE
    "CVE-2020-1472":  10.0,  # ZeroLogon
    "CVE-2022-0847":  7.8,   # Dirty Pipe
    "CVE-2023-4911":  7.8,   # glibc Looney Tunables
}

# Service + port heuristics: (service_substring, port_set) → base_cvss
_SERVICE_HEURISTICS: list[tuple] = [
    # (match_string, ports_hint, base_cvss_if_exploit, base_cvss_no_exploit)
    ("vsftpd",         {21},         9.0, 5.0),
    ("proftpd",        {21},         8.0, 4.5),
    ("samba",          {139, 445},   9.0, 6.0),
    ("smb",            {139, 445},   8.5, 5.5),
    ("ms-wbt-server",  {3389},       9.8, 7.5),   # RDP
    ("rdp",            {3389},       9.8, 7.5),
    ("mysql",          {3306},       7.0, 4.5),
    ("mssql",          {1433},       8.0, 5.0),
    ("apache",         {80, 443},    8.5, 5.0),
    ("nginx",          {80, 443},    6.5, 4.0),
    ("iis",            {80, 443},    7.5, 4.5),
    ("openssh",        {22},         5.3, 3.5),
    ("telnet",         {23},         8.0, 6.5),
    ("smtp",           {25, 587},    6.0, 3.5),
    ("snmp",           {161},        7.5, 5.0),
    ("tomcat",         {8080, 8443}, 9.0, 5.5),
    ("jenkins",        {8080},       9.8, 6.0),
    ("elasticsearch",  {9200},       9.8, 6.0),
    ("redis",          {6379},       9.8, 6.0),
    ("mongodb",        {27017},      9.8, 6.0),
    ("postgres",       {5432},       7.5, 4.5),
    ("ftp",            {21},         7.5, 4.0),
    ("vnc",            {5900},       9.0, 7.0),
]


def estimate_cvss_from_finding(finding: dict) -> float:
    """
    Heuristic CVSS estimator for auto-scored findings.

    Priority:
    1. CVE match in known lookup table  → exact score
    2. Service + port pattern match     → table lookup ± exploit bonus
    3. Fallback by severity field       → band midpoint
    4. Default                          → 5.0

    Args:
        finding : dict with any of: cve, service, port, severity,
                  exploit_available, title

    Returns:
        float CVSS estimate in [0.0, 10.0]
    """
    # [1] Known CVE
    cve = (finding.get("cve") or "").upper().strip()
    if cve and cve in _KNOWN_CVE_SCORES:
        return _KNOWN_CVE_SCORES[cve]

    exploit = bool(finding.get("exploit_available", False))
    service = (finding.get("service") or "").lower()
    title   = (finding.get("title") or "").lower()
    port    = int(finding.get("port") or 0)

    # [2] Service heuristic
    for match, ports_hint, cvss_exploit, cvss_no_exploit in _SERVICE_HEURISTICS:
        if match in service or match in title:
            base = cvss_exploit if exploit else cvss_no_exploit
            return round(min(base, 10.0), 1)

    # Port-only fallback (common attack surface)
    port_scores = {
        23:    8.0,   # Telnet
        21:    7.5,   # FTP
        3389:  7.5,   # RDP
        5900:  7.0,   # VNC
        161:   7.0,   # SNMP
        6379:  8.5,   # Redis (often unauth)
        27017: 8.5,   # MongoDB (often unauth)
        9200:  8.0,   # Elasticsearch (often unauth)
    }
    if port in port_scores:
        return port_scores[port]

    # [3] Severity band fallback
    sev = (finding.get("severity") or "").lower()
    sev_midpoints = {
        "critical": 9.5,
        "high":     8.0,
        "medium":   5.5,
        "low":      2.0,
        "info":     0.0,
    }
    if sev in sev_midpoints:
        return sev_midpoints[sev]

    # [4] Default
    return 5.0 if exploit else 3.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _roundup(x: float) -> float:
    """
    CVSS v3 Roundup function: rounds up to the nearest 0.1.
    Per spec: Roundup(x) = ⌈x × 10⌉ / 10
    """
    import math
    return math.ceil(x * 10) / 10


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== cvss.py smoke test ===\n")

    # [1] Known scores (spot-check against NVD)
    known_checks = [
        # AV  AC  PR  UI  S   C    I    A    expected  label
        ("N", "L", "N", "N", "U", "H", "H", "H", 9.8,  "Critical"),   # CVE-2021-41773
        ("N", "L", "L", "N", "U", "L", "N", "N", 4.3,  "Medium"),     # SSH low-priv read
        ("N", "L", "N", "N", "C", "H", "H", "H", 10.0, "Critical"),   # ZeroLogon-like
        ("L", "L", "L", "N", "U", "H", "H", "H", 7.8,  "High"),       # Local priv esc
        ("N", "H", "N", "N", "U", "N", "L", "N", 3.7,  "Low"),        # Low impact
        ("P", "H", "H", "R", "U", "N", "N", "N", 0.0,  "None"),       # Physical, no impact
    ]

    print("[1] Known CVSS calculations:")
    all_ok = True
    for av, ac, pr, ui, s, c, i, a, expected, exp_label in known_checks:
        score = calculate_cvss3(av, ac, pr, ui, s, c, i, a)
        label = severity_label(score)
        ok    = abs(score - expected) <= 0.1
        status = "OK" if ok else f"FAIL (got {score})"
        print(f"    AV:{av}/AC:{ac}/PR:{pr}/UI:{ui}/S:{s}/C:{c}/I:{i}/A:{a} "
              f"→ {score:4.1f} [{label:<8}]  {status}")
        if not ok:
            all_ok = False
    assert all_ok, "Some CVSS scores were off"

    # [2] Severity labels
    print("\n[2] severity_label():")
    label_checks = [
        (0.0, "None"), (0.1, "Low"), (3.9, "Low"), (4.0, "Medium"),
        (6.9, "Medium"), (7.0, "High"), (8.9, "High"), (9.0, "Critical"), (10.0, "Critical"),
    ]
    for score, expected in label_checks:
        got = severity_label(score)
        ok  = got == expected
        print(f"    {score:4.1f} → {got:<8} {'OK' if ok else f'FAIL (expected {expected})'}")
        assert ok, f"Label mismatch for {score}"

    # [3] estimate_cvss_from_finding — known CVE
    print("\n[3] estimate_cvss_from_finding — known CVE:")
    f1 = {"cve": "CVE-2021-41773", "service": "http", "port": 80, "exploit_available": True}
    s1 = estimate_cvss_from_finding(f1)
    print(f"    CVE-2021-41773 → {s1}  (expected 9.8)")
    assert s1 == 9.8, f"Expected 9.8, got {s1}"

    # [4] estimate — service heuristic
    print("\n[4] estimate_cvss_from_finding — service heuristic:")
    f2 = {"service": "vsftpd", "port": 21, "exploit_available": True}
    s2 = estimate_cvss_from_finding(f2)
    print(f"    vsftpd + exploit → {s2}  (expected 9.0)")
    assert s2 == 9.0, f"Expected 9.0, got {s2}"

    f3 = {"service": "openssh", "port": 22, "exploit_available": False}
    s3 = estimate_cvss_from_finding(f3)
    print(f"    openssh no-exploit → {s3}  (expected 3.5)")
    assert s3 == 3.5, f"Expected 3.5, got {s3}"

    # [5] estimate — severity fallback
    print("\n[5] estimate_cvss_from_finding — severity fallback:")
    f4 = {"severity": "high", "port": 9999}
    s4 = estimate_cvss_from_finding(f4)
    print(f"    severity=high → {s4}  (expected 8.0)")
    assert s4 == 8.0

    # [6] Invalid metric value
    print("\n[6] Invalid metric raises ValueError:")
    try:
        calculate_cvss3("X", "L", "N", "N", "U", "H", "H", "H")
        assert False, "Should have raised"
    except ValueError as e:
        print(f"    Correctly raised: {e}")

    print("\nAll tests passed.")