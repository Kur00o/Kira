"""
phase_controller.py — Kira Autonomous Penetration Testing Agent
Defines what "done" means for each phase and what tools the LLM is allowed to pick.
The planner reads phase focus text in every prompt — tight phase definitions
stop qwen2.5-coder:14b-instruct-q4_K_M from jumping to exploitation before it has finished enumerating.

Usage:
    from phase_controller import PHASE_CRITERIA, get_phase_prompt, is_phase_complete, get_allowed_tools
"""

# ─────────────────────────────────────────────
# Phase Criteria
# ─────────────────────────────────────────────

PHASE_CRITERIA = {
    "RECON": {
        "complete_when": lambda s: len(s.get("open_ports", [])) > 0,
        "focus": "Run nmap on heavily used ports (22,25,53,80,443,8080,3128,4443,4444,8090,8443) to discover service versions.",
        "allowed_tools": ["nmap_scan", "add_note"],
    },
    "ENUM": {
        "complete_when": lambda s: (
            len(s.get("findings", [])) > 0
            or len(s.get("web_paths", [])) > 0
            or any(
                a.get("tool") in ("searchsploit", "curl_probe", "whatweb", "enum4linux")
                for a in s.get("actions_taken", [])
            )
        ),
        "focus": "Enumerate every service. Gobuster on HTTP. Check FTP anon, SMB.",
        "allowed_tools": [
            "gobuster_dir",
            "enum4linux",
            "curl_probe",
            "whatweb",
            "searchsploit",
            "add_finding",
            "add_note",
            "advance_phase",
        ],
    },
    "VULN_SCAN": {
        "complete_when": lambda s: any(
            f.get("exploit_available") for f in s.get("findings", [])
        ),
        "focus": "Cross-reference all service versions. Find exploitable CVEs.",
        "allowed_tools": ["searchsploit", "add_finding", "add_note", "advance_phase"],
    },
    "EXPLOIT": {
        "complete_when": lambda s: len(s.get("sessions", [])) > 0,
        "focus": "Run the highest-CVSS MSF module. Get a shell.",
        "allowed_tools": ["msf_search", "msf_exploit", "shell_cmd", "add_note"],
    },
    "POST_EXPLOIT": {
        "complete_when": lambda s: s.get("is_root", False),
        "focus": "Run linpeas. Escalate to root.",
        "allowed_tools": ["shell_cmd", "linpeas", "add_note", "REPORT"],
    },
}

PHASE_ORDER = ["RECON", "ENUM", "VULN_SCAN", "EXPLOIT", "POST_EXPLOIT"]


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def get_phase_prompt(phase: str, state: dict) -> str:
    """
    Returns a focused instruction string injected into every LLM prompt.
    Tells qwen2.5-coder:14b-instruct-q4_K_M exactly what the current phase expects and what tools to use.

    Args:
        phase: current phase key e.g. "RECON"
        state: current state dict from StateManager

    Returns:
        Formatted string for LLM system/user prompt injection
    """
    criteria = PHASE_CRITERIA.get(phase)
    if not criteria:
        return f"Unknown phase: {phase}. Use advance_phase to continue."

    allowed = ", ".join(criteria["allowed_tools"])
    complete = is_phase_complete(phase, state)
    status = "COMPLETE — call advance_phase" if complete else "IN PROGRESS"

    return (
        f"Current phase: {phase} [{status}]\n"
        f"Goal: {criteria['focus']}\n"
        f"Allowed tools this phase: {allowed}\n"
        f"Do NOT use tools outside this list until the phase is complete."
    )


def is_phase_complete(phase: str, state: dict) -> bool:
    """
    Evaluate whether the current phase completion condition is met.

    Args:
        phase: phase key e.g. "RECON"
        state: current state dict from StateManager

    Returns:
        True if phase is done, False otherwise
    """
    criteria = PHASE_CRITERIA.get(phase)
    if not criteria:
        return False
    try:
        return bool(criteria["complete_when"](state))
    except Exception:
        return False


def get_allowed_tools(phase: str) -> list[str]:
    """
    Return the list of tools the LLM is allowed to call in this phase.

    Args:
        phase: phase key e.g. "ENUM"

    Returns:
        List of tool name strings
    """
    criteria = PHASE_CRITERIA.get(phase)
    if not criteria:
        return []
    return criteria["allowed_tools"]


def next_phase(current_phase: str) -> str | None:
    """
    Return the next phase key after current_phase.
    Returns None if already at the last phase.
    """
    try:
        idx = PHASE_ORDER.index(current_phase)
        if idx + 1 < len(PHASE_ORDER):
            return PHASE_ORDER[idx + 1]
    except ValueError:
        pass
    return None


def get_all_phases() -> list[str]:
    """Return ordered list of all phase keys."""
    return PHASE_ORDER.copy()


# ─────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Mock state dicts for each phase completion condition
    mock_states = {
        "RECON": {
            "empty":    {"open_ports": []},
            "complete": {"open_ports": [{"port": 22}, {"port": 80}]},
        },
        "ENUM": {
            "empty":    {"findings": []},
            "complete": {"findings": [{"service": "ftp", "exploit_available": False}]},
        },
        "VULN_SCAN": {
            "empty":    {"findings": [{"service": "ftp", "exploit_available": False}]},
            "complete": {"findings": [{"service": "vsftpd", "exploit_available": True}]},
        },
        "EXPLOIT": {
            "empty":    {"sessions": []},
            "complete": {"sessions": [{"id": 1, "type": "meterpreter"}]},
        },
        "POST_EXPLOIT": {
            "empty":    {"is_root": False},
            "complete": {"is_root": True},
        },
    }

    print("=" * 60)
    print("PHASE CONTROLLER SMOKE TEST")
    print("=" * 60)

    for phase in PHASE_ORDER:
        print(f"\n── {phase} ──")

        # Test is_phase_complete
        empty_state    = mock_states[phase]["empty"]
        complete_state = mock_states[phase]["complete"]
        not_done = is_phase_complete(phase, empty_state)
        done     = is_phase_complete(phase, complete_state)
        print(f"  is_phase_complete (empty):    {not_done}  (expected False)")
        print(f"  is_phase_complete (complete): {done}   (expected True)")

        # Test get_allowed_tools
        tools = get_allowed_tools(phase)
        print(f"  allowed_tools: {tools}")

        # Test get_phase_prompt
        prompt = get_phase_prompt(phase, complete_state)
        print(f"  prompt preview: {prompt.splitlines()[0]}")

        # Test next_phase
        nxt = next_phase(phase)
        print(f"  next_phase: {nxt}")

    print("\n[OK] All phases passed smoke test")