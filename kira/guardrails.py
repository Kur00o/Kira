"""
kira/guardrails.py — ScopeGuard
"""

from __future__ import annotations

import ipaddress
import re
import sys
from typing import Optional

_DESTRUCTIVE_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-[a-z]*r[a-z]*f\b", "rm -rf is destructive"),
    (r"\bdd\s+if=", "dd if= can wipe disks"),
    (r"\bmkfs\b", "mkfs formats filesystems"),
    (r"\bshred\b", "shred destroys files permanently"),
    (r"\bchmod\s+777\s+/\b", "chmod 777 / is globally destructive"),
    (r"\bchmod\s+-R\s+777\s+/\b", "chmod -R 777 / is globally destructive"),
    (r">\s*/dev/(sda|hda|nvme\w+)", "writing to raw disk device"),
    (r"\bwipefs\b", "wipefs erases filesystem signatures"),
    (r"\bsfdisk\b", "sfdisk modifies partition tables"),
    (r"\bparted\b.*rm\b", "parted rm deletes partitions"),
    (r"echo\s+.+\s*>\s*/etc/shadow", "overwriting /etc/shadow"),
    (r">\s*/etc/passwd\b", "overwriting /etc/passwd"),
    (r"\bfork\s*bomb\b|:\(\)\{.*\}\s*;", "fork bomb pattern"),
    (r"\bpoweroff\b|\bshutdown\b|\breboot\b", "shutdown/reboot command"),
    (r"\bkillall\s+-9\b", "killall -9 terminates all processes"),
]

_TARGET_ARG_TOOLS = {"nmap_scan", "gobuster_dir", "enum4linux", "curl_probe", "whatweb", "msf_exploit"}
_MIN_AUTH_LENGTH = 5


class ScopeGuard:
    def __init__(self, authorized_target: str, authorized_by: str):
        self.authorized_target = authorized_target.strip()
        self.authorized_by = authorized_by.strip()
        self._auth_network: Optional[ipaddress.IPv4Network] = None
        try:
            self._auth_network = ipaddress.ip_network(self.authorized_target, strict=False)
        except ValueError:
            pass

    def validate_startup(self, log=None) -> None:
        if not self.authorized_by:
            _die(
                "AUTHORIZATION REQUIRED\n"
                "You must pass --authorized-by with a written authorization.\n"
                "Example: --authorized-by 'Lab VM — authorized by John Smith'"
            )
        if len(self.authorized_by) < _MIN_AUTH_LENGTH:
            _die(
                f"--authorized-by is too short ('{self.authorized_by}').\n"
                f"Provide a meaningful authorization statement (min {_MIN_AUTH_LENGTH} chars)."
            )

        try:
            ipaddress.ip_address(self.authorized_target)
        except ValueError:
            try:
                ipaddress.ip_network(self.authorized_target, strict=False)
            except ValueError:
                _die(f"Invalid target: '{self.authorized_target}'")

        if log is not None:
            log.info(
                f"AUTHORIZATION RECORDED — target={self.authorized_target} "
                f"authorized_by='{self.authorized_by}'"
            )

    def check_action(self, action: dict) -> tuple[bool, str]:
        tool = action.get("tool", "")
        args = action.get("args", {})

        if tool in _TARGET_ARG_TOOLS:
            target_in_args = (
                args.get("target")
                or _extract_ip_from_url(args.get("url", ""))
                or args.get("options", {}).get("RHOSTS", "")
            )
            if target_in_args:
                blocked, reason = self._check_target_scope(target_in_args)
                if blocked:
                    return False, reason

        if tool == "shell_cmd":
            cmd = args.get("cmd", "")
            blocked, reason = self._check_destructive(cmd)
            if blocked:
                return False, reason

        if tool == "msf_exploit":
            rhosts = args.get("options", {}).get("RHOSTS", "")
            if rhosts:
                blocked, reason = self._check_target_scope(rhosts)
                if blocked:
                    return False, reason

        return True, ""

    def _check_target_scope(self, target: str) -> tuple[bool, str]:
        target = target.strip()
        if not target:
            return False, ""
        if target == self.authorized_target:
            return False, ""
        if target in ("127.0.0.1", "0.0.0.0", "localhost", "::1"):
            return False, ""
        try:
            target_ip = ipaddress.ip_address(target)
        except ValueError:
            return False, ""
        if self._auth_network and target_ip in self._auth_network:
            return False, ""
        try:
            auth_ip = ipaddress.ip_address(self.authorized_target)
            auth_net = ipaddress.ip_network(f"{auth_ip}/24", strict=False)
            if target_ip in auth_net:
                return False, ""
        except ValueError:
            pass
        return True, (
            f"SCOPE VIOLATION: target '{target}' is outside authorized scope "
            f"(authorized: {self.authorized_target})."
        )

    def _check_destructive(self, cmd: str) -> tuple[bool, str]:
        if not cmd:
            return False, ""
        for pattern, reason in _DESTRUCTIVE_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                return True, (
                    f"DESTRUCTIVE COMMAND BLOCKED: '{cmd[:80]}' — {reason}. "
                    "Kira will not execute commands that could cause irreversible damage."
                )
        return False, ""


def _extract_ip_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _die(msg: str) -> None:
    print(f"\nERROR: {msg}\n")
    sys.exit(1)
