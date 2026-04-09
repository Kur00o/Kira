"""
msf_client.py — Kira Autonomous Penetration Testing Agent
Metasploit RPC wrapper. All MSF operations go through this class.

msfrpcd is auto-started on first use — no manual startup needed.

Usage:
    from msf_client import MSFClient
    client = MSFClient()
    client.auto_start()   # launches msfrpcd if not running, then connects
"""

import logging
import socket
import subprocess
import time
from typing import Optional

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DEFAULT_HOST     = "127.0.0.1"
DEFAULT_PORT     = 55553
DEFAULT_PASSWORD = "kirapass"
POLL_INTERVAL    = 3    # seconds between job status checks
MAX_POLL_WAIT    = 120  # seconds before giving up on a running module


# ─────────────────────────────────────────────
# MSFClient
# ─────────────────────────────────────────────

class MSFClient:
    """
    Kira's only interface to Metasploit.
    All MSF operations — module search, option config,
    exploit execution, session management — go through here.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log     = logger or logging.getLogger("kira.msf")
        self._client = None   # pymetasploit3 MsfRpcClient instance
        self.connected = False

    # ── Auto-start ────────────────────────────

    def auto_start(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        password: str = DEFAULT_PASSWORD,
    ) -> bool:
        """
        Auto-start msfrpcd if not already running, then connect.
        Call this instead of connect() — no manual terminal needed.
        """
        if self._is_msfrpcd_running(port):
            self.log.info("[MSF] msfrpcd already running — connecting")
            return self.connect(host, port, password)

        self.log.info("[MSF] msfrpcd not running — starting it now...")
        try:
            subprocess.Popen(
                [
                    "msfrpcd",
                    "-P", password,
                    "-S",
                    "-a", host,
                    "-p", str(port),
                    "-f",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.log.error("[MSF] msfrpcd not found — is metasploit installed?")
            self.log.error("      sudo apt install -y metasploit-framework")
            return False

        # Wait up to 60s for msfrpcd to come up
        self.log.info("[MSF] Waiting for msfrpcd to start (up to 60s)...")
        for attempt in range(1, 13):
            time.sleep(5)
            self.log.debug(f"[MSF] Attempt {attempt}/12...")
            if self.connect(host, port, password):
                return True

        self.log.error("[MSF] msfrpcd failed to start after 60s")
        return False

    def _is_msfrpcd_running(self, port: int = DEFAULT_PORT) -> bool:
        """Check if something is already listening on the MSF RPC port."""
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            return False

    # ── Connection ────────────────────────────

    def connect(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        password: str = DEFAULT_PASSWORD,
    ) -> bool:
        """
        Connect to msfrpcd. Returns True on success.
        Prefer auto_start() over calling this directly.
        """
        try:
            from pymetasploit3.msfrpc import MsfRpcClient
            self._client = MsfRpcClient(
                password,
                server=host,
                port=port,
                ssl=False,
            )
            self.connected = True
            self.log.info(f"[MSF] Connected to msfrpcd at {host}:{port}")
            return True
        except Exception as e:
            self.log.error(f"[MSF] Connection failed: {e}")
            self.connected = False
            return False

    def _require_connection(self):
        if not self.connected or self._client is None:
            raise RuntimeError("MSFClient not connected. Call auto_start() first.")

    # ── Module search ─────────────────────────

    def search(self, query: str) -> list[dict]:
        """
        Search for MSF modules matching query string.
        Returns list of {module, name, rank, description, type}

        Example:
            results = client.search("vsftpd")
            # [{"module": "exploit/unix/ftp/vsftpd_234_backdoor", ...}]
        """
        self._require_connection()
        try:
            results = []
            module_types = ["exploits", "auxiliary", "post", "payloads"]
            for mtype in module_types:
                try:
                    modules = self._client.modules.list(mtype)
                    for mod_name in modules:
                        if query.lower() in mod_name.lower():
                            results.append({
                                "module":      mod_name,
                                "type":        mtype.rstrip("s"),
                                "name":        mod_name,
                                "rank":        "unknown",
                                "description": "",
                            })
                except Exception:
                    continue

            self.log.info(f"[MSF] search('{query}') -> {len(results)} results")
            return results[:20]

        except Exception as e:
            self.log.error(f"[MSF] search failed: {e}")
            return []

    def get_module_info(self, module_type: str, module_name: str) -> dict:
        """
        Get detailed info about a module including options and description.
        module_type: exploit / auxiliary / post / payload
        """
        self._require_connection()
        try:
            mod = self._client.modules.use(module_type, module_name)
            return {
                "name":             module_name,
                "description":      getattr(mod, "description", ""),
                "rank":             getattr(mod, "rank", "unknown"),
                "options":          mod.options if hasattr(mod, "options") else [],
                "required_options": mod.required if hasattr(mod, "required") else [],
            }
        except Exception as e:
            self.log.error(f"[MSF] get_module_info failed: {e}")
            return {}

    # ── Exploit execution ─────────────────────

    def run_module(
        self,
        module: str,
        options: dict,
        payload: Optional[str] = None,
    ) -> dict:
        """
        Execute an MSF module with given options.
        module format: "exploit/unix/ftp/vsftpd_234_backdoor"

        Returns:
            {
                "success":    bool,
                "session_id": int or None,
                "output":     str,
                "error":      str,
            }
        """
        self._require_connection()
        result = {
            "success":    False,
            "session_id": None,
            "output":     "",
            "error":      "",
        }

        try:
            parts       = module.split("/")
            module_type = parts[0]
            module_path = "/".join(parts[1:]) if len(parts) > 1 else module

            self.log.info(f"[MSF] Loading module: {module}")
            mod = self._client.modules.use(module_type, module_path)

            for key, value in options.items():
                mod[key] = value
                self.log.debug(f"[MSF] set {key} = {value}")

            if payload:
                mod["PAYLOAD"] = payload
                self.log.info(f"[MSF] Payload: {payload}")

            self.log.info(f"[MSF] Running {module} against {options.get('RHOSTS', '?')}")
            job    = mod.execute()
            job_id = job.get("job_id")
            uuid   = job.get("uuid", "")
            self.log.info(f"[MSF] Job started: id={job_id} uuid={uuid}")

            session_id = self._wait_for_session(uuid)
            if session_id:
                result["success"]    = True
                result["session_id"] = session_id
                result["output"]     = f"Session {session_id} opened"
                self.log.info(f"[MSF] Session opened: {session_id}")
            else:
                result["output"] = "Module ran but no session opened"
                self.log.info("[MSF] No session created")

        except Exception as e:
            result["error"] = str(e)
            self.log.error(f"[MSF] run_module failed: {e}")

        return result

    def _wait_for_session(self, uuid: str) -> Optional[int]:
        """Poll until a new session appears or timeout is reached."""
        existing = set(self._client.sessions.list.keys())
        elapsed  = 0

        while elapsed < MAX_POLL_WAIT:
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            current      = set(self._client.sessions.list.keys())
            new_sessions = current - existing
            if new_sessions:
                return int(list(new_sessions)[0])
            self.log.debug(f"[MSF] Waiting for session... ({elapsed}s)")

        return None

    # ── Session management ────────────────────

    def list_sessions(self) -> dict:
        """
        List all active sessions.
        Returns {session_id: {type, tunnel, info, target, via}}
        """
        self._require_connection()
        try:
            raw      = self._client.sessions.list
            sessions = {}
            for sid, info in raw.items():
                sessions[int(sid)] = {
                    "type":   info.get("type", "shell"),
                    "tunnel": info.get("tunnel_peer", ""),
                    "info":   info.get("info", ""),
                    "target": info.get("target_host", ""),
                    "via":    info.get("via_exploit", ""),
                }
            self.log.info(f"[MSF] Active sessions: {list(sessions.keys())}")
            return sessions
        except Exception as e:
            self.log.error(f"[MSF] list_sessions failed: {e}")
            return {}

    def shell_cmd(self, session_id: int, cmd: str, timeout: int = 30) -> str:
        """
        Run a shell command inside an active session.
        Works for both shell and meterpreter sessions.
        Returns command stdout as string.
        """
        self._require_connection()
        try:
            session      = self._client.sessions.session(str(session_id))
            session_type = self._client.sessions.list.get(
                str(session_id), {}
            ).get("type", "shell")

            if session_type == "meterpreter":
                result = session.run_with_output(cmd, timeout=timeout)
            else:
                session.write(cmd + "\n")
                time.sleep(2)
                result = session.read()

            self.log.info(f"[MSF] shell_cmd({session_id}, '{cmd}') -> {result[:80]}")
            return result

        except Exception as e:
            self.log.error(f"[MSF] shell_cmd failed: {e}")
            return ""

    def close_session(self, session_id: int) -> bool:
        """Kill an active session. Returns True on success."""
        self._require_connection()
        try:
            self._client.sessions.session(str(session_id)).stop()
            self.log.info(f"[MSF] Session {session_id} closed")
            return True
        except Exception as e:
            self.log.error(f"[MSF] close_session failed: {e}")
            return False

    # ── Convenience helpers ───────────────────

    def quick_exploit(self, target_ip: str, module: str, extra_options: dict = {}) -> dict:
        """
        Shortcut — set RHOSTS to target_ip and run module.
        Used by the agent's _do_metasploit() dispatcher.
        """
        options = {"RHOSTS": target_ip, **extra_options}
        return self.run_module(module, options)

    def is_connected(self) -> bool:
        return self.connected and self._client is not None


# ─────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="MSFClient smoke test")
    parser.add_argument("--host",     default=DEFAULT_HOST)
    parser.add_argument("--port",     default=DEFAULT_PORT, type=int)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--search",   default="vsftpd", help="Module to search for")
    args = parser.parse_args()

    client = MSFClient()

    print(f"\n[1] Auto-starting msfrpcd at {args.host}:{args.port}")
    ok = client.auto_start(args.host, args.port, args.password)
    if not ok:
        print("    FAIL — is metasploit installed?")
        print("    sudo apt install -y metasploit-framework")
        sys.exit(1)
    print("    PASS\n")

    print(f"[2] Searching for '{args.search}' modules")
    results = client.search(args.search)
    for r in results[:5]:
        print(f"    {r['module']}")
    print()

    print("[3] Listing active sessions")
    sessions = client.list_sessions()
    if sessions:
        for sid, info in sessions.items():
            print(f"    [{sid}] {info['type']} — {info['tunnel']}")
    else:
        print("    No active sessions")

    print("\n[OK] Smoke test complete")