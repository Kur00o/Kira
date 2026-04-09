"""
kira/tool_runner.py — ToolRunner
================================
All external tool execution flows through here.
Never call subprocess directly from any other module.

Responsibilities:
  - Spawn tool processes with timeout + stream capture
  - Save raw outputs to the session directory
  - Write every invocation to actions.jsonl (append-only audit log)
  - Return a clean ToolResult so callers never parse raw subprocess objects
  - Execute shell commands on live Metasploit sessions  ← NEW Day 3
  - Replay past session logs without re-running tools   ← NEW Day 3

Usage:
    runner = ToolRunner(session_dir="./sessions/my_scan")

    result = runner.nmap(target="10.10.10.5", flags="-sV -sC -p-")
    result = runner.gobuster(url="http://10.10.10.5", wordlist="/usr/share/wordlists/dirb/common.txt")
    result = runner.run(["echo", "hello"])          # raw fallback

    # Day 3 — attach MSF client after a session opens:
    #   from pymetasploit3.msfrpc import MsfRpcClient
    #   msf = MsfRpcClient("kira_msf_pass", port=55553, ssl=True)
    #   runner.attach_msf(msf)
    #   result = runner.shell_cmd("whoami", session_id=1)

    # Day 4 — replay a past session for reporting:
    #   entries = ToolRunner.load_action_log("./sessions/scan1/actions.jsonl")
    #   stats   = ToolRunner.summarise_action_log("./sessions/scan1/actions.jsonl")
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── ToolResult ─────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """
    Uniform return type for every tool call.
    Callers check .ok before reading .stdout / .artifact_path.
    """
    tool:          str
    cmd:           list
    ok:            bool           # True if returncode == 0 and no timeout
    stdout:        str   = ""
    stderr:        str   = ""
    returncode:    int   = -1
    elapsed_s:     float = 0.0
    timed_out:     bool  = False
    artifact_path: Optional[str] = None   # path to saved output file, if any
    error:         Optional[str] = None   # human-readable error message
    timestamp:     str   = field(default_factory=lambda: _ts())

    @property
    def summary(self) -> str:
        """One-line summary for action log + LLM context."""
        if self.timed_out:
            return f"TIMEOUT after {self.elapsed_s:.0f}s"
        if not self.ok:
            short_err = (self.stderr or self.error or "non-zero exit")[:120]
            return f"FAILED (rc={self.returncode}): {short_err}"
        lines   = [l for l in self.stdout.splitlines() if l.strip()]
        preview = lines[0][:120] if lines else "(no output)"
        return f"OK ({len(lines)} lines) — {preview}"

    def to_log_dict(self) -> dict:
        """Serialisable dict for JSONL logging."""
        return {
            "timestamp":     self.timestamp,
            "tool":          self.tool,
            "cmd":           " ".join(str(c) for c in self.cmd),
            "ok":            self.ok,
            "returncode":    self.returncode,
            "elapsed_s":     round(self.elapsed_s, 2),
            "timed_out":     self.timed_out,
            "artifact_path": self.artifact_path,
            "summary":       self.summary,
        }


# ── ToolRunner ─────────────────────────────────────────────────────────────────

class ToolRunner:
    """
    Central execution hub for all external tools Kira uses.

    session_dir : root directory for this scan session.
                  raw outputs  → session_dir/raw/
                  action log   → session_dir/actions.jsonl
    verbose     : stream tool output to stdout in real time
    msf         : optional MsfRpcClient — pass at construction or via attach_msf()
    """

    # Default timeouts (seconds) per tool class
    # Optimized for local environment with aggressive scanning
    TIMEOUTS = {
        "nmap":       180,   # Fast -sV with --min-rate 2000
        "gobuster":   120,   # Directory brute-force
        "ffuf":       120,
        "searchsploit": 15,
        "enum4linux": 60,
        "curl":        10,
        "whatweb":     15,
        "hydra":      180,
        "default":    60,
    }

    # ── __init__ ───────────────────────────────────────────────────────────────

    def __init__(self, session_dir: str, verbose: bool = True, msf=None):
        self.session_dir = Path(session_dir)
        self.raw_dir     = self.session_dir / "raw"
        self.log_path    = self.session_dir / "actions.jsonl"
        self.verbose     = verbose
        self._log_lock   = threading.Lock()

        # Metasploit RPC client.
        # None until attach_msf() is called or passed at construction.
        # Type: pymetasploit3.msfrpc.MsfRpcClient | None
        self.msf = msf                                          # ← NEW Day 3

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)

    # ── NEW Day 3: attach MSF client ──────────────────────────────────────────

    def attach_msf(self, msf_client) -> None:                  # ← NEW Day 3
        """
        Attach a live MsfRpcClient so shell_cmd() can execute on sessions.
        Call this after pymetasploit3 opens a Metasploit session.

        Example:
            from pymetasploit3.msfrpc import MsfRpcClient
            msf = MsfRpcClient("kira_msf_pass", port=55553, ssl=True)
            runner.attach_msf(msf)
        """
        self.msf = msf_client
        if self.verbose:
            _print_status("[MSF] MsfRpcClient attached to ToolRunner")

    # ── Low-level runner ───────────────────────────────────────────────────────

    def run(
        self,
        cmd:           list,
        tool_name:     str          = "shell",
        timeout:       Optional[int] = None,
        save_output:   bool         = True,
        output_suffix: str          = ".txt",
        env:           Optional[dict] = None,
        cwd:           Optional[str]  = None,
    ) -> ToolResult:
        """
        Execute any command and return a ToolResult.

        Parameters
        ----------
        cmd           : command + args as a list, e.g. ["nmap", "-sV", "10.10.10.5"]
        tool_name     : human label used in logs
        timeout       : seconds before SIGKILL; falls back to TIMEOUTS[tool_name]
        save_output   : write stdout to raw/<tool>_<ts>.txt
        output_suffix : file extension for saved output (".xml", ".txt", etc.)
        env           : extra environment variables (merged with os.environ)
        cwd           : working directory for the subprocess
        """
        if timeout is None:
            timeout = self.TIMEOUTS.get(tool_name, self.TIMEOUTS["default"])

        cmd       = [str(c) for c in cmd]
        start     = time.monotonic()
        timed_out = False
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        returncode = -1

        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        if self.verbose:
            _print_status(f"[RUN] {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=run_env,
                cwd=cwd,
                bufsize=1,
            )

            stdout_reader = _StreamReader(proc.stdout, stdout_chunks, self.verbose)
            stderr_reader = _StreamReader(proc.stderr, stderr_chunks, verbose=False)
            stdout_reader.start()
            stderr_reader.start()

            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                proc.wait()

            stdout_reader.join(timeout=5)
            stderr_reader.join(timeout=5)
            returncode = proc.returncode

        except FileNotFoundError:
            elapsed = time.monotonic() - start
            result  = ToolResult(
                tool=tool_name, cmd=cmd, ok=False,
                returncode=-1, elapsed_s=elapsed,
                error=f"Tool not found: '{cmd[0]}'. Is it installed and on PATH?",
            )
            self._log(result)
            return result

        except Exception as exc:
            elapsed = time.monotonic() - start
            result  = ToolResult(
                tool=tool_name, cmd=cmd, ok=False,
                returncode=-1, elapsed_s=elapsed,
                error=f"Unexpected error: {exc}",
            )
            self._log(result)
            return result

        elapsed = time.monotonic() - start
        stdout  = "".join(stdout_chunks)
        stderr  = "".join(stderr_chunks)
        ok      = (returncode == 0) and not timed_out

        artifact_path = None
        if save_output and (stdout or stderr):
            fname = f"{tool_name}_{_ts_file()}{output_suffix}"
            artifact_path = str(self.raw_dir / fname)
            with open(artifact_path, "w") as f:
                if stderr:
                    f.write("=== STDERR ===\n" + stderr + "\n=== STDOUT ===\n")
                f.write(stdout)

        result = ToolResult(
            tool=tool_name, cmd=cmd, ok=ok,
            stdout=stdout, stderr=stderr,
            returncode=returncode, elapsed_s=elapsed,
            timed_out=timed_out, artifact_path=artifact_path,
        )
        self._log(result)

        if self.verbose:
            status = "OK" if ok else ("TIMEOUT" if timed_out else f"FAIL rc={returncode}")
            _print_status(f"[{status}] {tool_name} finished in {elapsed:.1f}s")

        return result

    # ── Nmap ───────────────────────────────────────────────────────────────────

    def nmap(
        self,
        target:  str,
        flags:   str          = "-sV -sC",
        ports:   Optional[str] = None,
        extra:   list         = None,
        timeout: int          = 600,
    ) -> ToolResult:
        """
        Run nmap and save XML + grepable output.
        result.artifact_path → .xml file for nmap_parser.py
        """
        self._require("nmap")

        xml_out  = str(self.raw_dir / f"nmap_{_ts_file()}.xml")
        grep_out = str(self.raw_dir / f"nmap_{_ts_file()}.gnmap")

        cmd = ["nmap"] + flags.split()
        if ports:
            cmd += ["-p", ports]
        cmd += ["-oX", xml_out, "-oG", grep_out, "--open", "--reason"]
        if extra:
            cmd += extra
        cmd.append(target)

        result = self.run(cmd, tool_name="nmap", timeout=timeout,
                          save_output=True, output_suffix=".txt")
        if Path(xml_out).exists():
            result.artifact_path = xml_out
        return result

    # ── Gobuster ───────────────────────────────────────────────────────────────

    def gobuster(
        self,
        url:        str,
        wordlist:   str  = "/usr/share/wordlists/dirb/common.txt",
        extensions: str  = "php,html,txt,bak,zip",
        threads:    int  = 20,
        timeout:    int  = 300,
        extra:      list = None,
    ) -> ToolResult:
        """
        Run gobuster dir mode. Falls back to ffuf if gobuster is missing.
        result.stdout contains discovered paths one per line.
        """
        if shutil.which("gobuster"):
            return self._gobuster_native(url, wordlist, extensions, threads, timeout, extra)
        elif shutil.which("ffuf"):
            _print_status("[WARN] gobuster not found, falling back to ffuf")
            return self._ffuf_fallback(url, wordlist, threads, timeout)
        else:
            result = ToolResult(
                tool="gobuster", cmd=[], ok=False,
                error="Neither gobuster nor ffuf found on PATH. "
                      "Install: apt install gobuster",
            )
            self._log(result)
            return result

    def _gobuster_native(self, url, wordlist, extensions, threads, timeout, extra):
        out_file = str(self.raw_dir / f"gobuster_{_ts_file()}.txt")
        cmd = ["gobuster", "dir", "-u", url, "-w", wordlist, "-x", extensions,
               "-t", str(threads), "-o", out_file, "--no-error", "-q"]
        if extra:
            cmd += extra
        result = self.run(cmd, tool_name="gobuster", timeout=timeout,
                          save_output=True, output_suffix=".txt")
        if Path(out_file).exists():
            result.artifact_path = out_file
            if not result.stdout.strip():
                result.stdout = Path(out_file).read_text()
        return result

    def _ffuf_fallback(self, url, wordlist, threads, timeout):
        out_file = str(self.raw_dir / f"ffuf_{_ts_file()}.json")
        cmd = ["ffuf", "-u", url.rstrip("/") + "/FUZZ", "-w", wordlist,
               "-t", str(threads), "-of", "json", "-o", out_file, "-s"]
        result = self.run(cmd, tool_name="ffuf", timeout=timeout,
                          save_output=True, output_suffix=".txt")
        if Path(out_file).exists():
            result.artifact_path = out_file
        return result

    # ── Searchsploit ───────────────────────────────────────────────────────────

    def searchsploit(self, query: str, timeout: int = 30) -> ToolResult:
        """Search exploit-db for a service/version string."""
        self._require("searchsploit")
        return self.run(["searchsploit", "--json", query],
                        tool_name="searchsploit", timeout=timeout,
                        save_output=True, output_suffix=".json")

    # ── Enum4linux ─────────────────────────────────────────────────────────────

    def enum4linux(self, target: str, flags: str = "-a", timeout: int = 120) -> ToolResult:
        """Run enum4linux for SMB/LDAP enumeration."""
        self._require("enum4linux")
        return self.run(["enum4linux"] + flags.split() + [target],
                        tool_name="enum4linux", timeout=timeout,
                        save_output=True, output_suffix=".txt")

    # ── curl ───────────────────────────────────────────────────────────────────

    def curl(self, url: str, flags: str = "-sI", timeout: int = 20) -> ToolResult:
        """Quick HTTP probe — headers, redirect chain, server banner."""
        self._require("curl")
        return self.run(["curl"] + flags.split() + ["--max-time", "15", url],
                        tool_name="curl", timeout=timeout, save_output=False)

    # ── WhatWeb ────────────────────────────────────────────────────────────────

    def whatweb(self, url: str, timeout: int = 30) -> ToolResult:
        """Fingerprint web technologies (CMS, frameworks, server versions)."""
        self._require("whatweb")
        return self.run(["whatweb", "--color=never", "-a", "3", url],
                        tool_name="whatweb", timeout=timeout,
                        save_output=True, output_suffix=".txt")

    # ─────────────────────────────────────────────────────────────────────────
    # NEW Day 3 — shell_cmd: real MSFClient call (replaces Day 1 stub)
    # ─────────────────────────────────────────────────────────────────────────

    def shell_cmd(self, cmd: str, session_id: int = 1) -> ToolResult:   # ← UPDATED Day 3
        """
        Run a shell command on an active Metasploit session via MsfRpcClient.

        Requires attach_msf() to have been called first.
        Returns a clear error ToolResult (ok=False) if MSF is not attached —
        the agent loop never crashes on a missing client.

        Parameters
        ----------
        cmd        : shell command string, e.g. "whoami" or "cat /etc/passwd"
        session_id : integer session ID from Metasploit (default 1)

        Change from Day 1:
            Before — always returned ok=False with "not yet implemented" error.
            After  — calls self.msf.sessions.session(id).run_with_output(cmd).
        """
        t0 = time.monotonic()

        # Guard: MSF client not attached yet
        if self.msf is None:
            result = ToolResult(
                tool="shell_cmd",
                cmd=[cmd],
                ok=False,
                error=(
                    "MSFClient not attached. "
                    "Call runner.attach_msf(msf_client) after connecting to msfrpcd."
                ),
            )
            self._log(result)
            return result

        try:
            session = self.msf.sessions.session(str(session_id))
            output  = session.run_with_output(cmd, timeout=30)
            elapsed = time.monotonic() - t0

            result = ToolResult(
                tool="shell_cmd",
                cmd=[cmd],
                ok=True,
                stdout=output.strip(),
                returncode=0,
                elapsed_s=elapsed,
            )

            if self.verbose:
                _print_status(f"[MSF] session {session_id} $ {cmd}")
                preview = (output.strip().splitlines() or ["(no output)"])[0][:80]
                _print_status(f"[MSF] → {preview}")

        except Exception as exc:
            elapsed = time.monotonic() - t0
            result  = ToolResult(
                tool="shell_cmd",
                cmd=[cmd],
                ok=False,
                returncode=-1,
                elapsed_s=elapsed,
                error=f"shell_cmd error on session {session_id}: {exc}",
            )
            if self.verbose:
                _print_status(f"[MSF] shell_cmd FAILED: {exc}")

        self._log(result)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # NEW Day 3 — action log replay (used by reporter.py on Day 4)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def load_action_log(jsonl_path: str) -> list[dict]:             # ← NEW Day 3
        """
        Read a session's actions.jsonl and return all entries as a list of dicts.

        Used by reporter.py on Day 4 to reconstruct the full attack timeline
        without re-running any tools against the target.

        Parameters
        ----------
        jsonl_path : path to the actions.jsonl file
                     (typically session_dir/actions.jsonl)

        Returns
        -------
        list[dict]  one dict per logged action, in chronological order.
                    Keys: timestamp, tool, cmd, ok, returncode,
                          elapsed_s, timed_out, artifact_path, summary.
                    Returns [] if the file does not exist or is empty.

        Robustness: malformed lines (e.g. from a mid-write crash) are
                    skipped with a stderr warning — they never abort the load.

        Example:
            entries = ToolRunner.load_action_log("./sessions/scan1/actions.jsonl")
            for e in entries:
                print(e["timestamp"], e["tool"], e["summary"])
        """
        path = Path(jsonl_path)
        if not path.exists():
            return []

        entries: list[dict] = []
        with open(path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    # Corrupted/partial line — skip and keep going
                    print(
                        f"[load_action_log] skipping malformed line {line_num}: {exc}",
                        file=sys.stderr,
                    )

        return entries

    @staticmethod
    def summarise_action_log(jsonl_path: str) -> dict:             # ← NEW Day 3
        """
        Return high-level statistics about a session's action log.
        Feeds the executive summary section of the report on Day 4.

        Returns dict with keys:
            total_actions   : int
            successful      : int
            failed          : int
            tools_used      : list[str]  unique tool names in order of first use
            total_elapsed_s : float      cumulative tool runtime in seconds
            first_action_at : str | None ISO timestamp
            last_action_at  : str | None ISO timestamp
        """
        entries = ToolRunner.load_action_log(jsonl_path)
        if not entries:
            return {
                "total_actions": 0, "successful": 0, "failed": 0,
                "tools_used": [], "total_elapsed_s": 0.0,
                "first_action_at": None, "last_action_at": None,
            }

        # Preserve first-seen order for tools_used
        seen: dict[str, None] = {}
        for e in entries:
            seen.setdefault(e["tool"], None)

        return {
            "total_actions":   len(entries),
            "successful":      sum(1 for e in entries if e.get("ok")),
            "failed":          sum(1 for e in entries if not e.get("ok")),
            "tools_used":      list(seen.keys()),
            "total_elapsed_s": round(sum(e.get("elapsed_s", 0) for e in entries), 2),
            "first_action_at": entries[0].get("timestamp"),
            "last_action_at":  entries[-1].get("timestamp"),
        }

    # ── Tool availability check ────────────────────────────────────────────────

    def check_tools(self) -> dict:
        """
        Return availability of all tools Kira needs.
        Call at startup to warn about missing binaries.
        """
        tools = ["nmap", "gobuster", "ffuf", "searchsploit",
                 "enum4linux", "curl", "whatweb", "msfconsole"]
        return {t: shutil.which(t) is not None for t in tools}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _log(self, result: ToolResult) -> None:
        """Append a ToolResult to the JSONL action log (thread-safe)."""
        entry = result.to_log_dict()
        with self._log_lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _require(self, binary: str) -> None:
        """Raise a clear EnvironmentError if a required binary is missing."""
        if not shutil.which(binary):
            raise EnvironmentError(
                f"'{binary}' not found on PATH.\n"
                f"Install it:  sudo apt install {binary}   (Kali/Debian)\n"
                f"             brew install {binary}        (macOS)"
            )


# ── Stream reader thread ───────────────────────────────────────────────────────

class _StreamReader(threading.Thread):
    """
    Reads a subprocess stream line-by-line in a background thread.
    Optionally prints each line to stdout as it arrives (verbose mode).
    """
    def __init__(self, stream, chunks: list, verbose: bool = False):
        super().__init__(daemon=True)
        self.stream  = stream
        self.chunks  = chunks
        self.verbose = verbose

    def run(self):
        for line in self.stream:
            self.chunks.append(line)
            if self.verbose and line.strip():
                print(f"  {line}", end="", flush=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _ts_file() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _print_status(msg: str) -> None:
    try:
        from rich.console import Console
        Console().print(f"[dim]{msg}[/dim]")
    except ImportError:
        print(msg)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import shutil as _shutil

    print("=== ToolRunner smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        runner = ToolRunner(session_dir=tmp, verbose=True)

        # ── 1. Tool availability ───────────────────────────────────────────
        print("[1] Checking tool availability...")
        avail = runner.check_tools()
        for tool, found in avail.items():
            print(f"    {tool:<16} {'OK' if found else 'MISSING'}")
        print()

        # ── 2. Generic run ─────────────────────────────────────────────────
        print("[2] Generic run: echo")
        r = runner.run(["echo", "kira is alive"], tool_name="echo", save_output=True)
        assert r.ok and "kira" in r.stdout, f"echo failed: {r.error}"
        print(f"    stdout : {r.stdout.strip()}")
        print(f"    summary: {r.summary}")
        print()

        # ── 3. Timeout enforcement ─────────────────────────────────────────
        print("[3] Timeout test: sleep 10 with 1s timeout")
        r = runner.run(["sleep", "10"], tool_name="sleep", timeout=1, save_output=False)
        assert r.timed_out, "Expected timeout"
        print(f"    timed_out={r.timed_out}  elapsed={r.elapsed_s:.1f}s")
        print()

        # ── 4. Missing binary ──────────────────────────────────────────────
        print("[4] Missing binary test")
        r = runner.run(["definitely_not_a_real_tool_xyz"], tool_name="missing")
        assert not r.ok and r.error
        print(f"    error: {r.error}")
        print()

        # ── 5. JSONL log ───────────────────────────────────────────────────
        print("[5] Checking actions.jsonl...")
        log_path = tmp + "/actions.jsonl"
        assert Path(log_path).exists(), "actions.jsonl not created"
        raw_entries = [json.loads(l) for l in Path(log_path).read_text().strip().splitlines()]
        print(f"    {len(raw_entries)} entries logged")
        for e in raw_entries:
            print(f"    [{e['timestamp']}] {e['tool']:<12} ok={e['ok']}  {e['summary'][:55]}")
        print()

        # ── 6. Nmap (skip if not installed) ───────────────────────────────
        if _shutil.which("nmap"):
            print("[6] Nmap scan on localhost...")
            r = runner.nmap(target="127.0.0.1", flags="-sV --top-ports 10", timeout=60)
            print(f"    ok={r.ok}  artifact={r.artifact_path}")
        else:
            print("[6] nmap not installed — skipping")
        print()

        # ── 7. Raw dir contents ────────────────────────────────────────────
        print("[7] Raw output files saved:")
        for f in sorted(Path(tmp, "raw").iterdir()):
            print(f"    {f.name}  ({f.stat().st_size} bytes)")
        print()

        # ── 8. shell_cmd: no MSF attached → clean error ───────────────────
        # NEW Day 3
        print("[8] shell_cmd — no MSF attached (NEW Day 3)")
        runner_no_msf = ToolRunner(session_dir=tmp, verbose=False)
        r = runner_no_msf.shell_cmd("whoami", session_id=1)
        assert not r.ok,  "Expected ok=False when MSF not attached"
        assert r.error,   "Expected an error message"
        assert "MSFClient not attached" in r.error
        print(f"    ok={r.ok}")
        print(f"    error: {r.error}")
        print()

        # ── 9. shell_cmd: fake MsfRpcClient ───────────────────────────────
        # NEW Day 3
        print("[9] shell_cmd — fake MsfRpcClient wired in (NEW Day 3)")

        class _FakeSession:
            def run_with_output(self, cmd, timeout=30):
                return "uid=0(root) gid=0(root) groups=0(root)"

        class _FakeSessions:
            def session(self, sid):
                return _FakeSession()

        class _FakeMsf:
            sessions = _FakeSessions()

        runner_no_msf.attach_msf(_FakeMsf())
        r = runner_no_msf.shell_cmd("id", session_id=1)
        assert r.ok,              f"Expected ok=True, got error: {r.error}"
        assert "uid=0" in r.stdout
        assert r.tool == "shell_cmd"
        print(f"    ok={r.ok}")
        print(f"    stdout: {r.stdout}")
        print()

        # ── 10. load_action_log: reads entries correctly ───────────────────
        # NEW Day 3
        print("[10] load_action_log() (NEW Day 3)")
        entries = ToolRunner.load_action_log(log_path)
        assert len(entries) > 0, "Expected at least one entry"
        required_keys = {"timestamp", "tool", "ok", "summary"}
        for e in entries:
            missing = required_keys - e.keys()
            assert not missing, f"Entry missing keys: {missing}"
        print(f"    {len(entries)} entries loaded")
        print(f"    tools seen: {sorted({e['tool'] for e in entries})}")
        print()

        # ── 11. load_action_log: missing file returns [] ───────────────────
        # NEW Day 3
        print("[11] load_action_log — missing file returns [] (NEW Day 3)")
        result = ToolRunner.load_action_log("/nonexistent/path/actions.jsonl")
        assert result == []
        print("    correctly returned []")
        print()

        # ── 12. load_action_log: malformed line skipped ────────────────────
        # NEW Day 3
        print("[12] load_action_log — skips malformed JSON line (NEW Day 3)")
        import tempfile as _tf, os as _os
        with _tf.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as bf:
            bf.write(json.dumps({"tool": "echo", "ok": True,  "summary": "first"})  + "\n")
            bf.write("{not valid json at all\n")
            bf.write(json.dumps({"tool": "nmap", "ok": False, "summary": "third"})  + "\n")
            bad_path = bf.name
        entries = ToolRunner.load_action_log(bad_path)
        assert len(entries) == 2, f"Expected 2, got {len(entries)}"
        print(f"    2 valid entries parsed, 1 malformed line skipped")
        _os.unlink(bad_path)
        print()

        # ── 13. summarise_action_log ───────────────────────────────────────
        # NEW Day 3
        print("[13] summarise_action_log() (NEW Day 3)")
        stats = ToolRunner.summarise_action_log(log_path)
        assert stats["total_actions"] > 0
        assert stats["total_actions"] == stats["successful"] + stats["failed"]
        assert isinstance(stats["tools_used"], list)
        assert stats["first_action_at"] is not None
        print(f"    total={stats['total_actions']}  "
              f"ok={stats['successful']}  failed={stats['failed']}")
        print(f"    tools (first-seen order): {stats['tools_used']}")
        print(f"    cumulative elapsed: {stats['total_elapsed_s']}s")
        print(f"    first action: {stats['first_action_at']}")
        print(f"    last  action: {stats['last_action_at']}")

    print("\nAll tests passed.")