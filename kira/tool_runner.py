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

Usage:
    runner = ToolRunner(session_dir="./sessions/my_scan")

    result = runner.nmap(target="10.10.10.5", flags="-sV -sC -p-")
    result = runner.gobuster(url="http://10.10.10.5", wordlist="/usr/share/wordlists/dirb/common.txt")
    result = runner.run(["echo", "hello"])   # raw fallback
"""

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
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
    stdout:        str  = ""
    stderr:        str  = ""
    returncode:    int  = -1
    elapsed_s:     float = 0.0
    timed_out:     bool  = False
    artifact_path: Optional[str] = None   # path to saved output file, if any
    error:         Optional[str] = None   # human-readable error message
    timestamp:     str  = field(default_factory=lambda: _ts())

    @property
    def summary(self) -> str:
        """One-line summary for action log + LLM context."""
        if self.timed_out:
            return f"TIMEOUT after {self.elapsed_s:.0f}s"
        if not self.ok:
            short_err = (self.stderr or self.error or "non-zero exit")[:120]
            return f"FAILED (rc={self.returncode}): {short_err}"
        lines = [l for l in self.stdout.splitlines() if l.strip()]
        preview = lines[0][:120] if lines else "(no output)"
        total = len(lines)
        return f"OK ({total} lines) — {preview}"

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

    session_dir  : root directory for this scan session
                   raw outputs are saved to session_dir/raw/
                   action log  is at  session_dir/actions.jsonl
    verbose      : if True, stream tool output to stdout in real time
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

    def __init__(self, session_dir: str, verbose: bool = True):
        self.session_dir  = Path(session_dir)
        self.raw_dir      = self.session_dir / "raw"
        self.log_path     = self.session_dir / "actions.jsonl"
        self.verbose      = verbose
        self._log_lock    = threading.Lock()

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Low-level runner
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        cmd:          list,
        tool_name:    str  = "shell",
        timeout:      Optional[int] = None,
        save_output:  bool = True,
        output_suffix: str = ".txt",
        env:          Optional[dict] = None,
        cwd:          Optional[str]  = None,
    ) -> ToolResult:
        """
        Execute any command and return a ToolResult.

        Parameters
        ----------
        cmd           : command + args as a list, e.g. ["nmap", "-sV", "10.10.10.5"]
        tool_name     : human label used in logs (e.g. "nmap", "gobuster")
        timeout       : seconds before SIGKILL; falls back to TIMEOUTS[tool_name]
        save_output   : write stdout to raw/<tool>_<ts>.txt
        output_suffix : file extension for saved output (".xml", ".txt", etc.)
        env           : extra environment variables (merged with os.environ)
        cwd           : working directory for the subprocess
        """
        if timeout is None:
            timeout = self.TIMEOUTS.get(tool_name, self.TIMEOUTS["default"])

        cmd = [str(c) for c in cmd]   # ensure all args are strings
        start = time.monotonic()
        timed_out = False
        stdout_chunks = []
        stderr_chunks = []
        returncode = -1
        proc = None

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
                bufsize=1,           # line-buffered
            )

            # Stream stdout in real time; capture stderr separately
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
            # Tool binary not found on PATH
            elapsed = time.monotonic() - start
            result = ToolResult(
                tool=tool_name, cmd=cmd, ok=False,
                returncode=-1, elapsed_s=elapsed,
                error=f"Tool not found: '{cmd[0]}'. Is it installed and on PATH?",
            )
            self._log(result)
            return result

        except Exception as exc:
            elapsed = time.monotonic() - start
            result = ToolResult(
                tool=tool_name, cmd=cmd, ok=False,
                returncode=-1, elapsed_s=elapsed,
                error=f"Unexpected error: {exc}",
            )
            self._log(result)
            return result

        elapsed  = time.monotonic() - start
        stdout   = "".join(stdout_chunks)
        stderr   = "".join(stderr_chunks)
        ok       = (returncode == 0) and not timed_out

        # Save raw output to disk
        artifact_path = None
        if save_output and (stdout or stderr):
            fname = f"{tool_name}_{_ts_file()}{output_suffix}"
            artifact_path = str(self.raw_dir / fname)
            with open(artifact_path, "w") as f:
                if stderr:
                    f.write("=== STDERR ===\n" + stderr + "\n=== STDOUT ===\n")
                f.write(stdout)

        result = ToolResult(
            tool=tool_name,
            cmd=cmd,
            ok=ok,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            elapsed_s=elapsed,
            timed_out=timed_out,
            artifact_path=artifact_path,
        )
        self._log(result)

        if self.verbose:
            status = "OK" if ok else ("TIMEOUT" if timed_out else f"FAIL rc={returncode}")
            _print_status(f"[{status}] {tool_name} finished in {elapsed:.1f}s")

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Nmap
    # ─────────────────────────────────────────────────────────────────────────

    def nmap(
        self,
        target:    str,
        flags:     str  = "-sV -sC",
        ports:     Optional[str] = None,   # e.g. "22,80,443" or "1-65535"
        extra:     list = None,
        timeout:   int  = 600,
    ) -> ToolResult:
        """
        Run nmap against target and save XML + grepable output.

        Returns a ToolResult where:
          result.artifact_path  → path to the .xml file (feed to nmap_parser.py)
          result.stdout         → human-readable nmap output

        Example:
            result = runner.nmap("10.10.10.5", flags="-sV -sC -O")
            xml_path = result.artifact_path
        """
        self._require("nmap")

        xml_out  = str(self.raw_dir / f"nmap_{_ts_file()}.xml")
        grep_out = str(self.raw_dir / f"nmap_{_ts_file()}.gnmap")

        cmd = ["nmap"] + flags.split()

        if ports:
            cmd += ["-p", ports]

        cmd += [
            "-oX", xml_out,        # XML output  (for parser)
            "-oG", grep_out,       # grepable    (bonus)
            "--open",              # only show open ports
            "--reason",            # why nmap thinks port is open
        ]

        if extra:
            cmd += extra

        cmd.append(target)

        result = self.run(
            cmd,
            tool_name="nmap",
            timeout=timeout,
            save_output=True,
            output_suffix=".txt",  # normal output; XML is saved via -oX
        )

        # Point artifact_path at the XML (more useful to callers than .txt)
        if Path(xml_out).exists():
            result.artifact_path = xml_out

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Gobuster
    # ─────────────────────────────────────────────────────────────────────────

    def gobuster(
        self,
        url:       str,
        wordlist:  str  = "/usr/share/wordlists/dirb/common.txt",
        extensions: str = "php,html,txt,bak,zip",
        threads:   int  = 20,
        timeout:   int  = 300,
        extra:     list = None,
    ) -> ToolResult:
        """
        Run gobuster dir mode against a URL.

        result.stdout contains the discovered paths (one per line).
        Fallback to ffuf if gobuster is not installed.

        Example:
            result = runner.gobuster("http://10.10.10.5")
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
                      "Install with: apt install gobuster",
            )
            self._log(result)
            return result

    def _gobuster_native(self, url, wordlist, extensions, threads, timeout, extra):
        out_file = str(self.raw_dir / f"gobuster_{_ts_file()}.txt")
        cmd = [
            "gobuster", "dir",
            "-u", url,
            "-w", wordlist,
            "-x", extensions,
            "-t", str(threads),
            "-o", out_file,
            "--no-error",
            "-q",            # quiet — only results, no progress noise
        ]
        if extra:
            cmd += extra

        result = self.run(cmd, tool_name="gobuster", timeout=timeout,
                          save_output=True, output_suffix=".txt")
        if Path(out_file).exists():
            result.artifact_path = out_file
            # Also read back the saved file to fill stdout if gobuster was quiet
            if not result.stdout.strip():
                result.stdout = Path(out_file).read_text()
        return result

    def _ffuf_fallback(self, url, wordlist, threads, timeout):
        out_file = str(self.raw_dir / f"ffuf_{_ts_file()}.json")
        fuzz_url = url.rstrip("/") + "/FUZZ"
        cmd = [
            "ffuf",
            "-u", fuzz_url,
            "-w", wordlist,
            "-t", str(threads),
            "-of", "json",
            "-o", out_file,
            "-s",   # silent
        ]
        result = self.run(cmd, tool_name="ffuf", timeout=timeout,
                          save_output=True, output_suffix=".txt")
        if Path(out_file).exists():
            result.artifact_path = out_file
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Searchsploit
    # ─────────────────────────────────────────────────────────────────────────

    def searchsploit(self, query: str, timeout: int = 30) -> ToolResult:
        """
        Search exploit-db for a service/version string.

        Example:
            result = runner.searchsploit("Apache 2.4.49")
        """
        self._require("searchsploit")
        cmd = ["searchsploit", "--json", query]
        return self.run(cmd, tool_name="searchsploit", timeout=timeout,
                        save_output=True, output_suffix=".json")

    # ─────────────────────────────────────────────────────────────────────────
    # Enum4linux (SMB enumeration)
    # ─────────────────────────────────────────────────────────────────────────

    def enum4linux(self, target: str, flags: str = "-a", timeout: int = 120) -> ToolResult:
        """
        Run enum4linux for SMB/LDAP enumeration.

        Example:
            result = runner.enum4linux("10.10.10.5")
        """
        self._require("enum4linux")
        cmd = ["enum4linux"] + flags.split() + [target]
        return self.run(cmd, tool_name="enum4linux", timeout=timeout,
                        save_output=True, output_suffix=".txt")

    # ─────────────────────────────────────────────────────────────────────────
    # curl / HTTP probe
    # ─────────────────────────────────────────────────────────────────────────

    def curl(self, url: str, flags: str = "-sI", timeout: int = 20) -> ToolResult:
        """
        Quick HTTP probe — headers, redirect chain, server banner.

        Example:
            result = runner.curl("http://10.10.10.5", flags="-sI --max-redirs 3")
        """
        self._require("curl")
        cmd = ["curl"] + flags.split() + ["--max-time", "15", url]
        return self.run(cmd, tool_name="curl", timeout=timeout,
                        save_output=False)   # headers are short, no need to save

    # ─────────────────────────────────────────────────────────────────────────
    # WhatWeb (tech fingerprinting)
    # ─────────────────────────────────────────────────────────────────────────

    def whatweb(self, url: str, timeout: int = 30) -> ToolResult:
        """
        Fingerprint web technologies (CMS, frameworks, server versions).

        Example:
            result = runner.whatweb("http://10.10.10.5")
        """
        self._require("whatweb")
        cmd = ["whatweb", "--color=never", "-a", "3", url]
        return self.run(cmd, tool_name="whatweb", timeout=timeout,
                        save_output=True, output_suffix=".txt")

    # ─────────────────────────────────────────────────────────────────────────
    # Shell command on an active session (stub — wired up Day 3 via MSF RPC)
    # ─────────────────────────────────────────────────────────────────────────

    def shell_cmd(self, cmd: str, session_id: int = 1) -> ToolResult:
        """
        Run a command on an active Metasploit session.
        Stub until pymetasploit3 is wired in on Day 3.
        """
        return ToolResult(
            tool="shell_cmd",
            cmd=[cmd],
            ok=False,
            error="shell_cmd not yet implemented — wire up pymetasploit3 on Day 3",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Tool availability check
    # ─────────────────────────────────────────────────────────────────────────

    def check_tools(self) -> dict:
        """
        Return availability of all tools Kira needs.
        Call this at startup to warn the user about missing binaries.

        Returns:
            {"nmap": True, "gobuster": False, ...}
        """
        tools = [
            "nmap", "gobuster", "ffuf", "searchsploit",
            "enum4linux", "curl", "whatweb", "msfconsole",
        ]
        return {t: shutil.which(t) is not None for t in tools}

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _log(self, result: ToolResult) -> None:
        """Append a ToolResult to the JSONL action log (thread-safe)."""
        entry = result.to_log_dict()
        with self._log_lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _require(self, binary: str) -> None:
        """Raise a clear error if a required binary is missing."""
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
    """Filename-safe timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _print_status(msg: str) -> None:
    """Coloured status line. Falls back gracefully if rich isn't installed."""
    try:
        from rich.console import Console  # type: ignore
        Console().print(f"[dim]{msg}[/dim]")
    except ImportError:
        print(msg)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil as _shutil

    print("=== ToolRunner smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        runner = ToolRunner(session_dir=tmp, verbose=True)

        # ── 1. Tool availability ───────────────────────────────────────────
        print("[1] Checking tool availability...")
        avail = runner.check_tools()
        for tool, found in avail.items():
            mark = "OK" if found else "MISSING"
            print(f"    {tool:<16} {mark}")
        print()

        # ── 2. Generic run (echo — always available) ───────────────────────
        print("[2] Generic run: echo")
        r = runner.run(["echo", "kira is alive"], tool_name="echo", save_output=True)
        assert r.ok,           f"echo failed: {r.error}"
        assert "kira" in r.stdout
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
        assert not r.ok
        assert r.error is not None
        print(f"    error: {r.error}")
        print()

        # ── 5. JSONL log ───────────────────────────────────────────────────
        print("[5] Checking actions.jsonl...")
        log_path = tmp + "/actions.jsonl"
        assert Path(log_path).exists(), "actions.jsonl not created"
        entries = [json.loads(l) for l in Path(log_path).read_text().strip().splitlines()]
        print(f"    {len(entries)} entries logged")
        for e in entries:
            print(f"    [{e['timestamp']}] {e['tool']:<12} ok={e['ok']}  {e['summary'][:60]}")
        print()

        # ── 6. Nmap (skip if not installed) ───────────────────────────────
        if _shutil.which("nmap"):
            print("[6] Nmap scan on localhost...")
            r = runner.nmap(target="127.0.0.1", flags="-sV --top-ports 10", timeout=60)
            print(f"    ok={r.ok}  artifact={r.artifact_path}")
            if r.artifact_path and Path(r.artifact_path).exists():
                print(f"    XML size: {Path(r.artifact_path).stat().st_size} bytes")
        else:
            print("[6] nmap not installed — skipping nmap test")
        print()

        # ── 7. Raw dir contents ────────────────────────────────────────────
        print("[7] Raw output files saved:")
        for f in sorted(Path(tmp, "raw").iterdir()):
            print(f"    {f.name}  ({f.stat().st_size} bytes)")

    print("\nAll tests passed.")