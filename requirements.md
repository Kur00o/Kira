# Kira — Requirements

## Python Version
- **Minimum:** Python `3.10+` (code uses PEP 604 union syntax like `str | None` and built-in generics like `list[str]`).
- **Recommended:** Python `3.10` or `3.11` on Linux.

## Python Dependencies
```txt
requests>=2.31.0
rich>=13.0.0
jinja2>=3.1.0
pymetasploit3>=1.0.6
```

Notes:
- `requests` is required by `kira/llm.py` and `kira/reporter.py`.
- `rich` is imported inside `try/except` blocks for terminal UX; Kira falls back to plain printing if unavailable.
- `jinja2` is imported inside `try/except` in `kira/reporter.py`; fallback HTML rendering exists if unavailable.
- `pymetasploit3` is optional unless Metasploit RPC features are enabled.

## System Dependencies
| Tool | Install Command | Purpose |
|------|------------------|---------|
| nmap | `sudo apt install -y nmap` | Recon scanning (`ToolRunner.nmap`) and XML output consumed by parser flow. |
| gobuster | `sudo apt install -y gobuster` | Web directory brute force (`ToolRunner.gobuster`). |
| ffuf | `sudo apt install -y ffuf` | Fallback dir brute-force engine when `gobuster` is missing. |
| searchsploit | `sudo apt install -y exploitdb` | CVE/exploit lookup in vuln scan phase (`searchsploit --json`). |
| enum4linux | `sudo apt install -y enum4linux` | SMB enumeration (`enum4linux -a`). |
| curl | `sudo apt install -y curl` | HTTP probing and headers/banners (`curl -sI`). |
| whatweb | `sudo apt install -y whatweb` | Web tech fingerprinting. |
| mysql (client) | `sudo apt install -y mysql-client` | Optional unauthenticated MySQL check (`mysql -u root -e "SELECT 1;"`). |
| metasploit-framework (`msfrpcd`, `msfconsole`) | `sudo apt install -y metasploit-framework` | Exploitation and post-exploitation via RPC (`msfrpcd`, session commands). |

Version notes from static code:
- No explicit minimum versions are pinned for system binaries.
- Hardcoded default wordlist path is `/usr/share/wordlists/dirb/common.txt` (used by gobuster defaults).

## Ollama Setup
Detected planner model in code: `qwen2.5-coder:14b-instruct-q4_K_M` (from `OLLAMA_MODEL` in `kira/llm.py`).

Compatibility note:
- Code expects Ollama API endpoints `/api/chat` and `/api/tags`.
- No explicit Ollama version is pinned in code; use an Ollama release that supports both endpoints.
- No custom `Modelfile` or model options file is present in the repository.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required model
ollama pull qwen2.5-coder:14b-instruct-q4_K_M
```

## Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_HOST` | No | Ollama base URL. Default in code: `http://localhost:11434`. |
| `ANTHROPIC_API_KEY` | Conditional | Required only when `--provider anthropic` is used. |
| `OPENAI_API_KEY` | Conditional | Required only when `--provider openai` is used. |

Also configurable via CLI flags:
- `--api-key`
- `--ollama-host`
- `--model`
- `--provider`

## Required Directory Structure
```text
kira/
├── main.py
├── requirements.md
├── kira/
│   ├── llm.py
│   ├── planner.py
│   ├── tool_runner.py
│   ├── state.py
│   ├── reporter.py
│   ├── logger.py
│   ├── findings.py
│   ├── msf_client.py
│   ├── privesc.py
│   ├── phase_controller.py
│   ├── cvss.py
│   ├── parsers/
│   └── templates/
│       └── report.html.j2
└── sessions/                      # auto-created at runtime
    └── <target_or_custom_session>/
        ├── state.json
        ├── actions.jsonl
        ├── kira.log
        ├── report.md
        ├── report.html
        └── raw/
```

## Setup Instructions
1. Install Python 3.10+ on Linux.
2. Install system tools:
   - `sudo apt update`
   - `sudo apt install -y nmap gobuster ffuf exploitdb enum4linux curl whatweb mysql-client metasploit-framework`
3. Create and activate a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
4. Install Python packages:
   - `pip install requests rich jinja2 pymetasploit3`
5. Install and prepare Ollama:
   - `curl -fsSL https://ollama.com/install.sh | sh`
   - `ollama pull qwen2.5-coder:14b-instruct-q4_K_M`
6. (Optional) Export API keys for cloud providers:
   - `export ANTHROPIC_API_KEY=...`
   - `export OPENAI_API_KEY=...`
7. Ensure required filesystem paths are available:
   - Session output path is auto-created under `sessions/`
   - Gobuster default wordlist path expected: `/usr/share/wordlists/dirb/common.txt`
8. Run Kira (example):
   - `python main.py --target <ip-or-host> --authorized-by "<authorization-text>"`

## Notes & Warnings
- Kira is designed for Linux; many dependencies are Linux packages/CLI tools.
- Some scans and exploitation workflows may need elevated privileges (`sudo`) depending on target/network and tool configuration.
- Metasploit integration requires `msfrpcd` availability and reachable RPC settings.
- `rich`, `jinja2`, and `pymetasploit3` are optional in parts of the code, but missing them reduces functionality.
- No dependency conflict pins are declared in code; no explicit package conflicts are encoded in this repository.
