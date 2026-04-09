# Kira — Autonomous Penetration Testing Agent

Kira is an LLM-driven agent that autonomously executes the full penetration testing lifecycle — reconnaissance, enumeration, vulnerability discovery, exploitation, privilege escalation, and professional report generation — from a single command.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py  (CLI)                           │
│   args → ScopeGuard → StateManager → LLMClient → Planner.run()  │
└──────────────────────────────┬──────────────────────────────────┘
                               │ observe → think → act loop
        ┌──────────────────────┼──────────────────────────┐
        ▼                      ▼                          ▼
   LLMClient              ToolRunner                 KiraLogger
   (llm.py)               (tool_runner.py)           (logger.py)
   Ollama/Anthropic/       subprocess wrapper         JSONL event log
   OpenAI                  nmap, gobuster,            kira.log
   JSON action output      searchsploit, MSF
        │                      │
        ▼                      ▼
   StateManager          parsers/
   (state.py)             nmap_parser.py
   state.json             gobuster_parser.py
   actions.jsonl          vuln_scanner.py
   thread-safe            service_enum.py
        │
        ▼
   KnowledgeBase          PrivescEngine          ScopeGuard
   (findings.py)          (privesc.py)           (guardrails.py)
   dedup, CVSS            linpeas analysis       target scope check
   severity groups        14 vector detectors    destructive cmd block
        │
        ▼
   ReportGenerator        cvss.py
   (reporter.py)          NVD CVSS v3.1 formula
   Markdown + HTML        estimate_cvss_from_finding()
   LLM-written narrative
   Jinja2 template
```

**Data flow per iteration:**
```
StateManager.get_context_summary()
    → LLMClient.next_action()      # Gemma/Claude/GPT picks a tool
    → ScopeGuard.check_action()    # safety pre-flight
    → Planner._dispatch()          # runs the tool
    → ToolRunner.*()               # subprocess execution
    → parser (nmap/gobuster/etc.)  # structured output
    → StateManager.update()        # persist to state.json
    → KiraLogger.action()          # log to kira.log
    → KnowledgeBase.add()          # deduplicated findings
    → phase gate check             # advance phase if complete
```

---

## Prerequisites

| Requirement       | Version    | Notes                                     |
|-------------------|------------|-------------------------------------------|
| OS                | Kali Linux | Recommended; Ubuntu works with tool setup |
| Python            | 3.11+      | f-strings, `match`, `tomllib`             |
| Ollama            | latest     | For local LLM (Gemma 3 4B)               |
| Nmap              | 7.x+       | `sudo apt install nmap`                   |
| Gobuster          | 3.x+       | `sudo apt install gobuster`               |
| Searchsploit      | latest     | `sudo apt install exploitdb`              |
| Metasploit        | 6.x+       | Optional — `sudo apt install metasploit-framework` |
| rich              | latest     | Terminal UI — `pip install rich`          |
| jinja2            | latest     | HTML reports — `pip install jinja2`       |
| weasyprint        | latest     | PDF export (optional) — `pip install weasyprint` |

---

## Setup

```bash
# 1. Clone / copy kira/ to your Kali machine
cd kira/

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install requests rich jinja2

# Optional — PDF export:
sudo apt install -y libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0
pip install weasyprint

# 4. Pull the LLM model (local Ollama)
ollama pull gemma3:4b

# 5. Verify tools are installed
nmap --version
gobuster version
searchsploit --version
```

---

## LAN Setup (Ollama on a teammate's machine)

Run on your **teammate's machine** (the one with the GPU):

```bash
# Allow remote connections
export OLLAMA_HOST=0.0.0.0
ollama serve

# Ensure the model is pulled
ollama pull gemma3:4b
```

Run Kira on your machine, pointing at teammate's Ollama:

```bash
python main.py \
  --target 10.10.10.5 \
  --authorized-by "Lab VM" \
  --ollama-host http://TEAMMATE_IP:11434
```

---

## Quickstart

```bash
# Basic run — local Ollama, no Metasploit
python main.py --target 10.10.10.5 --authorized-by "Lab VM — authorized" --no-msf

# Expected output:
#  ✓ Session dir : sessions/10_10_10_5_20260409_120000/
#  ✓ [ollama] gemma3:4b — ready
#  ⚠  MSF disabled — exploitation limited
#  ─── AGENT LOOP ───
#  --- iter 1/50  phase=RECON ---
#    THINK  tool=nmap_scan  args={'target': '10.10.10.5', 'flags': '-sV -sC'}
#    RESULT  Found 3 open ports: [22, 80, 445] ...
#  ...
#  ─── GENERATING REPORT ───
#  ✓ Markdown : sessions/.../report.md
#  ✓ HTML     : sessions/.../report.html
```

After the run, open `sessions/<session>/report.html` in a browser.

---

## CLI Reference

```
python main.py [OPTIONS]

Required:
  --target IP              Target IP address (e.g. 10.10.10.5)
  --authorized-by TEXT     Written authorization (e.g. "Lab VM — auth'd by Alice")

LLM:
  --provider PROVIDER      LLM backend: ollama | anthropic | openai (default: ollama)
  --ollama-host URL        Ollama server URL (default: http://localhost:11434)
  --model MODEL            Override model name (e.g. gemma3:12b)
  --api-key KEY            API key for Anthropic or OpenAI

Session:
  --session-dir PATH       Custom session directory (auto-generated if not set)
  --max-iter N             Max agent loop iterations (default: 50)

Flags:
  --no-msf                 Disable Metasploit (safe for enumeration-only runs)
  --no-report              Skip automatic report generation
  --quiet                  Suppress verbose terminal output
```

**Provider switching examples:**

```bash
# Anthropic Claude (cloud)
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --target 10.10.10.5 --authorized-by "Lab" --provider anthropic

# OpenAI GPT-4o-mini (cloud)
export OPENAI_API_KEY=sk-...
python main.py --target 10.10.10.5 --authorized-by "Lab" --provider openai

# Local Gemma on LAN
python main.py --target 10.10.10.5 --authorized-by "Lab" \
               --ollama-host http://192.168.1.42:11434
```

---

## Module Map

```
kira/
├── main.py              CLI entry point — wires all modules, runs the session
├── state.py             StateManager — single source of truth, thread-safe JSON store
├── llm.py               LLMClient — Ollama/Anthropic/OpenAI, JSON action parsing,
│                                    few-shot prompting, per-phase temperature
├── planner.py           Planner — observe→think→act loop, phase controller, dispatch
├── tool_runner.py       ToolRunner — all subprocess calls, timeout, JSONL action log
├── findings.py          Finding + KnowledgeBase — dedup, CVSS sort, state integration
├── logger.py            KiraLogger — typed JSONL event log (kira.log)
├── cvss.py              CVSS v3.1 formula + severity labels + heuristic estimator
├── privesc.py           PrivescEngine — linpeas parser, 14 escalation vector detectors
├── reporter.py          ReportGenerator — reads session artefacts, LLM narrative,
│                                          outputs report.md + report.html (+ PDF)
├── guardrails.py        ScopeGuard — target scope enforcement, destructive cmd blocking
├── msf_client.py        MSFClient — Metasploit RPC wrapper, auto-start msfrpcd
├── parsers/
│   ├── nmap_parser.py   Parse nmap XML → structured Host/Service/NmapResult
│   ├── gobuster_parser.py  Parse gobuster stdout → paths, juicy paths, auto-findings
│   ├── service_enum.py  Service enumeration helpers (FTP anon, SMB null session)
│   └── vuln_scanner.py  searchsploit JSON parser → Finding objects with CVE + CVSS
└── templates/
    └── report.html.j2   Dark-theme Jinja2 HTML report template
```

**Session directory contents after a run:**
```
sessions/10_10_10_5_20260409_120000/
├── state.json        Full agent state — all findings, ports, sessions
├── actions.jsonl     Raw tool execution log (timestamp, cmd, result, elapsed)
├── kira.log          Typed agent event log (phase transitions, findings, errors)
├── report.md         Markdown pentest report
├── report.html       HTML pentest report (open in browser)
├── report.pdf        PDF report (if weasyprint installed)
└── raw/              Raw tool output files
    ├── nmap_*.xml
    ├── gobuster_*.txt
    └── ...
```

---

## Build Log

| Phase | What was built |
|-----|---------------|
| 1   | `state.py` (StateManager), `tool_runner.py` (ToolRunner), `llm.py` (LLMClient + Ollama), basic nmap integration |
| 2   | `findings.py` (Finding + KnowledgeBase), `parsers/nmap_parser.py`, `parsers/gobuster_parser.py`, `parsers/service_enum.py` |
| 3   | `msf_client.py`, `planner.py` (full agent loop + phase controller + all dispatch handlers), `parsers/vuln_scanner.py` |
| 4   | `privesc.py` (PrivescEngine), `logger.py` (KiraLogger), `cvss.py` (CVSS v3.1), `reporter.py` (ReportGenerator), `templates/report.html.j2` |
| 5   | `guardrails.py` (ScopeGuard), full orchestration in `main.py`, planner wired with logger+guard, few-shot prompts + DONT_DO constraints in `llm.py`, `README.md` |

---

## Ethical Use Statement

**Kira is designed exclusively for authorized security testing.**

- You **must** provide `--authorized-by` with a written authorization statement before Kira will run.
- Kira enforces a scope guard that prevents scanning targets outside the authorized IP range.
- Kira blocks destructive commands (`rm -rf`, `dd if=`, `mkfs`, etc.) even when operating on a live session.
- The authorization statement is permanently recorded in `kira.log` at session start.

**Authorized environments only:**
- CTF machines (HTB, VulnHub, TryHackMe — retired/intended-for-testing only)
- Your own VMs and lab networks
- Systems you own or have explicit written permission to test

**Never point Kira at systems you do not own or have explicit written authorization to test.** Unauthorized access to computer systems is illegal in most jurisdictions regardless of intent.

---

## Recommended CTF Test Targets

| Target            | Setup            | What it tests |
|-------------------|------------------|---------------|
| Metasploitable 2  | Local VirtualBox | FTP, SMB, MySQL, Apache, full lifecycle |
| HTB: Lame         | HTB VPN          | SMB CVE-2007-2447, MSF integration |
| VulnHub: Basic 1  | Local VirtualBox | Web + privesc, RECON→ROOT flow |

Avoid Active Directory targets — multi-hop pivoting is not yet implemented.

---

## Troubleshooting

**Ollama not reachable:**
```bash
ollama serve                      # start Ollama
ollama pull gemma3:4b             # pull the model
curl http://localhost:11434/api/tags   # verify API is up
```

**Gobuster wordlist not found:**
```bash
sudo apt install wordlists
ls /usr/share/wordlists/dirb/    # verify common.txt exists
```

**Metasploit RPC won't start:**
```bash
sudo apt install metasploit-framework
which msfrpcd                    # must be on PATH
```

**Report HTML looks broken:**
```bash
pip install jinja2               # required for templated HTML
```
