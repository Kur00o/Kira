<div align="center">

```
   _  ___
 | |/ (_)_ _ __ _
  | ' <| | '_/ _` |
  |_|\_\_|_| \__,_|
```

**AUTONOMOUS PENETRATION TESTING AGENT**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Kali%20Linux-557C94?style=flat-square&logo=kalilinux&logoColor=white)](https://kali.org)
[![LLM](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash%20%7C%20Gemma%203-FF6B35?style=flat-square)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*LLM-driven agent that autonomously runs the full pentest lifecycle —*
*recon → enum → vuln scan → exploit → privesc → report*

> ⚠️ **Authorized environments only. Unauthorized use is illegal.**

</div>

---

## What Kira Does

```
Target IP
   │
   ▼
┌─────────┐    ┌──────────┐    ┌────────────┐    ┌─────────┐    ┌──────────────┐    ┌──────────────┐
│  RECON  │───▶│   ENUM   │───▶│ VULN SCAN  │───▶│ EXPLOIT │───▶│ POST EXPLOIT │───▶│    REPORT    │
│         │    │          │    │            │    │         │    │              │    │              │
│ nmap    │    │ curl     │    │searchsploit│    │   MSF   │    │   linpeas    │    │  report.html │
│ 2-stage │    │ whatweb  │    │ CVE lookup │    │ modules │    │   privesc    │    │  report.md   │
│  scan   │    │ gobuster │    │ findings   │    │ shells  │    │    root      │    └──────────────┘
└─────────┘    └──────────┘    └────────────┘    └─────────┘    └──────────────┘
```

Kira runs as a **conversational agent** — start it, chat with it, tell it a target, and it runs autonomously. Each phase is driven by an LLM that decides the next tool based on live session state. The planner enforces tool sequencing so the agent never loops.

---

## How It Works

**RECON — Two-stage nmap:**
1. Fast connect scan on 22 common ports (~5-10s, no root needed)
2. If nothing found → full sweep of all 65535 ports (~60-120s)
3. Version scan (`-sV -sC`) runs only on confirmed open ports

**ENUM — Programmatic sequencer:**
Enforces `curl_probe → whatweb → searchsploit → gobuster_dir` regardless of LLM choice. Prevents the model from looping on already-completed tools.

**VULN SCAN — Searchsploit with fallback:**
Query chain `Apache httpd 2.4.25` → `Apache 2.4.25` → `Apache 2.4` until results are found. Results converted into scored findings with CVSS and severity.

**EXPLOIT — Real module lookup:**
Queries live Metasploit RPC for actual module names before attempting exploitation. Never uses hallucinated module paths.

**REPORT — HTML + Markdown:**
Generated from all findings using a Jinja2 template with LLM-written executive summary and per-finding analysis.

---

## LLM Backends

Kira auto-selects the backend from your `.env`:

| Backend | When used | Model |
|---------|-----------|-------|
| **Gemini API** | `GEMINI_API_KEY` is set | `gemini-2.5-flash` |
| **Ollama (local)** | `GEMINI_API_KEY` is absent | `gemma3:4b` |

To use Gemini: set `GEMINI_API_KEY` in `.env`  
To use Gemma locally: comment out `GEMINI_API_KEY`, set `OLLAMA_MODEL=gemma3:4b`

---

## Requirements

| Tool | Purpose | Install |
|------|---------|---------|
| **Kali Linux** | Recommended OS | — |
| **Python 3.11+** | Runtime | pre-installed on Kali |
| **Nmap** | Port scanning | `sudo apt install nmap` |
| **Gobuster** | Dir brute-force | `sudo apt install gobuster` |
| **Searchsploit** | CVE lookup | `sudo apt install exploitdb` |
| **Metasploit** | Exploitation | `sudo apt install metasploit-framework` |
| **enum4linux** | SMB enum | `sudo apt install enum4linux` |
| **WhatWeb** | Web fingerprint | `sudo apt install whatweb` |
| **Ollama** | Local LLM (optional) | [ollama.com/install](https://ollama.com/install.sh) |

---

## Setup on Kali

### 1 — System tools

```bash
sudo apt update && sudo apt install -y \
    nmap gobuster exploitdb metasploit-framework \
    enum4linux whatweb curl wordlists \
    python3-pip python3-venv
```

### 2 — Clone + Python deps

```bash
git clone <repo-url> kira
cd kira

python3 -m venv venv
source venv/bin/activate

pip install requests rich jinja2 pymetasploit3 python-dotenv
```

### 3 — Configure .env

```bash
cp .env.example .env   # or edit .env directly
```

**Option A — Gemini API (cloud, better reasoning):**
```ini
GEMINI_API_KEY=AIzaSy...          # from https://aistudio.google.com/apikey
GEMINI_MODEL=gemini-2.5-flash
```

**Option B — Ollama/Gemma (local, no API key):**
```ini
# Leave GEMINI_API_KEY unset or commented out
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
```

### 4 — Ollama setup (if using local model)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma3:4b
```

### 5 — Start Metasploit RPC

```bash
msfrpcd -P kirapass -p 55553 -a 127.0.0.1 -f

# Verify
ss -tlnp | grep 55553
```

### 6 — Verify everything

```bash
nmap --version
gobuster version
searchsploit --version
msfconsole --version
curl http://localhost:11434/api/tags   # Ollama only
```

---

## Test Target — DVWA

[DVWA](https://github.com/digininja/DVWA) is the recommended target for testing Kira.

### Docker (fastest)

```bash
sudo systemctl start docker

sudo docker run -d \
    -p 8080:80 \
    --name dvwa \
    vulnerables/web-dvwa

curl -s http://127.0.0.1:8080/ | grep -i dvwa
```

### Manual install on Kali

```bash
sudo apt install -y apache2 php php-mysqli mariadb-server

sudo git clone https://github.com/digininja/DVWA /var/www/html/dvwa
sudo cp /var/www/html/dvwa/config/config.inc.php.dist \
        /var/www/html/dvwa/config/config.inc.php
sudo sed -i "s/p\@ssw0rd//" /var/www/html/dvwa/config/config.inc.php
sudo chmod -R 777 /var/www/html/dvwa/hackable/uploads/
sudo systemctl start apache2 mariadb
```

Visit `http://127.0.0.1/dvwa/setup.php` → **Create / Reset Database**  
Login: `admin` / `password` → set security level to **Low**

---

## Running Kira

```bash
source venv/bin/activate

# Basic — interactive mode, provide target via chat
sudo -E venv/bin/python main.py

# With target pre-set
sudo -E venv/bin/python main.py --target 10.163.172.51 --authorized-by "Lab VM"

# No Metasploit
sudo -E venv/bin/python main.py --target 10.163.172.51 --authorized-by "Lab VM" --no-msf

# More iterations
sudo -E venv/bin/python main.py --target 10.163.172.51 --authorized-by "Lab VM" --max-iter 50

# Override model
sudo -E venv/bin/python main.py --target 10.163.172.51 --authorized-by "Lab VM" --model gemini-2.5-pro
```

### Chat interface

Once running, Kira presents a conversational prompt:

```
kira> find vulnerabilities at 10.163.172.51
```

Kira detects the IP + trigger word and starts the autonomous scan. You can control the iteration count inline:

```
kira> find vulnerabilities at 10.163.172.51 iterations = 50
kira> scan 10.163.172.51 iter=100
kira> start pentest on 10.163.172.51 max_iter=30
```

You can also ask questions at any time:

```
kira> what is CVE-2021-41773?
kira> explain the findings so far
kira> what should I do next?
kira> show report
kira> open report
```

Type `exit` to quit.

### View the report

```bash
xdg-open sessions/*/report.html
```

---

## CLI Reference

```
sudo -E venv/bin/python main.py [OPTIONS]

Target:
  --target IP              Target IP (optional — can be set via chat)
  --authorized-by TEXT     Authorization statement (optional — prompted if omitted)

LLM:
  --api-key KEY            Gemini API key (overrides GEMINI_API_KEY env var)
  --model MODEL            Override model (e.g. gemini-2.5-pro, gemma3:12b)

Scan:
  --max-iter N             Max agent loop iterations (default: 20)
  --session-dir PATH       Custom session directory
  --no-msf                 Disable Metasploit (RECON → VULN_SCAN only)
  --no-report              Skip report generation

Metasploit RPC:
  --msf-host HOST          msfrpcd host (default: 127.0.0.1)
  --msf-port PORT          msfrpcd port (default: 55553)
  --msf-pass PASS          msfrpcd password (default: kirapass)
  --msf-no-ssl             Disable SSL for msfrpcd

Output:
  --verbose / -v           Stream raw tool output
  --version                Print version and exit
```

---

## Session Output

Every run creates a timestamped directory under `sessions/`:

```
sessions/kira_20260415_120000/
├── state.json       ← full agent state (findings, ports, sessions)
├── actions.jsonl    ← every tool call with args, result, timing
├── kira.log         ← phase transitions, errors, events
├── report.md        ← markdown pentest report
├── report.html      ← HTML report (open in browser)
└── raw/
    ├── nmap_*.xml
    ├── gobuster_*.txt
    └── searchsploit_*.json
```

---

## Architecture

```
main.py  ──▶  KiraChat.start()
               │
               ├── chat mode: LLM.generate_text() — free-form Q&A
               │
               └── scan trigger: Planner.run()
                    │
                    ├── LLMClient.next_action()
                    │    ├── Gemini REST API  (if GEMINI_API_KEY set)
                    │    └── Ollama /api/generate  (fallback)
                    │
                    ├── ENUM sequencer  (curl → whatweb → searchsploit → gobuster)
                    │
                    ├── ScopeGuard.check_action()
                    │
                    ├── Planner._dispatch()
                    │    ├── ToolRunner.nmap()        two-stage discovery
                    │    ├── ToolRunner.gobuster()
                    │    ├── ToolRunner.searchsploit() query fallback chain
                    │    ├── ToolRunner.whatweb()
                    │    ├── ToolRunner.curl()
                    │    ├── MSFClient.search()       real module lookup
                    │    ├── MSFClient.run_module()   exploit + session tracking
                    │    └── ToolRunner.shell_cmd()   post-exploit
                    │
                    ├── KnowledgeBase.add()    dedup findings by (title, port)
                    ├── StateManager.update()  persist to state.json
                    └── PhaseController        auto-advance phases
```

---

## Troubleshooting

**nmap permission denied:**
```bash
sudo -E venv/bin/python main.py --target <IP> --authorized-by "Lab VM"
```

**Gemini API key error:**
```bash
# Check key is set
echo $GEMINI_API_KEY
# Or test directly
python -c "from kira.llm import LLMClient; c = LLMClient(); print(c.ping())"
```

**Ollama not reachable:**
```bash
ollama serve
curl http://localhost:11434/api/tags
ollama pull gemma3:4b
```

**Gobuster wordlist missing:**
```bash
sudo apt install wordlists
ls /usr/share/wordlists/dirb/common.txt
```

**Metasploit RPC not connecting:**
```bash
msfrpcd -P kirapass -p 55553 -a 127.0.0.1 -f
ss -tlnp | grep 55553
```

**Report not generating:**
```bash
pip install jinja2
# Check for template errors in the output — they are now printed with traceback
```

**Review why a session stopped:**
```bash
cat sessions/<dir>/actions.jsonl | python3 -m json.tool | less
```

---

## Ethical Use

Kira is built for **authorized security testing only**.

- `--authorized-by` is required — Kira prompts for it if not provided
- A scope guard blocks scanning IPs outside the authorized target
- Destructive shell commands are blocked even on live sessions
- All actions are permanently logged to `actions.jsonl` and `kira.log`

**Only use Kira against:**
- Your own machines and lab VMs
- CTF platforms (HTB, VulnHub, TryHackMe) on machines you are assigned
- Systems you have explicit written permission to test

Unauthorized access to computer systems is illegal regardless of intent.
