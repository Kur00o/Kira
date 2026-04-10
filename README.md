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
[![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20Claude%20%7C%20GPT-FF6B35?style=flat-square)](https://ollama.com)
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
┌─────────┐    ┌──────────┐    ┌───────────┐    ┌─────────┐    ┌──────────────┐
│  RECON  │───▶│   ENUM   │───▶│ VULN SCAN │───▶│ EXPLOIT │───▶│ POST EXPLOIT │
│         │    │          │    │           │    │         │    │              │
│ nmap    │    │ gobuster │    │searchsploit│   │   MSF   │    │   linpeas    │
│ 65535   │    │ whatweb  │    │ CVE lookup │    │ modules │    │   privesc    │
│  ports  │    │ curl     │    │ findings  │    │ shells  │    │    root      │
└─────────┘    └──────────┘    └───────────┘    └─────────┘    └──────┬───────┘
                                                                       │
                                                                       ▼
                                                               ┌──────────────┐
                                                               │    REPORT    │
                                                               │              │
                                                               │  report.html │
                                                               │  report.md   │
                                                               └──────────────┘
```

Each phase is driven by an LLM (local Gemma or cloud Claude/GPT) that decides
the next tool to run based on live session state. The planner enforces tool
sequencing so the agent never gets stuck.

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
| **whatweb** | Web fingerprint | `sudo apt install whatweb` |
| **Ollama** | Local LLM | [ollama.com/install](https://ollama.com/install.sh) |

---

## Setup

### Step 1 — System tools

```bash
sudo apt update && sudo apt install -y \
    nmap gobuster exploitdb metasploit-framework \
    enum4linux whatweb curl wordlists \
    python3-pip python3-venv docker.io
```

### Step 2 — Ollama + model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma3:4b
```

> 💡 Better results with a larger model if you have the VRAM:
> `ollama pull gemma3:12b` or `ollama pull qwen2.5-coder:14b`

### Step 3 — Clone + Python deps

```bash
git clone <repo-url> kira && cd kira

python3 -m venv venv && source venv/bin/activate

pip install requests rich jinja2 pymetasploit3
```

### Step 4 — Start Metasploit RPC

```bash
msfrpcd -P kirapass -p 55553 -a 127.0.0.1 -f

# Verify
ss -tlnp | grep 55553
```

### Step 5 — Verify everything

```bash
nmap --version && gobuster version && searchsploit --version
msfconsole --version && ollama list
```

---

## Test Target — DVWA Setup

[DVWA](https://github.com/digininja/DVWA) (Damn Vulnerable Web Application) is
the recommended target for testing Kira. It's intentionally vulnerable and safe
to attack in a lab.

### Docker (recommended — 30 seconds)

```bash
sudo systemctl start docker

sudo docker run -d \
    -p 8080:80 \
    --name dvwa \
    vulnerables/web-dvwa

# Confirm it's running
curl -s http://127.0.0.1:8080/ | grep -i dvwa
```

### Manual install on Kali

```bash
sudo apt install -y apache2 php php-mysqli mariadb-server

sudo git clone https://github.com/digininja/DVWA /var/www/html/dvwa

sudo cp /var/www/html/dvwa/config/config.inc.php.dist \
        /var/www/html/dvwa/config/config.inc.php

# Blank out the DB password
sudo sed -i "s/p\@ssw0rd//" /var/www/html/dvwa/config/config.inc.php

sudo chmod -R 777 /var/www/html/dvwa/hackable/uploads/
sudo systemctl start apache2 mariadb
```

Then visit `http://127.0.0.1/dvwa/setup.php` → click **Create / Reset Database**.

### Configure DVWA for testing

1. Go to `http://127.0.0.1:8080/`
2. Login: `admin` / `password`
3. Navigate to **DVWA Security** → set level to **Low**
4. Note the IP address of your DVWA machine

---

## Running Kira

```bash
source venv/bin/activate
```

**Local DVWA (Docker on same machine):**
```bash
python main.py \
  --target 127.0.0.1 \
  --authorized-by "Local DVWA lab"
```

**DVWA on a LAN VM:**
```bash
python main.py \
  --target 10.163.172.51 \
  --authorized-by "Lab VM"
```

**LAN VM + Ollama on a separate GPU machine:**
```bash
python main.py \
  --target 10.163.172.51 \
  --authorized-by "Lab VM" \
  --ollama-host http://10.163.172.253:11434
```

**Enumeration only (no Metasploit):**
```bash
python main.py \
  --target 10.163.172.51 \
  --authorized-by "Lab VM" \
  --no-msf
```

**More iterations + verbose output:**
```bash
python main.py \
  --target 10.163.172.51 \
  --authorized-by "Lab VM" \
  --max-iter 30 \
  --verbose
```

**Cloud model (better reasoning):**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python main.py \
  --target 10.163.172.51 \
  --authorized-by "Lab VM" \
  --provider anthropic
```

**View the report after the run:**
```bash
xdg-open sessions/10_163_172_51_*/report.html
```

---

## CLI Reference

```
python main.py --target IP --authorized-by "TEXT" [OPTIONS]

  --target IP              Target IP address
  --authorized-by TEXT     Written authorization statement (required)

  --provider PROVIDER      ollama | anthropic | openai  (default: ollama)
  --ollama-host URL        Ollama URL  (default: http://localhost:11434)
  --model MODEL            Override model  (e.g. gemma3:12b)
  --api-key KEY            API key for Anthropic / OpenAI

  --max-iter N             Max loop iterations  (default: 20)
  --session-dir PATH       Custom session directory
  --no-msf                 Disable Metasploit
  --no-report              Skip report generation
  --verbose / -v           Stream tool output
```

---

## Session Output

Every run creates a timestamped session directory:

```
sessions/10_163_172_51_20260410_120000/
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
main.py  ──▶  Planner.run()
               │
               ├── LLMClient.next_action()     ollama / anthropic / openai
               │    └── SYSTEM_PROMPT + state context → JSON action
               │
               ├── ScopeGuard.check_action()   blocks out-of-scope targets
               │
               ├── Planner._dispatch()
               │    ├── ToolRunner.nmap()       two-stage: sweep + version scan
               │    ├── ToolRunner.gobuster()
               │    ├── ToolRunner.searchsploit()
               │    ├── ToolRunner.whatweb()
               │    ├── ToolRunner.curl()
               │    ├── MSFClient.run_module()  exploit + session tracking
               │    └── ToolRunner.shell_cmd()  post-exploit commands
               │
               ├── KnowledgeBase.add()          dedup findings by (title, port)
               ├── StateManager.update()        persist to state.json
               ├── KiraLogger.action()          append to actions.jsonl
               └── PhaseController             auto-advance phases
```

---

## Troubleshooting

**nmap SYN scan needs root:**
```bash
sudo python main.py --target 10.163.172.51 --authorized-by "Lab VM"
```

**Ollama not reachable:**
```bash
ollama serve
curl http://localhost:11434/api/tags
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

**DVWA Docker not starting:**
```bash
sudo systemctl start docker
sudo docker logs dvwa
```

---

## Ethical Use

Kira is built for **authorized security testing only**.

- `--authorized-by` is required — Kira won't start without it
- A scope guard prevents scanning IPs outside the authorized target
- Destructive shell commands are blocked even on live sessions
- All actions are permanently logged to `kira.log`

**Only use Kira against:**
- Your own machines and lab VMs
- CTF platforms (HTB, VulnHub, TryHackMe) on machines you're assigned
- Systems you have explicit written permission to test

Unauthorized access to computer systems is illegal regardless of intent.
