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
[![LLM](https://img.shields.io/badge/LLM-Gemini%20%7C%20Ollama%20%7C%20Claude%20%7C%20GPT-FF6B35?style=flat-square)](https://ollama.com)
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
┌─────────┐   ┌──────────┐   ┌────────────┐   ┌─────────┐   ┌──────────────┐   ┌──────────┐
│  RECON  │──▶│   ENUM   │──▶│ VULN SCAN  │──▶│ EXPLOIT │──▶│ POST EXPLOIT │──▶│  REPORT  │
│         │   │          │   │            │   │         │   │              │   │          │
│ nmap    │   │ gobuster │   │searchsploit│   │   MSF   │   │   linpeas    │   │ HTML+MD  │
│ 65535   │   │ whatweb  │   │ CVE lookup │   │ modules │   │   privesc    │   │ report   │
│  ports  │   │ curl     │   │ findings   │   │ shells  │   │    root      │   │          │
└─────────┘   └──────────┘   └────────────┘   └─────────┘   └──────────────┘   └──────────┘
```

The planner enforces tool sequencing at each phase so the agent never loops or
gets stuck — even with a small local model like gemma3:4b.

---

## Requirements

| Tool | Purpose | Install |
|------|---------|---------|
| Kali Linux | Recommended OS | — |
| Python 3.11+ | Runtime | pre-installed on Kali |
| Nmap | Port scanning | `sudo apt install nmap` |
| Gobuster | Dir brute-force | `sudo apt install gobuster` |
| Searchsploit | CVE lookup | `sudo apt install exploitdb` |
| Metasploit | Exploitation | `sudo apt install metasploit-framework` |
| enum4linux | SMB enum | `sudo apt install enum4linux` |
| whatweb | Web fingerprint | `sudo apt install whatweb` |
| Ollama | Local LLM | [ollama.com](https://ollama.com/install.sh) |

---

## Setup

### 1 — Install system tools

```bash
sudo apt update && sudo apt install -y \
    nmap gobuster exploitdb metasploit-framework \
    enum4linux whatweb curl wordlists \
    python3-pip python3-venv docker.io
```

### 2 — Clone and install Python dependencies

```bash
git clone <repo-url> kira
cd kira

python3 -m venv venv
source venv/bin/activate

pip install requests rich jinja2 pymetasploit3 python-dotenv
```

### 3 — Configure your API key

Create a `.env` file in the project root (never committed to git):

```bash
cp .env .env.local   # or just create it manually
```

```env
# .env
GEMINI_API_KEY=AIza...
```

Get a free Gemini key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

Alternatively export it in your shell (no `.env` file needed):

```bash
echo 'export GEMINI_API_KEY=AIza...' >> ~/.bashrc
source ~/.bashrc
```

### 4 — Set up Ollama (local model, optional)

Only needed if you want to run without a cloud API key.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma3:4b
```

### 5 — Start Metasploit RPC

Required for the EXPLOIT phase. Skip with `--no-msf` if you only want recon/enum.

```bash
msfrpcd -P kirapass -p 55553 -a 127.0.0.1 -f

# Verify it's listening
ss -tlnp | grep 55553
```

### 6 — Verify

```bash
nmap --version
gobuster version
searchsploit --version
msfconsole --version
```

---

## Test Target — DVWA

[DVWA](https://github.com/digininja/DVWA) (Damn Vulnerable Web Application) is
the target used for development and testing of Kira. It runs Apache + PHP with
intentional vulnerabilities and is safe to attack in a lab environment.

### Spin up DVWA with Docker

```bash
sudo systemctl start docker

sudo docker run -d \
    -p 8080:80 \
    --name dvwa \
    vulnerables/web-dvwa

# Confirm it's up
curl -s http://127.0.0.1:8080/ | grep -i dvwa
```

### Configure DVWA

1. Open `http://127.0.0.1:8080/` in a browser
2. Login with `admin` / `password`
3. Go to **Setup / Reset DB** → click **Create / Reset Database**
4. Go to **DVWA Security** → set level to **Low**

---

## Running Kira

Activate the venv first:

```bash
source venv/bin/activate
```

**Against local DVWA (Gemini — recommended):**
```bash
python main.py \
  --target 127.0.0.1 \
  --authorized-by "Local DVWA lab" \
  --provider gemini
```

**Against a DVWA VM on your LAN:**
```bash
python main.py \
  --target 10.163.172.51 \
  --authorized-by "Lab VM" \
  --provider gemini
```

**With Ollama on a separate GPU machine on your LAN:**
```bash
python main.py \
  --target 10.163.172.51 \
  --authorized-by "Lab VM" \
  --ollama-host http://10.163.172.253:11434
```

**Enumeration only — no Metasploit:**
```bash
python main.py \
  --target 10.163.172.51 \
  --authorized-by "Lab VM" \
  --provider gemini \
  --no-msf
```

**More iterations + verbose tool output:**
```bash
python main.py \
  --target 10.163.172.51 \
  --authorized-by "Lab VM" \
  --provider gemini \
  --max-iter 30 \
  --verbose
```

**Open the report when done:**
```bash
xdg-open sessions/10_163_172_51_*/report.html
```

---

## LLM Providers

| Provider | Flag | Key env var | Notes |
|----------|------|-------------|-------|
| **Gemini** | `--provider gemini` | `GEMINI_API_KEY` | Recommended — free tier, large context |
| Ollama (local) | *(default)* | — | No internet needed, needs GPU |
| Anthropic Claude | `--provider anthropic` | `ANTHROPIC_API_KEY` | Best reasoning, paid |
| OpenAI | `--provider openai` | `OPENAI_API_KEY` | Paid |

Recommended models by provider:
- Gemini: `gemini-2.0-flash` (default) or `gemini-2.5-pro`
- Ollama: `gemma3:12b` or `qwen2.5-coder:14b` (4b works but loops more)
- Claude: `claude-haiku-4-5-20251001`
- OpenAI: `gpt-4o-mini`

Override the model with `--model <name>`.

---

## CLI Reference

```
python main.py --target IP --authorized-by "TEXT" [OPTIONS]

  --target IP              Target IP address
  --authorized-by TEXT     Written authorization statement (required)

  --provider PROVIDER      gemini | ollama | anthropic | openai  (default: ollama)
  --ollama-host URL        Ollama URL  (default: http://localhost:11434)
  --model MODEL            Override model name
  --api-key KEY            API key (or set via .env / env var)

  --max-iter N             Max agent loop iterations  (default: 20)
  --session-dir PATH       Custom session directory
  --no-msf                 Disable Metasploit
  --no-report              Skip report generation
  --verbose / -v           Stream tool output to terminal
```

---

## Session Output

Every run saves a timestamped session directory:

```
sessions/10_163_172_51_20260410_120000/
├── state.json       full agent state — findings, ports, sessions
├── actions.jsonl    every tool call with args, result, and timing
├── kira.log         phase transitions, errors, events
├── report.md        markdown pentest report
├── report.html      HTML report — open in browser
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
               ├── LLMClient.next_action()     gemini / ollama / anthropic / openai
               │    └── SYSTEM_PROMPT + state context → JSON action
               │
               ├── ScopeGuard.check_action()   blocks out-of-scope targets
               │
               ├── ENUM sequencer              enforces tool order regardless of LLM
               │
               ├── Planner._dispatch()
               │    ├── ToolRunner.nmap()       two-stage: full sweep + version scan
               │    ├── ToolRunner.gobuster()
               │    ├── ToolRunner.searchsploit()
               │    ├── ToolRunner.whatweb() / curl()
               │    ├── MSFClient.search()      validates module names via live RPC
               │    ├── MSFClient.run_module()  exploit + session tracking
               │    └── ToolRunner.shell_cmd()  post-exploit commands
               │
               ├── KnowledgeBase.add()          dedup findings by (title, port)
               ├── StateManager.update()        persist to state.json
               └── PhaseController             auto-advance phases
```

---

## Troubleshooting

**nmap SYN scan needs root:**
```bash
sudo python main.py --target 10.163.172.51 --authorized-by "Lab VM" --provider gemini
```

**Gemini key not found:**
```bash
# Check your .env has the key
cat .env | grep GEMINI
# Or export directly
export GEMINI_API_KEY=AIza...
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
sudo docker ps -a
sudo docker logs dvwa
```

---

## Ethical Use

Kira is built for **authorized security testing only**.

- `--authorized-by` is required — Kira will not start without it
- A scope guard prevents scanning IPs outside the authorized target
- Destructive shell commands are blocked even on live sessions
- All actions are permanently logged to `kira.log`

Only use Kira against systems you own, lab VMs, or CTF machines you are
explicitly assigned to test. Unauthorized access to computer systems is
illegal regardless of intent.
