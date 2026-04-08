---
title: SOC Triage Environment
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - cybersecurity
---

# SOC Triage Environment (OpenEnv)

## Description & Motivation
The SOC Triage Environment is a production-grade, trajectory-aware Reinforcement Learning simulator designed to evaluate an AI agent's ability to act as a Cybersecurity Security Operations Center (SOC) Analyst. 

Most cybersecurity benchmarks test single-step classification. This environment tests **long-horizon correlation**. Agents must monitor network traffic over time, distinguish between noisy benign traffic and subtle distributed attacks, and learn the critical SOC protocol of investigating ("flagging") ambiguous traffic before taking destructive network actions ("blocking").

## Observation Space
`SocTriageObservation` represents a simulated JSON network log event.
* `timestamp` (float): Unix timestamp of the event.
* `source_ip` (str): The origin IP address.
* `request_payload` (str): The payload or URI of the request.
* `http_status` (int): HTTP response status code.
* `user_agent` (str): User-Agent header value.

## Action Space
`SocTriageAction` supports exactly three strict triage commands:
1. `block_ip`: Deploys a firewall block. Used for confidently malicious activity.
2. `allow_traffic`: Ignores the log. Used for benign activity.
3. `flag_for_review`: Escalates the log. Used for ambiguous activity requiring investigation.

## Tasks & Difficulty
1. **Easy (Brute Force):** Repeated frequent identical failure payloads from a single source IP. High noise, single-event complexity.
2. **Medium (SQL Injection):** Intermittent suspicious SQL syntax payloads mixed with normal traffic. Medium noise, short-sequence complexity.
3. **Hard (Low and Slow):** Coordinated subtle anomalous requests across grouped subnets over an extended duration. High noise, requires trajectory memory. *Agents are actively penalized for blocking ambiguous IPs in this task without flagging them first.*

## Setup & Execution
**1. Install dependencies:**
```bash
pip install openai huggingface_hub
uv sync
