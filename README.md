# BankBot: Intelligent Banking Support Chatbot

## Overview

**BankBot** is a prototype chatbot framework designed to process free-form, multi-lingual user messages (WhatsApp, SMS, chat UI) and automatically perform:

1. **Intent Recognition**: Identifies user intent (e.g., `balance_check`, `emi_issue`, `report_fraud`, `card_upgrade`, `kyc_pending`, etc.)
2. **Entity Extraction**: Pulls out key data points (account number, amount, date, merchant) via an on‑device DistilBERT NER tagger.
3. **Urgency Classification**: Categorizes requests into `critical`, `medium`, or `low` urgency.
4. **Action Suggestion**: Maps `(intent + urgency)` to discrete actions (e.g., `connect_to_fraud_desk`, `send_emi_schedule_pdf`, etc.) via a finite JSON lookup (`action_map.json`).
5. **Clarification Generation**: When confidence is low or data is missing, asks the user to clarify—either via fixed prompts, a menu of intents, or an LM‑powered follow‑up using a small T5 model.
6. **Active Learning Loop**: Collects user‑corrected spans and intent confirmations nightly to retrain/update models, demonstrating self‑improvement.

> This repo is intended as a demo for a production‑ready banking support pipeline, showcasing how to wire together lightweight transformer models, rule‑based extractors, and active learning in pure Python.

SpeedRun: The goal is to get this as close to production as possible in under a week. Fun.

---

## Table of Contents

* [Getting Started](#getting-started)
* [Folder Structure](#folder-structure)
* [Core Components](#core-components)
* [Model Architecture & Choices](#model-architecture--choices)
* [Usage & Smoke Tests](#usage--smoke-tests)
* [Training & Retraining](#training--retraining)
* [Self‑Improving Loop](#self-improving-loop)
* [Docker & Deployment](#docker--deployment)
* [Further Improvements](#further-improvements)
* [License & Acknowledgements](#license--acknowledgements)

---

## Getting Started

1. **Clone** the repository:

   ```bash
   git clone https://github.com/<your-user>/bankbot.git
   cd bankbot
   ```

2. **Create & activate** a Python venv:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download models** (scripts/prepare\_models.sh):

   * DistilBERT NER tagger (multilingual)
   * T5 small for clarifier

5. **Start the FastAPI server**:

   ```bash
   uvicorn serve:app --reload
   ```

Your API is now live at `http://127.0.0.1:8000`.

---

## Folder Structure

```
bankbot/             # Python package
├── clarifier/       # LM‑powered clarification logic (T5)
├── entity/          # ONNX tagger + labels (DistilBERT NER)
├── monitor/         # Drift logging & entropy tracking (Redis stub)
├── models/          # Downloaded transformer snapshots
└── __init__.py      # Glues together parser + pipeline

scripts/             # Training & maintenance utilities
├── train_intent.py  # Fine‑tune intent classifier (bankbot_intent)
├── train_tagger.py  # Train/export ONNX NER tagger
└── merge_labels.py  # Consolidate new span corrections

data/                # Training data & user‑corrected spans
└── span_training/
    └── spans.jsonl  # Manual + bootstrapped entity spans

action_map.json      # Finite mapping from (intent_urgency) → action
serve.py             # FastAPI endpoints
train_intent.py      # CLI entrypoint for intent tuning
README.md            # (this file)
```

---

## Core Components

### 1. Intent Recognizer (`bankbot_intent`)

* A lightweight transformer classifier fine‑tuned on synthetic banking bot conversation data across three languages (Hindi, English, Hinglish).
* Outputs a softmax over \~11 intents plus a confidence score.
* Internally uses `INTENT_MAP` to alias/filter model outputs before mapping to actions.

### 2. Entity Extractor (`bankbot/entity/extractor.py`)

* **ONNX** runtime DistilBERT token‑classification model:

  * Trained on 5000 human‑annotated spans (account, amount, date, merchant).
  * Exported to `/bankbot/entity/tagger.onnx` and labels in `tagger.labels`.
  * Fast inference via `onnxruntime` and maps spans → `[ (type, text) ]`.

### 3. Urgency & Action Mapping

* Heuristic urgency rules (e.g., fraud → `critical`, EMI issues → `medium`).
* Finite `action_map.json` translates `(intent_urgency)` to downstream actions.

### 4. Clarifier (`bankbot/clarifier/clarifier.py`)

* **Very low confidence**: shows static menu of all intents + “other.”
* **Missing amount**: asks a fixed prompt “Could you please tell me the transaction amount?”
* **Moderate confidence**: uses a CPU‑based T5‑small LM (via 🤗 `pipeline`) to generate a single follow-up question.
* **High confidence**: no clarification.

---

## Model Architecture & Choices

| Component             | Model                                    | Host                 | Training Data                    |
| --------------------- | ---------------------------------------- | -------------------- | -------------------------------- |
| **Intent Classifier** | i4bharat/indic-bert (custom fine‑tune)   | MacBook GPU          | 10 000 synthetic banking samples |
| **Entity Tagger**     | DistilBERT‑base‑multilingual‑cased       | ONNX/CPUMacBook GPU  | 5 000 human + bootstrapped spans |
| **Clarifier**         | T5‑small (text2text‑generation)          | CPU                  | Zero‑shot prompt engineering     |

We chose **DistilBERT** for NER to balance size & speed. **T5‑small** (≈60 M parameters) runs comfortably on CPU for a single question.

---

## Usage & Smoke Tests

Use any REST client or `curl`. For example:

```bash
curl -s -X POST http://127.0.0.1:8000/message \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","text":"₹500 debited on 01/10/2025 at Zomato"}' | jq
```

Look for:

* `intent`
* `confidence`
* `entities`
* `urgency`
* `action`
* `clarify` (or `buttons`)

Refer to the **Smoke Test** section in `README.md` for a full set of examples.

---

## Training & Retraining

1. **Intent**: `python train_intent.py --num_train_epochs 3`
2. **Tagger**: `python scripts/train_tagger.py` → produces `bankbot/entity/tagger.onnx` + `tagger.labels`.
3. **Nightly Loop** (cron/GitHub Actions):

   * Detect new entries in `data/span_training/spans.jsonl` or user‑confirmed intents.
   * Re-run training scripts.
   * Deploy updated ONNX & intent snapshots automatically.

A sample GitHub Actions workflow is provided in `.github/workflows/retrain.yml` (optional).

---

## Self‑Improving Loop

Demonstrate by editing/adding one JSON record in `data/span_training/spans.jsonl`, then:

```bash
python scripts/train_tagger.py
# restart serve
# re-run curl → new entity span appears
```

---

## Docker & Deployment (Optional)

For a containerized demo:

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["uvicorn", "serve:app", "--host", "0.0.0.0"]
```

Build & run:

```bash
docker build -t bankbot-demo .
docker run -p 8000:8000 bankbot-demo
```

---

## Further Improvements

* Replace rule‑based urgency with ML classifier.
* Expand NER to free‑form via a sequence‑to‑sequence extractor.
* Implement language detection (Hindi/Hinglish) and normalization pipeline.
* Add interactive UI (buttons + text) in a web chat or Slack integration.

---

## License & Acknowledgements

This project is provided under the MIT License.Built using FastAPI, Transformers, and ONNX Runtime.

This project is provided under the MIT License.
Built using [FastAPI](https://fastapi.tiangolo.com/), [Transformers](https://github.com/huggingface/transformers), and [ONNX Runtime](https://onnxruntime.ai/).

---
