# bankbot/clarifier/clarifier.py
from typing import Dict, List
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
qg = pipeline("text2text-generation", model="t5-small", device=0)


# ----- load LM once --------------------------------------------------------
# _QG_DIR = Path("models/question_gen")           # put the 7-B model here
# tok_qg  = AutoTokenizer.from_pretrained(_QG_DIR)
# qg      = AutoModelForSeq2SeqLM.from_pretrained(_QG_DIR, device_map="auto")
_QG_DIR = Path("bankbot/models/question_gen")
if _QG_DIR.exists() and any(_QG_DIR.iterdir()):
    tok_qg  = AutoTokenizer.from_pretrained(_QG_DIR)
    qg      = AutoModelForSeq2SeqLM.from_pretrained(_QG_DIR, device_map="auto")
else:
    tok_qg = qg = None          # ← disables LM questions

INTENT_CHOICES = [
    "balance_check", "emi_issue", "report_fraud",
    "card_upgrade", "kyc_pending", "other"
]

# ---------------------------------------------------------------------------
# def _lm_question(prompt: str) -> str:
#     inputs = tok_qg(prompt, return_tensors="pt").to("mps")
#     out    = qg.generate(**inputs, max_new_tokens=32)
#     return tok_qg.decode(out[0], skip_special_tokens=True).strip()
def _lm_question(prompt: str) -> str:
    if qg is None:
        return "Could you give me a bit more detail?"
    inputs = tok_qg(prompt, return_tensors="pt").to("mps")
    out    = qg.generate(**inputs, max_new_tokens=32)
    return tok_qg.decode(out[0], skip_special_tokens=True).strip()


def _need_menu(conf: float) -> bool:
    return conf < 0.3

def _need_lm(conf: float, missing_entities: bool) -> bool:
    return conf < 0.6 or missing_entities

def build_clarifier(parse: Dict) -> Dict:
    """
    Returns {"question": str, "buttons": List[str] | None}
    """
    conf   = parse["confidence"]
    found  = {typ for typ, _ in parse["entities"]}
    missing_amount = "amount" not in found

    if _need_menu(conf):
        return {
            "question": ("I'm not sure what this is about. "
                         "Please reply with one option."),
            "buttons": INTENT_CHOICES
        }

    if _need_lm(conf, missing_amount):
        prompt = (f"User message: {parse['raw_text']}\n"
                  f"Bot intent: {parse['intent']} (conf {conf:.2f}).\n"
                  f"Missing amount: {missing_amount}\n"
                  "Ask ONE short clarifying question:")
        return {"question": _lm_question(prompt), "buttons": None}

    return {"question": "", "buttons": None}

def apply_reply(user_text: str, last_parse: Dict) -> Dict:
    choice = user_text.strip().lower()
    if choice in INTENT_CHOICES:
        last_parse["intent"] = "report_fraud" if choice == "other" else choice
        last_parse["confidence"] = 1.0
    return last_parse
