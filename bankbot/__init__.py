import os, torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

import re, json
from fastlid import fastlid                          # language detector
from indic_transliteration import sanscript, detect  # Hinglish→Hindi
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastlid import fastlid
from .entity.extractor import extract_entities   # stub extractor is fine

tok   = AutoTokenizer.from_pretrained("bankbot_intent")
model = AutoModelForSequenceClassification.from_pretrained("bankbot_intent").to(device) #send to GPU
label = model.config.id2label                        # id → name

# -------------------------------------------------------

# rule snippets to pull values out of free text
PATTERNS = {
    "account":  re.compile(r"\b\d{9,16}\b"),
    # "amount":   re.compile(r"[₹$]?\s?\d{3,}[0-9,]*(?:\.\d{1,2})?"),
    "amount":   re.compile(r"[₹$]\s?\d{3,}[0-9,]*(?:\.\d{1,2})?"),
    "date":     re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
    "merchant": re.compile(r"\b(amazon|flipkart|swiggy|zomato|myntra)\b", re.I),
}
ACTIONS = json.load(open("action_map.json"))  # your workflow map

def _prep(text: str) -> str:
    # if user mixes Hindi words in Latin letters, flip to Devanagari
    if detect.detect(text) == sanscript.ITRANS:
        text = sanscript.transliterate(text,
                                        sanscript.ITRANS,
                                        sanscript.DEVANAGARI)
    return text.lower().strip()

def _extract(text: str):
    found = []
    for name, pat in PATTERNS.items():
        found += [(name, m.group()) for m in pat.finditer(text)]
    return found

@torch.inference_mode()
def parse(message: str) -> dict:
    inputs = tok(message, return_tensors="pt",
                 truncation=True, max_length=128).to(device)
    probs  = torch.softmax(model(**inputs).logits, dim=-1)[0]
    clean = _prep(message)
    # -------- intent ------------
    tokens = tok(clean, return_tensors="pt",
                 truncation=True, max_length=128).to(device) # Move the input tensor to the same device (GPU or CPU)
    probs  = torch.softmax(model(**tokens).logits, dim=-1)[0]
    intent_id = int(probs.argmax())
    intent = label[intent_id] # use what the model predicts
    conf      = float(probs[intent_id])

    # make the full vector available to the drift monitor
    probs_list = probs.tolist()          # ← convert to plain Python list
# ----------------------------------------------------------------

    # -------- entities ----------
    ents = _extract(clean)

    # -------- urgency -----------
    urgency = ("critical" if intent == "report_fraud"
               else "medium"   if intent == "emi_issue"
               else "low")

    # -------- action ------------
    action = ACTIONS.get(f"{intent}_{urgency}", "manual_review")

    # -------- clarification -----
    need_clarify = conf < 0.4 or (intent == "emi_issue" and not ents)
    clarify = ("Could you confirm the EMI month?" if need_clarify else None)

    return dict(raw_text=message,
                intent=intent,
                confidence=round(conf, 2),
                entities=ents,
                urgency=urgency,
                action=action,
                probs=probs_list,
                clarify=clarify)