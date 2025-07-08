# def extract_entities(text: str, regex_hits: list):
#     """Placeholder – just return the regex list unchanged."""
#     return regex_hits
# bankbot/entity/extractor.py
import re, json, pathlib, numpy as np, onnxruntime as ort
# from bankbot import PATTERNS            # reuse your regex dict
import re
PATTERNS = {
  "account":  re.compile(r"\b\d{9,16}\b"),
  "amount":   re.compile(r"[₹$]\s?\d{3,}[0-9,]*(?:\.\d{1,2})?"),
  "date":     re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
  "merchant": re.compile(r"\b(amazon|flipkart|swiggy|zomato|myntra)\b", re.I),
}

# --- load ONNX ------------------------------------------------
T_DIR  = pathlib.Path(__file__).with_suffix(".onnx")
if T_DIR.exists():
    sess = ort.InferenceSession(T_DIR.as_posix(),
                                providers=["CPUExecutionProvider"])
    id2lbl = json.load(open(T_DIR.with_suffix(".labels")))
else:
    sess = None; id2lbl = []

def _tokenize(text, tok):
    out = tok(text, return_offsets_mapping=True, truncation=True,
              return_tensors="np")
    return dict(input_ids=out["input_ids"],
                attention_mask=out["attention_mask"],
                offsets=out["offset_mapping"])

def extract_entities(text: str, regex_hits: list):
    """merge regex + tagger"""
    found = regex_hits[:]

    # 1. regex masking
    mask = list(text)
    for _,span in regex_hits:
        s = text.find(span)
        if s!=-1:
            mask[s:s+len(span)] = " "*len(span)
    residual = "".join(mask)

    # 2. tag only if model present
    if sess:
        import bankbot  # to reuse tokenizer already in memory
        tok = bankbot.tok
        tok_out = _tokenize(residual, tok)
        logits = sess.run(None, tok_out)[0][0]      # [seq,labels]
        probs  = (np.exp(logits)/np.exp(logits).sum(-1,keepdims=True))
        labels = probs.argmax(-1)

        # BIO to spans
        cur=None
        for i,lid in enumerate(labels):
            lbl = id2lbl[lid]
            if lbl=="O": 
                if cur: _close(cur,i)
                cur=None
            elif lbl.startswith("B-"):
                if cur: _close(cur,i)
                cur=[lbl[2:],i]
            elif lbl.startswith("I-") and cur and lbl[2:]==cur[0]:
                pass
            else:
                if cur: _close(cur,i); cur=None
        if cur: _close(cur,len(labels))

    return found

    # --- helpers ------------------------
    def _close(cur, i):
        typ,start_tok = cur
        offsets = tok_out["offsets"][0]
        s = offsets[start_tok][0]; e = offsets[i-1][1]
        span = text[s:e]
        conf = float(probs[start_tok:i,:].mean())
        if conf>0.7 and all(t!=typ for t,_ in found):
            found.append((typ, span))
