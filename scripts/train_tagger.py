import json, numpy as np, torch, datasets, transformers, os, pathlib
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer)
TOK_NAME = "distilbert-base-multilingual-cased"
DATA     = "data/span_training/spans.jsonl"
OUTDIR   = "bankbot/entity"

# 1. load data -------------------------------------------------
def read():
    with open(DATA) as f:
        for j in map(json.loads, f):
            yield j["text"], j["spans"]

texts, spans = zip(*list(read()))
tok = AutoTokenizer.from_pretrained(TOK_NAME)

def encode(example):
    text, gold = example["text"], example["spans"]
    enc = tok(text, truncation=True, return_offsets_mapping=True, padding="max_length", max_length=128)
    labels = ["O"]*len(enc["input_ids"])
    for s in gold:
        for i,(st,en) in enumerate(enc["offset_mapping"]):
            if st>=s["start"] and en<=s["end"]:
                labels[i] = "B-"+s["type"] if st==s["start"] else "I-"+s["type"]
    enc["labels"] = [label_map[x] for x in labels]
    return enc

tags = sorted({s["type"] for span in spans for s in span})
label_list = ["O"]+[b+"-"+t for t in tags for b in ("B","I")]
label_map  = {l:i for i,l in enumerate(label_list)}

ds = datasets.load_dataset("json", data_files=DATA, split="train")
ds = ds.map(lambda ex: {"text":ex["text"],"spans":ex["spans"]})
ds = ds.map(encode, batched=False, remove_columns=["text","spans"])

# 2. train ------------------------------------------------------
model = AutoModelForTokenClassification.from_pretrained(
            TOK_NAME, num_labels=len(label_list))
args = TrainingArguments(
   output_dir="tmp_tagger", per_device_train_batch_size=16,
   num_train_epochs=3, learning_rate=2e-5, logging_steps=100)
trainer = Trainer(model, args, train_dataset=ds)
trainer.train()

###################
######HERE
##################

# # 3. export to ONNX --------------------------------------------
# # from optimum.exporters.onnx import main_export
# # main_export(model_name_or_path=model, feature="token-classification", tokenizer=tok, output=str(OUTDIR+"/tagger.onnx"))
# # with open(OUTDIR/"tagger.labels","w") as f:
# #     json.dump(label_list, open(OUTDIR+"/tagger.labels","w"))
# # print("✅ tagger.onnx + tagger.labels written to", OUTDIR)
# from optimum.exporters.onnx import main_export
# # OUTDIR.mkdir(parents=True, exist_ok=True)
# pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)
# # export to ONNX
# main_export(
#     model_name_or_path="distilbert-base-multilingual-cased",
#     feature="token-classification",
#     tokenizer=tok,
#     output=str(pathlib.Path(OUTDIR) / "tagger.onnx")
# )

# # after export, move/rename ONNX to tagger.onnx
# import shutil
# shutil.move(str(pathlib.Path(OUTDIR, "model.onnx")),
#             str(pathlib.Path(OUTDIR, "tagger.onnx")))

# # write labels
# with open(pathlib.Path(OUTDIR, "tagger.labels"), "w") as f:
#     json.dump(label_list, f)
# print("✅ onnx/model.onnx → tagger.onnx, tagger.labels written to", OUTDIR)


# #-------------------------HERE---------------------

from optimum.exporters.onnx import main_export
import pathlib, shutil, json

# ensure OUTDIR is a directory
OUTDIR = "bankbot/entity"
pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)

# 1) export into that folder (will create onnx/model.onnx etc.)
main_export(
    model_name_or_path="distilbert-base-multilingual-cased",
    feature="token-classification",
    tokenizer=tok,
    output=OUTDIR,                # ← DIRECTORY, not a file
)

# 2) locate the ONNX file, wherever it landed
dirp = pathlib.Path(OUTDIR)
candidates = [dirp/"model.onnx", dirp/"onnx"/"model.onnx"]
src = next(p for p in candidates if p.exists())

# 3) move/rename
dst = dirp/"tagger.onnx"
shutil.move(str(src), str(dst))

# 4) write labels
with open(dirp/"tagger.labels", "w") as f:
    json.dump(label_list, f)

print("✅ tagger.onnx + tagger.labels written to", OUTDIR)
