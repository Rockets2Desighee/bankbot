# --- Run MacBook GPU ---
import os, torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
#keep above "device = ... " line forever through prototype and production - no need to edit when ship to cloud or another laptop.

# ------------- train_intent.py --------------------------------------------
import pandas as pd, torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

MODEL_NAME = "ai4bharat/indic-bert"   # multilingual model already knowing Hindi & English
CSV_FILE   = "synthetic_dataset.csv"

# LABELS = [
#     "balance_check",
#     "emi_issue",
#     "report_fraud",
#     "card_upgrade",
#     "kyc_pending"
# ]

# --- tokenizer ---
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# 1. read your spreadsheet
df = pd.read_csv(CSV_FILE, usecols=["raw_text", "intent"])

# drop the header that leaked into the data
df = df[df.intent != "intent"]

# decide whether stratification is possible
strat_col = df.intent if df.intent.value_counts().min() > 1 else None

LABELS = sorted(df.intent.unique().tolist())
print("Intents found in CSV:", LABELS)

# 2. split 90 % for learning, 10 % for checking
# train_df, val_df = train_test_split(df, test_size=0.1,
#                                     stratify=df.intent, random_state=42)
train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        stratify=strat_col,
        random_state=42
)

# 3. turn rows into HuggingFace Dataset objects
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- encode rows ---
label2id = {lab: i for i, lab in enumerate(LABELS)}
id2label = {i: lab for lab, i in label2id.items()}

def encode(batch):
    return tok(batch["raw_text"],
               truncation=True,
               padding="max_length",
               max_length=128)

def encode_row(row):
    text   = row["raw_text"]              # preserve original fields
    intent = row["intent"]

    tok_out = tok(
        text,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tok_out["labels"] = label2id[intent]  # str → int
    return tok_out

train_ds = (
    Dataset.from_pandas(train_df)
    .map(encode_row, remove_columns=["raw_text", "intent"])
)

val_ds = (
    Dataset.from_pandas(val_df)
    .map(encode_row, remove_columns=["raw_text", "intent"])
)

# rename “intent” to the reserved word “labels”
# train_ds = train_ds.rename_column("intent", "labels").class_encode_column("labels")
# val_ds   = val_ds.rename_column("intent", "labels").class_encode_column("labels")

# train_ds = train_ds.rename_column("labels").class_encode_column(
#               "labels", class_labels=LABELS)
# val_ds   = val_ds.rename_column("labels").class_encode_column(
#               "labels", class_labels=LABELS)

# train_ds = (Dataset.from_pandas(train_df)
#             .map(encode_row)
#             .remove_columns(["raw_text", "intent"]))

# val_ds   = (Dataset.from_pandas(val_df)
#             .map(encode_row)
#             .remove_columns(["raw_text", "intent"]))

# 4. load the base model, attach a fresh classifier layer sized for our labels
model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            # num_labels=train_ds.features["labels"].num_classes).to(device) #send to Mac GPU
            num_labels=len(LABELS)).to(device)

# 5. tell the Trainer how to learn
args = TrainingArguments(
        output_dir="chk",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        logging_steps=50)

# 6. run
trainer = Trainer(model=model,
                  args=args,
                  train_dataset=train_ds,
                  eval_dataset=val_ds,
                  tokenizer=tok)

trainer.train()          # 25 min on Mac CPU, <3 min on a cloud GPU, warnings such as “kernel launched on the CPU because it is not supported on the MPS device”, are normal; those ops transparently fall back to the CPU.

# bankbot can’t translate the numeric class ID
# into the real intent name, so downstream rules will fail.
# Applied permanent fix (4 Lines) below:
# label2id = {lab: i for i, lab in enumerate(LABELS)}
# id2label = {i: lab for lab, i in label2id.items()}
model.config.label2id = label2id      # NEW
model.config.id2label = id2label      # NEW

trainer.save_model("bankbot_intent")
tok.save_pretrained("bankbot_intent")
