#!/usr/bin/env python
import subprocess, pathlib, requests, os, json

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")
BEST_FILE = pathlib.Path("models/best_f1.txt")
DATA_CSV  = "data/labeled_merged.csv"

def read_best(): return float(BEST_FILE.read_text()) if BEST_FILE.exists() else 0
def write_best(v): BEST_FILE.write_text(str(v))

def main():
    subprocess.run(["python", "pipeline/merge_labels.py"], check=True)
    new_f1 = float(subprocess.check_output(
        ["python","train_intent.py","--data",DATA_CSV,"--eval_only"],
        text=True).strip().split()[-1])
    if new_f1 >= read_best() - 0.5:
        subprocess.run(["python","train_intent.py","--data",DATA_CSV])
        write_best(new_f1)
        pathlib.Path("/tmp/bankbot_reload").touch()
        msg = f":white_check_mark: deployed nightly model – F1 {new_f1:.3f}"
    else:
        msg = f":x: nightly model rejected – F1 {new_f1:.3f}"
    requests.post(SLACK_WEBHOOK, json={"text": msg})

if __name__ == "__main__":
    main()
