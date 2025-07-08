import subprocess, pathlib, os, requests

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")
BEST_FILE = pathlib.Path("models/best_tagger_f1.txt")

def read_best(): return float(BEST_FILE.read_text()) if BEST_FILE.exists() else 0
def write_best(v): BEST_FILE.write_text(str(v))

# retrain
proc = subprocess.run(["python","scripts/train_tagger.py","--eval_only"],
                      capture_output=True, text=True)
new_f1 = float(proc.stdout.strip().split()[-1])
if new_f1 >= read_best() - 0.5:
    subprocess.run(["python","scripts/train_tagger.py"], check=True)
    write_best(new_f1)
    msg = f":white_check_mark: tagger deployed. F1={new_f1:.3f}"
else:
    msg = f":x: tagger rejected. F1={new_f1:.3f}"
requests.post(SLACK_WEBHOOK, json={"text": msg})
