# bankbot/monitor/drift.py for INTENT DRIFT DETECTION

import redis, math, datetime as dt, os, requests

REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SLACK_WEBHOOK  = os.getenv("SLACK_WEBHOOK")  # export this in your shell
rdb            = redis.from_url(REDIS_URL)

def _entropy(probs):
    return -sum(p * math.log(p + 1e-9) for p in probs)

def log(parse):
    ent   = _entropy(parse["probs"])      # ensure parse keeps full softmax
    day   = dt.date.today().isoformat()
    rdb.rpush(f"entropy:{day}", ent)

def daily_check():
    today = dt.date.today()
    e7  = _avg_entropy(days=7,  today=today)
    e28 = _avg_entropy(days=28, today=today)
    if e28 and e7 / e28 > 1.3:
        msg = f":warning: entropy jump: 7-day={e7:.2f}, 28-day={e28:.2f}"
        requests.post(SLACK_WEBHOOK, json={"text": msg})

def _avg_entropy(days:int, today):
    keys = [f"entropy:{(today - dt.timedelta(d)).isoformat()}" for d in range(days)]
    vals = [float(x) for k in keys for x in rdb.lrange(k, 0, -1)]
    return sum(vals) / max(1, len(vals))
