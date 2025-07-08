from fastapi import FastAPI
from pydantic import BaseModel
import importlib, pathlib, time, os
import bankbot
from bankbot.clarifier.clarifier import build_clarifier, apply_reply
from bankbot.monitor.drift     import log as log_drift

app = FastAPI()
_sessions = {}

class Msg(BaseModel):
    session_id: str
    text: str

# hot-reload after nightly swap
@app.on_event("startup")
def reload_if_flag():
    flag = pathlib.Path("/tmp/bankbot_reload")
    if flag.exists() and time.time() - flag.stat().st_mtime < 60:
        importlib.reload(bankbot)
        flag.unlink()

@app.post("/message")
def message(m: Msg):
    res  = bankbot.parse(m.text)
    log_drift(res)
    clar = build_clarifier(res)
    _sessions[m.session_id] = res
    return {**res, **clar}

@app.post("/clarify")
def clarify(m: Msg):
    last = _sessions.get(m.session_id)
    if not last:
        return {"error": "session expired"}
    res = apply_reply(m.text, last)
    _sessions[m.session_id] = res
    return res

@app.get("/ping")
def ping(): return {"status": "ok"}
