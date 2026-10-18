"""Send messages through Pipdon BOT. Without secrets (e.g. local runs) it only prints."""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from .config import BOT_NAME


def send(text: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    msg = f"{BOT_NAME} · {text}"
    if not token or not chat:
        print("[telegram skipped: no secrets]\n" + msg)
        return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": msg, "disable_web_page_preview": "true"}).encode()
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=20) as r:
            ok = json.load(r).get("ok", False)
    except Exception as e:  # never let an alert failure crash the data job
        print(f"[telegram error] {e}")
        return False
    print("[telegram sent]" if ok else "[telegram refused]")
    return ok
