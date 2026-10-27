"""Send messages through Pipdon BOT. Without secrets (e.g. local runs) it only prints."""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from .config import BOT_NAME


def send(text: str, html: bool = False, silent: bool = False) -> bool:
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    msg = f"{BOT_NAME} · {text}"
    if not token or not chat:
        print("[telegram skipped: no secrets]\n" + msg)
        return False
    fields = {"chat_id": chat, "text": msg[:4000], "disable_web_page_preview": "true"}
    if silent:
        fields["disable_notification"] = "true"
    if html:
        fields["parse_mode"] = "HTML"
    data = urllib.parse.urlencode(fields).encode()
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=20) as r:
            ok = json.load(r).get("ok", False)
    except Exception as e:  # never let an alert failure crash the data job
        print(f"[telegram error] {e}")
        return False
    print("[telegram sent]" if ok else "[telegram refused]")
    return ok


def send_photo(path: str, caption: str = "", html: bool = True, silent: bool = False) -> bool:
    """A picture with a caption (≤ 1024 characters) through Pipdon BOT. Multipart upload, no extra packages."""
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print(f"[telegram photo skipped: no secrets] {path}\n{caption}")
        return False
    import uuid
    b = uuid.uuid4().hex
    parts = [("chat_id", chat), ("caption", caption[:1024])] + ([("parse_mode", "HTML")] if html else []) + \
            ([("disable_notification", "true")] if silent else [])
    body = b"".join(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode() for k, v in parts)
    with open(path, "rb") as f:
        body += (f"--{b}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"card.png\"\r\n"
                 f"Content-Type: image/png\r\n\r\n").encode() + f.read() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendPhoto", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            ok = json.load(r).get("ok", False)
    except Exception as e:
        print(f"[telegram photo error] {e}")
        return False
    print("[telegram photo sent]" if ok else "[telegram photo refused]")
    return ok
