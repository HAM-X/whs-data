"""On-demand robot run (Pipdon BOT /refresh): fresh bars → scan → news → cards → images + decisions to the bot's KV,
then the board + Today (or one chart card) straight to Telegram. Commits nothing; the hourly robot stays the record.

Run:  python -m jobs.now            → the board + Today
      python -m jobs.now 6E 15m     → that chart's card
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from jobs import download as J
from whs import cfkv, telegram
from whs.sources import yahoo


def main(argv: list[str]) -> None:
    now = pd.Timestamp.now(tz="UTC")
    J.run(now=now, send=lambda *a, **k: print("[now: not sent]", a[0][:120] if a else ""), ltf_fetch=yahoo.fetch)
    made = J.write_cards(now, True, alerts=False)
    if not made:
        telegram.send("⚠️ Fresh read failed — the hourly board is still there: /today")
        return
    dec, imgs = made
    cfkv.put_json("dec:latest", dec)
    if len(argv) >= 1:
        key = argv[0].upper()
        tf = argv[1] if len(argv) > 1 else "4h"
        name, k = f"card:{key}:{tf}", f"{key} {tf}"
        if name in imgs and k in dec:
            telegram.send_photo(imgs[name], "⚡ fresh · " + dec[k]["caption"])
            return
    telegram.send_photo(imgs["board"], "⚡ fresh · " + dec["_today"])


if __name__ == "__main__":
    main(sys.argv[1:])
