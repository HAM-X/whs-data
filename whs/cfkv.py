"""Pipdon BOT's Cloudflare KV ("pipdon-desk"), written by the robot: the chart cards (img:<name>) the bot sends.
Needs CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID (repository secrets); without them it does nothing."""
from __future__ import annotations

import json
import os
import urllib.request

API = "https://api.cloudflare.com/client/v4/accounts/{acc}/storage/kv/namespaces"
_NS = {}


def _req(url, method="GET", data=None, ctype=None):
    h = {"Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}"}
    if ctype:
        h["Content-Type"] = ctype
    with urllib.request.urlopen(urllib.request.Request(url, data=data, method=method, headers=h), timeout=30) as r:
        return json.loads(r.read().decode() or "{}")


def namespace(title: str = "pipdon-desk") -> str | None:
    if title in _NS:
        return _NS[title]
    acc = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not acc or not os.environ.get("CLOUDFLARE_API_TOKEN"):
        return None
    res = _req(API.format(acc=acc) + "?per_page=100").get("result") or []
    _NS[title] = next((x["id"] for x in res if x.get("title") == title), None)
    return _NS[title]


def put_files(files: dict, prefix: str = "img:") -> int:
    """{name: local path} → KV keys prefix+name. Returns how many were written."""
    ns = namespace()
    if not ns:
        print("[kv skipped: no Cloudflare secrets or namespace]")
        return 0
    acc, n = os.environ["CLOUDFLARE_ACCOUNT_ID"], 0
    for name, path in files.items():
        try:
            with open(path, "rb") as f:
                _req(API.format(acc=acc) + f"/{ns}/values/{prefix}{name}", "PUT", f.read(), "application/octet-stream")
            n += 1
        except Exception as e:
            print(f"[kv error] {name}: {type(e).__name__}: {e}")
    return n


def put_bundle(files: dict, key: str = "img:bundle") -> int:
    """All images in ONE KV value ({name: base64 PNG}) — one write per hour instead of one per image
    (the free plan allows 1,000 KV writes a day)."""
    import base64
    ns = namespace()
    if not ns:
        print("[kv skipped: no Cloudflare secrets or namespace]")
        return 0
    blob = {}
    for name, path in files.items():
        try:
            with open(path, "rb") as f:
                blob[name] = base64.b64encode(f.read()).decode()
        except Exception as e:
            print(f"[kv bundle] {name}: {e}")
    try:
        _req(API.format(acc=os.environ["CLOUDFLARE_ACCOUNT_ID"]) + f"/{ns}/values/{key}", "PUT",
             json.dumps(blob).encode(), "application/json")
        return len(blob)
    except Exception as e:
        print(f"[kv error] bundle: {type(e).__name__}: {e}")
        return 0


def put_json(key: str, obj) -> bool:
    ns = namespace()
    if not ns:
        return False
    try:
        _req(API.format(acc=os.environ["CLOUDFLARE_ACCOUNT_ID"]) + f"/{ns}/values/{key}", "PUT",
             json.dumps(obj, default=str).encode(), "application/json")
        return True
    except Exception as e:
        print(f"[kv error] {key}: {type(e).__name__}: {e}")
        return False
