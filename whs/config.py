"""Instruments the robot follows. One place to change symbols or tick sizes."""

# key: our short name. yahoo: Yahoo Finance ticker. tick: minimum price step (CME spec).
# kind: "future" (exchange volume, used for effort) or "spot" (price only, used to translate to MT5).
FUTURES = {
    "6E": {"yahoo": "6E=F", "tick": 0.00005, "name": "Euro FX"},
    "6B": {"yahoo": "6B=F", "tick": 0.0001, "name": "British Pound"},
    "6A": {"yahoo": "6A=F", "tick": 0.00005, "name": "Australian Dollar"},
    "6N": {"yahoo": "6N=F", "tick": 0.00005, "name": "New Zealand Dollar"},
    "6J": {"yahoo": "6J=F", "tick": 0.0000005, "name": "Japanese Yen"},
    "6C": {"yahoo": "6C=F", "tick": 0.00005, "name": "Canadian Dollar"},
    "6S": {"yahoo": "6S=F", "tick": 0.00005, "name": "Swiss Franc"},
    "GC": {"yahoo": "GC=F", "tick": 0.10, "name": "Gold"},
    "DX": {"yahoo": "DX=F", "tick": 0.005, "name": "US Dollar Index (ICE)"},
}

SPOT = {
    "EURUSD": {"yahoo": "EURUSD=X"},
    "GBPUSD": {"yahoo": "GBPUSD=X"},
    "AUDUSD": {"yahoo": "AUDUSD=X"},
    "NZDUSD": {"yahoo": "NZDUSD=X"},
    "USDJPY": {"yahoo": "JPY=X"},
    "USDCAD": {"yahoo": "CAD=X"},
    "USDCHF": {"yahoo": "CHF=X"},
    "XAUUSD": {"yahoo": "XAUUSD=X", "optional": True},  # Yahoo coverage of spot gold is patchy
}

# Futures checked for effort-related health (volume). DX is context only.
VOLUME_CHECKED = [k for k in FUTURES if k != "DX"]

EXCHANGE_TZ = "America/Chicago"   # all sessions are keyed to CME time
LOCAL_TZ = "Africa/Nairobi"        # Hamza's clock, UTC+3 all year
SESSION_OPEN_HOUR_CT = 17          # CME Globex FX/metals session opens 17:00 CT
SESSION_CLOSE_HOUR_CT = 16         # and closes 16:00 CT (one-hour halt)

BOT_NAME = "Pipdon BOT"
