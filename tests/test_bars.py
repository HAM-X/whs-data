

def test_normalise_keeps_every_intraday_bar():
    """21–27 Oct 2026 bug: the hourly-grid rule kept one 5m bar per hour."""
    import pandas as pd
    from whs import bars
    idx = pd.date_range("2026-10-27T07:00Z", periods=24, freq="5min")
    raw = pd.DataFrame({"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 10}, index=idx)
    assert len(bars.normalise(raw, now="2026-10-27T10:00Z", step=pd.Timedelta(minutes=5))) == 24
    assert len(bars.normalise(raw.iloc[::3], now="2026-10-27T10:00Z", step=pd.Timedelta(minutes=15))) == 8
    assert len(bars.normalise(raw, now="2026-10-27T10:00Z")) == 2           # 1h keeps the hourly grid
