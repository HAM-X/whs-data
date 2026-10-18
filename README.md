# whs-data

Public data robot for the WHS Wyckoff system (CME FX futures + gold, 1h/4h).

GitHub Actions jobs collect market and news data on a schedule and save it here.
Nothing personal lives in this repository; the journal and stats live in the private `whs-desk`.

Planned layout:

| Folder | What goes there |
| --- | --- |
| `data/` | Stored bars, one file per symbol and timeframe, Chicago exchange time |
| `engine/` | Python copy of WHS Core + Engine |
| `jobs/` | One script per task (download, health check, scan, alerts, news) |
| `.github/workflows/` | The schedules (cron in UTC; local time is UTC+3) |

Secrets used (set in Settings → Secrets and variables → Actions, never in code):
`TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.

Alerts are delivered by the Telegram bot **Pipdon BOT**.

## Jobs

| Workflow | When | What it does |
| --- | --- | --- |
| `robot-hourly` | hh:12 every hour, Sun–Fri (UTC; same minute in UTC+3) | Downloads 1h bars for 9 futures + 8 spot pairs from Yahoo, stores them, rebuilds 4h bars, checks data health, alerts through Pipdon BOT |
| `tests` | on every code change | Offline tests of the session clock, storage and 4h builder |
| `telegram-test` | by hand | Sends one test message |

Data health alerts: download failures, a feed 3+ hours behind, missing hours, frozen prices, zero volume.
Each problem is reported once, and again when fixed. One "data healthy" line per trading day after 08:00 UTC+3.

4h bars are built from 1h bars in blocks from the 17:00 Chicago session open, exactly like TradingView
(checked: 34,585 of 34,585 bars identical across 8 futures).

Status: Stage 1 (robot foundation).
