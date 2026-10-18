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

Status: Stage 0 setup. First job: `telegram-test` (run it from the Actions tab).
