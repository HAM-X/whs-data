# CLAUDE.md — whs-data (PUBLIC repository)

This repository is public. It holds only the data robot (GitHub Actions jobs), the Pipdon BOT
(Telegram) code and public market/news data.

Rules for working here:
- Anything private stays out: TradingView/CME exports, Pine code, trade or P&L data, webhook URLs, keys.
  The project manual and private material live in the private sister repo (whs-desk); read its
  CLAUDE.md first when it is available.
- Tokens are GitHub Actions secrets only (`TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, Cloudflare). Never write
  a token, key or secret URL into code, commits or logs.
- The bot is always called "Pipdon BOT".
- Schedules are cron in UTC; local time is UTC+3.
