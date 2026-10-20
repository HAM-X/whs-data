#!/usr/bin/env bash
# Pipdon BOT robot loop. GitHub's own hourly schedule turned out to be unreliable (runs 3–7 hours apart),
# so one run now stays alive for ~5h40m and does the work itself at hh:12 every hour, then hands over to
# the next run (workflow_dispatch). The hourly cron stays only as a backup starter.
set -u
END=$(( $(date +%s) + ${LOOP_MINUTES:-340} * 60 ))
ARGS="${FIRST_ARGS:-}"
first=1
save() {
  git add data
  if git diff --cached --quiet; then echo "No new bars."; return 0; fi
  git commit -q -m "data: $(TZ=Africa/Nairobi date '+%a %d %b %H:%M') UTC+3"
  for i in 1 2 3; do
    git pull -q --rebase origin main && git push -q origin HEAD:main && return 0
    sleep 5
  done
  echo "::error::Could not push data after 3 tries"
  return 1
}
while :; do
  if [ "$first" = 0 ]; then
    now=$(date +%s)
    next=$(( now / 3600 * 3600 + 12 * 60 ))
    [ "$next" -le "$now" ] && next=$(( next + 3600 ))
    [ "$next" -ge "$END" ] && break
    echo "next run at $(date -u -d @"$next" '+%H:%M') UTC"
    sleep $(( next - now ))
  fi
  git pull -q --rebase origin main || true
  echo "=== $(date -u '+%a %H:%M') UTC ==="
  python -m jobs.download $ARGS || echo "::warning::download job failed this hour"
  save || true
  ARGS=""
  first=0
done
