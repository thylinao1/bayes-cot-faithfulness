#!/bin/bash
# ssh soc with a hard timeout and a bounded retry, for a lane whose VPN drops.
# macOS has no coreutils `timeout`, so the wall is enforced by a background child and a
# poll loop; the child is killed rather than left to hang on banner exchange.
#
#   bcf/ssh_retry.sh [--wall SECONDS] [--tries N] [--gap SECONDS] -- <command...>
# Exit 0 on success, 99 when every try timed out or failed (the caller writes BLOCKED).
set -uo pipefail
WALL=90; TRIES=6; GAP=30
while [ $# -gt 0 ]; do
  case "$1" in
    --wall) WALL="$2"; shift 2 ;;
    --tries) TRIES="$2"; shift 2 ;;
    --gap) GAP="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: ssh_retry.sh [--wall S] [--tries N] [--gap S] -- <cmd...>" >&2; exit 2; }
try=1
while [ "$try" -le "$TRIES" ]; do
  tmp="$(mktemp)"
  ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o BatchMode=yes soc "$@" >"$tmp" 2>&1 &
  pid=$!
  waited=0
  while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$WALL" ]; do
    sleep 2; waited=$(( waited + 2 ))
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null; sleep 1; kill -KILL "$pid" 2>/dev/null
    echo "[ssh_retry] try ${try}/${TRIES} TIMED OUT after ${WALL}s" >&2
    rc=124
  else
    wait "$pid"; rc=$?
  fi
  cat "$tmp"; rm -f "$tmp"
  [ "$rc" -eq 0 ] && exit 0
  echo "[ssh_retry] try ${try}/${TRIES} exit ${rc}" >&2
  try=$(( try + 1 ))
  [ "$try" -le "$TRIES" ] && sleep "$GAP"
done
echo "[ssh_retry] GAVE UP after ${TRIES} tries" >&2
exit 99
