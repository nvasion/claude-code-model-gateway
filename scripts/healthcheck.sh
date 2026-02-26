#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# healthcheck.sh — Health check for claude-code-model-gateway
# ---------------------------------------------------------------------------
# Usage:
#   ./scripts/healthcheck.sh                   # Check default port 8080
#   ./scripts/healthcheck.sh --port 9090       # Check custom port
#   ./scripts/healthcheck.sh --host 10.0.0.1   # Check remote host
# ---------------------------------------------------------------------------
# Exit codes:
#   0  Service is healthy
#   1  Service is unhealthy
# ---------------------------------------------------------------------------

set -euo pipefail

HOST="${GATEWAY_HOST:-127.0.0.1}"
PORT="${GATEWAY_PORT:-8080}"
TIMEOUT=5

while [ $# -gt 0 ]; do
    case "$1" in
        --host)    HOST="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--host HOST] [--port PORT] [--timeout SEC]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Try a basic TCP connection using bash built-ins or Python
check_tcp() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import socket, sys
try:
    s = socket.create_connection(('${HOST}', ${PORT}), timeout=${TIMEOUT})
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
"
    elif command -v nc >/dev/null 2>&1; then
        nc -z -w "${TIMEOUT}" "${HOST}" "${PORT}"
    elif command -v bash >/dev/null 2>&1; then
        (echo > /dev/tcp/"${HOST}"/"${PORT}") 2>/dev/null
    else
        echo "No suitable tool found for TCP check" >&2
        return 1
    fi
}

if check_tcp; then
    echo "OK: claude-code-model-gateway is listening on ${HOST}:${PORT}"
    exit 0
else
    echo "FAIL: claude-code-model-gateway is NOT responding on ${HOST}:${PORT}"
    exit 1
fi
