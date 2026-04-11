#!/bin/bash
# Launch Claude Code routed through the model gateway.
#
# Prerequisites:
#   1. Gateway running: bash scripts/start-test-gateway.sh  (in another terminal)
#   2. OPENROUTER_API_KEY set in the gateway terminal
#
# This script sets ANTHROPIC_BASE_URL so Claude Code talks to the local gateway
# instead of directly to Anthropic. The gateway translates Anthropic format to
# OpenAI format and routes to OpenRouter's free Qwen3 Coder model.

set -e

GATEWAY_URL="http://127.0.0.1:8080"

# Check gateway is running
if ! curl -s "$GATEWAY_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Gateway is not running at $GATEWAY_URL"
    echo ""
    echo "Start it first in another terminal:"
    echo "  cd claude-code-model-gateway"
    echo "  export OPENROUTER_API_KEY=\"sk-or-v1-your-key\""
    echo "  bash scripts/start-test-gateway.sh"
    exit 1
fi

echo "Gateway health check: OK"
echo "Routing Claude Code through $GATEWAY_URL → OpenRouter → Qwen3 Coder"
echo ""

# Claude Code needs ANTHROPIC_API_KEY to be set (it validates the format).
# If you don't have one set, use a placeholder — the gateway will use
# OPENROUTER_API_KEY for the actual upstream call.
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Note: ANTHROPIC_API_KEY not set, using placeholder for gateway mode."
    export ANTHROPIC_API_KEY="sk-ant-placeholder-for-gateway-mode"
fi

export ANTHROPIC_BASE_URL="$GATEWAY_URL"

# Pass through any args (e.g., --model, project path)
exec claude "$@"
