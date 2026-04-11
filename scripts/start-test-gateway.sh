#!/bin/bash
# Start the gateway in test mode with OpenRouter → Qwen3 Coder (free)
#
# Prerequisites:
#   export OPENROUTER_API_KEY="sk-or-v1-your-key"
#
# Then in another terminal:
#   bash scripts/start-claude-via-gateway.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY is not set."
    echo ""
    echo "  export OPENROUTER_API_KEY=\"sk-or-v1-your-key\""
    echo ""
    echo "Get one at: https://openrouter.ai/keys"
    exit 1
fi

echo "============================================"
echo "  Factory Model Gateway - Test Mode"
echo "============================================"
echo "  Provider : OpenRouter"
echo "  Model    : qwen/qwen3-coder:free"
echo "  Listen   : http://127.0.0.1:8080"
echo "  Config   : gateway-test.yaml"
echo "============================================"
echo ""
echo "In another terminal, run:"
echo "  cd $(pwd)"
echo "  bash scripts/start-claude-via-gateway.sh"
echo ""
echo "Starting gateway..."
echo ""

claude-code-model-gateway route serve \
    -c gateway-test.yaml \
    --host 127.0.0.1 \
    --port 8080 \
    --timeout 120 \
    --max-retries 5 \
    --retry-delay 3.0 \
    --no-auth \
    --verbose \
    --log-format colored
