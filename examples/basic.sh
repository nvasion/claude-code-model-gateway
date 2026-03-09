#!/bin/bash
# Basic usage examples for claude-code-model-gateway

echo "=== claude-code-model-gateway Examples ==="
echo

# Show help
echo "1. Showing help:"
claude-code-model-gateway --help
echo

# Hello command
echo "2. Hello command:"
claude-code-model-gateway hello
echo

# Greet command
echo "3. Greet command (default):"
claude-code-model-gateway greet
echo

echo "4. Greet command (with name):"
claude-code-model-gateway greet "User"
echo

# Version
echo "5. Version:"
claude-code-model-gateway version
echo

echo "=== Examples complete ==="
