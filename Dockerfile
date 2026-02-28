# ===========================================================================
# Dockerfile — claude-code-model-gateway container image
# ===========================================================================
# Build:
#   docker build -t claude-code-model-gateway .
#
# Run:
#   docker run -d \
#     --name gateway \
#     -p 8080:8080 \
#     -e ANTHROPIC_API_KEY=sk-ant-... \
#     claude-code-model-gateway
# ===========================================================================

# --- Stage 1: Build --------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml requirements.txt ./
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# --- Stage 2: Runtime ------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Anonymous"
LABEL description="Claude Code Model Gateway — HTTP model gateway/proxy service"
LABEL version="0.1.0"

# Create non-root service user
RUN groupadd --system claude-gateway && \
    useradd --system --no-create-home --gid claude-gateway claude-gateway

# Install built package from builder stage
COPY --from=builder /install /usr/local

# Create runtime directories
RUN mkdir -p /var/log/claude-code-model-gateway \
             /var/lib/claude-code-model-gateway \
             /etc/claude-code-model-gateway && \
    chown claude-gateway:claude-gateway \
        /var/log/claude-code-model-gateway \
        /var/lib/claude-code-model-gateway

# Copy default configuration
COPY service/conf/gateway.yaml /etc/claude-code-model-gateway/gateway.yaml

# Default environment
ENV GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8080 \
    GATEWAY_TIMEOUT=300 \
    GATEWAY_LOG_FORMAT=json \
    GATEWAY_LOG_LEVEL=info

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import socket; s=socket.create_connection(('127.0.0.1',${GATEWAY_PORT:-8080}),5); s.close()" || exit 1

USER claude-gateway

ENTRYPOINT ["claude-code-model-gateway-service"]
