#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# install.sh — Install claude-code-model-gateway as a system service
# ---------------------------------------------------------------------------
# Usage:
#   sudo ./scripts/install.sh              # Full install (systemd preferred)
#   sudo ./scripts/install.sh --initd      # Force SysV init.d install
#   sudo ./scripts/install.sh --no-service # Install binary only, skip service
#   sudo ./scripts/install.sh --prefix /custom/path  # Custom install path
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
INSTALL_PREFIX="/opt/claude-code-model-gateway"
CONFIG_DIR="/etc/claude-code-model-gateway"
LOG_DIR="/var/log/claude-code-model-gateway"
DATA_DIR="/var/lib/claude-code-model-gateway"
RUN_DIR="/var/run/claude-code-model-gateway"
SERVICE_USER="claude-gateway"
SERVICE_GROUP="claude-gateway"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_INITD=false
SKIP_SERVICE=false

# --- Colours ----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { printf "${BLUE}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[ OK ]${NC}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[ERR]${NC}  %s\n" "$*" >&2; }

# --- Argument parsing -------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --initd)       USE_INITD=true; shift ;;
        --no-service)  SKIP_SERVICE=true; shift ;;
        --prefix)      INSTALL_PREFIX="$2"; shift 2 ;;
        --python)      PYTHON_BIN="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: sudo $0 [--initd] [--no-service] [--prefix PATH] [--python PATH]"
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Pre-flight checks ------------------------------------------------------
if [ "$(id -u)" -ne 0 ]; then
    err "This script must be run as root (use sudo)."
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    err "Python 3 not found at '${PYTHON_BIN}'. Install Python >= 3.11 first."
    exit 1
fi

PY_VERSION=$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info.minor)')
if [ "${PY_MAJOR}" -lt 3 ] || { [ "${PY_MAJOR}" -eq 3 ] && [ "${PY_MINOR}" -lt 11 ]; }; then
    err "Python >= 3.11 required (found ${PY_VERSION})."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

info "Installing claude-code-model-gateway"
info "  Source:  ${PROJECT_DIR}"
info "  Prefix:  ${INSTALL_PREFIX}"
info "  Python:  ${PYTHON_BIN} (${PY_VERSION})"

# --- Create service user ----------------------------------------------------
if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
    info "Creating service user '${SERVICE_USER}' ..."
    useradd --system --no-create-home --home-dir "${INSTALL_PREFIX}" \
        --shell /usr/sbin/nologin --comment "Claude Code Model Gateway" \
        "${SERVICE_USER}"
    ok "Service user created."
else
    ok "Service user '${SERVICE_USER}' already exists."
fi

# --- Create directories -----------------------------------------------------
info "Creating directories ..."
mkdir -p "${INSTALL_PREFIX}"
mkdir -p "${CONFIG_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${DATA_DIR}"
mkdir -p "${RUN_DIR}"

chown "${SERVICE_USER}:${SERVICE_GROUP}" "${LOG_DIR}"
chown "${SERVICE_USER}:${SERVICE_GROUP}" "${DATA_DIR}"
chown "${SERVICE_USER}:${SERVICE_GROUP}" "${RUN_DIR}"
chmod 0750 "${LOG_DIR}"
chmod 0750 "${DATA_DIR}"
chmod 0755 "${RUN_DIR}"
ok "Directories created."

# --- Install application into virtualenv ------------------------------------
info "Setting up Python virtual environment ..."
"${PYTHON_BIN}" -m venv "${INSTALL_PREFIX}/venv"
"${INSTALL_PREFIX}/venv/bin/pip" install --upgrade pip setuptools wheel >/dev/null 2>&1
ok "Virtual environment ready."

info "Installing application package ..."
"${INSTALL_PREFIX}/venv/bin/pip" install "${PROJECT_DIR}" >/dev/null 2>&1
ok "Application installed."

# Verify installation
"${INSTALL_PREFIX}/venv/bin/claude-code-model-gateway" version
ok "Binary verified."

# --- Copy source for reference (optional) -----------------------------------
info "Copying source tree to ${INSTALL_PREFIX}/src ..."
cp -r "${PROJECT_DIR}/src" "${INSTALL_PREFIX}/src"
cp "${PROJECT_DIR}/pyproject.toml" "${INSTALL_PREFIX}/"
cp "${PROJECT_DIR}/requirements.txt" "${INSTALL_PREFIX}/"
ok "Source tree copied."

# --- Install configuration --------------------------------------------------
info "Installing configuration files ..."
if [ ! -f "${CONFIG_DIR}/environment" ]; then
    cp "${PROJECT_DIR}/service/conf/environment" "${CONFIG_DIR}/environment"
    chmod 0640 "${CONFIG_DIR}/environment"
    chown root:"${SERVICE_GROUP}" "${CONFIG_DIR}/environment"
    ok "Environment file installed."
else
    warn "Environment file already exists — not overwriting."
    cp "${PROJECT_DIR}/service/conf/environment" "${CONFIG_DIR}/environment.dist"
fi

if [ ! -f "${CONFIG_DIR}/gateway.yaml" ]; then
    cp "${PROJECT_DIR}/service/conf/gateway.yaml" "${CONFIG_DIR}/gateway.yaml"
    chmod 0640 "${CONFIG_DIR}/gateway.yaml"
    chown root:"${SERVICE_GROUP}" "${CONFIG_DIR}/gateway.yaml"
    ok "Gateway config installed."
else
    warn "Gateway config already exists — not overwriting."
    cp "${PROJECT_DIR}/service/conf/gateway.yaml" "${CONFIG_DIR}/gateway.yaml.dist"
fi

# --- Install helper wrapper to /usr/local/bin --------------------------------
info "Installing CLI wrapper to /usr/local/bin ..."
cat > /usr/local/bin/claude-code-model-gateway <<WRAPPER
#!/usr/bin/env bash
# Wrapper script for claude-code-model-gateway
# Loads environment and delegates to the virtualenv binary.

if [ -f "${CONFIG_DIR}/environment" ]; then
    set -a
    . "${CONFIG_DIR}/environment"
    set +a
fi

exec "${INSTALL_PREFIX}/venv/bin/claude-code-model-gateway" "\$@"
WRAPPER
chmod 0755 /usr/local/bin/claude-code-model-gateway
ok "CLI wrapper installed."

# --- Install service --------------------------------------------------------
if [ "${SKIP_SERVICE}" = true ]; then
    warn "Skipping service installation (--no-service)."
else
    # Detect init system
    HAS_SYSTEMD=false
    if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
        HAS_SYSTEMD=true
    fi

    if [ "${HAS_SYSTEMD}" = true ] && [ "${USE_INITD}" = false ]; then
        # ---- systemd ----
        info "Installing systemd service unit ..."
        cp "${PROJECT_DIR}/service/systemd/claude-code-model-gateway.service" \
            /etc/systemd/system/claude-code-model-gateway.service

        if [ -f "${PROJECT_DIR}/service/systemd/claude-code-model-gateway.tmpfiles" ]; then
            cp "${PROJECT_DIR}/service/systemd/claude-code-model-gateway.tmpfiles" \
                /etc/tmpfiles.d/claude-code-model-gateway.conf
        fi

        if [ -f "${PROJECT_DIR}/service/systemd/claude-code-model-gateway.sysusers" ]; then
            cp "${PROJECT_DIR}/service/systemd/claude-code-model-gateway.sysusers" \
                /etc/sysusers.d/claude-code-model-gateway.conf
        fi

        systemctl daemon-reload
        systemctl enable claude-code-model-gateway.service
        ok "systemd service installed and enabled."
        info "Start with: sudo systemctl start claude-code-model-gateway"
    else
        # ---- SysV init ----
        info "Installing SysV init script ..."
        cp "${PROJECT_DIR}/service/initd/claude-code-model-gateway" \
            /etc/init.d/claude-code-model-gateway
        chmod 0755 /etc/init.d/claude-code-model-gateway

        if command -v update-rc.d >/dev/null 2>&1; then
            update-rc.d claude-code-model-gateway defaults
        elif command -v chkconfig >/dev/null 2>&1; then
            chkconfig --add claude-code-model-gateway
            chkconfig claude-code-model-gateway on
        fi
        ok "SysV init script installed and enabled."
        info "Start with: sudo service claude-code-model-gateway start"
    fi
fi

# --- Install logrotate config -----------------------------------------------
if [ -d /etc/logrotate.d ]; then
    info "Installing logrotate configuration ..."
    cat > /etc/logrotate.d/claude-code-model-gateway <<'LOGROTATE'
/var/log/claude-code-model-gateway/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 claude-gateway claude-gateway
    sharedscripts
    postrotate
        systemctl reload claude-code-model-gateway >/dev/null 2>&1 || \
        service claude-code-model-gateway reload >/dev/null 2>&1 || true
    endscript
}
LOGROTATE
    ok "Logrotate configuration installed."
fi

# --- Summary -----------------------------------------------------------------
echo ""
echo "============================================================"
printf "${GREEN}Installation complete!${NC}\n"
echo "============================================================"
echo ""
echo "  Install prefix:   ${INSTALL_PREFIX}"
echo "  Config directory:  ${CONFIG_DIR}"
echo "  Log directory:     ${LOG_DIR}"
echo "  Data directory:    ${DATA_DIR}"
echo "  CLI wrapper:       /usr/local/bin/claude-code-model-gateway"
echo "  Service user:      ${SERVICE_USER}"
echo ""
echo "Next steps:"
echo "  1. Edit ${CONFIG_DIR}/environment to set API keys"
echo "  2. Edit ${CONFIG_DIR}/gateway.yaml to configure providers"
echo "  3. Start the service:"
if [ "${SKIP_SERVICE}" = false ]; then
    if [ "${HAS_SYSTEMD}" = true ] && [ "${USE_INITD}" = false ]; then
        echo "       sudo systemctl start claude-code-model-gateway"
        echo "       sudo systemctl status claude-code-model-gateway"
    else
        echo "       sudo service claude-code-model-gateway start"
        echo "       sudo service claude-code-model-gateway status"
    fi
else
    echo "       claude-code-model-gateway gateway --host 0.0.0.0 --port 8080"
fi
echo "  4. View logs:"
echo "       tail -f ${LOG_DIR}/gateway.log"
echo "       journalctl -u claude-code-model-gateway -f  (systemd)"
echo ""
