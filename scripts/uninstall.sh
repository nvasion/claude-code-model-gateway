#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# uninstall.sh — Remove claude-code-model-gateway system service
# ---------------------------------------------------------------------------
# Usage:
#   sudo ./scripts/uninstall.sh              # Uninstall (keep config)
#   sudo ./scripts/uninstall.sh --purge      # Uninstall and remove all data
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
INSTALL_PREFIX="/opt/claude-code-model-gateway"
CONFIG_DIR="/etc/claude-code-model-gateway"
LOG_DIR="/var/log/claude-code-model-gateway"
DATA_DIR="/var/lib/claude-code-model-gateway"
RUN_DIR="/var/run/claude-code-model-gateway"
SERVICE_USER="claude-gateway"
PURGE=false

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
        --purge) PURGE=true; shift ;;
        -h|--help)
            echo "Usage: sudo $0 [--purge]"
            echo "  --purge  Remove configuration, logs, and data as well"
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

info "Uninstalling claude-code-model-gateway ..."

# --- Stop and disable service ------------------------------------------------
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet claude-code-model-gateway 2>/dev/null; then
    info "Stopping systemd service ..."
    systemctl stop claude-code-model-gateway || true
fi

if command -v systemctl >/dev/null 2>&1 && systemctl is-enabled --quiet claude-code-model-gateway 2>/dev/null; then
    info "Disabling systemd service ..."
    systemctl disable claude-code-model-gateway || true
fi

# Remove systemd unit files
if [ -f /etc/systemd/system/claude-code-model-gateway.service ]; then
    info "Removing systemd unit file ..."
    rm -f /etc/systemd/system/claude-code-model-gateway.service
    rm -f /etc/tmpfiles.d/claude-code-model-gateway.conf
    rm -f /etc/sysusers.d/claude-code-model-gateway.conf
    systemctl daemon-reload 2>/dev/null || true
    ok "systemd service removed."
fi

# Remove SysV init script
if [ -f /etc/init.d/claude-code-model-gateway ]; then
    info "Stopping SysV service ..."
    service claude-code-model-gateway stop 2>/dev/null || true

    if command -v update-rc.d >/dev/null 2>&1; then
        update-rc.d -f claude-code-model-gateway remove 2>/dev/null || true
    elif command -v chkconfig >/dev/null 2>&1; then
        chkconfig --del claude-code-model-gateway 2>/dev/null || true
    fi

    rm -f /etc/init.d/claude-code-model-gateway
    ok "SysV init script removed."
fi

# --- Remove CLI wrapper ------------------------------------------------------
if [ -f /usr/local/bin/claude-code-model-gateway ]; then
    info "Removing CLI wrapper ..."
    rm -f /usr/local/bin/claude-code-model-gateway
    ok "CLI wrapper removed."
fi

# --- Remove logrotate config -------------------------------------------------
if [ -f /etc/logrotate.d/claude-code-model-gateway ]; then
    info "Removing logrotate configuration ..."
    rm -f /etc/logrotate.d/claude-code-model-gateway
    ok "Logrotate config removed."
fi

# --- Remove application directory --------------------------------------------
if [ -d "${INSTALL_PREFIX}" ]; then
    info "Removing application directory ${INSTALL_PREFIX} ..."
    rm -rf "${INSTALL_PREFIX}"
    ok "Application directory removed."
fi

# --- Remove runtime directory ------------------------------------------------
if [ -d "${RUN_DIR}" ]; then
    rm -rf "${RUN_DIR}"
fi

# --- Purge (optional) -------------------------------------------------------
if [ "${PURGE}" = true ]; then
    warn "Purging all configuration, logs, and data ..."

    if [ -d "${CONFIG_DIR}" ]; then
        info "Removing configuration directory ${CONFIG_DIR} ..."
        rm -rf "${CONFIG_DIR}"
        ok "Configuration removed."
    fi

    if [ -d "${LOG_DIR}" ]; then
        info "Removing log directory ${LOG_DIR} ..."
        rm -rf "${LOG_DIR}"
        ok "Logs removed."
    fi

    if [ -d "${DATA_DIR}" ]; then
        info "Removing data directory ${DATA_DIR} ..."
        rm -rf "${DATA_DIR}"
        ok "Data removed."
    fi

    # Remove service user
    if id -u "${SERVICE_USER}" >/dev/null 2>&1; then
        info "Removing service user '${SERVICE_USER}' ..."
        userdel "${SERVICE_USER}" 2>/dev/null || true
        ok "Service user removed."
    fi
else
    info "Configuration preserved in ${CONFIG_DIR}"
    info "Logs preserved in ${LOG_DIR}"
    info "Data preserved in ${DATA_DIR}"
    info "Use --purge to remove everything."
fi

echo ""
echo "============================================================"
printf "${GREEN}Uninstallation complete.${NC}\n"
echo "============================================================"
