#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# update.sh — In-place upgrade of claude-code-model-gateway
# ---------------------------------------------------------------------------
# Updates the installed application to the version in the current project
# directory (or a specified source archive) without touching existing
# configuration files, log files, or user data.
#
# Usage:
#   sudo ./scripts/update.sh              # Upgrade from the current source tree
#   sudo ./scripts/update.sh --python /usr/bin/python3.12
#   sudo ./scripts/update.sh --backup     # Keep a timestamped backup of the venv
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
INSTALL_PREFIX="/opt/claude-code-model-gateway"
CONFIG_DIR="/etc/claude-code-model-gateway"
LOG_DIR="/var/log/claude-code-model-gateway"
SERVICE_NAME="claude-code-model-gateway"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAKE_BACKUP=false
RESTART_SERVICE=true

# --- Colours ----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { printf "${BLUE}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[ OK ]${NC}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[ERR]${NC}  %s\n" "$*" >&2; }
bold()  { printf "${BOLD}%s${NC}\n" "$*"; }

# --- Argument parsing -------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --python)      PYTHON_BIN="$2"; shift 2 ;;
        --prefix)      INSTALL_PREFIX="$2"; shift 2 ;;
        --backup)      MAKE_BACKUP=true; shift ;;
        --no-restart)  RESTART_SERVICE=false; shift ;;
        -h|--help)
            echo "Usage: sudo $0 [--python PATH] [--prefix DIR] [--backup] [--no-restart]"
            echo ""
            echo "Options:"
            echo "  --python PATH    Python interpreter to use (default: python3)"
            echo "  --prefix DIR     Installation prefix (default: /opt/claude-code-model-gateway)"
            echo "  --backup         Keep a timestamped backup of the current venv"
            echo "  --no-restart     Do not restart the service after upgrade"
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

if [ ! -d "${INSTALL_PREFIX}" ]; then
    err "Installation directory not found: ${INSTALL_PREFIX}"
    err "Run scripts/install.sh to install the service first."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Detect new version from the source tree
if command -v python3 >/dev/null 2>&1; then
    NEW_VERSION=$(python3 -c "
import re, sys
try:
    with open('${PROJECT_DIR}/pyproject.toml') as f:
        for line in f:
            m = re.match(r'^version\s*=\s*[\"\'](.*?)[\"\']', line.strip())
            if m:
                print(m.group(1))
                sys.exit(0)
except Exception:
    pass
print('unknown')
" 2>/dev/null || echo "unknown")
else
    NEW_VERSION="unknown"
fi

# Detect currently installed version
OLD_VERSION="(unknown)"
if [ -x "${INSTALL_PREFIX}/venv/bin/claude-code-model-gateway" ]; then
    OLD_VERSION=$("${INSTALL_PREFIX}/venv/bin/claude-code-model-gateway" version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
fi

bold "Upgrading claude-code-model-gateway"
echo "  From:    ${OLD_VERSION}"
echo "  To:      ${NEW_VERSION}"
echo "  Source:  ${PROJECT_DIR}"
echo "  Prefix:  ${INSTALL_PREFIX}"
echo ""

# --- Detect init system -----------------------------------------------------
HAS_SYSTEMD=false
IS_MACOS=false
HAS_OPENRC=false

case "$(uname -s)" in
    Darwin)
        IS_MACOS=true
        ;;
    Linux)
        if command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
            HAS_SYSTEMD=true
        elif command -v rc-service >/dev/null 2>&1; then
            HAS_OPENRC=true
        fi
        ;;
esac

# --- Stop the service -------------------------------------------------------
SERVICE_WAS_RUNNING=false

if [ "${RESTART_SERVICE}" = true ]; then
    if [ "${HAS_SYSTEMD}" = true ]; then
        if systemctl is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
            SERVICE_WAS_RUNNING=true
            info "Stopping systemd service ..."
            systemctl stop "${SERVICE_NAME}" || true
            ok "Service stopped."
        fi
    elif [ "${HAS_OPENRC}" = true ]; then
        if rc-service "${SERVICE_NAME}" status >/dev/null 2>&1; then
            SERVICE_WAS_RUNNING=true
            info "Stopping OpenRC service ..."
            rc-service "${SERVICE_NAME}" stop || true
            ok "Service stopped."
        fi
    elif [ "${IS_MACOS}" = true ]; then
        if launchctl print "system/com.anthropic.${SERVICE_NAME}" >/dev/null 2>&1; then
            SERVICE_WAS_RUNNING=true
            info "Stopping launchd service ..."
            launchctl kill TERM "system/com.anthropic.${SERVICE_NAME}" || true
            sleep 2
            ok "Service stopped."
        fi
    elif [ -f "/etc/init.d/${SERVICE_NAME}" ]; then
        if service "${SERVICE_NAME}" status >/dev/null 2>&1; then
            SERVICE_WAS_RUNNING=true
            info "Stopping SysV service ..."
            service "${SERVICE_NAME}" stop || true
            ok "Service stopped."
        fi
    fi
fi

# --- Optional venv backup ---------------------------------------------------
if [ "${MAKE_BACKUP}" = true ] && [ -d "${INSTALL_PREFIX}/venv" ]; then
    BACKUP_TS=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="${INSTALL_PREFIX}/venv.bak.${BACKUP_TS}"
    info "Backing up current venv to ${BACKUP_DIR} ..."
    cp -a "${INSTALL_PREFIX}/venv" "${BACKUP_DIR}"
    ok "Backup created: ${BACKUP_DIR}"
fi

# --- Upgrade the Python package in the existing venv ------------------------
info "Upgrading application package ..."
"${INSTALL_PREFIX}/venv/bin/pip" install --upgrade --quiet "${PROJECT_DIR}"
ok "Package upgraded."

# --- Refresh the source tree (for direct module execution) ------------------
info "Refreshing source tree ..."
rm -rf "${INSTALL_PREFIX}/src"
cp -r "${PROJECT_DIR}/src" "${INSTALL_PREFIX}/src"
cp "${PROJECT_DIR}/pyproject.toml" "${INSTALL_PREFIX}/"
cp "${PROJECT_DIR}/requirements.txt" "${INSTALL_PREFIX}/" 2>/dev/null || true
ok "Source tree refreshed."

# --- Verify the upgrade -----------------------------------------------------
NEW_INSTALLED_VERSION=$("${INSTALL_PREFIX}/venv/bin/claude-code-model-gateway" version 2>/dev/null \
    | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
ok "Installed version: ${NEW_INSTALLED_VERSION}"

# --- Refresh service unit files (but NOT config) ----------------------------
if [ "${HAS_SYSTEMD}" = true ]; then
    if [ -f "${PROJECT_DIR}/service/systemd/${SERVICE_NAME}.service" ]; then
        info "Refreshing systemd unit file ..."
        cp "${PROJECT_DIR}/service/systemd/${SERVICE_NAME}.service" \
            "/etc/systemd/system/${SERVICE_NAME}.service"
        # Also update the template unit if present
        if [ -f "${PROJECT_DIR}/service/systemd/${SERVICE_NAME}@.service" ]; then
            cp "${PROJECT_DIR}/service/systemd/${SERVICE_NAME}@.service" \
                "/etc/systemd/system/${SERVICE_NAME}@.service"
        fi
        systemctl daemon-reload
        ok "systemd unit files refreshed."
    fi
elif [ "${HAS_OPENRC}" = true ]; then
    if [ -f "${PROJECT_DIR}/service/openrc/${SERVICE_NAME}" ]; then
        info "Refreshing OpenRC init script ..."
        cp "${PROJECT_DIR}/service/openrc/${SERVICE_NAME}" \
            "/etc/init.d/${SERVICE_NAME}"
        chmod 0755 "/etc/init.d/${SERVICE_NAME}"
        ok "OpenRC init script refreshed."
    fi
elif [ "${IS_MACOS}" = true ]; then
    PLIST_SRC="${PROJECT_DIR}/service/launchd/com.anthropic.${SERVICE_NAME}.plist"
    PLIST_DEST="/Library/LaunchDaemons/com.anthropic.${SERVICE_NAME}.plist"
    if [ -f "${PLIST_SRC}" ] && [ -f "${PLIST_DEST}" ]; then
        info "Refreshing launchd plist ..."
        cp "${PLIST_SRC}" "${PLIST_DEST}"
        ok "launchd plist refreshed."
    fi
elif [ -f "/etc/init.d/${SERVICE_NAME}" ]; then
    if [ -f "${PROJECT_DIR}/service/initd/${SERVICE_NAME}" ]; then
        info "Refreshing SysV init script ..."
        cp "${PROJECT_DIR}/service/initd/${SERVICE_NAME}" \
            "/etc/init.d/${SERVICE_NAME}"
        chmod 0755 "/etc/init.d/${SERVICE_NAME}"
        ok "SysV init script refreshed."
    fi
fi

# --- Update the CLI wrapper (preserves existing config paths) ---------------
if [ -f /usr/local/bin/claude-code-model-gateway ]; then
    info "Refreshing CLI wrapper at /usr/local/bin/claude-code-model-gateway ..."
    cat > /usr/local/bin/claude-code-model-gateway <<WRAPPER
#!/usr/bin/env bash
# Wrapper script for claude-code-model-gateway
if [ -f "${CONFIG_DIR}/environment" ]; then
    set -a
    . "${CONFIG_DIR}/environment"
    set +a
fi
exec "${INSTALL_PREFIX}/venv/bin/claude-code-model-gateway" "\$@"
WRAPPER
    chmod 0755 /usr/local/bin/claude-code-model-gateway
    ok "CLI wrapper updated."
fi

# --- Install any NEW config templates as .dist files (never overwrite) ------
for conf_file in "${PROJECT_DIR}/service/conf/"*; do
    fname="$(basename "${conf_file}")"
    dest="${CONFIG_DIR}/${fname}"
    if [ ! -f "${dest}" ]; then
        info "Installing new config file: ${dest}"
        cp "${conf_file}" "${dest}"
        chmod 0640 "${dest}"
    else
        # Lay down a .dist copy so the admin can diff it
        cp "${conf_file}" "${dest}.dist"
    fi
done

# --- Restart the service (unless --no-restart) ------------------------------
if [ "${RESTART_SERVICE}" = true ] && [ "${SERVICE_WAS_RUNNING}" = true ]; then
    if [ "${HAS_SYSTEMD}" = true ]; then
        info "Restarting systemd service ..."
        systemctl start "${SERVICE_NAME}"
        sleep 2
        systemctl is-active --quiet "${SERVICE_NAME}" && ok "Service restarted." || warn "Service failed to restart — check: journalctl -u ${SERVICE_NAME}"
    elif [ "${HAS_OPENRC}" = true ]; then
        info "Restarting OpenRC service ..."
        rc-service "${SERVICE_NAME}" start
        ok "Service restarted."
    elif [ "${IS_MACOS}" = true ]; then
        info "Restarting launchd service ..."
        launchctl kickstart -k "system/com.anthropic.${SERVICE_NAME}" || true
        ok "Service restarted."
    elif [ -f "/etc/init.d/${SERVICE_NAME}" ]; then
        info "Restarting SysV service ..."
        service "${SERVICE_NAME}" start
        ok "Service restarted."
    fi
elif [ "${RESTART_SERVICE}" = false ]; then
    warn "Service not restarted (--no-restart). Start manually when ready."
fi

# --- Summary ----------------------------------------------------------------
echo ""
bold "============================================================"
printf "${GREEN}Upgrade complete!${NC}\n"
bold "============================================================"
echo ""
echo "  Previous version: ${OLD_VERSION}"
echo "  New version:      ${NEW_INSTALLED_VERSION}"
echo ""
echo "Configuration files were NOT modified. To review new defaults:"
echo "  diff ${CONFIG_DIR}/environment ${CONFIG_DIR}/environment.dist"
echo "  diff ${CONFIG_DIR}/gateway.yaml ${CONFIG_DIR}/gateway.yaml.dist"
echo ""
if [ "${RESTART_SERVICE}" = true ] && [ "${SERVICE_WAS_RUNNING}" = false ]; then
    echo "The service was not running before upgrade. Start it with:"
    if [ "${HAS_SYSTEMD}" = true ]; then
        echo "  sudo systemctl start ${SERVICE_NAME}"
    elif [ "${IS_MACOS}" = true ]; then
        echo "  sudo launchctl kickstart system/com.anthropic.${SERVICE_NAME}"
    else
        echo "  sudo service ${SERVICE_NAME} start"
    fi
    echo ""
fi
