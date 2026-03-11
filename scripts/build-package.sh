#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build-package.sh — Build a self-contained distributable archive
# ---------------------------------------------------------------------------
# Creates a versioned .tar.gz that bundles:
#   - The Python source package (sdist wheel)
#   - All service/startup files (systemd, initd, openrc, launchd)
#   - Install / uninstall / update / healthcheck scripts
#   - Configuration templates
#
# The resulting archive can be copied to any target system and installed with:
#   tar -xzf claude-code-model-gateway-<version>.tar.gz
#   sudo ./claude-code-model-gateway-<version>/scripts/install.sh
#
# Usage:
#   ./scripts/build-package.sh                  # Build with auto-detected version
#   ./scripts/build-package.sh --version 1.2.3  # Override version
#   ./scripts/build-package.sh --output /tmp     # Custom output directory
#   ./scripts/build-package.sh --wheel           # Also build Python wheel/sdist
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Colours ----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()    { printf "${BLUE}[INFO]${NC}  %s\n" "$*"; }
ok()      { printf "${GREEN}[ OK ]${NC}  %s\n" "$*"; }
warn()    { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()     { printf "${RED}[ERR]${NC}  %s\n" "$*" >&2; }
bold()    { printf "${BOLD}%s${NC}\n" "$*"; }

# --- Defaults ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/dist"
BUILD_WHEEL=false
VERSION=""

# --- Argument parsing -------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --version)  VERSION="$2"; shift 2 ;;
        --output)   OUTPUT_DIR="$2"; shift 2 ;;
        --wheel)    BUILD_WHEEL=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--version X.Y.Z] [--output DIR] [--wheel]"
            echo ""
            echo "Options:"
            echo "  --version X.Y.Z  Override version string (default: from pyproject.toml)"
            echo "  --output DIR     Output directory for the archive (default: dist/)"
            echo "  --wheel          Also build the Python sdist/wheel packages"
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Auto-detect version from pyproject.toml --------------------------------
if [ -z "${VERSION}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        VERSION=$(python3 -c "
import re, sys
with open('${PROJECT_DIR}/pyproject.toml') as f:
    for line in f:
        m = re.match(r'^version\s*=\s*[\"\'](.*?)[\"\']', line.strip())
        if m:
            print(m.group(1))
            sys.exit(0)
print('0.0.0')
" 2>/dev/null || echo "0.0.0")
    else
        VERSION="0.0.0"
    fi
fi

PACKAGE_NAME="claude-code-model-gateway-${VERSION}"
ARCHIVE_NAME="${PACKAGE_NAME}.tar.gz"

bold "Building distributable package: ${PACKAGE_NAME}"
echo "  Source:  ${PROJECT_DIR}"
echo "  Output:  ${OUTPUT_DIR}/${ARCHIVE_NAME}"
echo ""

# --- Build Python wheel/sdist (optional) ------------------------------------
if [ "${BUILD_WHEEL}" = true ]; then
    info "Building Python sdist and wheel ..."
    if ! python3 -m build --outdir "${OUTPUT_DIR}/python" "${PROJECT_DIR}" 2>&1; then
        warn "Python build failed — continuing without wheel."
    else
        ok "Python packages built in ${OUTPUT_DIR}/python/"
    fi
fi

# --- Stage the package tree -------------------------------------------------
STAGE_DIR="$(mktemp -d)"
STAGE="${STAGE_DIR}/${PACKAGE_NAME}"
mkdir -p "${STAGE}"

info "Staging package contents ..."

# Core source
mkdir -p "${STAGE}/src"
cp -r "${PROJECT_DIR}/src/"*       "${STAGE}/src/"
cp    "${PROJECT_DIR}/pyproject.toml" "${STAGE}/"
cp    "${PROJECT_DIR}/requirements.txt" "${STAGE}/" 2>/dev/null || true

# README
cp "${PROJECT_DIR}/README.md" "${STAGE}/" 2>/dev/null || true

# Scripts
mkdir -p "${STAGE}/scripts"
for script in install.sh uninstall.sh healthcheck.sh; do
    if [ -f "${PROJECT_DIR}/scripts/${script}" ]; then
        cp "${PROJECT_DIR}/scripts/${script}" "${STAGE}/scripts/"
        chmod 0755 "${STAGE}/scripts/${script}"
    fi
done
# Include update.sh if present
if [ -f "${PROJECT_DIR}/scripts/update.sh" ]; then
    cp "${PROJECT_DIR}/scripts/update.sh" "${STAGE}/scripts/"
    chmod 0755 "${STAGE}/scripts/update.sh"
fi

# Service files
mkdir -p "${STAGE}/service/systemd"
mkdir -p "${STAGE}/service/initd"
mkdir -p "${STAGE}/service/openrc"
mkdir -p "${STAGE}/service/launchd"
mkdir -p "${STAGE}/service/conf"

# systemd
for f in "${PROJECT_DIR}/service/systemd/"*; do
    [ -f "$f" ] && cp "$f" "${STAGE}/service/systemd/"
done

# SysV init
for f in "${PROJECT_DIR}/service/initd/"*; do
    [ -f "$f" ] && cp "$f" "${STAGE}/service/initd/" && chmod 0755 "${STAGE}/service/initd/$(basename "$f")"
done

# OpenRC
for f in "${PROJECT_DIR}/service/openrc/"*; do
    [ -f "$f" ] && cp "$f" "${STAGE}/service/openrc/" && chmod 0755 "${STAGE}/service/openrc/$(basename "$f")"
done

# launchd (macOS)
for f in "${PROJECT_DIR}/service/launchd/"*; do
    [ -f "$f" ] && cp "$f" "${STAGE}/service/launchd/"
done

# Config templates
for f in "${PROJECT_DIR}/service/conf/"*; do
    [ -f "$f" ] && cp "$f" "${STAGE}/service/conf/"
done

# Makefile
cp "${PROJECT_DIR}/Makefile" "${STAGE}/" 2>/dev/null || true

# Docker files
cp "${PROJECT_DIR}/Dockerfile"        "${STAGE}/" 2>/dev/null || true
cp "${PROJECT_DIR}/docker-compose.yaml" "${STAGE}/" 2>/dev/null || true

# Stamp version into a plain text file inside the archive
echo "${VERSION}" > "${STAGE}/VERSION"

# Write a quick-start README for the archive
cat > "${STAGE}/INSTALL.md" <<INSTALL_README
# Installation — claude-code-model-gateway ${VERSION}

## Quick Start (Linux — systemd or SysV)

\`\`\`bash
# 1. Extract the archive
tar -xzf claude-code-model-gateway-${VERSION}.tar.gz
cd claude-code-model-gateway-${VERSION}

# 2. Install (auto-detects systemd vs SysV init)
sudo ./scripts/install.sh

# 3. Set your API key
sudo editor /etc/claude-code-model-gateway/environment

# 4. Start the service
sudo systemctl start claude-code-model-gateway    # systemd
# or
sudo service claude-code-model-gateway start       # SysV

# 5. Verify
./scripts/healthcheck.sh
\`\`\`

## macOS (launchd)

\`\`\`bash
sudo ./scripts/install.sh --launchd
\`\`\`

## Alpine Linux / OpenRC

\`\`\`bash
sudo ./scripts/install.sh --openrc
\`\`\`

## Docker

\`\`\`bash
docker compose up -d
\`\`\`

## Upgrade existing installation

\`\`\`bash
sudo ./scripts/update.sh
\`\`\`

## Uninstall

\`\`\`bash
sudo ./scripts/uninstall.sh           # keep config/logs
sudo ./scripts/uninstall.sh --purge   # remove everything
\`\`\`

## Documentation

See README.md for full configuration reference.
INSTALL_README

ok "Package staged at ${STAGE}"

# --- Create the archive -----------------------------------------------------
info "Creating archive ${ARCHIVE_NAME} ..."
mkdir -p "${OUTPUT_DIR}"
(cd "${STAGE_DIR}" && tar -czf "${OUTPUT_DIR}/${ARCHIVE_NAME}" "${PACKAGE_NAME}")
ok "Archive created: ${OUTPUT_DIR}/${ARCHIVE_NAME}"

# --- Compute checksum -------------------------------------------------------
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${OUTPUT_DIR}/${ARCHIVE_NAME}" > "${OUTPUT_DIR}/${ARCHIVE_NAME}.sha256"
    ok "SHA-256 checksum: ${OUTPUT_DIR}/${ARCHIVE_NAME}.sha256"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${OUTPUT_DIR}/${ARCHIVE_NAME}" > "${OUTPUT_DIR}/${ARCHIVE_NAME}.sha256"
    ok "SHA-256 checksum: ${OUTPUT_DIR}/${ARCHIVE_NAME}.sha256"
fi

# --- Cleanup staging area ---------------------------------------------------
rm -rf "${STAGE_DIR}"

# --- Summary ----------------------------------------------------------------
echo ""
bold "============================================================"
printf "${GREEN}Package ready!${NC}\n"
bold "============================================================"
echo ""
echo "  Archive:   ${OUTPUT_DIR}/${ARCHIVE_NAME}"
ARCHIVE_SIZE=$(du -sh "${OUTPUT_DIR}/${ARCHIVE_NAME}" 2>/dev/null | cut -f1 || echo "?")
echo "  Size:      ${ARCHIVE_SIZE}"
echo "  Version:   ${VERSION}"
echo ""
echo "Distribute and install on any target system:"
echo "  scp ${OUTPUT_DIR}/${ARCHIVE_NAME} user@target:/tmp/"
echo "  ssh user@target 'cd /tmp && tar -xzf ${ARCHIVE_NAME} && sudo ./${PACKAGE_NAME}/scripts/install.sh'"
echo ""
