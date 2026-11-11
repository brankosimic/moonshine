#!/usr/bin/env bash
#
# Install moonshine from a local source build.
#
# Companion to moonshine-install.sh, which installs a prebuilt release tarball.
# This script builds the workspace and installs the resulting artifacts to the
# same system paths as nfpm.yaml, so an existing moonshine@<user>.service keeps
# working with no changes.
#
# Intended for development: it tracks whatever is checked out in the working
# tree, not a tagged release. Files installed this way are not owned by pacman,
# so package updates will neither replace nor remove them.
#
# Usage: moonshine-build-install.sh [options]
#   --user USER   Service instance to restart (default: current user)
#   --no-restart  Install without restarting the service
#   --debug       Build the debug profile instead of release
#   --non-interactive  Never prompt for a sudo password (needs NOPASSWD sudo)
#   --help        Show this message
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RESTART=true
PROFILE="release"
TARGET_USER=""
NONINTERACTIVE=false
SUDO=(sudo)

print_help() {
  sed -n '3,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) TARGET_USER="$2"; shift 2 ;;
    --no-restart) RESTART=false; shift ;;
    --debug) PROFILE="debug"; shift ;;
    --non-interactive) NONINTERACTIVE=true; shift ;;
    --help|-h) print_help ;;
    *) echo "unknown option: $1" >&2; print_help ;;
  esac
done

if [[ $EUID -eq 0 ]]; then
  echo "do not run with sudo. Run as your normal user." >&2
  exit 1
fi

if ! command -v sudo &>/dev/null; then
  echo "sudo not found" >&2
  exit 1
fi

if $NONINTERACTIVE; then
  SUDO=(sudo -n)
  if ! sudo -n /usr/bin/systemctl daemon-reload 2>/dev/null; then
    echo "sudo requires a password; --non-interactive needs NOPASSWD for the install commands" >&2
    exit 1
  fi
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "moonshine only supports x86_64" >&2
  exit 1
fi

TARGET_USER="${TARGET_USER:-$USER}"
if ! id -u "$TARGET_USER" &>/dev/null; then
  echo "user '$TARGET_USER' does not exist" >&2
  exit 1
fi

if ! command -v cargo &>/dev/null; then
  echo "cargo not found in PATH; install a Rust toolchain first" >&2
  exit 1
fi

BIN_SRC="${REPO_ROOT}/target/${PROFILE}/moonshine"

echo ":: Building moonshine (${PROFILE})"
# `debug` is cargo's default `dev` profile and has no flag; `release` does.
CARGO_PROFILE_FLAG=()
if [[ "$PROFILE" == "release" ]]; then
  CARGO_PROFILE_FLAG=(--release)
fi
cargo build --manifest-path "${REPO_ROOT}/Cargo.toml" "${CARGO_PROFILE_FLAG[@]}" -p moonshine

for artifact in "$BIN_SRC"; do
  if [[ ! -f "$artifact" ]]; then
    echo "build did not produce ${artifact}" >&2
    exit 1
  fi
done

echo ":: Built"
echo "   ${BIN_SRC}"

# Destination paths mirror nfpm.yaml: the shipped moonshine@.service hardcodes
# /usr/bin/start-moonshine.sh. (The old WSI Vulkan layer is gone — upstream
# replaced moonshine-wsi with ferriscope, so there is no layer to install.)
echo ":: Installing (sudo)"
echo ""

run_root() {
  printf '  %s\n' "$*"
  "${SUDO[@]}" "$@"
}

run_root_optional() {
  printf '  %s\n' "$*"
  "${SUDO[@]}" "$@" || true
}

run_root /usr/bin/install -Dm755 "$BIN_SRC" /usr/bin/moonshine
run_root /usr/bin/install -Dm755 "$REPO_ROOT/dist/start-moonshine.sh" /usr/bin/start-moonshine.sh
run_root /usr/bin/install -Dm644 "$REPO_ROOT/dist/moonshine@.service" /usr/lib/systemd/system/moonshine@.service
run_root /usr/bin/install -Dm644 "$REPO_ROOT/dist/60-moonshine.rules" /usr/lib/udev/rules.d/60-moonshine.rules
run_root /usr/bin/install -Dm644 "$REPO_ROOT/dist/moonshine-modules.conf" /usr/lib/modules-load.d/moonshine.conf
run_root /usr/bin/install -Dm644 "$REPO_ROOT/dist/moonshine-sysusers.conf" /usr/lib/sysusers.d/moonshine.conf
run_root /usr/bin/install -Dm644 "$REPO_ROOT/dist/50-moonshine-inhibit-sleep.rules" /usr/share/polkit-1/rules.d/50-moonshine-inhibit-sleep.rules
run_root /usr/bin/install -Dm644 "$REPO_ROOT/LICENSE" /usr/share/licenses/moonshine/LICENSE
run_root /usr/bin/systemctl daemon-reload
# Creates the 'moonshine' group referenced by the sleep-inhibit polkit rule.
run_root_optional /usr/bin/systemd-sysusers
run_root_optional /usr/bin/udevadm control --reload
run_root_optional /usr/bin/udevadm trigger
# Virtual input backends used by inputtino, so no reboot is needed.
run_root_optional /usr/bin/modprobe uinput
run_root_optional /usr/bin/modprobe uhid
run_root_optional /usr/bin/systemctl reload-or-restart polkit.service

if $RESTART; then
  run_root /usr/bin/systemctl restart "moonshine@${TARGET_USER}"
fi

echo ""
echo "moonshine ${PROFILE} build installed"
echo "  status  systemctl status moonshine@${TARGET_USER}"
echo "  config  /home/${TARGET_USER}/.config/moonshine/config.toml"
echo ""
echo "note: these files are not owned by pacman, so 'paru -Syu' will not"
echo "      update or remove them. Re-run this script after rebuilding."
