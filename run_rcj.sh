#!/usr/bin/env bash

set -eo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${PACKAGE_DIR}/../.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
PACKAGE_NAME="rcj_localization"
PACKAGE_IN_WORKSPACE="${WORKSPACE_DIR}/src/${PACKAGE_NAME}"
NESTED_BUILD_DIR="${PACKAGE_DIR}/build"
NESTED_INSTALL_DIR="${PACKAGE_DIR}/install"
NESTED_LOG_DIR="${PACKAGE_DIR}/log"

if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "ROS setup file not found: ${ROS_SETUP}" >&2
    exit 1
fi

if [[ "${PACKAGE_DIR}" != "${PACKAGE_IN_WORKSPACE}" ]]; then
    echo "Package directory does not match expected workspace layout." >&2
    echo "Expected: ${PACKAGE_IN_WORKSPACE}" >&2
    echo "Actual:   ${PACKAGE_DIR}" >&2
    exit 1
fi

if [[ ! -d "${WORKSPACE_DIR}/src" ]]; then
    echo "Workspace src directory not found: ${WORKSPACE_DIR}/src" >&2
    exit 1
fi

if [[ -d "${NESTED_BUILD_DIR}" || -d "${NESTED_INSTALL_DIR}" || -d "${NESTED_LOG_DIR}" ]]; then
    echo "[run_rcj] Warning: found nested colcon artifacts inside the package directory." >&2
    echo "[run_rcj] These should be removed to avoid sourcing the wrong workspace:" >&2
    [[ -d "${NESTED_BUILD_DIR}" ]] && echo "  ${NESTED_BUILD_DIR}" >&2
    [[ -d "${NESTED_INSTALL_DIR}" ]] && echo "  ${NESTED_INSTALL_DIR}" >&2
    [[ -d "${NESTED_LOG_DIR}" ]] && echo "  ${NESTED_LOG_DIR}" >&2
fi

set +u
source "${ROS_SETUP}"
set -u

cd "${WORKSPACE_DIR}"
echo "[run_rcj] Building ${PACKAGE_NAME} from workspace root ${WORKSPACE_DIR}"
colcon build --symlink-install --packages-select "${PACKAGE_NAME}"

set +u
source "${WORKSPACE_DIR}/install/setup.bash"
set -u

if [[ $# -gt 0 ]]; then
    echo "[run_rcj] Running: $*"
    exec "$@"
fi

echo "[run_rcj] Build complete."
echo "[run_rcj] ROS and workspace were sourced inside this script only."
echo "[run_rcj] Current shell will continue as usual after the script exits."
