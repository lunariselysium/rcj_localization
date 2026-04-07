#!/usr/bin/env bash

set -eo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${PACKAGE_DIR}/../.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
PACKAGE_NAME="rcj_localization"

if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "ROS setup file not found: ${ROS_SETUP}" >&2
    exit 1
fi

set +u
source "${ROS_SETUP}"
set -u

cd "${WORKSPACE_DIR}"
echo "[run_rcj] Building ${PACKAGE_NAME} in ${WORKSPACE_DIR}"
colcon build --packages-select "${PACKAGE_NAME}"

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
