#!/usr/bin/env bash
# Stop the rendezvous mission (and optionally the whole container).
#   ./stop.sh            # stop the running mission/Gazebo, keep container idle
#   ./stop.sh --all      # also remove the container
set -uo pipefail

CONTAINER_NAME="fuel-gz-run"

if [ -n "$(docker ps -q -f name="^${CONTAINER_NAME}$")" ]; then
    docker exec "$CONTAINER_NAME" bash -c \
        'pkill -9 -f "ros2 launch"; pkill -9 -f parameter_bridge; pkill -9 -f fuel_rendezvous; pkill -9 ruby; pkill -9 -f "gz sim"' 2>/dev/null
    echo "Mission and Gazebo stopped; container '$CONTAINER_NAME' still idle."
else
    echo "Container '$CONTAINER_NAME' is not running."
fi

if [ "${1:-}" = "--all" ]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 && echo "Container removed."
fi
