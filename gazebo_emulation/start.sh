#!/usr/bin/env bash
# Build and start the ROS 2 Jazzy + Gazebo Harmonic rendezvous emulation.
# Launches an idle container with X11 forwarded; run the mission inside with:
#   ros2 launch fuel_rendezvous rendezvous.launch.py            # with GUI
#   ros2 launch fuel_rendezvous rendezvous.launch.py headless:=true
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE_NAME="fuel-gz"
CONTAINER_NAME="fuel-gz-run"

echo "Building image..."
docker build -t "$IMAGE_NAME" .

echo "Allowing local Docker containers to access the X server..."
xhost +local:docker >/dev/null

if [ -z "$(docker ps -q -f name="^${CONTAINER_NAME}$")" ]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    echo "Starting idle container '$CONTAINER_NAME'..."
    docker run -d --name "$CONTAINER_NAME" \
      -e DISPLAY="$DISPLAY" \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      --ipc=host \
      -e LIBGL_ALWAYS_SOFTWARE=1 \
      -e QT_X11_NO_MITSHM=1 \
      "$IMAGE_NAME" sleep infinity >/dev/null
else
    echo "Container '$CONTAINER_NAME' is already running."
fi

echo
echo "Inside the container:"
echo "  ros2 launch fuel_rendezvous rendezvous.launch.py               # GUI"
echo "  ros2 launch fuel_rendezvous rendezvous.launch.py headless:=true"
echo

if [ -t 0 ]; then
    exec docker exec -it "$CONTAINER_NAME" bash
else
    echo "Attach with:  docker exec -it $CONTAINER_NAME bash"
fi
