#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE_NAME="dronesim"
CONTAINER_NAME="dronesim-run"

echo "Building image..."
docker build -t "$IMAGE_NAME" .

echo "Allowing local Docker containers to access the X server..."
xhost +local:docker >/dev/null

# Start the container idle if it isn't already running; the app is NOT
# launched automatically — run `python main.py` inside the container.
if [ -z "$(docker ps -q -f name="^${CONTAINER_NAME}$")" ]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    echo "Starting idle container '$CONTAINER_NAME'..."
    docker run -d --name "$CONTAINER_NAME" \
      -e DISPLAY="$DISPLAY" \
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
      -v "$PWD":/app \
      --ipc=host \
      -e QT_XCB_GL_INTEGRATION=none \
      -e LIBGL_ALWAYS_SOFTWARE=1 \
      "$IMAGE_NAME" sleep infinity >/dev/null
else
    echo "Container '$CONTAINER_NAME' is already running."
fi

echo
echo "Inside the container, launch the sim with:  python main.py"
echo "(repo is live-mounted at /app — code edits apply without rebuilding)"
echo

if [ -t 0 ]; then
    exec docker exec -it "$CONTAINER_NAME" bash
else
    echo "Attach with:  docker exec -it $CONTAINER_NAME bash"
fi
