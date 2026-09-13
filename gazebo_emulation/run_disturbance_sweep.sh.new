#!/usr/bin/env bash
# Run the ROS 2 / Gazebo case study under a set of disturbance conditions
# (wind, gusts, GNSS noise, telemetry loss) and collect per-cycle rendezvous
# metrics into ../results/gazebo/<condition>/.
#
#   ./run_disturbance_sweep.sh                 # all conditions, 300 s each
#   DURATION=120 ./run_disturbance_sweep.sh calm wind6_gust2
#
# Requires the idle container from ./start.sh. Sources are synced into the
# container and rebuilt (python only, a few seconds) before the runs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONTAINER="fuel-gz-run"
DURATION="${DURATION:-300}"           # wall-clock seconds per condition (RTF 1)
OUT="${OUT:-../results/gazebo}"

# name | launch arguments
declare -A COND=(
  [calm]=""
  [wind3]="wind_speed:=3.0 wind_dir_deg:=45.0"
  [wind6_gust2]="wind_speed:=6.0 wind_dir_deg:=45.0 wind_gust:=2.0"
  [wind9_gust3]="wind_speed:=9.0 wind_dir_deg:=45.0 wind_gust:=3.0"
  [gnss05]="pos_noise:=0.5"
  [drop20]="drop_prob:=0.2"
  [combined]="wind_speed:=6.0 wind_dir_deg:=45.0 wind_gust:=2.0 pos_noise:=0.5 drop_prob:=0.1"
  # Same wind conditions with integral action in the drone position loop
  [wind6_gust2_pi]="wind_speed:=6.0 wind_dir_deg:=45.0 wind_gust:=2.0 pos_ki:=0.3"
  [wind9_gust3_pi]="wind_speed:=9.0 wind_dir_deg:=45.0 wind_gust:=3.0 pos_ki:=0.3"
  [combined_pi]="wind_speed:=6.0 wind_dir_deg:=45.0 wind_gust:=2.0 pos_noise:=0.5 drop_prob:=0.1 pos_ki:=0.3"
)
ORDER=(calm wind3 wind6_gust2 wind9_gust3 gnss05 drop20 combined wind6_gust2_pi wind9_gust3_pi combined_pi)
if [ $# -gt 0 ]; then ORDER=("$@"); fi

if [ -z "$(docker ps -q -f name="^${CONTAINER}$")" ]; then
  echo "container ${CONTAINER} is not running; run ./start.sh first" >&2; exit 1
fi

echo "syncing sources and rebuilding in ${CONTAINER}..."
docker cp ws/src/fuel_rendezvous "${CONTAINER}:/ws/src/" \
  && docker exec "${CONTAINER}" bash -c '. /opt/ros/jazzy/setup.sh && cd /ws && colcon build --symlink-install > /tmp/build.log 2>&1 && tail -1 /tmp/build.log' \
  || { echo "build failed"; docker exec "${CONTAINER}" tail -20 /tmp/build.log; exit 1; }

for name in "${ORDER[@]}"; do
  args="${COND[$name]-}"
  echo "=== ${name}: ${DURATION}s  [${args}]"
  ./stop.sh > /dev/null 2>&1
  docker exec "${CONTAINER}" bash -c "rm -f /tmp/rendezvous_metrics_${name}_*.csv /tmp/wind_${name}.csv /tmp/launch_${name}.log"
  docker exec "${CONTAINER}" bash -c ". /opt/ros/jazzy/setup.sh && . /ws/install/setup.sh && \
      timeout --signal=INT --kill-after=15 ${DURATION}s ros2 launch fuel_rendezvous rendezvous.launch.py \
      headless:=true run_tag:=${name} ${args} > /tmp/launch_${name}.log 2>&1"
  ./stop.sh > /dev/null 2>&1
  mkdir -p "${OUT}/${name}"
  docker exec "${CONTAINER}" bash -c "ls /tmp/rendezvous_metrics_${name}_*.csv /tmp/wind_${name}.csv 2>/dev/null" | while read -r f; do
    docker cp "${CONTAINER}:${f}" "${OUT}/${name}/" > /dev/null
  done
  docker cp "${CONTAINER}:/tmp/launch_${name}.log" "${OUT}/${name}/launch.log" > /dev/null
  cycles=$(cat "${OUT}/${name}"/rendezvous_metrics_*.csv 2>/dev/null | grep -vc '^cycle' || true)
  echo "    cycles: ${cycles}  dispatches: $(grep -c DISPATCH "${OUT}/${name}/launch.log")  strandings: $(grep -c STRANDED "${OUT}/${name}/launch.log")"
done
echo "done -> ${OUT}"
