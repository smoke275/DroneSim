FROM osrf/ros:jazzy-desktop

# Gazebo Harmonic (modern gz-sim) from the OSRF repository + ROS<->Gazebo bridge
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg lsb-release \
    && curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    gz-harmonic \
    ros-jazzy-ros-gz \
    && rm -rf /var/lib/apt/lists/*

# Pre-fetch the X3 quadcopter model from the Gazebo Fuel library so runtime
# needs no network. (Formerly fuel.ignitionrobotics.org.)
RUN gz fuel download -u "https://fuel.gazebosim.org/1.0/OpenRobotics/models/X3 UAV" || true

# Confine DDS to this container: without this, ROS nodes in sibling
# containers on the docker bridge discover each other and cross-talk.
ENV ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

WORKDIR /ws
COPY ws/src /ws/src
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

RUN echo '. /opt/ros/jazzy/setup.sh && . /ws/install/setup.sh' >> /root/.bashrc

CMD ["bash"]
