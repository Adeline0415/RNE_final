FROM registry.screamtrumpet.csie.ncku.edu.tw/screamlab/pros_base_image:latest
ENV ROS2_WS /workspaces
ENV ROS_DOMAIN_ID=1
ENV ROS_DISTRO=humble
ARG THREADS=4
ARG TARGETPLATFORM

SHELL ["/bin/bash", "-c"]

##### Copy Source Code #####
COPY ./src /tmp

##### Environment Settings #####
WORKDIR ${ROS2_WS}

# System Upgrade
RUN apt update && \
    apt upgrade -y && \
    apt autoremove -y && \
    apt autoclean -y && \

    pip3 install --no-cache-dir --upgrade pip

##### colcon Installation #####
# Copy Source Code
RUN mkdir -p ${ROS2_WS}/src && \
    mv /tmp/src/* ${ROS2_WS}/src && \

# Bootstrap rosdep and setup colcon mixin and metadata ###
    rosdep update --rosdistro $ROS_DISTRO && \
    colcon mixin update && \
    colcon metadata update && \

# Install the system dependencies for all ROS packages located in the `src` directory.
    rosdep install -q -y -r --from-paths src --ignore-src

### Lidar Transformer Installation ###
RUN colcon build --packages-select unity_transformer --symlink-install --parallel-workers ${THREADS} --mixin release && \

##### Post-Settings #####
# Clear tmp and cache
    rm -rf /tmp/* && \
    rm -rf /temp/* && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/ros_entrypoint.bash"]
CMD ["bash", "-l"]
