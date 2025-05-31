# From PROS dev image
FROM ghcr.io/otischung/pros_ai_image:latest
ENV ROS2_WS /workspaces
ENV ROS_DOMAIN_ID=1

COPY ./requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt
RUN apt-get update && apt-get install -y sl

# TODO install dependencies 
# RUN apt install -y packages_to_install

# Build your ROS packages
# We use mount instead of copy
# COPY ./src ${ROS2_WS}/src
# WORKDIR ${ROS2_WS}
# RUN . /opt/ros/humble/setup.sh && colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=Release
# You could decide wheather to delete source code

ENTRYPOINT [ "/ros_entrypoint.bash" ]
CMD ["bash","-l"]
