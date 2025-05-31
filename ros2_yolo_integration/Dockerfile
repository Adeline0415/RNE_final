# 使用 ultralytics 的 Jetson JetPack 6 映像作為基底
FROM ultralytics/ultralytics:latest-jetson-jetpack6

# 設定環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# 更新套件並安裝基礎依賴
RUN apt-get update && apt-get install -y \
    software-properties-common \
    locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 移除可能已安裝的舊版 OpenCV 開發包，防止衝突
RUN apt-get remove -y '*opencv*' || true

RUN apt-get update && apt-get install -y curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list

# 安裝 ROS 2 Humble 基本工具和依賴
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-argcomplete \
    --fix-missing \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 安裝 OpenCV 和 Boost 的依賴項
RUN apt-get update && apt-get install -y \
    python3-opencv \
    libopencv-dev \
    libboost-python-dev \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 初始化 rosdep
RUN rosdep init && rosdep update

# 設置工作空間並克隆特定版本的 cv_bridge 源碼（humble 分支）
WORKDIR /workspace
RUN mkdir -p ~/workspace/src && cd ~/workspace/src && \
    git clone -b humble https://github.com/ros-perception/vision_opencv.git

# 設定 SHELL 為 /bin/bash
SHELL ["/bin/bash", "-c"]

# 使用 colcon 編譯 cv_bridge
RUN cd ~/workspace && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --packages-select cv_bridge

# 設定環境變數以啟用 ROS 2 和 cv_bridge
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source ~/workspace/install/setup.bash" >> ~/.bashrc

# **升級 setuptools，確保與 Humble 相容**
RUN python3 -m pip install --no-cache-dir --upgrade setuptools==58.2.0

# **新增 `r` 指令，讓使用者輸入 `r` 就能執行 colcon build**
RUN echo 'alias r="cd /workspace && colcon build && source ./install/setup.bash"' >> ~/.bashrc

# 確保容器啟動時也降級 setuptools
CMD ["bash", "-c", "python3 -m pip install --upgrade setuptools==58.2.0 && bash"]
