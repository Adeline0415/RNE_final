# PROS Application

Authors:

- 陳麒麟
- 曾裕翔
- 林庭琮
- 鍾博丞

Advising professor:

- 蘇文鈺



## Workflow Diagram

![workflow_diagram](./img/workflow_diagram.png)



## Feature

This project contains the following 3 features shown above.

- SLAM
  - SLAM for real car
  - SLAM for Unity

- Store SLAM map
- Localization and Navigation
  - Localization and Navigation for real car
  - Localization and Navigation for Unity

- Depth Camera
  - Astra camera
  - Dabai camera




## Get Started

Execute `control.sh` to start.

- Main menu
  - Enter number to execute the script.
  - Enter `s` to show running processes.
  - Enter `d` to shutdown all child processes.
  - Enter `q` to quit the control process.
- Running process state
  - You will enter this state after executing scripts.
  - Single press `b` to return to the main menu without terminating the script.
  - Single press `q` to terminate the script and return to the main menu.



## Dev Note

- Compatible with `docker-compose` and `docker compose`.

- The destructor for each unit shell script is written in `utils.sh`, which is triggered by SIGINT.

- The `control.sh` triggers each destructor of the child process by sending SIGINT to them.

- The command for removing items in the bash array has a known issue. It will remain an empty string after removing the item.

  ```bash
  # Remove the process from child_pids array
  child_pids=("${child_pids[@]/$script_pid}")
  ```

- The code structure

  ```
  control.sh ---> camera_astra.sh			---> utils.sh
              |-> camera_dabai.sh			-|
              |-> localization_unity.sh	-|
              |-> localization_ydlidar.sh	-|
              |-> localization.sh			-|
              |-> slam_unity.sh			-|
              |-> slam_ydlidar.sh			-|
              |-> slam.sh					-|
              |-> store_map.sh			-|
  ```

  

## 設定 USB Rule

將 lidar、ESP32 的 USB設定成讀取以下名稱 (可查閱 udev 相關資料)

- 前面的 esp32: usb_front_wheel
- 後面的 esp32: usb_rear_wheel
- lidar: usb_lidar
- [Reference](https://inegm.medium.com/persistent-names-for-usb-serial-devices-in-linux-dev-ttyusbx-dev-custom-name-fd49b5db9af1)



## 設定 Lidar Baud Rate

每個 lidar 都有屬於 baud rate 需要設定，若執行 slam 的 compose 檔案發現 lidar 不能啟動時，請看 lidar 型號查詢該 lidar 的 baud rate 是多少

ex. [SLAMTEC_rplidarkit_usermanual_S2_v1.1_en.pdf](https://bucket-download.slamtec.com/1d6d308d60e27da6c910177b06370a1fe901defd/SLAMTEC_rplidarkit_usermanual_S2_v1.1_en.pdf)



## 啟動 Compose 檔案順序

```bash
docker-compose -f <your-compose-file.yml> up
```

1. 啟動 pros_app :

   1. [docker-compose_slam.yml](https://github.com/otischung/pros_app/blob/main/docker-compose_slam.yml) (做並使用 fox glove 的 rosbridge 連接觀看點雲圖)

2. 啟動 pros_car :

   1. 根據車型啟動 [car_control_2.sh](https://github.com/otischung/pros_car/blob/main/car_control_2.sh) or [car_control_4.sh](https://github.com/otischung/pros_car/blob/main/car_control_2.sh) 腳本
   2. 控制車子掃描地圖 (可用fox glove觀察)

3. 啟動 pros_app:

   1. [docker-compose_store_map.yml](https://github.com/otischung/pros_app/blob/main/docker-compose_store_map.yml)，將目前 slam 製作成地圖

4. 關閉 pros_app 的 [docker-compose_slam.yml](https://github.com/otischung/pros_app/blob/main/docker-compose_slam.yml)，並開啟 pros_app (因為 slam 和 localization 不能同時開) :

   1. [docker-compose_localization.yml](https://github.com/otischung/pros_app/blob/main/docker-compose_localization.yml)，並用 fox glove 查看目前車子的位置並做定位後修正目前點雲圖

   [https://inegm.medium.com/persistent-names-for-usb-serial-devices-in-linux-dev-ttyusbx-dev-custom-name-fd49b5db9af1](https://inegm.medium.com/persistent-names-for-usb-serial-devices-in-linux-dev-ttyusbx-dev-custom-name-fd49b5db9af1)



## Docker Bridge Network (doc written at 2024.02.29)

使用 port 對應會造成網路嚴重延遲，為了解決這個問題，我們新增了自定義的 [bridge network](https://godleon.github.io/blog/Docker/docker-network-bridge/)

```yaml
networks:
  my_bridge_network:
    driver: bridge
```

執行時，若機器上的 docker 原本沒有 `pros_app_my_bridge_network`，則會自動新增

若是需要在執行 docker-compose 前就先行使用此 bridge network，亦可自行新增

```bash
docker network create --driver bridge pros_app_my_bridge_network
```



## ROS2 Image Transport Plugins

我們使用了 `ros-humble-image-transport-plugins ros-humble-theora-image-transport ros-humble-compressed-depth-image-transport ros-humble-compressed-image-transport` 來實現影像壓縮，這些可以從 `apt` 進行安裝

原本傳送 640\*480@30Hz 的原始影像，有 640\*480\*30Hz\*8bits\*3channels/1024/1024 = 210.9375 Mbps，經由影像壓縮之後，已經成功壓到 24~32Mbps 左右，壓縮率為驚人的 15.17%

在 Linux 系統中，可以使用 `bmon` 得知網路頻寬流量

這個 node 是讀取 `/camera` 的 topic，經處理後 redirect 到 `/out` 這個新的 topic，使用者只須從 rviz 或 foxglove 選取壓縮之後的 topic 即可