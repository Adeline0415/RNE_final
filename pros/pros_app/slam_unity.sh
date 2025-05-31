#!/bin/bash

source "./utils.sh"
main "./docker/compose/docker-compose_rplidar_unity.yml" "./docker/compose/docker-compose_slam_unity.yml"
