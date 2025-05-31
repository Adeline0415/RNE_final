#!/bin/bash

source "./utils.sh"
main "./docker/compose/docker-compose_ydlidar.yml" "./docker/compose/docker-compose_slam.yml"
