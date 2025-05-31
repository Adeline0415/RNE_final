#!/bin/bash

source "./utils.sh"
main "./docker/compose/docker-compose_rplidar_unity.yml" "./docker/compose/docker-compose_localization_unity.yml" "./docker/compose/docker-compose_navigation_unity.yml"
