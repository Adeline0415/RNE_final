@echo off

docker "compose" "-f" "%CD%\docker\compose\docker-compose_rplidar_unity.yml" "up" "-d"
docker "compose" "-f" "%CD%\docker\compose\docker-compose_slam_unity.yml" "up" "-d"
trap "cleanup" "SIGINT"
echo "Press Ctrl+C to stop..."
echo "Listening to docker-compose_slam_unity.yml logs. Press Ctrl+C to stop..."
docker "compose" "-f" "%CD%\docker\compose\docker-compose_slam_unity.yml" "logs" "-f"
wait

EXIT /B %ERRORLEVEL%

:cleanup
echo "Shutting down docker compose services..."
REM UNKNOWN: {"type":"Pipeline","commands":[{"type":"Command","name":{"text":"docker","type":"Word"},"suffix":[{"text":"ps","type":"Word"},{"text":"-aq","type":"Word"},{"text":"--filter","type":"Word"},{"text":"name=scripts*","type":"Word"}]},{"type":"Command","name":{"text":"xargs","type":"Word"},"suffix":[{"text":"docker","type":"Word"},{"text":"rm","type":"Word"},{"text":"-f","type":"Word"}]}]}
exit "0"
EXIT /B 0
