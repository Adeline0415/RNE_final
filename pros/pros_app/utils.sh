#!/bin/bash

cleanup() {
    local scripts=("$@")  # Accept all arguments as an array
    echo "Shutting down docker compose services..."
    for script in "${scripts[@]}"; do
        echo "Stopping services for $script..."
        $DOCKER_COMPOSE_COMMAND -f "$script" down --timeout 0
    done
    exit 0
}

main() {
    # Declare an array to hold all scripts to be cleaned up
    SCRIPTS=()

    # Determine which docker compose command to use
    if command -v docker-compose &> /dev/null
    then
        DOCKER_COMPOSE_COMMAND="docker-compose"
    elif docker compose version &> /dev/null
    then
        DOCKER_COMPOSE_COMMAND="docker compose"
    else
        echo "Neither 'docker-compose' nor 'docker compose' is installed. Please install Docker Compose."
        exit 1
    fi

    # Iterate over each script passed as an argument
    for script in "$@"; do
        echo "Starting services for $script..."
        $DOCKER_COMPOSE_COMMAND -f "$script" up -d

        # Add script to the array for cleanup later
        SCRIPTS+=("$script")

        echo "Listening to $script logs. Press Ctrl+C to stop..."
        $DOCKER_COMPOSE_COMMAND -f "$script" logs -f &
    done

    # Set the trap to handle cleanup for all scripts
    trap 'cleanup "${SCRIPTS[@]}"' SIGINT  # Expand SCRIPTS array when passing

    # Wait for all background processes
    wait
}
