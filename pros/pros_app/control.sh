#!/bin/bash

# List of shell scripts to choose from
scripts=(
    "./slam.sh"
    "./slam_unity.sh"
    "./slam_ydlidar.sh"
    "./store_map.sh"
    "./localization.sh"
    "./localization_unity.sh"
    "./localization_ydlidar.sh"
    "./camera_astra.sh"
    "./camera_dabai.sh"
    "./rosbridge_server.sh"
    "./gps.sh"
)

# Array to keep track of child process IDs
child_pids=()

# Function to display the menu
show_menu() {
    clear
    echo "Choose a script to run:"
    for i in "${!scripts[@]}"; do
        echo "$((i+1)). ${scripts[i]}"
    done
    echo "s. Show running processes"
    echo "d. Shutdown all child processes"
    echo "q. Quit"
}

# Function to show running PIDs and their commands
show_running_processes() {
    if [[ ${#child_pids[@]} -eq 0 ]]; then
        echo "No child processes are currently running."
    else
        echo "Running processes:"
        for pid in "${child_pids[@]}"; do
            # Check if the process is still running before retrieving its command
            if kill -0 $pid 2>/dev/null; then
                command=$(ps -p $pid -o args= 2>/dev/null)
                echo "PID: $pid - Command: $command"
            else
                # If process is not found, remove it from child_pids
                child_pids=("${child_pids[@]/$pid}")
            fi
        done
    fi
}

# Function to handle running the script
run_script() {
    local script=$1
    local print_logs=$2  # Boolean to control whether to print logs or not

    if [[ $print_logs == true ]]; then
        echo "Running $script with logs... Press 'b' to go back to menu without terminating, or 'q' to quit and terminate the process."
        (exec "$script") &
    else
        echo "Running $script without logs... Press 'b' to go back to menu without terminating, or 'q' to quit and terminate the process."
        (exec "$script") > /dev/null 2>&1 &
    fi
    script_pid=$!
    child_pids+=("$script_pid")

    # Wait for user to press 'q' or 'b'
    while :; do
        read -n 1 input
        if [[ $input == "q" ]]; then
            echo -e "\nTerminating $script: $script_pid..."
            kill -SIGINT $script_pid
            wait $script_pid 2>/dev/null
            # Remove the process from child_pids array
            child_pids=("${child_pids[@]/$script_pid}")
            break
        elif [[ $input == "b" ]]; then
            echo -e "\nGoing back to the menu. The process $script: $script_pid will continue running."
            break
        fi
    done
}

# Function to shut down all child processes
shutdown_all_children() {
    if [[ ${#child_pids[@]} -eq 0 ]]; then
        echo "No child processes are running."
    else
        echo "Shutting down all child processes..."
        for pid in "${child_pids[@]}"; do
            # Check if the process is still running before retrieving its command
            if kill -0 $pid 2>/dev/null; then
                echo "Terminating process: $pid"
                kill -SIGINT $pid
                wait $pid 2>/dev/null
            fi
        done
        child_pids=()
    fi
}

# Function to display help message
print_help() {
    echo "Usage: $0 [options] <script>"
    echo "Options:"
    echo "  -s, --silent      Run the script in silent mode (suppress logs)"
    echo "  -h, --help        Show this help message and exit"
    echo "Script:"
    echo "  The script to execute (e.g., ./store_map.sh)."
    exit 0
}

# Default value for silent mode
silent=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--silent) silent=true ;;  # Set silent mode
        -h|--help) print_help ;;  # Print help and exit
    esac
    shift
done

# Main loop
while true; do
    show_menu

    if [[ "$silent" == true ]]; then
        echo "===== Running in silent mode ====="
    fi

    read -p "Enter your choice: " choice

    if [[ $choice == "q" ]]; then
        echo "Shutting down all child processes."
        shutdown_all_children
        echo "Exiting..."
        break
    elif [[ $choice == "d" ]]; then
        shutdown_all_children
    elif [[ $choice == "s" ]]; then
        show_running_processes
    elif [[ $choice =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#scripts[@]} )); then
        selected_script="${scripts[$((choice-1))]}"
        
        # Determine if we want to print logs
        if [[ "$silent" == false || $selected_script == "./store_map.sh" ]]; then
            run_script "$selected_script" true  # Print logs if not in silent mode or if running store_map.sh
        else
            run_script "$selected_script" false  # Suppress logs for other scripts
        fi
    else
        echo "Invalid choice. Please try again."
    fi
    echo "Press any key to continue..."
    read -n 1 -s
done
