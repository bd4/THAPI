#!/bin/bash
# Usage: THAPI_BIN_DIR=/path/to/iprof/bin THAPI_SYNC_DAEMON=mpi|fs TEST_EXE=clinfo test.sh

# Get base real-time signal number
SIGRTMIN=$(kill -l SIGRTMIN)

# Set signals as defined in MPI daemon code
RT_SIGNAL_READY=$((SIGRTMIN + 0))
RT_SIGNAL_GLOBAL_BARRIER=$((SIGRTMIN + 1))
RT_SIGNAL_LOCAL_BARRIER=$((SIGRTMIN + 2))
RT_SIGNAL_FINISH=$((SIGRTMIN + 3))

# Initialize a variable to track signal reception
SIGNAL_RECEIVED="false"
# Signal handler for capturing signals
handle_signal() {
    echo "Received signal $1 from mpi_daemon"
    if [ "$1" == "RT_SIGNAL_READY" ]; then
        SIGNAL_RECEIVED="true"
    fi
}

# Setup trap for ready signal sent from signal daemon
trap 'handle_signal RT_SIGNAL_READY' $RT_SIGNAL_READY

# Function to wait for RT_SIGNAL_READY
wait_for_signal() {
    SIGNAL_RECEIVED="false"
    while [[ "$SIGNAL_RECEIVED" == "false" ]]; do
        sleep 0.1  # Small sleep to prevent busy looping
    done
}

# Function to send signals, using adjusted SIGRTMIN corresponding to MPI signal daemon defines
send_signal_blocking() {
    kill -$1 $DAEMON_PID
    wait_for_signal
}

# Get the PID of this script
PARENT_PID=$$
# Start sync daemon in the background
${THAPI_BIN_DIR}/sync_daemon_${THAPI_SYNC_DAEMON} $PARENT_PID &
DAEMON_PID=$!
# Wait for daemon to be ready
wait_for_signal
# Send signals to mpi_daemon to test synchronization
send_signal_blocking $RT_SIGNAL_LOCAL_BARRIER
send_signal_blocking $RT_SIGNAL_GLOBAL_BARRIER

# Run test program
$TEST_EXE

# Final synchronization after mpi_hello_world execution
send_signal_blocking $RT_SIGNAL_LOCAL_BARRIER
send_signal_blocking $RT_SIGNAL_GLOBAL_BARRIER
# Signal to terminate the mpi_daemon
send_signal_blocking $RT_SIGNAL_FINISH
wait $DAEMON_PID  # Ensure daemon exits cleanly
