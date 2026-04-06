#!/bin/bash

# ===================================================================
# ---            SEQUENTIAL SCRIPT LAUNCHER                     ---
# ===================================================================

# Define and create the main log directory.
LOG_DIR="sequential_run_logs_$(date +%F_%H-%M-%S)"
mkdir -p "$LOG_DIR"
echo "✅ All output will be logged in the directory: $LOG_DIR"
echo "------------------------------------------------------------------"

# --- Run Scripts Sequentially ---
# The '&&' ensures the second script only runs if the first one succeeds.

echo "➡️ Starting sequential run..."

echo "[Step 1/2] Running ncn_cgle_runtime_check.sh with --run 3..."
bash cgle_ncn_runtime_check.sh > "$LOG_DIR/cgle_ncnc.log" 2>&1 && \. ###Error
echo "[Step 2/2] Running ncnc_cgle_runtime_check.sh with --run 3..." && \
bash cgle_ncnc_runtime_check.sh > "$LOG_DIR/cgle_ncn.log" 2>&1. ##Error

# Check the exit status of the last command to determine the final message.
if [ $? -eq 0 ]; then
  echo "🎉 All tasks have finished successfully."
else
  echo "❌ A task failed. Please check the logs in '$LOG_DIR' for details."
fi

echo "------------------------------------------------------------------"