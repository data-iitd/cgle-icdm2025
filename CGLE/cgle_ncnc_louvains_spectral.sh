#!/bin/bash

# --- Configuration ---
max_jobs=5 # Max number of parallel processes allowed
pids=()

# --- Function to run commands in parallel ---
run_command() {
    local cmd="$1"
    echo "Spawning command: $cmd"
    eval "$cmd &"
    pids+=("$!")

    # Wait if the max number of jobs are already running
    if [ "${#pids[@]}" -ge "$max_jobs" ]; then
        # Wait for the oldest job to finish
        wait "${pids[0]}"
        # Remove the finished PID from the array
        pids=("${pids[@]:1}")
    fi
}

# --- Create results directories for all datasets ---
echo "Creating result directories..."
for dataset in "cora" "citeseer" "actor" "crocodile"
do
    mkdir -p "results/$dataset"
done
echo "Directories created."
echo "---"

# # --- Louvain clustering for 4 datasets ---
# echo "Starting Louvain clustering experiments..."
# for dataset in "cora" "citeseer" "actor" "crocodile"
# do
#     output_file="results/$dataset/cgle_ncn_4_louvain.txt"
#     # Updated: --clustering -> --cluster_method
#     command="python main_ncn.py --dataset $dataset --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 8192 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --cluster --cluster_method louvain | tee \"$output_file\""
#     run_command "$command"
# done
# echo "---"

# --- Spectral clustering for 4 datasets with different k ---
echo "Starting Spectral clustering experiments for k in {5, 10, 15, 20}..."
for dataset in "cora" "citeseer" "actor" "crocodile"
do
    for k in 5 10 15 20
    do
        output_file="results/$dataset/cgle_ncn_4_spectral_k${k}.txt"
        # Updated: --clustering -> --cluster_method and --k_means -> --k_clusters
        command="python main_ncn.py --dataset $dataset --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 8192 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --cluster --cluster_method spectral --k_clusters $k --spectral_mode feature | tee \"$output_file\""
        run_command "$command"
    done
done
echo "---"

# --- Wait for all remaining background jobs to complete ---
echo "All commands have been spawned. Waiting for the last running jobs to finish..."
wait
echo "All jobs completed. ✅"