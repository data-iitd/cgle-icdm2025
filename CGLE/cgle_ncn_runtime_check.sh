#!/bin/bash

# ===================================================================
# ---               CONFIGURATION (CHANGE THESE)                  ---
# ===================================================================
# 1. List the GPU IDs that you want the script to use.
#    The script will cycle through these sequentially for the jobs.
GPUS=(0)

# 2. Define the main directory for all log files for this run.
#    A timestamp is added to keep different runs separate.
LOG_DIR="ncn_addon_logs_$(date +%F_%H-%M-%S)"
# ===================================================================

# Create the main log directory
mkdir -p "$LOG_DIR"

# --- Define all commands in an array ---
# Each command now includes '--runs 5' and '--addon'
COMMANDS=(
  # Command 0: Pubmed
   "python main_ncn.py --dataset pubmed --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --runs 5"
  # Command 1: Citeseer
  "python main_ncn.py --dataset citeseer --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 1e-7 --predp 0.5 --gnndp 0.5 --mplayers 1 --nnlayers 2 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0 --probscale 6.5 --proboffset 4.4 --alpha 0.4 --ln --lnnn --model puregcn --testbs 512 --maskinput --jk --use_xlin --tailact --twolayerlin --addon --runs 5"
  # Command 2: Cora
  "python main_ncn.py --dataset Cora --gnnlr 0.01 --prelr 0.01 --l2 1e-4 --predp 0.1 --gnndp 0.1 --mplayers 2 --nnlayers 1 --hiddim 128 --testbs 512 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --model puregcn --maskinput --jk --use_xlin --tailact --addon --runs 5"
  # Command 3: DBLP
  "python main_ncn.py --dataset DBLP --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --runs 5"
  # Command 4: Facebook
  "python main_ncn.py --dataset Facebook --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --runs 5"
  # Command 5: Physics
  "python main_ncn.py --dataset Physics --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --runs 5"
  # Command 6: Roman-empire
  "python main_ncn.py --dataset Roman-empire --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --runs 5"
  # Command 7: Amazon-ratings
  "python main_ncn.py --dataset Amazon-ratings --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --runs 5"
  # Command 8: Questions
  "python main_ncn.py --dataset Questions --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --runs 5"
  # Command 9: FB Page-Page
  "python main_ncn_fb.py --dataset fb --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --runs 5"
  # Command 10: Chameleon
  "python main_ncn.py --dataset chameleon --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --runs 5"
  # Command 11: Actor
  "python main_ncn.py --dataset actor --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --runs 5"
)

# --- Execution Logic ---
num_gpus=${#GPUS[@]}
echo "Starting sequential experiments. Logs will be saved in the directory: $LOG_DIR"
echo "Using GPUs: ${GPUS[*]}"
echo "========================================================================"

# Loop through all the commands and launch them sequentially
for i in "${!COMMANDS[@]}"; do
  cmd="${COMMANDS[i]}"
  gpu_id=${GPUS[i % num_gpus]}
  
  # Extract dataset name for the individual log file
  dataset_name=$(echo "$cmd" | grep -oP '(?<=--dataset )[^ ]+')
  individual_log_file="${LOG_DIR}/${dataset_name}.log"

  echo "Starting job $((i+1)): Dataset '$dataset_name' on GPU $gpu_id. Log: $individual_log_file"

  # The command to run, with output redirected to its own file
  (
    echo "--- [START] Dataset: $dataset_name on GPU $gpu_id (5 runs) ---"
    time CUDA_VISIBLE_DEVICES=$gpu_id $cmd
    echo "--- [END] Dataset: $dataset_name ---"
  ) >> "$individual_log_file" 2>&1

  echo "Finished job $((i+1))."
  echo "------------------------------------------------------------------------"
done

echo "========================================================================"
echo "All experiments finished. Logs saved in the directory: $LOG_DIR"