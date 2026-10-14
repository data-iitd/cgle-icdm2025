import subprocess
import time
import os

# ==========================================
# 1. CONFIGURE RUN
# ==========================================

# Define the tuple of datasets you want to run in parallel.
# Add or remove names from this tuple to control what runs.
DATASETS_TO_RUN = ("facebook", "physics", "roman_empire", "amazon_ratings",
                   "questions", "chameleon") 

# List of GPU IDs available. 
# If you have 1 GPU, use [0]. If you have 4, use [0, 1, 2, 3].
GPU_IDS = [0] 

# ==========================================
# 2. COMMAND DEFINITIONS
# ==========================================

commands = {
    "pubmed": "python main_ncn.py  --dataset pubmed --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact",
    
    "citeseer": "python  main_ncn.py  --dataset citeseer  --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 1e-7   --predp 0.5 --gnndp 0.5  --mplayers 1 --nnlayers 2   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10  --batch_size 1024 --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0  --probscale 6.5 --proboffset 4.4 --alpha 0.4 --ln --lnnn   --model puregcn  --testbs 512  --maskinput  --jk  --use_xlin  --tailact  --twolayerlin",
    
    "cora": "python main_ncn.py  --dataset cora  --gnnlr 0.01 --prelr 0.01 --l2 1e-4  --predp 0.1 --gnndp 0.1  --mplayers 2 --nnlayers 1 --hiddim 128 --testbs 512 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024     --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4    --probscale 4.3 --proboffset 2.8 --alpha 1.0    --ln --lnnn --predictor cn1 --runs 10 --model puregcn   --maskinput  --jk  --use_xlin  --tailact",
    
    "dblp": "python main_ncn.py  --dataset DBLP --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact",
    
    "facebook": "python main_ncn.py  --dataset Facebook --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact",
    
    "physics": "python main_ncn.py  --dataset Physics --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact",
    
    "roman_empire": "python main_ncn.py  --dataset Roman-empire --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact",
    
    "amazon_ratings": "python main_ncn.py  --dataset Amazon-ratings --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact",
    
    "questions": "python main_ncn.py  --dataset Questions --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact",
    
    "fb_page": "python main_ncn_fb.py  --dataset fb --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact",
    
    "chameleon": "python main_ncn.py  --dataset chameleon --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact",
    
    "actor": "python main_ncn.py  --dataset actor --predictor cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact"
}

# ==========================================
# 3. EXECUTION LOOP
# ==========================================

processes = []

print(f"Starting {len(DATASETS_TO_RUN)} experiments in parallel...")

for i, dataset_key in enumerate(DATASETS_TO_RUN):
    if dataset_key not in commands:
        print(f"Error: Dataset '{dataset_key}' not found in command dictionary. Skipping.")
        continue

    base_cmd = commands[dataset_key]
    
    # Assign GPU round-robin
    gpu_id = GPU_IDS[i % len(GPU_IDS)]
    
    # Prepare environment with specific GPU visibility
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[{i+1}/{len(DATASETS_TO_RUN)}] Launching {dataset_key} on GPU {gpu_id}...")
    
    # Launch the process
    p = subprocess.Popen(base_cmd, shell=True, env=env)
    processes.append(p)
    
    # Stagger start slightly (5s) to prevent initial CPU/IO bottlenecks
    time.sleep(5) 

print("\nAll processes launched. Waiting for completion...")

# Wait for all processes to finish
for p in processes:
    p.wait()

print("All experiments finished.")