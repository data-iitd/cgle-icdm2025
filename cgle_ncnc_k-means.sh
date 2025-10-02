max_jobs=5 #Max no. of parallel processes allowed
pids=()

run_command() {
    local cmd="$1"
    eval "$cmd &"
    pids+=("$!")

    # Wait if max jobs are running
    if [ "${#pids[@]}" -ge "$max_jobs" ]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
}

datasets=("pubmed" "citeseer" "cora" "DBLP" "Facebook" "Physics" "Roman-empire" "Amazon-ratings" "Questions" "actor" "chameleon" "fb")

for dataset in "${datasets[@]}"
do
    for x in 5 10 15 20
    do
        if [ "$dataset" == "fb" ]; then
            output_file="results/$dataset/cgle_ncn_4_k_means_$x.txt"
            run_command "python main_ncn_fb.py --dataset fb --predictor incn1cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 8192 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --cluster --k_means $x | tee "$output_file""
        else
            output_file="results/$dataset/cgle_ncn_4_k_means_$x.txt"
            run_command "python main_ncn.py --dataset $dataset --predictor cn1 --gnnlr 0.001 --prelr 0.001 --l2 0 --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3 --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 8192 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0 --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn --model puregcn --testbs 8192 --maskinput --jk --use_xlin --tailact --addon --cluster --k_means $x | tee "$output_file""
        fi
    done

done

# Wait for remaining processes
wait
