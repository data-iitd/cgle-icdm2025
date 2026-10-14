#Pubmed
python ncnc_concat_y.py  --dataset pubmed --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --concat_label

#Citeseer
python  ncnc_concat_y.py  --dataset citeseer  --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 1e-7   --predp 0.5 --gnndp 0.5  --mplayers 1 --nnlayers 2   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10  --batch_size 1024 --xdp 0.4 --tdp 0.0 --pt 0.75 --gnnedp 0.0 --preedp 0.0  --probscale 6.5 --proboffset 4.4 --alpha 0.4 --ln --lnnn   --model puregcn  --testbs 512  --maskinput  --jk  --use_xlin  --tailact  --twolayerlin --concat_label

#Cora
python ncnc_concat_y.py  --dataset cora  --gnnlr 0.01 --prelr 0.01 --l2 1e-4  --predp 0.1 --gnndp 0.1  --mplayers 2 --nnlayers 1 --hiddim 128 --testbs 512 --epochs 9999 --kill_cnt 10 --eval_steps 5  --batch_size 1024     --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4    --probscale 4.3 --proboffset 2.8 --alpha 1.0    --ln --lnnn --predictor incn1cn1 --runs 10 --model puregcn   --maskinput  --jk  --use_xlin  --tailact --concat_label

#DBLP
python ncnc_concat_y.py  --dataset DBLP --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --concat_label

#Facebook
python ncnc_concat_y.py  --dataset Facebook --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --concat_label

#Physics
python ncnc_concat_y.py  --dataset Physics --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --concat_label

#Roman-empire
python ncnc_concat_y.py  --dataset Roman-empire --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --concat_label


#Amazon-ratings
python ncnc_concat_y.py  --dataset Amazon-ratings --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --concat_label


#Questions
python ncnc_concat_y.py  --dataset Questions --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --concat_label

#FB Page-Page
python main_ncn_fb.py  --dataset fb --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --concat_label

#Chameleon
python ncnc_concat_y.py  --dataset chameleon --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --concat_label

#Actor
python ncnc_concat_y.py  --dataset actor --predictor incn1cn1   --gnnlr 0.001 --prelr 0.001 --l2 0  --predp 0.3 --gnndp 0.3 --mplayers 3 --nnlayers 3   --hiddim 256 --epochs 9999 --eval_steps 5 --kill_cnt 10 --batch_size 1024 --xdp 0.3 --tdp 0.0 --pt 0.5 --gnnedp 0.0 --preedp 0.0   --probscale 5.3 --proboffset 0.5 --alpha 0.3 --ln --lnnn   --model puregcn  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --concat_label
