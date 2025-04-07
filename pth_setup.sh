conda create -n open_clip python=3.8
conda activate open_clip
pip install open_clip_torch
CUDA_VISIBLE_DEVICES=0 python clip_eval.py