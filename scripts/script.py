import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

if opt.train:
    os.system("python train.py \
        --name rctnet_LoL_batch8 \
        --model rct_net \
        --data_root ./datasets/LoL/train \
        --dataset_mode pair \
        --which_direction AtoB \
        --batch_size 8 \
        --num_workers 4 \
        --resize_or_crop resize \
        --fine_size 256 \
        --scale_width 512 \
        --number_filter 16 \
        --fusion_filter 128 \
        --represent_feature 16 \
        --ngf 64 \
        --nlf 16 \
        --mesh_size 31 \
        --vgg_choose conv5_1 \
        --display_port=" + opt.port)
