#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python  ./train.py  \
                    --data_path /220019054/Dataset/SUN-SEG \
                    --save_path  ./result \
                    --model_name ECCPolypDet\
                    --backbone pvt_v2_b2 \
                    --epoch 20 \
                    --lr 1e-4 \
                    --scheduler cos \
                    --batch_size 12 \

