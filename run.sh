#!/bin/bash

python task1.py --dataset-name coco-nomic-768-normalized &> result/log-coco-nomic-768-normalized.txt
python task1.py --dataset-name imagenet-align-640-normalized &> result/log-imagenet-align-640-normalized.txt
python task1.py --dataset-name laion-clip-512-normalized &> result/log-laion-clip-512-normalized.txt
python task1.py --dataset-name llama-128-ip &> result/log-llama-128-ip.txt
python task1.py --dataset-name yandex-200-cosine &> result/log-yandex-200-cosine.txt
python task1.py --dataset-name yi-128-ip &> result/log-yi-128-ip.txt

echo "Done"