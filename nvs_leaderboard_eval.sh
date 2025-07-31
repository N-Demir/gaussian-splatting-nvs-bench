#!/bin/bash

datasets_and_scenes=(
    "examples/kitchen"
    # "mipnerf360/bicycle"
)

iterations=10

for dataset_and_scene in "${datasets_and_scenes[@]}"; do
    python train.py -s /nvs-leaderboard-data/$dataset_and_scene/train -m /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting/ --iterations $iterations
    
    # Render the "train" scene and move the renders out
    python render.py -s /nvs-leaderboard-data/$dataset_and_scene/train -m /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting/ 
    mv /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting/train/ours_$iterations/renders /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting/renders_train
    
    python render.py -s /nvs-leaderboard-data/$dataset_and_scene/test -m /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting/ 
    mv /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting/test/ours_$iterations/renders /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting/renders_test
done