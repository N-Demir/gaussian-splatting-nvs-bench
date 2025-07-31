#!/bin/bash

# Check if dataset_and_scene argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset/scene>"
    echo "Example: $0 mipnerf360/bicycle"
    exit 1
fi

dataset_and_scene=$1

iterations=30000

# Remove the output folder if it already exists
rm -rf /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting

# Train -- note that we do not pass in "--eval" so that the full train split is used
python train.py -s /nvs-leaderboard-data/$dataset_and_scene/train -m /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting/ --iterations $iterations

# Render the test split -- note how technically gaussian-splatting places them in the "train" output folder despite them coming from the test split folder
python render.py -s /nvs-leaderboard-data/$dataset_and_scene/test -m /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting/ 
mv /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting/train/ours_$iterations/renders /nvs-leaderboard-output/$dataset_and_scene/gaussian_splatting/renders_test