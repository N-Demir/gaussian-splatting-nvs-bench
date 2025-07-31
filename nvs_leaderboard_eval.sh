#!/bin/bash

# Check if scene argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset/scene>"
    echo "Example: $0 mipnerf360/bicycle"
    exit 1
fi

scene=$1

method="gaussian-splatting"
iterations=10

# Remove the output folder if it already exists
rm -rf /nvs-leaderboard-output/$scene/$method

# Record start time
start_time=$(date +%s)

# Train -- note that we do not pass in "--eval" so that the full train split is used
python train.py -s /nvs-leaderboard-data/$scene/train -m /nvs-leaderboard-output/$scene/$method/ --iterations $iterations

# Render the test split -- note how technically gaussian-splatting places them in the "train" output folder despite them coming from the test split folder
python render.py -s /nvs-leaderboard-data/$scene/test -m /nvs-leaderboard-output/$scene/$method/ 
# Move the renders to the expected nvs-leaderboard output folder
mv /nvs-leaderboard-output/$scene/$method/train/ours_$iterations/renders /nvs-leaderboard-output/$scene/$method/renders_test

# Record end time and show duration
end_time=$(date +%s)
echo $((end_time - start_time)) > /nvs-leaderboard-output/$scene/$method/training_time.txt
