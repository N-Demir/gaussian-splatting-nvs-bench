import os
from beam import function, Image, Volume

TRAIN_COMMAND = "python train.py -s ~/data/{capture_name} -m ~/output/{capture_name}_gaussian_splatting/ --eval --iterations 10"

@function(
    gpu="RTX4090",
    image=Image.from_dockerfile("./Dockerfile"),
    volumes=[
        Volume("data", mount_path="./data"),
    ]
)
def run(capture_name: str):
    print(f"Running gaussian-splatting on {capture_name}")
    os.system(TRAIN_COMMAND.format(capture_name=capture_name))

run("examples/kitchen")