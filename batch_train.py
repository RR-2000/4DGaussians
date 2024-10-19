import os
import sys
from tqdm import tqdm
import argparse

def train_sequence(name, exp_tag = ''):
    num_frames = len(os.listdir(os.path.join("./data", name, "frames_1", "cam00")))

    try:
        os.system(f"CUDA_LAUNCH_BLOCKING=1 python train.py -s \
              ./data/{name}/ -m ./output/{'brics' + exp_tag}/{name}/ \
              --eval --white_background --load2gpu_on_the_fly --dataloader \
              --num_t {num_frames} --load_image_on_the_fly  > ./output/{exp_tag + name}_train.log")
    except:
        print(f"Encountered error while training {name}")
    
def render_sequence(name, exp_tag = ''):
    num_frames = len(os.listdir(os.path.join("./data", name, "frames_1", "cam00")))

    try:
        os.system(f"CUDA_LAUNCH_BLOCKING=1 python render.py -s \
              ./data/{name}/ -m ./output/{'brics' + exp_tag}/{name}/ \
              --eval --white_background --load2gpu_on_the_fly \
              --num_t {num_frames} --load_image_on_the_fly  > ./output/{exp_tag + name}_render.log")
    except:
        print(f"Encountered error while rendering {name}")


if __name__ == "__main__":
    sequences = sorted(os.listdir("./data"))[:5]
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", help = "Default will train all sequnces", default = False, action = 'store_true')
    parser.add_argument("--skip_render", help = "Default will render all sequnces", default = False, action = 'store_true')
    parser.add_argument("--exp_tag", help = "Experiment Name", default = '', action = 'store')
    args = parser.parse_args()

    for sequence in tqdm(sequences):
        if not args.skip_train:
            train_sequence(sequence, args.exp_tag)


        if not args.skip_render:
            render_sequence(sequence, args.exp_tag)

