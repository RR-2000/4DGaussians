#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures

def multithread_write(image_list, path, file_names, idx_offset = 0):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path, file_name):
        try:
            if file_name:
                torchvision.utils.save_image(image, os.path.join(path, file_name + '.png'))
            else:
                torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index + idx_offset, path, file_names[index] if len(file_names) > 0 else None))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index + idx_offset, path, file_names[index])
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, load2gpu_on_the_fly, batch_size, generate_mp4 = True):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    idx_offset = 0
    file_path = []
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()


        if load2gpu_on_the_fly:
            view.load2device()
        
        rendering = render(view, gaussians, pipeline, background,cam_type=cam_type)["render"]
        if generate_mp4:
            render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering.to("cpu"))

        if load2gpu_on_the_fly:
            view.load2device("cpu")

        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                if view.original_image is not None:
                    gt = view.original_image[0:3, :, :]
                    file_path.append(view.image_name)
                    os.makedirs(os.path.join(gts_path, view.image_name.split('/')[0]), exist_ok = True)
                    os.makedirs(os.path.join(render_path, view.image_name.split('/')[0]), exist_ok = True)
                else:
                    gt = None
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

        # Avoiding keeping all images in RAM
        if len(render_list) >= batch_size:
            if not gt_list == []:
                multithread_write(gt_list, gts_path, file_path, idx_offset)

            multithread_write(render_list, render_path, file_path, idx_offset)
            
            idx_offset = idx
            file_path = []
            gt_list = []
            render_list = []


    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    if len(render_list) > 0:
        if not gt_list == []:
            multithread_write(gt_list, gts_path, file_path, idx_offset)

        multithread_write(render_list, render_path, file_path, idx_offset)

    if generate_mp4:
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)
    return os.path.join(model_path, name, "ours_{}".format(iteration))

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, batch_size: int, generate_mp4: False):

    hyperparam.kplanes_config['resolution'] = [64, 64, 64, int(dataset.num_t/2)]
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_path = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type, load2gpu_on_the_fly = dataset.load2gpu_on_the_fly, batch_size = batch_size, generate_mp4 = generate_mp4)
        if not skip_test:
            render_path = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type, load2gpu_on_the_fly = dataset.load2gpu_on_the_fly, batch_size = batch_size, generate_mp4 = generate_mp4)
        if not skip_video:
            render_path = render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type, load2gpu_on_the_fly = dataset.load2gpu_on_the_fly, batch_size = batch_size)
            
            # os.system(f"ffmpeg -y -framerate 24 -i {render_path}/renders/%05d.png -pix_fmt yuv420p {render_path}/video.mp4 -y")
            # print('Saved:', f"{render_path}/video.mp4")
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--generate_mp4", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.batch_size, args.generate_mp4)