from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type,
        resolution_scale = 1
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
        self.resolution_scale = resolution_scale
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None

                return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu',
                              time=time,
                              mask=mask)
            except:
                caminfo = self.dataset[index]
                # image = caminfo.image
                # R = caminfo.R
                # T = caminfo.T
                # FovX = caminfo.FovX
                # FovY = caminfo.FovY
                # time = caminfo.time
                # K = caminfo.K
    
                # mask = caminfo.mask
                return loadCam(self.args, index, caminfo, self.resolution_scale)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
