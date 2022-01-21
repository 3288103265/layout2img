## input label and bbox, output images
import argparse
from collections import OrderedDict

from utils.fid import calculate_fid_given_paths
from utils import misc
import torch
import torch.nn.functional as F
from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator_v2 import *
from utils.util import *
import tqdm
import os
import glob
import natsort
import warnings
warnings.filterwarnings("ignore")


def generate(model，labels, bboxs, noise=None):# ResnetGenerator可以满足这个条件。
    """
    input:
        bbox: (N * O * 4):[img1的bbox,..,]
        objs: (N * O):[img1bbox的label,...]
    output:
        imgs:(NCHW)
    """


def sample_image(netG, bbox, objs, image_id=None, z=None, sample_num=5, max_obj=8, real_images=None):
    
    
    assert len(bbox) == len(objs) == len()
    assert bbox.shape[1] <= max_obj
    assert bbox.shape[2] == 4



    batch_size = real_images.shape[0]
    objs = objs.long().unsqueeze(-1)
    bbox = bbox.float().cuda()

    if real_images:
        real_images = real_images.cuda()
        
    for s_i in range(sample_num):  # sample_num=5
        z_obj = torch.randn(batch_size, num_o, 128, device=real_images.device)
    
        z_im = torch.randn(batch_size, 128, device=real_images.device)
        fake_images = netG.forward(
            z_obj, bbox, z_im, label.squeeze(dim=-1))
        
        for j,img in enumerate(fake_images):
            # misc.imsave("{save_path}/images/sample{".format(save_path=sample_path,
            #             id=idx*batch_size+j, s_i=s_i), img.cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)
            misc.imsave("{save_path}/images/sample{id}_{s_i}.jpg".format(save_path=sample_path,
                        id=idx*batch_size+j, s_i=s_i), img.cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)
        # if save_gt:
        #     for k, img in enumerate(real_images):
        #         misc.imsave("{save_path}/images_gt/sample{id}.jpg".format(save_path=sample_path,
        #                                                                 s_i=s_i, id=idx*args.batch_size+k), img.cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)

    paths = [f"{sample_path}/images", GT_IMAGES_PATH]
    print('>>>calculating fid score...')
    return calculate_fid_given_paths(paths, batch_size=50, device=torch.cuda.current_device(), dims=2048)

