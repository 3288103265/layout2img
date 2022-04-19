## input label and bbox, output images
## Drespert
import argparse
import os
import warnings
from collections import OrderedDict

import numpy as np
import torch
import tqdm
from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator_v2 import *
from utils import misc
from utils.util import *

warnings.filterwarnings("ignore")

def load_json(json_path):
    with open(json_path, 'r') as f:
        conds = json.load(f)
    return conds

def load_model(model_path, num_classes, device):
    netG = ResnetGenerator128_posi(num_classes=num_classes, output_dim=3).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k,
                       v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)
    
    return netG


def sample_from_layout(netG, bbox, objs, z_dim, sample_num=5, max_obj=8): 
    assert len(bbox) == len(objs)
    assert len(bbox) <= max_obj
    assert len(bbox[0]) == 4
    
    for _ in range(len(objs), max_obj): # padding
        objs.append(0)
        bbox.append(np.array([-0.6, -0.6, 0.5, 0.5]))
        
    device = netG.device

    bbox = np.vstack(bbox)
    bbox = torch.from_numpy(bbox, device=device)
    objs = torch.LongTensor(objs, device=device)
    
    # 将采样的个数作为第一个维度
    bbox = bbox.repeat(sample_num, 1, 1)# sample_image * max_obj * 4 
    objs = objs.repeat(sample_num, 1)# sample_image * max_obj 
    z_obj = torch.randn(sample_num, max_obj, z_dim, device=device) # sample * max_obj* z_dim
    z_im = torch.randn(sample_num, z_dim, device=device) # sample * z_dim
    
    fake_images = netG.forward(
        z_obj, bbox, z_im, objs.squeeze(dim=-1))
    
    return fake_images

def sample_from_json(args):
    assert os.path.isfile(args.json_path)
    assert os.path.isfile(args.model_path)
    assert not os.path.exists(args.out_path)
    os.mkdirs(fake_path:=os.path.join(args.out_path,"fake_images"))
    layouts = load_json(args.json_path)
    args.device = torch.device(args.device)
    
    with open(os.path.join(args.out_path,".json"),'w') as f:
        json.dump(layouts, layouts)
    netG = load_model(args.model_path, num_classes=args.num_classes, device=args.device)
    netG.eval()

    for layout in tqdm.tqdm(layouts, total=len(layouts)):
        image_id = layout["image_id"]
        bbox = layout["bbox"]
        objs = layout["objs"]
        fake_images = sample_from_layout(netG, bbox, objs, args.z_dim, args.sample_num, args.max_obj)
        
        for j,img in enumerate(fake_images):# save image.
            misc.imsave("{save_path}/{image_id}_{s_i}.jpg".format(save_path=fake_path,
                        image_id=image_id, s_i=j), img.cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)

def modify_args(args):
    if args.datasets == "coco":
        args.z_dim = 128
        args.num_classes = 182
        args.sample_num = 5
        args.max_obj = 8
    elif args.datasets == "vg":
        args.z_dim = 128
        args.num_classes = 179
        args.sample_num = 5
        args.max_obj = 30
    else:
        raise NotImplementedError

    return args
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "vg"])
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    args = modify_args(args)
    sample_from_json(args)
    
    