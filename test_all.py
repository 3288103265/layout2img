import argparse
from collections import OrderedDict

from torch._C import device
from utils.fid import calculate_fid_given_paths
import numpy as np
from utils import misc
import imageio
import torch
import torch.nn as nn
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

## single batch_size

def get_dataloader(dataset='coco', img_size=128, batch_size=1):
    if dataset == 'coco':
        dataset = CocoSceneGraphDataset(image_dir='./datasets/coco/val2017/',
                                        instances_json='./datasets/coco/annotations/instances_val2017.json',
                                        stuff_json='./datasets/coco/annotations/stuff_val2017.json',
                                        stuff_only=True, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == 'vg':
        with open("./datasets/vg/vocab.json", "r") as read_file:
            vocab = json.load(read_file)
        dataset = VgSceneGraphDataset(vocab=vocab,
                                      h5_path='./datasets/vg/val.h5',
                                      image_dir='./datasets/vg/images/',
                                      image_size=(128, 128), left_right_flip=False, max_objects=30)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        drop_last=True, shuffle=False, num_workers=1)
    return dataloader


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    for arg in vars(args):
        print("%15s : %-15s" % (arg, str(getattr(args, arg))))
    num_classes = 184 if args.dataset == 'coco' else 179
    num_o = 8 if args.dataset == 'coco' else 31

    dataloader = get_dataloader(args.dataset, args.image_size)

    if args.image_size == 128:
        netG = ResnetGenerator128(
            num_classes=num_classes, output_dim=3).cuda()
    elif args.image_size == 256:
        netG = ResnetGenerator256(num_classes=num_classes, output_dim=3).cuda()

    ckpt_path_list = glob.glob(os.path.join(args.model_root, 'model/G*.pth'))

    exist_model = glob.glob(os.path.join(args.model_root, 'samples_G*/'))
    exist_model = natsort.natsorted(exist_model)

    print(exist_model)
    sample_start = len(exist_model)
    if args.sample_start > 0:
        sample_start = args.sample_start

    test_res = []
    ckpt_path_list = natsort.natsorted(ckpt_path_list)
    print(ckpt_path_list)

    ckpt_path_list = ckpt_path_list[sample_start:]

    sample_path_list = [os.path.join(args.model_root, 'samples_' + os.path.basename(
        p).split('.')[0].replace('_', '')) for p in ckpt_path_list]
    for idx, (ckpt_path, sample_path) in enumerate(zip(ckpt_path_list, sample_path_list)):
        print(
            f'>>>[{idx+1}/{len(ckpt_path_list)}]Sampling images using ckpt: {ckpt_path}')
        res = test_ckpt(ckpt_path, netG=netG,
                        sample_path=sample_path, dataloader=dataloader, num_o=num_o)
        print(f">>>result:{res}")
        test_res.append(res)

    res_dict = dict(zip(ckpt_path_list, test_res))

    print("******Evaluation results******")
    for k, v in res_dict.items():
        print("%15s : %-15s" % (k.split('/')[-1], str(v)))
    with open(os.path.join(args.model_root, 'test_all.json'), 'a') as f:
        json.dump(res_dict, f)


def test_ckpt(model_path, netG, sample_path, dataloader, num_o):
    if not os.path.isfile(model_path):
        print('Model path invalid')
        return

    state_dict = torch.load(model_path, map_location="cuda:0")

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k,
                       v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()

    if not os.path.exists(os.path.join(sample_path, 'images')):
        os.makedirs(os.path.join(sample_path, 'images'))
    if not os.path.exists(os.path.join(sample_path, 'images_gt')):
        os.makedirs(os.path.join(sample_path, 'images_gt'))

    thres = 2.0
    sample_num = 5
    for idx, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # if idx == 50:
        #     break
        real_images, label, bbox = data
        real_images, label = real_images.cuda(), label.long().unsqueeze(-1).cuda()
        bbox = bbox.float().cuda()
        for s_i in range(sample_num):  # sample_num=5
            z_obj = torch.from_numpy(truncted_random(
                num_o=num_o, thres=thres)).float().cuda()
            z_im = torch.from_numpy(truncted_random(
                num_o=1, thres=thres)).view(1, -1).float().cuda()
            fake_images = netG.forward(
                z_obj, bbox, z_im, label.squeeze(dim=-1))

            misc.imsave("{save_path}/images/sample{idx}_{s_i}.jpg".format(save_path=sample_path,
                        idx=idx, s_i=s_i), fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)

        misc.imsave("{save_path}/images_gt/sample{idx}.jpg".format(save_path=sample_path,
                                                                   s_i=s_i, idx=idx), real_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)

    paths = [f"{sample_path}/images", f"{sample_path}/images_gt"]
    print('>>>calculating fid score...')
    return calculate_fid_given_paths(paths, batch_size=50, device=torch.cuda.current_device(), dims=2048)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    # parser.add_argument('--model_path', type=str, default='pretrained_model/netGv2_coco128.pth',
    #                     help='which epoch to load')
    # parser.add_argument('--sample_path', type=str, default='samples',
    #                     help='path to save generated images')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--use_trans_enc', type=int, default=1)
    parser.add_argument('--model_root', type=str, help='Root dir contains model/',
                        default='outputs/train_debug')
    parser.add_argument('--batch_size', type=int,
                        help='test batch size', default=1)
    parser.add_argument('--sample_start', type=int,
                        help='Start sample from {sample_start}-th sample', default=0)
    args = parser.parse_args()
    main(args)
