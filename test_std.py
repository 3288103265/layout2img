import argparse
from collections import OrderedDict
from model.resnet_generator_context import context_aware_generator
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
from skimage import img_as_ubyte



def get_dataloader(dataset='coco', img_size=128):
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
        dataset, batch_size=1,
        drop_last=True, shuffle=False, num_workers=1)
    return dataloader


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    for arg in vars(args):
        print("%15s : %-15s" % (arg, str(getattr(args, arg))))
    num_classes = 184 if args.dataset == 'coco' else 179
    num_o = 8 if args.dataset == 'coco' else 31

    dataloader = get_dataloader(args.dataset, args.image_size)

    # if args.image_size == 128:
    #     netG = ResnetGenerator128(
    #         num_classes=num_classes, output_dim=3, use_trans_enc=args.use_trans_enc).cuda()
    # elif args.image_size == 256:
    #     netG = ResnetGenerator256(num_classes=num_classes, output_dim=3).cuda()
    
    netG = context_aware_generator(
        num_classes=num_classes, output_dim=3).cuda()# for test context
    # netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).cuda()


    if not os.path.isfile(args.model_path):
        print('Model path invalid')
        return
    state_dict = torch.load(args.model_path)

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

    if not os.path.exists(os.path.join(args.sample_path, 'images')):
        os.makedirs(os.path.join(args.sample_path, 'images'))
    if not os.path.exists(os.path.join(args.sample_path, 'images_gt')):
        os.makedirs(os.path.join(args.sample_path, 'images_gt'))
    thres = 2.0
    sample_num = 5
    for idx, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        real_images, label, bbox = data
        # src_mask = None
        # if args.use_trans_enc:
        #     src_mask = torch.bmm(label.unsqueeze(2), label.unsqueeze(1))
        #     src_mask = src_mask != 0
        #     src_mask = src_mask.cuda()
        real_images, label = real_images.cuda(), label.long().unsqueeze(-1).cuda()
        bbox = bbox.float().cuda()
        for s_i in range(sample_num):  # sample_num=5
            z_obj = torch.from_numpy(truncted_random(
                num_o=num_o, thres=thres)).float().cuda()
            z_im = torch.from_numpy(truncted_random(
                num_o=1, thres=thres)).view(1, -1).float().cuda()
            fake_images = netG.forward(
                z_obj, bbox, z_im, label.squeeze(dim=-1))
            fake_images_uint = img_as_ubyte(
                fake_images[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
            imageio.imwrite("{save_path}/images/sample_{idx}_numb_{numb}.jpg".format(
                save_path=args.sample_path, idx=idx, numb=s_i), fake_images_uint)

        # imageio.imwrite("{save_path}/images_gt/sample{idx}.jpg".format(save_path=args.sample_path,
        #                                                            s_i=s_i, idx=idx), real_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)
        real_images_uint = img_as_ubyte(
            real_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)
        imageio.imwrite("{save_path}/images_gt/sample{idx}.jpg".format(save_path=args.sample_path,
                                                                    idx=idx), real_images_uint)
    print("Runing fid test.")
    print("=======================")
    print("python -m pytorch_fid {save_path}/images {save_path}/images_gt/".format(
        save_path=args.sample_path))
    print("=======================")

    os.system(
        "python -m pytorch_fid {save_path}/images {save_path}/images_gt/".format(save_path=args.sample_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--model_path', type=str, default='pretrained_model/netGv2_coco128.pth',
                        help='which epoch to load')
    parser.add_argument('--sample_path', type=str, default='samples',
                        help='path to save generated images')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--use_trans_enc', type=int, default=1)
    args = parser.parse_args()
    main(args)
