## input label and bbox, output images
## TODO: refactor
import argparse
import json
import os
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm
from data.vg import *
# import io
from torch.utils.data import Dataset
from model.resnet_generator_v2 import *
from utils import misc
from utils.util import *

warnings.filterwarnings("ignore")



class CocoSceneGraphDataset(Dataset):
    def __init__(self, image_dir, instances_json, stuff_json=None,
                 stuff_only=True, image_size=(64, 64), mask_size=16,
                 normalize_images=True, max_samples=None,
                 include_relationships=True, min_object_size=0.02,
                 min_objects_per_image=3, max_objects_per_image=8, left_right_flip=False,
                 include_other=False, instance_whitelist=None, stuff_whitelist=None):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.
    
        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
        """
        super(Dataset, self).__init__()

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')

        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_samples = max_samples
        self.max_objects_per_image = max_objects_per_image
        self.normalize_images = normalize_images
        self.include_relationships = include_relationships
        self.left_right_flip = left_right_flip
        self.set_image_size(image_size)

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        object_idx_to_name = {}
        all_instance_categories = []
        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
        all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id

        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

        # Add object data from instances
        self.image_id_to_objects = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            # box_area = object_data['area'] / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                self.image_id_to_objects[image_id].append(object_data)

        # Add object data from stuff
        if stuff_data:
            image_ids_with_stuff = set()
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                image_ids_with_stuff.add(image_id)
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                # box_area = object_data['area'] / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok and (object_data['iscrowd'] != 1):
                    self.image_id_to_objects[image_id].append(object_data)

            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id)
                self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)

        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

        self.vocab['pred_idx_to_name'] = [
            '__in_image__',
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx
        
        # with open("datasets/coco/vocab.json", 'w') as f:
        #     json.dump(self.vocab, f)

    def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        if self.max_samples is None:
            if self.left_right_flip:
                return len(self.image_ids)*2
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.
    
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)        
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, w, h) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        flip = False
        # index = 1292
        if index >= len(self.image_ids):
            index = index - len(self.image_ids)
            flip = True
        image_id = self.image_ids[index]

        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                if flip:
                    image = PIL.ImageOps.mirror(image)
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        objs, boxes, masks = [], [], []
        # obj_masks = []
        for object_data in self.image_id_to_objects[image_id]:
            objs.append(object_data['category_id'])
            # print(self.vocab['object_idx_to_name'][object_data['category_id']])
            x, y, w, h = object_data['bbox']
            x0 = x / WW
            y0 = y / HH
            w = (w) / WW
            h = (h) / HH
            if flip:
                x0 = 1 - (x0 + w)
            boxes.append(np.array([x0, y0, w, h]))

        for _ in range(len(objs), self.max_objects_per_image):
            objs.append(self.vocab['object_name_to_idx']['__image__'])
            boxes.append(np.array([-0.6, -0.6, 0.5, 0.5]))
     

        objs = torch.LongTensor(objs)
        boxes = np.vstack(boxes)
        return image, objs, boxes, image_id## , b_map #, None # obj_masks #, obj_masks # , b_map # masks # , triples


# class ResnetGenerator128(nn.Module):
#     def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
#         super(ResnetGenerator128, self).__init__()
#         self.num_classes = num_classes

#         self.label_embedding = nn.Embedding(num_classes, 180)

#         num_w = 128 + 180
#         self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))

#         self.res1 = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_w)
#         self.res2 = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_w)
#         self.res3 = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_w)
#         self.res4 = ResBlock(ch * 4, ch * 2, upsample=True, num_w=num_w, psp_module=True)
#         self.res5 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False)
#         self.final = nn.Sequential(BatchNorm(ch),
#                                    nn.ReLU(),
#                                    conv2d(ch, output_dim, 3, 1, 1),
#                                    nn.Tanh())

#         # mapping function
#         mapping = list()
#         self.mapping = nn.Sequential(*mapping)

#         self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
#         self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
#         self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
#         self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))

#         self.sigmoid = nn.Sigmoid()

#         self.mask_regress = MaskRegressNetv2(num_w)
#         self.init_parameter()

#     def forward(self, z, bbox, z_im=None, y=None):
#         b, o = z.size(0), z.size(1)
#         label_embedding = self.label_embedding(y)

#         z = z.view(b * o, -1)
#         label_embedding = label_embedding.view(b * o, -1)

#         latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

#         w = self.mapping(latent_vector.view(b * o, -1))
#         # preprocess bbox
#         bmask = self.mask_regress(w, bbox)

#         if z_im is None:
#             z_im = torch.randn((b, 128), device=z.device)

#         bbox_mask_ = bbox_mask(z, bbox, 64, 64)

#         # 4x4
#         x = self.fc(z_im).view(b, -1, 4, 4)
#         # 8x8
#         x, stage_mask = self.res1(x, w, bmask)

#         # 16x16
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
#         alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
#         x, stage_mask = self.res2(x, w, stage_bbox)

#         # 32x32
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
#         alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
#         x, stage_mask = self.res3(x, w, stage_bbox)

#         # 64x64
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
#         alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
#         x, stage_mask = self.res4(x, w, stage_bbox)

#         # 128x128
#         hh, ww = x.size(2), x.size(3)
#         seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
#         seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
#         alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
#         stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
#         x, _ = self.res5(x, w, stage_bbox)

#         # save_path1 = 'samples/tmp/edit/apponly/1292_bmask2_0.npy'
#         # save_path2 = 'samples/tmp/edit/apponly/1292_stage2_bbox_0.npy'
#         # np.save(save_path1, bmask.cpu().detach().numpy())
#         # np.save(save_path2, stage_bbox.cpu().detach().numpy())

#         # to RGB
#         x = self.final(x)
#         return x

#     def init_parameter(self):
#         for k in self.named_parameters():
#             if k[1].dim() > 1:
#                 torch.nn.init.orthogonal_(k[1])
#             if k[0][-4:] == 'bias':
#                 torch.nn.init.constant_(k[1], 0)


# class ResBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, upsample=False, num_w=128, predict_mask=True, psp_module=False):
#         super(ResBlock, self).__init__()
#         self.upsample = upsample
#         self.h_ch = h_ch if h_ch else out_ch
#         self.conv1 = conv2d(in_ch, self.h_ch, ksize, pad=pad)
#         self.conv2 = conv2d(self.h_ch, out_ch, ksize, pad=pad)
#         self.b1 = SpatialAdaptiveSynBatchNorm2d(in_ch, num_w=num_w, batchnorm_func=BatchNorm)
#         self.b2 = SpatialAdaptiveSynBatchNorm2d(self.h_ch, num_w=num_w, batchnorm_func=BatchNorm)
#         self.learnable_sc = in_ch != out_ch or upsample
#         if self.learnable_sc:
#             self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
#         self.activation = nn.ReLU()

#         self.predict_mask = predict_mask
#         if self.predict_mask:
#             if psp_module:
#                 self.conv_mask = nn.Sequential(PSPModule(out_ch, 100),
#                                                nn.Conv2d(100, 184, kernel_size=1))
#             else:
#                 self.conv_mask = nn.Sequential(nn.Conv2d(out_ch, 100, 3, 1, 1),
#                                                BatchNorm(100),
#                                                nn.ReLU(),
#                                                nn.Conv2d(100, 184, 1, 1, 0, bias=True))

#     def residual(self, in_feat, w, bbox):
#         x = in_feat
#         x = self.b1(x, w, bbox)
#         x = self.activation(x)
#         if self.upsample:
#             x = F.interpolate(x, scale_factor=2, mode='nearest')
#         x = self.conv1(x)
#         x = self.b2(x, w, bbox)
#         x = self.activation(x)
#         x = self.conv2(x)
#         return x

#     def shortcut(self, x):
#         if self.learnable_sc:
#             if self.upsample:
#                 x = F.interpolate(x, scale_factor=2, mode='nearest')
#             x = self.c_sc(x)
#         return x

#     def forward(self, in_feat, w, bbox):
#         out_feat = self.residual(in_feat, w, bbox) + self.shortcut(in_feat)
#         if self.predict_mask:
#             mask = self.conv_mask(out_feat)
#         else:
#             mask = None
#         return out_feat, mask


# def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
#     conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
#     if spectral_norm:
#         return nn.utils.spectral_norm(conv, eps=1e-4)
#     else:
#         return conv


# def batched_index_select(input, dim, index):
#     expanse = list(input.shape)
#     expanse[0] = -1
#     expanse[dim] = -1
#     index = index.expand(expanse)
#     return torch.gather(input, dim, index)


# def bbox_mask(x, bbox, H, W):
#     b, o, _ = bbox.size()
#     N = b * o

#     bbox_1 = bbox.float().view(-1, 4)
#     x0, y0 = bbox_1[:, 0], bbox_1[:, 1]
#     ww, hh = bbox_1[:, 2], bbox_1[:, 3]

#     x0 = x0.contiguous().view(N, 1).expand(N, H)
#     ww = ww.contiguous().view(N, 1).expand(N, H)
#     y0 = y0.contiguous().view(N, 1).expand(N, W)
#     hh = hh.contiguous().view(N, 1).expand(N, W)

#     X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).cuda(device=x.device)
#     Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).cuda(device=x.device)

#     X = (X - x0.to(X.device)) / ww.to(X.device)
#     Y = (Y - y0.to(Y.device)) / hh.to(Y.device)

#     X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
#     Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)

#     out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1)
#     return out_mask.view(b, o, H, W)


# class PSPModule(nn.Module):
#     """
#     Reference:
#         Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
#     """

#     def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
#         super(PSPModule, self).__init__()

#         self.stages = []
#         self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
#             BatchNorm(out_features),
#             nn.ReLU(),
#             nn.Dropout2d(0.1)
#         )

#     def _make_stage(self, features, out_features, size):
#         prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
#         conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
#         bn = nn.BatchNorm2d(out_features)
#         return nn.Sequential(prior, conv, bn, nn.ReLU())

#     def forward(self, feats):
#         h, w = feats.size(2), feats.size(3)
#         priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
#         bottle = self.bottleneck(torch.cat(priors, 1))
#         return bottle


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
        drop_last=True, shuffle=False, num_workers=8)
    return dataloader

def load_model(model_kargs ,model_path, num_classes, device):
    # config from model_kargs.
   
    netG = globals()[f'{model_kargs["G_arch"]}{model_kargs["img_size"]}_{model_kargs["G_type"]}'](num_classes=num_classes, output_dim=3).to(device)
    
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


def sample_from_layout(netG, bbox, objs, z_dim, sample_num=5, max_obj=8, device="cuda:0"): 
   
    # 将采样的个数作为第一个维度
    bbox = bbox.repeat(sample_num, 1, 1)# sample_image * max_obj * 4 
    objs = objs.repeat(sample_num, 1)# sample_image * max_obj 
    z_obj = torch.randn(sample_num, max_obj, z_dim, device=device) # sample * max_obj* z_dim
    z_im = torch.randn(sample_num, z_dim, device=device) # sample * z_dim

    fake_images = netG.forward(
        z_obj, bbox, z_im, objs)
    
    return fake_images

def sample_from_dataset(num_classes, dataset, img_size, model_path, out_path, device, z_dim=128, sample_num=5, max_obj=8, save_gt=True):
    assert os.path.isfile(model_path)
    os.makedirs(fake_path:=os.path.join(out_path,"fake_images"))
    os.makedirs(real_path:=os.path.join(out_path,"real_images"))
    
    
    device = torch.device(device)
    
    dataloader = get_dataloader(dataset=dataset, img_size=img_size, batch_size=1)
    
    model_kargs = dict(
        G_arch = "ResnetGenerator",
        G_type = "base",
        D_arch = "CombineDiscriminator",
        D_type = "app",
        img_size = 128
    )
    
    netG = load_model(model_kargs, model_path, num_classes=num_classes, device=device)
    netG.eval()

    layouts = []
    for i,data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        real_images, objs, bbox, image_id = data
        layout = dict()
        layout["image_id"] = image_id.squeeze().item()
        layout["bbox"] = bbox.squeeze().tolist()
        layout["objs"] = objs.squeeze().tolist()
        # layout["objs_name"] = [vocab["id_to_pred_name"][id] for id in layout["objs"]]#TODO
        layouts.append(layout)
        # if i==100:
        #     break
        
        objs = objs.to(device)
        bbox = bbox.float().to(device)
        
        fake_images = sample_from_layout(netG, bbox, objs, z_dim, sample_num, max_obj, device)
        
        for j, img in enumerate(fake_images):# save image.
            misc.imsave("{save_path}/{image_id}_{s_i}.jpg".format(save_path=fake_path,
                        image_id=image_id.item(), s_i=j), img.cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)
        
        if save_gt:
            misc.imsave("{save_path}/{image_id}.jpg".format(save_path=real_path,
                        image_id=image_id.item()), real_images.squeeze().cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)
        
    with open(os.path.join(out_path, "layouts.json"), 'w') as f:
        json.dump(layouts, f)
            
            
def modify_args(args):
    if args.datasets == "coco":
        args.z_dim = 64 # base2中generator的噪声维度改成了64，以往是128
        args.num_classes = 184
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
    parser.add_argument("--datasets", type=str, default="coco", choices=["coco", "vg"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--img_size" ,type=int ,default=128)
    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    args = modify_args(args)
    sample_from_dataset(args.num_classes, args.datasets, args.img_size, args.model_path, args.out_path, args.device, args.z_dim, args.sample_num, args.max_obj, args.save_gt)

if __name__ == "__main__":
    main()
    
