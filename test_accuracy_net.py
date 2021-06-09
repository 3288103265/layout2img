from ast import parse
import numpy as np
import argparse
import csv
import json
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import tqdm
from scene_generation.bilinear import crop_bbox_batch
from scene_generation.data.coco import CocoSceneGraphDataset, coco_collate_fn
from scene_generation.utils import bool_flag, int_tuple, str_tuple
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# python test_accuracy_net.py -op outputs/app/logits_coco_128_G230_pred.csv --coco_val_image_dir outputs/app/samples_G230_ori_size/ -b 1 -g 0


parser = argparse.ArgumentParser(description='Training a pytorch model to classify different plants')
parser.add_argument('-idl', '--input_data_loc', help='', default='data/training_data')
parser.add_argument('-mo', '--model_name', default="resnet101")
parser.add_argument('-mp', '--model_path', default="model/accuracy net/resnet101_172_classes_128_pixels_0.623.pth")
parser.add_argument('-op','--out_path', required=True)
parser.add_argument('-f', '--freeze_layers', default=True, action='store_false', help='Bool type')
parser.add_argument('-fi', '--freeze_initial_layers', default=True, action='store_false', help='Bool type')
parser.add_argument('-ep', '--epochs', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=4, type=int)
parser.add_argument('-is', '--input_shape', default=224, type=int)
parser.add_argument('-sl', '--save_loc', default="models/")
# parser.add_argument("-g", '--use_gpu', default=True, action='store_false', help='Bool type gpu')
parser.add_argument('-g', '--gpu_id', default="0")
parser.add_argument("-p", '--use_parallel', default=False, action='store_false', help='Bool type to use_parallel')

parser.add_argument('--dataset', default='coco', type=str)
parser.add_argument('--mask_size', default=32, type=int)
parser.add_argument('--image_size', default='128,128', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# COCO-specific options
COCO_DIR = os.path.expanduser('datasets/coco')
parser.add_argument('--coco_train_image_dir',
                    default=os.path.join(COCO_DIR, 'train2017'))
parser.add_argument('--coco_val_image_dir',
                    default=os.path.join(COCO_DIR, 'val2017'))
parser.add_argument('--coco_train_instances_json',
                    default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--coco_train_stuff_json',
                    default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_val_instances_json',
                    default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--coco_val_stuff_json',
                    default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)


def all_pretrained_models(n_class, name="resnet101", pretrained=True):
    if pretrained:
        weights = "imagenet"
    else:
        weights = False

    if name == "resnet18":
        print("[Building resnet18]")
        model_conv = torchvision.models.resnet18(pretrained=weights)
    elif name == "resnet34":
        print("[Building resnet34]")
        model_conv = torchvision.models.resnet34(pretrained=weights)
    elif name == "resnet50":
        print("[Building resnet50]")
        model_conv = torchvision.models.resnet50(pretrained=weights)
    elif name == "resnet101":
        print("[Building resnet101]")
        model_conv = torchvision.models.resnet101(pretrained=weights)
    elif name == "resnet152":
        print("[Building resnet152]")
        model_conv = torchvision.models.resnet152(pretrained=weights)
    else:
        raise ValueError

    for i, param in model_conv.named_parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, n_class)

    if "resnet" in name:
        print("[Resnet: Freezing layers only till layer1 including]")
        ct = []
        for name, child in model_conv.named_children():
            if "layer1" in ct:
                for params in child.parameters():
                    params.requires_grad = True
            ct.append(name)

    return model_conv


def build_loaders(args):
    vocab, val_dset = build_coco_dsets(args)
    collate_fn = coco_collate_fn

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
        'collate_fn': collate_fn,
    }
    # train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = args.shuffle_val
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, val_loader


def build_coco_dsets(args):
    dset_kwargs = {
        'image_dir': args.coco_train_image_dir,
        'instances_json': args.coco_train_instances_json,
        'stuff_json': args.coco_train_stuff_json,
        'stuff_only': args.coco_stuff_only,
        'image_size': args.image_size,
        'mask_size': args.mask_size,
        'max_samples': args.num_train_samples,
        'min_object_size': args.min_object_size,
        'min_objects_per_image': args.min_objects_per_image,
        'instance_whitelist': args.instance_whitelist,
        'stuff_whitelist': args.stuff_whitelist,
        'include_other': args.coco_include_other,
        # 'include_relationships': args.include_relationships,
        'no__img__': True
    }
    # train_dset = CocoSceneGraphDataset(**dset_kwargs)
    # num_objs = train_dset.total_objects()
    # num_imgs = len(train_dset)
    # print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
    # print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    dset_kwargs['image_dir'] = args.coco_val_image_dir
    dset_kwargs['instances_json'] = args.coco_val_instances_json
    dset_kwargs['stuff_json'] = args.coco_val_stuff_json
    dset_kwargs['max_samples'] = None
    val_dset = CocoSceneGraphDataset(**dset_kwargs)
    num_imgs = len(val_dset)
    num_objs = val_dset.total_objects()
    print('Validation dataset has %d images and %d objects' % (num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    # assert train_dset.vocab == val_dset.vocab
    vocab = json.loads(json.dumps(val_dset.vocab))

    return vocab, val_dset

def load_model(model_path):
    model_name = 'resnet101'
    model = all_pretrained_models(172, name=model_name).cuda()
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()
    return model


def test_model(model, val_dataloader, input_shape=224, out_path=None):
    since = time.time()

    device = 'cuda'

    model.train(False)  # Set model to evaluate mode
    dataloader = val_dataloader

    running_corrects = 0
    objects_len = 0


    with open(out_path, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        columns = ['label'] + ['pred'] + ["logits"+str(i+1) for i in range(172)]
        writer.writerow(columns)
        
        # Iterate over data.
        for data in tqdm.tqdm(dataloader, total=len(dataloader)):
            # get the inputs
            imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, attributes = data
            imgs = imgs.to(device)
            boxes = boxes.to(device)
            obj_to_img = obj_to_img.to(device)
            labels = objs.to(device)

            objects_len += obj_to_img.shape[0]

            with torch.no_grad():
                crops = crop_bbox_batch(imgs, boxes, obj_to_img, input_shape)

            # forward
            outputs = model(crops)
            if type(outputs) == tuple:
                outputs, _ = outputs
            outputs = F.softmax(outputs, dim=-1)
           
            _, preds = torch.max(outputs, 1)

            # statistics
            running_corrects += torch.sum(preds.view(-1,1) == labels.view(-1,1))
            
            ## save logits
            labels = labels.detach().cpu().numpy().reshape(-1,1)
            outputs = outputs.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy().reshape(-1,1)
            
            writer.writerows(np.concatenate((labels,preds,outputs), axis=1).tolist())
            


    epoch_acc = running_corrects.item() / objects_len

    print('{} Acc: {:.4f}'.format("val", epoch_acc))

  
    time_elapsed = time.time() - since
    print('Validation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    
    # device = 'cuda' if args.use_gpu else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = 'cuda'
    vocab, val_loader = build_loaders(args)
    print(len(vocab['object_to_idx']))
    num_objs = 172  # len(vocab['object_to_idx'])
    print("[Load the model...]")
    # Parameters of newly constructed modules have requires_grad=True by default
    print(
        "Loading model using class: {}, use_gpu: {}, freeze_layers: {}, freeze_initial_layers: {}, name_of_model: {}".format(
            num_objs, args.gpu_id, args.freeze_layers, args.freeze_initial_layers, args.model_name))
    # model_conv = all_pretrained_models(num_objs, name=args.model_name).to(device)
    
    print("[Load model from: {}]".format(args.model_path))
    model_conv = load_model(args.model_path)
    
    
    
    if args.use_parallel:
        print("[Using all the available GPUs]")
        model_conv = nn.DataParallel(model_conv, device_ids=[0, 1])


    print("[Validate the model begun ....]")
    model_ft = test_model(model_conv, val_loader, input_shape=args.input_shape, out_path=args.out_path)
