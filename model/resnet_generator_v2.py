import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_module import *
from .mask_regression import *
import numpy as np
from .components import *



class ResnetGenerator128_base2(nn.Module):
    def __init__(self, ch=64, z_dim=64, num_classes=10, output_dim=3):
        super(ResnetGenerator128_base2, self).__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.label_embedding = nn.Embedding(num_classes, 180)
        self.pos_emb = nn.utils.spectral_norm(nn.Linear(4, 64))

        num_w = 64 + 180 + 64# noise + label + position
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))

        self.res1 = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_w)
        self.res2 = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch * 4, ch * 2, upsample=True, num_w=num_w, psp_module=True)
        self.res5 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False)
        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))

        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()

    def forward(self, z, bbox, z_im=None, y=None):
        """
        z: z_obj, bs*obj*z_dim
        bbox:[x0,y0,w,h]
        """

        b, o = z.size(0), z.size(1)
        label_embedding = self.label_embedding(y)
        pos_embedding = self.pos_emb(bbox)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)
        pos_embedding = pos_embedding.view(b * o, -1)
      

        latent_vector = torch.cat((z, label_embedding, pos_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1))
        # preprocess bbox
        bmask = self.mask_regress(w, bbox)

        if z_im is None:
            z_im = torch.randn((b, self.z_dim), device=z.device)

        bbox_mask_ = bbox_mask(z, bbox, 64, 64)

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x, _ = self.res5(x, w, stage_bbox)

        # save_path1 = 'samples/tmp/edit/apponly/1292_bmask2_0.npy'
        # save_path2 = 'samples/tmp/edit/apponly/1292_stage2_bbox_0.npy'
        # np.save(save_path1, bmask.cpu().detach().numpy())
        # np.save(save_path2, stage_bbox.cpu().detach().numpy())

        # to RGB
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

class ResnetGenerator128(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator128, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128 + 180
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))

        self.res1 = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_w)
        self.res2 = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch * 4, ch * 2, upsample=True, num_w=num_w, psp_module=True)
        self.res5 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False)
        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))

        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()

    def forward(self, z, bbox, z_im=None, y=None):
        b, o = z.size(0), z.size(1)
        label_embedding = self.label_embedding(y)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1))
        # preprocess bbox
        bmask = self.mask_regress(w, bbox)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        bbox_mask_ = bbox_mask(z, bbox, 64, 64)

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x, _ = self.res5(x, w, stage_bbox)

        # save_path1 = 'samples/tmp/edit/apponly/1292_bmask2_0.npy'
        # save_path2 = 'samples/tmp/edit/apponly/1292_stage2_bbox_0.npy'
        # np.save(save_path1, bmask.cpu().detach().numpy())
        # np.save(save_path2, stage_bbox.cpu().detach().numpy())

        # to RGB
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

class ResnetGenerator128_lama(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator128_lama, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 160)
        self.z_dim = z_dim
        num_w = z_dim + self.label_embedding.embedding_dim
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4*4*8*ch))

        self.res1 = ResBlockG(ch*8, ch*8, upsample=True, num_w=num_w)
        self.res2 = ResBlockG(ch*8, ch*4, upsample=True, num_w=num_w)
        self.res3 = ResBlockG(ch*4, ch*4, upsample=True, num_w=num_w)
        self.res4 = ResBlockG(ch*4, ch*2, upsample=True, num_w=num_w)
        self.res5 = ResBlockG(ch*2, ch*1, upsample=True, num_w=num_w)
        self.final = nn.Sequential(nn.BatchNorm2d(ch),
                                   nn.LeakyReLU(0.01),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.mask_regress = MaskRegressNet(num_w+2, map_size=128)

        self.style_mapping = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.01),
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.01),
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim))
        )
        self.init_parameter()
        print(f"ResnetGenerator128 initialized")

    def forward(self, z, bbox, z_im=None, y=None, return_mask=False):
        b, o = z.size(0), z.size(1)
        z, bbox = z.cuda(), bbox.cuda()
        
        label_embedding = self.label_embedding(y)

        mask_latent_vector = torch.cat([label_embedding, z, bbox[:,:,2:]], dim=2) # b*o*(num_w+2)
        if return_mask:
            mask, raw_mask = self.mask_regress(mask_latent_vector, bbox, return_raw=True)
        else:
            mask = self.mask_regress(mask_latent_vector, bbox)
        w = torch.cat( [label_embedding, self.style_mapping(z.view(b*o, -1)).view(b,o,-1)], dim=2)  # b*o*num_w
        
        if z_im is None:
            z_im = torch.randn((b, self.z_dim), device=z.device)

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        x = self.res1(x, w, mask)
        # 16x16
        x = self.res2(x, w, mask)
        # 32x32
        x = self.res3(x, w, mask)
        # 64x64
        x = self.res4(x, w, mask)
        # 128x128
        x = self.res5(x, w, mask)
        # to RGB
        x = self.final(x)
        return x if not return_mask else [x, mask, raw_mask]

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

class ResnetGenerator256(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator256, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128 + 180
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))

        self.res1 = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_w)
        self.res2 = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch * 8, ch * 8, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_w)
        self.res5 = ResBlock(ch * 4, ch * 2, upsample=True, num_w=num_w)
        self.res6 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False)
        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha5 = nn.Parameter(torch.zeros(1, 184, 1))
        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()

    def forward(self, z, bbox, z_im=None, y=None, include_mask_loss=False):
        b, o = z.size(0), z.size(1)

        label_embedding = self.label_embedding(y)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1))

        # preprocess bbox
        bmask = self.mask_regress(w, bbox)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        bbox_mask_ = bbox_mask(z, bbox, 128, 128)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)
        w = self.mapping(latent_vector.view(b * o, -1))

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        # label mask
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x, stage_mask = self.res5(x, w, stage_bbox)

        # 256x256
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))  # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha5 = torch.gather(self.sigmoid(self.alpha5).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha5) + seman_bbox * alpha5
        x, _ = self.res6(x, w, stage_bbox)
        # to RGB
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

