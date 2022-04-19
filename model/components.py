from torch import nn
from torch.nn import functional as F
from .norm_module import *
from .sync_batchnorm import SynchronizedBatchNorm2d
BatchNorm = SynchronizedBatchNorm2d


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, upsample=False, num_w=128, predict_mask=True, psp_module=False):
        super(ResBlock, self).__init__()
        self.upsample = upsample
        self.h_ch = h_ch if h_ch else out_ch
        self.conv1 = conv2d(in_ch, self.h_ch, ksize, pad=pad)
        self.conv2 = conv2d(self.h_ch, out_ch, ksize, pad=pad)
        self.b1 = SpatialAdaptiveSynBatchNorm2d(in_ch, num_w=num_w, batchnorm_func=BatchNorm)
        self.b2 = SpatialAdaptiveSynBatchNorm2d(self.h_ch, num_w=num_w, batchnorm_func=BatchNorm)
        self.learnable_sc = in_ch != out_ch or upsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()

        self.predict_mask = predict_mask
        if self.predict_mask:
            if psp_module:
                self.conv_mask = nn.Sequential(PSPModule(out_ch, 100),
                                               nn.Conv2d(100, 184, kernel_size=1))
            else:
                self.conv_mask = nn.Sequential(nn.Conv2d(out_ch, 100, 3, 1, 1),
                                               BatchNorm(100),
                                               nn.ReLU(),
                                               nn.Conv2d(100, 184, 1, 1, 0, bias=True))

    def residual(self, in_feat, w, bbox):
        x = in_feat
        x = self.b1(x, w, bbox)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.b2(x, w, bbox)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.c_sc(x)
        return x

    def forward(self, in_feat, w, bbox):
        out_feat = self.residual(in_feat, w, bbox) + self.shortcut(in_feat)
        if self.predict_mask:
            mask = self.conv_mask(out_feat)
        else:
            mask = None
        return out_feat, mask



def batched_index_select(input, dim, index):
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def bbox_mask(x, bbox, H, W):
    b, o, _ = bbox.size()
    N = b * o

    bbox_1 = bbox.float().view(-1, 4)
    x0, y0 = bbox_1[:, 0], bbox_1[:, 1]
    ww, hh = bbox_1[:, 2], bbox_1[:, 3]

    x0 = x0.contiguous().view(N, 1).expand(N, H)
    ww = ww.contiguous().view(N, 1).expand(N, H)
    y0 = y0.contiguous().view(N, 1).expand(N, W)
    hh = hh.contiguous().view(N, 1).expand(N, W)

    X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).cuda(device=x.device)
    Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).cuda(device=x.device)

    X = (X - x0.to(X.device)) / ww.to(X.device)
    Y = (Y - y0.to(Y.device)) / hh.to(Y.device)

    X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
    Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)

    out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1)
    return out_mask.view(b, o, H, W)


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn, nn.ReLU())

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


# adopted from https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L280
# LAMA used
class NoiseInjection(nn.Module):
    def __init__(self, full=False):
        super().__init__()
        self.noise_weight_seed = nn.Parameter(torch.tensor(0.0))
        self.full = full

    def forward(self, image, noise=None):
        if noise is None:
            batch, channel, height, width = image.shape
            noise = image.new_empty(batch, [1, channel][self.full], height, width).normal_()
        return image + F.softplus(self.noise_weight_seed) * noise

class ChannelwiseNoiseInjection(nn.Module):
    def __init__(self, num_channels, full=False):
        super().__init__()
        self.noise_weight_seed = nn.Parameter(torch.zeros((1, num_channels, 1, 1)))
        self.num_channels = num_channels
        self.full = full

    def forward(self, image, noise=None):
        if noise is None:
            batch, channel, height, width = image.shape
            noise = image.new_empty(batch, [1, channel][self.full], height, width).normal_()
        return image + F.softplus(self.noise_weight_seed) * noise
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_channels) + ')'

class ResBlockG(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, upsample=False, num_w=128):
        super().__init__()
        self.upsample = upsample
        self.h_ch = h_ch if h_ch else out_ch
        self.conv1 = conv2d(in_ch, self.h_ch, ksize, pad=pad, bias=False)
        self.conv2 = conv2d(self.h_ch, out_ch, ksize, pad=pad, bias=False)
        self.b1 = SpatialAdaptiveSynBatchGroupNorm2d(in_ch, num_w=num_w)
        self.b2 = SpatialAdaptiveSynBatchGroupNorm2d(self.h_ch, num_w=num_w)
        self.learnable_sc = in_ch != out_ch or upsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.LeakyReLU(0.01)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.out_ch = out_ch
        # self.noise1 = NoiseInjection()
        # self.noise2 = NoiseInjection()
        self.noise1 = ChannelwiseNoiseInjection(self.h_ch)
        self.noise2 = ChannelwiseNoiseInjection(out_ch)
        
    def residual(self, in_feat, w, bbox):
        x = in_feat
        x = self.b1(x, w, bbox)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.b2(x, w, bbox)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.noise2(x)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.c_sc(x)
        return x

    def forward(self, in_feat, w, bbox):
        return self.alpha * self.residual(in_feat, w, bbox) + self.shortcut(in_feat)



class ResBlockD(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super().__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.activation = nn.LeakyReLU(0.01)
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def residual(self, in_feat):
        x = in_feat
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, in_feat):
        return self.alpha.clamp(-1,1) * self.residual(in_feat) + self.shortcut(in_feat)

def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True, bias=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


class MaskRegressBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, bias = False):
        super().__init__()
        conv = list()
        conv.append(nn.BatchNorm2d(channels))
        conv.append(nn.LeakyReLU(0.01))
        conv.append(conv2d(channels, channels, kernel_size, bias = bias))
        self.conv = nn.Sequential(*conv)
        self.alpha = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        return x + self.alpha * self.conv(x)


# BGN+SPADE 
class SpatialAdaptiveSynBatchGroupNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512):
        super().__init__()
        self.num_features = num_features
        self.weight_proj = nn.utils.spectral_norm(
            nn.Linear(num_w, num_features))
        self.bias_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features))
        self.batch_norm2d = nn.BatchNorm2d(num_features, eps=1e-5, affine=False,
                            momentum=0.1, track_running_stats=True)

        self.group_norm = nn.GroupNorm(4, num_features, eps=1e-5, affine=False)
        self.rho = nn.Parameter(torch.tensor(0.1)) # the ratio of GN

        self.alpha = nn.Parameter(torch.tensor(0.0)) # the scale of the affined 

    def forward(self, x, vector, bbox):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        self.batch_norm2d._check_input_dim(x)
        # use BGN
        output_b = self.batch_norm2d(x)
        output_g = self.group_norm(x)
        output = output_b + self.rho.clamp(0,1) * (output_g - output_b)

        b, o, _, _ = bbox.size()
        _, _, h, w = x.size()
        bbox = F.interpolate(bbox, size=(h, w), mode='bilinear', align_corners=False) # b o h w
        weight, bias = self.weight_proj(vector), self.bias_proj(vector) # b*o d

        bbox_non_spatial = bbox.view(b, o, -1) # b o h*w
        bbox_non_spatial_margin = bbox_non_spatial.sum(dim=1, keepdim=True) + torch.tensor(1e-4) # b 1 h*w
        bbox_non_spatial.div_(bbox_non_spatial_margin)
        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1) # b o d
        weight.transpose_(1, 2), bias.transpose_(1, 2) # b d o
        weight, bias = torch.bmm(weight, bbox_non_spatial), torch.bmm(bias, bbox_non_spatial) # b d h*w
        # weight.div_(bbox_non_spatial_margin), bias.div_(bbox_non_spatial_margin) # b d h*w
        weight, bias = weight.view(b, -1, h, w), bias.view(b, -1, h, w)

        # weight = torch.sum(bbox * weight, dim=1, keepdim=False) / \
        #     (torch.sum(bbox, dim=1, keepdim=False) + 1e-6) # b d h w
        # bias = torch.sum(bbox * bias, dim=1, keepdim=False) / \
        #     (torch.sum(bbox, dim=1, keepdim=False) + 1e-6) # b d h w
        affined = weight * output + bias
        return output + self.alpha.clamp(-1, 1) * affined

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
