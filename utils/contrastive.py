from torch import autograd
from torch.nn import DataParallel
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import spectral_norm
# from utils.model_ops import snlinear, linear
import numpy as np
import torch
import numpy as np


def linear(in_features, out_features, bias=True):
    return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)


def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias), eps=1e-6)

def make_mask(labels, n_cls, mask_negatives):
    # return a one-hot  likt mask. can use scatter instead.
    device = labels.device
    labels = labels.detach().cpu().numpy()
    n_samples = labels.shape[0]
    if mask_negatives:
        mask_multi, target = np.zeros([n_cls, n_samples]), 1.0
    else:
        mask_multi, target = np.ones([n_cls, n_samples]), 0.0

    for c in range(n_cls):
        c_indices = np.where(labels == c)
        mask_multi[c, c_indices] = target

    return torch.tensor(mask_multi).type(torch.long).to(device)


def set_temperature(conditional_strategy, tempering_type, start_temperature, end_temperature, step_count, tempering_step, total_step):
    if conditional_strategy == 'ContraGAN':
        if tempering_type == 'continuous':
            t = start_temperature + step_count * \
                (end_temperature - start_temperature)/total_step
        elif tempering_type == 'discrete':
            tempering_interval = total_step//(tempering_step + 1)
            t = start_temperature + \
                (step_count//tempering_interval) * \
                (end_temperature-start_temperature)/tempering_step
        else:
            t = start_temperature
    else:
        t = 'no'
    return t


class Conditional_Contrastive_loss(torch.nn.Module):
    def __init__(self, batch_size, pos_collected_numerator):
        super(Conditional_Contrastive_loss, self).__init__()
        self.batch_size = batch_size
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        # self.device = device

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        # mask = np.ones((h, w)) - np.eye(h)
        mask = torch.ones(h, w)-torch.eye(h)
        mask = (mask).type(torch.bool)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, inst_embed, proxy, negative_mask, labels, temperature, margin):
        # inst_embed: instance feature: l(x) in paper Eq.8
        # proxy: label embeding -> e(y) in Eq.8
        # negative_mask: shape?
        negative_mask = negative_mask.T  # batch first.
        similarity_matrix = self.calculate_similarity_matrix(
            inst_embed, inst_embed)
        instance_zone = torch.exp(
            (self.remove_diag(similarity_matrix) - margin)/temperature)  # 分母第二项

        inst2proxy_positive = torch.exp((self.cosine_similarity(
            inst_embed, proxy) - margin)/temperature)  # 分子和分母第一项
        if self.pos_collected_numerator:
            mask_4_remove_negatives = negative_mask[labels]
            mask_4_remove_negatives = self.remove_diag(mask_4_remove_negatives)
            inst2inst_positives = instance_zone*mask_4_remove_negatives  # 分子第二项

            numerator = inst2proxy_positive + inst2inst_positives.sum(dim=1)
        else:
            numerator = inst2proxy_positive  # no data-to-date, paper Eq.7

        denominator = torch.cat(
            [torch.unsqueeze(inst2proxy_positive, dim=1), instance_zone], dim=1).sum(dim=1)
        criterion = -torch.log(temperature*(numerator/denominator)).mean()
        return criterion


class Temperature_Conditional_Contrastive_loss(torch.nn.Module):
    def __init__(self, batch_size, temperature_dict, pos_collected_numerator):
        super(Temperature_Conditional_Contrastive_loss, self).__init__()
        self.batch_size = batch_size
        # self.temperature_dict = temperature_dict#
        self.register_buffer("temperature_dict", temperature_dict)
        # self.intra_temperature = 1
        self.pos_collected_numerator = pos_collected_numerator
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        # self.device = device

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        # mask = np.ones((h, w)) - np.eye(h)
        mask = torch.ones(h, w)-torch.eye(h)
        mask = (mask).type(torch.bool)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, inst_embed, proxy, negative_mask, labels, temperature, margin):
        # inst_embed: instance feature: l(x) in paper Eq.8
        # proxy: label embeding -> e(y) in Eq.8
        # negative_mask: shape?
        negative_mask = negative_mask.T  # batch first.
        similarity_matrix = self.calculate_similarity_matrix(
            inst_embed, inst_embed)
        temperature_table = self.temperature_dict[labels][:, labels]
        assert temperature_table.shape == similarity_matrix.shape
        instance_zone = torch.exp(
            (self.remove_diag(similarity_matrix) - margin)/self.remove_diag(temperature_table))  # 分母第二项
        # TODO: change temperature into a matrix

        inst2proxy_positive = torch.exp((self.cosine_similarity(
            inst_embed, proxy) - margin)/temperature)  # 分子和分母第一项
        if self.pos_collected_numerator:
            mask_4_remove_negatives = negative_mask[labels]
            mask_4_remove_negatives = self.remove_diag(mask_4_remove_negatives)
            inst2inst_positives = instance_zone*mask_4_remove_negatives  # 分子第二项，同类别样本

            numerator = inst2proxy_positive + inst2inst_positives.sum(dim=1)
        else:
            numerator = inst2proxy_positive  # no data-to-date, paper Eq.7

        denominator = torch.cat(
            [torch.unsqueeze(inst2proxy_positive, dim=1), instance_zone], dim=1).sum(dim=1)
        criterion = -torch.log(temperature*(numerator/denominator)).mean()
        return criterion


# DCGAN loss
def loss_dcgan_dis(dis_out_real, dis_out_fake):
    device = dis_out_real.get_device()
    ones = torch.ones_like(dis_out_real, device=device, requires_grad=False)
    dis_loss = -torch.mean(nn.LogSigmoid()(dis_out_real) +
                           nn.LogSigmoid()(ones - dis_out_fake))
    return dis_loss


def loss_dcgan_gen(gen_out_fake):
    return -torch.mean(nn.LogSigmoid()(gen_out_fake))


def loss_lsgan_dis(dis_out_real, dis_out_fake):
    dis_loss = 0.5*(dis_out_real - torch.ones_like(dis_out_real)
                    )**2 + 0.5*(dis_out_fake)**2
    return dis_loss.mean()


def loss_lsgan_gen(dis_out_fake):
    gen_loss = 0.5*(dis_out_fake - torch.ones_like(dis_out_fake))**2
    return gen_loss.mean()


def loss_hinge_dis(dis_out_real, dis_out_fake):
    return torch.mean(F.relu(1. - dis_out_real)) + torch.mean(F.relu(1. + dis_out_fake))


def loss_hinge_gen(gen_out_fake):
    return -torch.mean(gen_out_fake)


def loss_wgan_dis(dis_out_real, dis_out_fake):
    return torch.mean(dis_out_fake - dis_out_real)


def loss_wgan_gen(gen_out_fake):
    return -torch.mean(gen_out_fake)


def latent_optimise(zs, fake_labels, gen_model, dis_model, conditional_strategy, latent_op_step, latent_op_rate,
                    latent_op_alpha, latent_op_beta, trans_cost, default_device):
    batch_size = zs.shape[0]
    for step in range(latent_op_step):
        drop_mask = (torch.FloatTensor(batch_size, 1).uniform_()
                     > 1 - latent_op_rate).to(default_device)
        z_gradients, z_gradients_norm = calc_derv(
            zs, fake_labels, dis_model, conditional_strategy, default_device, gen_model)
        delta_z = latent_op_alpha*z_gradients / \
            (latent_op_beta + z_gradients_norm)
        zs = torch.clamp(zs + drop_mask*delta_z, -1.0, 1.0)

        if trans_cost:
            if step == 0:
                transport_cost = (delta_z.norm(2, dim=1)**2).mean()
            else:
                transport_cost += (delta_z.norm(2, dim=1)**2).mean()

    if trans_cost:
        return zs, trans_cost
    else:
        return zs


class Cross_Entropy_loss(torch.nn.Module):
    def __init__(self, in_features, out_features, spectral_norm=True):
        super(Cross_Entropy_loss, self).__init__()

        if spectral_norm:
            self.layer = snlinear(in_features=in_features,
                                  out_features=out_features, bias=True)
        else:
            self.layer = linear(in_features=in_features,
                                out_features=out_features, bias=True)
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, embeds, labels):
        logits = self.layer(embeds)
        return self.ce_loss(logits, labels)


class Proxy_NCA_loss(torch.nn.Module):
    def __init__(self, device, embedding_layer, num_classes, batch_size):
        super(Proxy_NCA_loss, self).__init__()
        self.device = device
        self.embedding_layer = embedding_layer
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _get_positive_proxy_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        rvs_one_hot_target = np.ones(
            [self.num_classes, self.num_classes]) - np.eye(self.num_classes)
        rvs_one_hot_target = rvs_one_hot_target[labels]
        mask = torch.from_numpy((rvs_one_hot_target)).type(torch.bool)
        return mask.to(self.device)

    def forward(self, inst_embed, proxy, labels):
        all_labels = torch.tensor([c for c in range(self.num_classes)]).type(
            torch.long).to(self.device)
        positive_proxy_mask = self._get_positive_proxy_mask(labels)
        negative_proxies = torch.exp(
            torch.mm(inst_embed, self.embedding_layer(all_labels).T))*positive_proxy_mask

        inst2proxy_positive = torch.exp(
            self.cosine_similarity(inst_embed, proxy))
        numerator = inst2proxy_positive
        denomerator = negative_proxies.sum(dim=1)
        criterion = -torch.log(numerator/denomerator).mean()
        return criterion


class NT_Xent_loss(torch.nn.Module):
    def __init__(self, batch_size, use_cosine_similarity=True):
        super(NT_Xent_loss, self).__init__()
        self.batch_size = batch_size
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(
            use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 *
                    self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 *
                    self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, temperature):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(
            representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)


class XT_Xent_loss(torch.nn.Module):
    def __init__(self, use_cosine_similarity=True):
        super(XT_Xent_loss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_fn = self._get_similarity_function(
            use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, num_o):
        mask = torch.cat([torch.arange(num_o)
                           for i in range(2)], dim=0)
        mask = (mask.unsqueeze(0) == mask.unsqueeze(1))
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    @staticmethod
    def _cosine_simililarity(x):
        x = F.normalize(x, dim=1)
        v = torch.mm(x, x.T)# better performance
        return v

    def forward(self, feats, feats_gt):
        similarity_matrix = self.similarity_fn(feats)
        similarity_matrix_gt = self.similarity_fn(feats_gt)

        num_o = feats.shape[0]//2
        corr_mask = self._get_correlated_mask(num_o).to(feats.device)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(corr_mask.shape[0], dtype=torch.bool, device=feats.device)
        corr_mask = corr_mask[~mask].view(corr_mask.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)
        similarity_matrix_gt = similarity_matrix_gt[~mask].view(
            similarity_matrix_gt.shape[0], -1)
        
        # select and combine multiple positives
        positives = similarity_matrix[corr_mask].view(corr_mask.shape[0], -1)
        positives_gt = similarity_matrix_gt[corr_mask].view(corr_mask.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~corr_mask.bool()].view(
            similarity_matrix.shape[0], -1)
        negatives_gt = similarity_matrix_gt[~corr_mask.bool()].view(
            similarity_matrix_gt.shape[0], -1)
        
    
        logits = torch.cat((positives, negatives), dim=1)
        temperature = torch.cat((positives_gt, negatives_gt), dim=1)
        logits /= temperature

        labels = torch.zeros(2 * num_o, dtype=torch.long, device=feats.device)
        loss = self.criterion(logits, labels)
        return loss / (2 * num_o)


def calc_derv4gp(netD, conditional_strategy, real_data, fake_data, real_labels, device):
    batch_size, c, h, w = real_data.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_data.nelement() //
                         batch_size).contiguous().view(batch_size, c, h, w)
    alpha = alpha.to(device)

    real_data = real_data.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    if conditional_strategy in ['ContraGAN', "Proxy_NCA_GAN", "NT_Xent_GAN"]:
        _, _, disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy in ['ProjGAN', 'no']:
        disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy == 'ACGAN':
        _, disc_interpolates = netD(interpolates, real_labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calc_derv4dra(netD, conditional_strategy, real_data, real_labels, device):
    batch_size, c, h, w = real_data.shape
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.to(device)

    real_data = real_data.to(device)
    differences = 0.5*real_data.std()*torch.rand(real_data.size()).to(device)

    interpolates = real_data + (alpha*differences)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    if conditional_strategy in ['ContraGAN', "Proxy_NCA_GAN", "NT_Xent_GAN"]:
        _, _, disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy in ['ProjGAN', 'no']:
        disc_interpolates = netD(interpolates, real_labels)
    elif conditional_strategy == 'ACGAN':
        _, disc_interpolates = netD(interpolates, real_labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calc_derv(inputs, labels, netD, conditional_strategy, device, netG=None):
    zs = autograd.Variable(inputs, requires_grad=True)
    fake_images = netG(zs, labels)

    if conditional_strategy in ['ContraGAN', "Proxy_NCA_GAN", "NT_Xent_GAN"]:
        _, _, dis_out_fake = netD(fake_images, labels)
    elif conditional_strategy in ['ProjGAN', 'no']:
        dis_out_fake = netD(fake_images, labels)
    elif conditional_strategy == 'ACGAN':
        _, dis_out_fake = netD(fake_images, labels)
    else:
        raise NotImplementedError

    gradients = autograd.grad(outputs=dis_out_fake, inputs=zs,
                              grad_outputs=torch.ones(
                                  dis_out_fake.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients_norm = torch.unsqueeze((gradients.norm(2, dim=1) ** 2), dim=1)
    return gradients, gradients_norm
