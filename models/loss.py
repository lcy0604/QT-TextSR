import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import List
from math import exp
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class CELoss(nn.Module):
    """Implementation of loss module for encoder-decoder based text recognition
    method with CrossEntropy loss.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
    """

    def __init__(self, ignore_index=-1, reduction='mean'):
        super().__init__()
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor,
                lengths: List[int]):
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            targets (Tensor): A tensor of shape :math:`(N, T)`.
            lengths (list[int]): A list of length :math:`(N)`. Unused.
        Returns:
            dict: A loss dict with the key ``loss_ce``.
        """
        outputs = outputs[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        losses = loss_ce #dict(ce_loss=loss_ce)

        return losses

class ImageLoss(nn.Module):
    def __init__(self, gradient=True, loss_weight=[20, 1e-4]):
        super(ImageLoss, self).__init__()
        # self.mse = nn.L1Loss(reduce=False)
        self.mse = nn.MSELoss(reduce=False)
        if gradient:
            self.GPLoss = GradientPriorLoss()
        self.gradient = gradient
        self.loss_weight = loss_weight

    def forward(self, out_images, target_images, grad_mask=None):

        # if not grad_mask is None:
        #     out_images *= grad_mask
        #     target_images *= grad_mask
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        if self.gradient:

            mse_loss = self.mse(out_images, target_images).mean(1).mean(1).mean(1)

            loss = self.loss_weight[0] * mse_loss + \
                   self.loss_weight[1] * self.GPLoss(out_images[:, :3, :, :], target_images[:, :3, :, :])
        else:
            loss = self.loss_weight[0] * mse_loss

        return loss


class GradientPriorLoss(nn.Module):
    def __init__(self, ):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss(reduce=False)

    def forward(self, out_images, target_images):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        map_out = self.gradient_map(out_images)
        map_target = self.gradient_map(target_images)

        g_loss = self.func(map_out, map_target)

        return g_loss.mean(1).mean(1).mean(1)

    @staticmethod
    def gradient_map(x):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        batch_size, channel, h_x, w_x = x.size()
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2)+1e-6, 0.5)
        return xgrad

class SemanticLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(SemanticLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.margin = margin

        self.lambda1 = 1.0
        self.lambda2 = 1.0
        
        self.kl_loss = torch.nn.KLDivLoss(reduction = 'mean')
        

    def forward(self, pred_vec, gt_vec):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # pred_vec: [N, C]
        # gt_vec: [N, C]
        # mean_sim = torch.mean(self.cos_sim(gt_vec, pred_vec))
        # sim_loss = 1 - mean_sim
        
        #noise =  Variable(torch.rand(pred_vec.shape)) * 0.1 - 0.05

        #normed_pred_vec = pred_vec + noise.to(pred_vec.device)
        # print("pred_vec:", pred_vec.shape)
        # import pdb;pdb.set_trace()
        norm_vec = torch.abs(gt_vec - pred_vec)
        margin_loss = torch.mean(norm_vec) #

        # pr int("sem_loss:", float(margin_loss.data), "sim_loss:", float(sim_loss.data))
        ce_loss = self.kl_loss(torch.log(pred_vec + 1e-20), gt_vec + 1e-20)
        # print("sem_loss:", float(margin_loss.data), "sim_loss:", float(sim_loss.data))

        return self.lambda1 * margin_loss + self.lambda2 * ce_loss# ce_loss #margin_loss # + ce_loss #  + sim_loss #margin_loss +

    def cross_entropy(self, pred_vec, gt_vec, l=1e-5):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cal = gt_vec * torch.log(pred_vec+l) + (1 - gt_vec) * torch.log(1 - pred_vec+l)
        #print("cal:", cal)
        return -cal


class TextSRLoss_addreg_MS(nn.Module):
    def __init__(self):
        super(TextSRLoss_addreg_MS, self).__init__()
        self.img_loss = ImageLoss()
        self.rec_loss = CELoss()

    def ssim_loss(self, img1, img2, window_size=11, channel=3, size_average = True):
        window = create_window(window_size, channel)
        window = window.cuda()
        # import pdb;pdb.set_trace()
        loss = _ssim(img1, img2, window, window_size, channel, size_average)
        return loss

    def forward(self, preds, gt, text_label, text_lenghth):
        # mask_loss = self.mask_loss(preds['mask'], 1 - mask_gt)
        # D_loss = self.discriminator_loss(preds['real_prob'], preds['fake_prob_D'])
        # D_fake = -torch.mean(preds['fake_prob_G'])
        # import pdb;pdb.set_trace()
        msr_loss = self.MSR_loss(preds['output'], gt) * 1
        prc_loss = self.percetual_loss(preds['feat_output'][0], preds['feat_gt']) + self.percetual_loss(preds['feat_output'][1], preds['feat_gt']) + self.percetual_loss(preds['feat_output'][2], preds['feat_gt'])
        style_loss = self.style_loss(preds['feat_output'][0], preds['feat_gt']) + self.style_loss(preds['feat_output'][1], preds['feat_gt']) + self.style_loss(preds['feat_output'][2], preds['feat_gt'])
        
        img_qua_loss = self.img_loss(preds['output'][0], gt) * 1 + self.img_loss(preds['output'][1], gt) * 3 + self.img_loss(preds['output'][2], gt) * 5
        # import pdb;pdb.set_trace()
        
        rec_loss = self.rec_loss(preds['rec_output'][0], text_label, text_lenghth) * 1 + self.rec_loss(preds['rec_output'][1], text_label, text_lenghth) * 3 + self.rec_loss(preds['rec_output'][2], text_label, text_lenghth) * 5

        loss_ssim = 1 * (1 - self.ssim_loss(preds['output'][0], gt)) + 3 * (1 - self.ssim_loss(preds['output'][1], gt)) + 5 * (1 - self.ssim_loss(preds['output'][2], gt))
        
        # losses = {'MSR_loss': msr_loss, 'prc_loss': prc_loss, 'style_loss': style_loss,
        #          'img_qua_loss': img_qua_loss.mean(), 'ssim_loss': loss_ssim}
        losses = {'MSR_loss': msr_loss, 'prc_loss': prc_loss, 'style_loss': style_loss,
                 'img_qua_loss': img_qua_loss.mean(), 'rec_loss': rec_loss, 'ssim_loss': loss_ssim}
        return losses

    def mask_loss(self, mask_pred, mask_label):
        return dice_loss(mask_pred, mask_label)
    
    @staticmethod
    def discriminator_loss(real_prob, fake_prob):
        return hinge_loss(real_prob, 1) + hinge_loss(fake_prob, -1)
        
    
    def percetual_loss(self, feat_output, feat_gt):
        pcr_losses = []
        for i in range(3):
            pcr_losses.append(F.l1_loss(feat_output[i], feat_gt[i]))
        return sum(pcr_losses)
    
    def style_loss(self, feat_output, feat_gt):
        style_losses = []
        for i in range(3):
            style_losses.append(F.l1_loss(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i])))
        return sum(style_losses)

    def MSR_loss(self, outputs, gt, weights=[4,6,8]):
        msr_loss = []
        for i in range(3):
            msr_loss.append(F.l1_loss(outputs[i], gt) * weights[i])
        return sum(msr_loss)



def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def hinge_loss(input, target):
    return torch.mean(F.relu(1 - target * input))

def dice_loss(input, target):
    input = torch.sigmoid(input)
    input = input.flatten(1)
    target = target.flatten(1)

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(input * target, 1) + 0.001
    dice_loss = (2 * a) / (b + c)
    dice_loss = torch.mean(dice_loss)
    return 1 - dice_loss