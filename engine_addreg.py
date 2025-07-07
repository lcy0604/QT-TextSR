# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import os
import sys
import cv2
import math
import json
import torch
import numpy as np
import util.misc as utils

import re

from typing import Iterable
from tqdm import tqdm
from util.visualize import tensor_to_cv2image
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F

def cal_psnr(img1, img2):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # img1 and img2 have range [0, 1]

    mse = ((img1[:,:3,:,:]*255 - img2[:,:3,:,:]*255)**2).mean()
    # mse = (((img1[:,:3,:,:] + 1) / 2.0 * 255.0 - (img2[:,:3,:,:] + 1) / 2.0 * 255.0)**2).mean()
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2,
        groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        img1 = img1[:,:3,:,:]
        img2 = img2[:,:3,:,:]
        # img1 = (img1[:,:3,:,:] + 1) / 2.0
        # img2 = (img2[:,:3,:,:] + 1) / 2.0
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def visual(image):
    from PIL import Image
    im =(image).transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).save('1.jpg')

def train_one_epoch(model: torch.nn.Module, 
                    criterion: torch.nn.Module, data_loader: Iterable, convertor, optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    lr_scheduler: list = [0], print_freq: int = 10, debug: bool = False,
                    optim_with_mask: bool = False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # print_freq = 10
    # optimizer['D'].param_groups[0]['lr'] = lr_scheduler[epoch]
    optimizer['G'].param_groups[0]['lr'] = lr_scheduler[epoch]
    optimizer['G'].param_groups[1]['lr'] = lr_scheduler[epoch] * 0.1

    if debug:
        count = 0
        save_folder = '../debug/20220112-vis-scutens-train-textmask'
        os.makedirs(save_folder, exist_ok=True)
        for data in tqdm(data_loader):
            for image, label, mask, mask_gt in zip(data['image'], data['label'], data['mask'], data['mask_gt']):
                image = tensor_to_cv2image(image, False)
                label = tensor_to_cv2image(label, False)
                mask = tensor_to_cv2image(mask, False)
                mask_gt = tensor_to_cv2image(mask_gt, False)
                cv2.imwrite(os.path.join(save_folder, f'{count:06d}-image.jpg'), image)
                cv2.imwrite(os.path.join(save_folder, f'{count:06d}-label.jpg'), label)
                cv2.imwrite(os.path.join(save_folder, f'{count:06d}-mask.jpg'), mask)
                cv2.imwrite(os.path.join(save_folder, f'{count:06d}-mask_gt.jpg'), mask_gt)
                count += 1
        return

    for data in metric_logger.log_every(data_loader, print_freq, header):
        
        img_HR, img_lr = data['img_HR'].to(device), data['img_lr'].to(device) #, data[4].to(device), data[5].to(device)
        label = data['label']
        # import pdb;pdb.set_trace()
        outputs = model(img_lr, img_HR)
        
        # real_prob = discriminator(img_HR)
        
        targets, lengths = convertor.str2tensor(label)
        targets = targets.to(device)
        # import pdb;pdb.set_trace()
        # fake_prob_D = discriminator(outputs['output'].contiguous().detach())
        
        # D_loss = criterion.discriminator_loss(real_prob, fake_prob_D)
        # optimizer['D'].zero_grad()
        # D_loss.backward()
        # optimizer['D'].step()
        
        # fake_prob_G = discriminator(outputs['output'])
        # outputs['real_prob'] = real_prob
        # outputs['fake_prob_D'] = fake_prob_D
        # outputs['fake_prob_G'] = fake_prob_G   
        
        loss_dict = criterion(outputs, img_HR, targets, lengths)     

        # loss_dict['D_loss'] = D_loss
        weight_dict = {'MSR_loss': 1, 'prc_loss': 0.01, 'style_loss': 50, 'rec_loss': 1, 'img_qua_loss':1, 'ssim_loss': 1} 
        # weight_dict = {'MSR_loss': 1, 'prc_loss': 0.01, 'style_loss': 50, 'img_qua_loss':1, 'ssim_loss': 1}
        for k in loss_dict.keys():
            loss_dict[k] *= weight_dict[k]

        G_loss = sum([loss_dict[k] for k in loss_dict.keys() if k != 'D_loss'])
        # import pdb;pdb.set_trace()
        optimizer['G'].zero_grad()
        G_loss.backward()
        optimizer['G'].step()

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()} 
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items()}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
    
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(lr=optimizer['G'].param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, convertor, data_loader, device, output_dir, chars, start_index, visualize=False):
    model.eval()
    criterion.eval()
    chars = list(chars)

    index = 0
    all_psnr = 0.0
    all_ssim = 0.0
    acc = 0

    save_folder = os.path.join(output_dir, 'textzoom')
    os.makedirs(save_folder, exist_ok=True)
    im_folder = os.path.join(save_folder, 'sr')
    os.makedirs(im_folder, exist_ok=True)
    lr_im_folder = os.path.join(save_folder, 'lr')
    os.makedirs(lr_im_folder, exist_ok=True)
    recog_label_file = save_folder + '/label.txt'
    f1 = open(recog_label_file, 'w')

    psnr_list = []

    for data in tqdm(data_loader):
        # img_HR, img_lr, img_lr_up, mask = data['img_HR'].to(device), data['img_lr'].to(device), data['img_lr_up'].to(device), data['mask'].to(device)
        # images = data['image'].to(device)
        img_HR, img_lr = data['img_HR'].to(device), data['img_lr'].to(device) #, data[4].to(device), data[5].to(device)
        label = data['label']

        # file_name = data['file_name'][0].split('/')[-1]
        
        outputs, rec_logits = model(img_lr)
        # import pdb;pdb.set_trace()
        psnr = cal_psnr(outputs, img_HR)
        ssim = SSIM()(outputs, img_HR)
        
        
        pred_indexes, _ =convertor.tensor2idx(rec_logits)
        out = convertor.idx2str(pred_indexes)

        psnr_list.append(psnr)
        
        all_psnr += psnr
        all_ssim += ssim

        # outputs = model(images)
        
        img_HR = img_HR[0].cpu()
        img_HR = tensor_to_cv2image(img_HR)
        # w = img_HR.shape[1]

        output = outputs[0]
        output = output.cpu() #.clamp(min=0, max=1)
        output = tensor_to_cv2image(output)
        
        h, w = img_HR.shape[0], img_HR.shape[1]

        # if output.shape[1] >= w:
        #     output = output[:, :w, :]
        # else:
        #     output = output #cv2.resize(output, (w, h))
        
        
        img_lr_saved = img_lr[0].cpu()
        img_lr_saved = tensor_to_cv2image(img_lr_saved)
        
        # mask_pred = mask_pred[0].cpu()
        # mask_pred_saved = tensor_to_cv2image(mask_pred)
        # save_path_results = os.path.join(im_folder, file_name)
        # save_path_lr = os.path.join(lr_im_folder, file_name)
        # import pdb;pdb.set_trace()

        save_path_lr = os.path.join(im_folder, '%09d_lr.jpg' % index)
        # save_path_mask = os.path.join(im_folder, '%09d_hr.jpg' % index)

        save_path_results = os.path.join(im_folder, '%09d_sr.jpg' % index)
        # save_path_lr = os.path.join(im_folder, '%09d_lr.jpg' % index)
        # save_path_mask = os.path.join(im_folder, '%09d_hr.jpg' % index)
        
        f1.writelines('%09d_sr.jpg' % index + ' ' + label[0] + '\n')
        
        cv2.imwrite(save_path_results, output)
        cv2.imwrite(save_path_lr, img_lr_saved)
        # cv2.imwrite(save_path_mask, img_HR)
        index += 1
    print(index)
    print('psnr: ', all_psnr/index)
    print('ssim: ', all_ssim/index)
