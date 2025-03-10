# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import math
from typing import Text
import torch
import torch.nn.functional as F

from torch import nn
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer
from .deconv_decoder import build_feature_extractor
from .pixel_decoder import build_pixel_deocder
from .loss import TextSRLoss_addreg_MS
from .converter import build_convertor

class Pix2Seq(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, deconv_decoder, feature_extractor, pixel_embed_dim, num_queries):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super(Pix2Seq, self).__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Linear(hidden_dim, pixel_embed_dim)

        
        self.fcn_decoder = deconv_decoder
        # self.discriminator = discriminator
        self.vgg16 = feature_extractor

        self.backbone = backbone
        self.init_size = int(math.sqrt(num_queries))
        self.pixel_embed_dim = pixel_embed_dim
        
        self.rec_query_embed = nn.Embedding(25, hidden_dim)
        self.sr_query_embed = nn.Embedding(64, hidden_dim)
        
        self.input_proj = nn.Conv2d(768, hidden_dim, kernel_size=1)
        
        self.classifier = nn.Linear(768, 94)
        

    def forward(self, samples, img_hr=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos, middle = self.backbone(samples)
        # import pdb;pdb.set_trace()
        src, mask = features[-1].decompose()
        b, N, C = src.shape[:]
        src = src.permute(0, 2, 1)
        src = src.view(b, C, 8, 32)
        assert mask is not None
        
        
        rec_query_embed = self.rec_query_embed.weight.unsqueeze(1).repeat(
            1, b, 1).permute(1, 0, 2)
        sr_query_embed = self.sr_query_embed.weight.unsqueeze(1).repeat(
            1, b, 1).permute(1, 0, 2)
        query_embed = torch.cat((rec_query_embed, sr_query_embed), dim=1) 
        hs = self.transformer(self.input_proj(src), mask, query_embed, pos[-1])[0]
        
        rec_logit = []
        pixel_output = []
        
        for i in range(3,6):
            query_output = self.query_embed(hs[i]).permute(1, 0, 2)
            rec_query, sr_query = torch.split(query_output, [25, 64], dim=1)  
            logit = self.classifier(rec_query)
            sr_output = self.fcn_decoder(src, sr_query, middle)
            
            pixel_output.append(sr_output)
            rec_logit.append(logit)
        # import pdb;pdb.set_trace()
        # query_output = self.query_embed(hs[-1]).permute(1, 0, 2)
        # rec_query, sr_query = torch.split(query_output, [25, 64], dim=1) 
        # logit = self.classifier(rec_query)
        # sr_output = self.fcn_decoder(src, sr_query, middle)

        pixel_output.append(sr_output)
        rec_logit.append(logit)

        if not self.training:
            return pixel_output[-1], rec_logit[-1]

        feat_output = []
        for p_o in pixel_output:
            feat_output.append(self.vgg16(p_o))
        feat_gt = self.vgg16(img_hr)
        
        preds = {
            'output': pixel_output,
            'rec_output': rec_logit,
            'feat_output': feat_output,
            'feat_gt': feat_gt,
        }   
        
        return preds     
    


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)
    deconv_decoder = build_pixel_deocder(args)
    # discriminator = build_discriminator(args)
    feature_extractor = build_feature_extractor(args)
    
    # transformer = build_tp_encoder(args)

    # recognizer_hr = build_recog_hr(args)
    # recognizer_lr = build_recog_lr(args)

    model = Pix2Seq(backbone, transformer, deconv_decoder, feature_extractor, args.pixel_embed_dim, args.pix2pix_queries)
    
    convertor = build_convertor(args)
    # weight = torch.ones(num_classes)
    # weight[args.end_index] = args.end_loss_coef; weight[args.noise_index] = args.noise_loss_coef; weight[args.pad_rec_index] = args.pad_rec_loss_coef
    # criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=args.padding_index)
    # criterion.to(device)
    criterion = TextSRLoss_addreg_MS().to(device)

    return model, convertor, criterion