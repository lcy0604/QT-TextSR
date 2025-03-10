# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from collections import OrderedDict

import timm.models.vision_transformer
import torch
import torch.nn as nn

from util.misc import NestedTensor
import torch.nn.functional as F

# from unitext.builder import BACKBONES

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding.
    """

    def __init__(self,
                 img_size=(32, 128),
                 patch_size=(32, 16),
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})." # noqa
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self,
                 global_pool=False,
                 in_chans=3,
                 patch_size=4,
                 img_size=(32, 128),
                 embed_dim=768,       # VIT-base: 768  VIT-small: 384
                 depth=12,
                 num_heads=12,         # VIT-base: 12  VIT-small: 6
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 pretrained=None,
                 ignore_loding_patch_embed=False,
                 with_patch_embed=True,
                 pretrained_img_size=(56, 224),
                 freeze=False,
                 **kwargs):
        super(VisionTransformer, self).__init__(
            patch_size=patch_size,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            **kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.reset_classifier(0)

        # use a more flexible pathch embedding
        self.with_patch_embed = with_patch_embed
        self.in_chans = in_chans
        if with_patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=(patch_size, patch_size),
                in_chans=in_chans,
                embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches
        else:
            num_patches = img_size[0] * img_size[1]
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.apply(self._init_weights)
        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % pretrained)
            try:
                checkpoint_model = checkpoint['model']
            except:
                checkpoint_model = checkpoint
            state_dict = self.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[
                        k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            if ignore_loding_patch_embed:
                print("Ignore patch embedding")
                for k in ['patch_embed.proj.weight', 'patch_embed.proj.bias']:
                    if k in checkpoint_model:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
            self.interpolate_pos_embed(checkpoint_model)
            msg = self.load_state_dict(checkpoint_model, strict=False)
            # print(msg)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward_features(self, x):
        B = x.shape[0]
        if self.with_patch_embed:
            x = self.patch_embed(x)
        else:
            # features are from hybrid model in shape (N, C, 1, W)
            assert x.shape[
                2] == 1, "Input features must have shape (N, C, 1, W)"
            assert x.shape[1] == self.in_chans, \
                f"Input features must have shape (N, {self.in_chans}, 1,  W)"
            x = x.squeeze(2).permute(0, 2, 1)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        middle = []
        for blk in self.blocks:
            x = blk(x)
            middle.append(x[:, 1:])
            
        x = self.norm(x)
        # remove the cls token
        x = x[:, 1:]
        return x, middle

    def forward(self, tensor_list: NestedTensor):
        output = []
        x = tensor_list.tensors
        mask = tensor_list.mask
        mask = F.interpolate(mask.to(torch.float).unsqueeze(1), (8, 32)).to(torch.bool).squeeze(1)    ### change
        out, middle = self.forward_features(x)
        output.append(NestedTensor(out, mask))
        return output, middle
    # def forward(self, x):
    #     return self.forward_features(x)

    def interpolate_pos_embed(self, checkpoint_model):
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.patch_embed.num_patches
            num_extra_tokens = self.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int(
                (pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches**0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" %
                      (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                                embedding_size).permute(
                                                    0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode='bicubic',
                    align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed



if __name__ == '__main__':
    model = PatchEmbed(
        img_size=(32, 1024), patch_size=(32, 8), in_chans=3, embed_dim=192)
    input = torch.randn(1, 3, 32, 1024)
    print(model(input).shape())
