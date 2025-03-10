from .block import Joiner 
from .vit import VisionTransformer
from .position_encoding import build_position_encoding
from collections import OrderedDict



# __all__ = ['VisionTransformer']
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    if args.backbone == 'vit':
        backbone = VisionTransformer(pretrained=None)    ### ViT-based
        
    model = Joiner(backbone, position_embedding)
    
    return model
        
