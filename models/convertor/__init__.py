from .attn import AttnConvertor
from .base import BaseConvertor
from .ctc import CTCConvertor

def build_convertor(args):
    if args.rec_type == 'attn':
        # backbone = VisionTransformer_v2(pretrained="/home/pci/disk1/lcy/unitext/SR_single/SwinErase-main/pretrained/300w_jinsanadd_crop_20w_jinsantest_crop.pth")
        converter = AttnConvertor()
    if args.rec_type == 'ctc':
        converter = CTCConvertor()
    
    # model = Joiner(backbone, position_embedding)
    
    return converter

# __all__ = ['BaseConvertor', 'CTCConvertor', 'AttnConvertor']
