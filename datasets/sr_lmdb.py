from multiprocessing.util import sub_debug
import os
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import sampler
# from torch.nn import functional as F
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import torchvision.transforms as transforms
import lmdb
import six
import sys
import bisect
import warnings
from PIL import Image
import numpy as np
import string
import math
from typing import List, Tuple, Optional
from collections.abc import Sequence
import imgaug.augmenters as iaa
from pathlib import Path

def buf2PIL(txn, key, type='RGB'):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    return im

class lmdbDataset(Dataset):
    def __init__(self, root=None, img_w = 128, img_h = 32, voc_type='upper', max_len=100, phase='train'):
        super(lmdbDataset, self).__init__()
        self.env = lmdb.open(
            str(root),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b'num-samples'))
            self.nSamples = nSamples
        self.voc_type = voc_type
        self.max_len = max_len
        self.phase = phase
        # self.toTensor = transforms.ToTensor()

        sometimes = lambda aug: iaa.Sometimes(0.25, aug)
        if self.phase == 'train':
            aug = iaa.SomeOf((1,2),[
                iaa.GaussianBlur(sigma=(0.0, 3.0)),
                iaa.AverageBlur(k=(1, 5)),
                # iaa.MedianBlur(k=(3, 7)),
                iaa.BilateralBlur(
                    d=(3, 9), sigma_color=(10, 250), sigma_space=(10, 250)),
                iaa.MotionBlur(k=5),
                # iaa.MeanShiftBlur(),
                # iaa.Superpixels(p_replace=(0.1, 0.5), n_segments=(1, 7))

            ], random_order=True)
        else:
            aug = None
        self.aug = iaa.Sequential(sometimes(aug))
        # aug = iaa.OneOf([
        #     # iaa.GaussianBlur(sigma=(0.0, 3.0)),
        #     # iaa.AverageBlur(k=(1, 5)),
        #     # iaa.MedianBlur(k=(3, 7)),
        #     # iaa.BilateralBlur(
        #         # d=(3, 9), sigma_color=(10, 250), sigma_space=(10, 250)),
        #     # iaa.MotionBlur(k=5),

        #     # iaa.MeanShiftBlur(),
        #     # iaa.Superpixels(p_replace=(0.1, 0.5), n_segments=(1, 7))

        # ])
        # import pdb;pdb.set_trace()
        # self.aug = iaa.Sequential(sometimes(aug))


        # self.transform = RandomResizedCrop_Flip((img_h, img_w),scale=(0.2, 1.0), interpolation=3)  # 3 is bicubic
        self.preprocess_lr = resizeNormalize((img_w, img_h),aug = self.aug)
        self.preprocess_hr = resizeNormalize((img_w, img_h))
        


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # import pdb;pdb.set_trace()
        index += 1
        # if index == 128:
        #     import pdb;pdb.set_trace()
        txn = self.env.begin(write=False)
        label_key = b'label-%09d' % index
        label = str(txn.get(label_key).decode())
        # if label =='products,':
        #     import pdb;pdb.set_trace()
        img_HR_key = b'image_hr-%09d' % index  # 128*32
        img_lr_key = b'image_lr-%09d' % index  # 64*16
        embed_vec_key = b'embed-%09d' % index
        embed_vec = txn.get(embed_vec_key)
        if embed_vec is not None:
            embed_vec = embed_vec.decode()
        else:
            embed_vec = ' '.join(['0'*300])
        embed_vec = np.array(embed_vec.split()).astype(np.float32)
        # if embed_vec.shape[0] != 300:
        #     return self[index + 1]
        embed_vec = torch.FloatTensor(embed_vec)
        
        # f = open('./train_image.txt','a')
        try:
            img_HR = buf2PIL(txn, img_HR_key, 'RGB')
            img_lr = buf2PIL(txn, img_lr_key, 'RGB')
            
        except IOError or len(label) > self.max_len:
            # import pdb;pdb.set_trace()
            return self[index + 1]
        # label_str = str_filt(word, self.voc_type)
        # import pdb;pdb.set_trace()
        # img_HR, img_lr = self.transform(img_HR, img_lr)
        # import pdb;pdb.set_trace()
        img_HR = self.preprocess_hr(img_HR)
        img_lr = self.preprocess_lr(img_lr)
        
        return {'img_HR':img_HR, 'img_lr': img_lr, 'label':label} # img_HR, img_lr, label, embed_vec
        # return {'img_HR':img_HR, 'img_lr': img_lr, 'label':label}


class resizeNormalize(object):
    def __init__(self, size, mask=False, interpolation=Image.BICUBIC, aug =None):
        self.size = size
        self.interpolation = interpolation
        # self.toTensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std= [0.5, 0.5, 0.5])
            ])
        # self.mask = mask
        self.aug = aug

    def __call__(self, img):
        # import pdb;pdb.set_trace()
        img = img.resize(self.size, self.interpolation)  
        # img.save('5.png')
        if not self.aug is None:
            img_np = np.array(img)
            # print("imgaug_np:", imgaug_np.shape)
            imgaug_np = self.aug(images=img_np[None, ...])
            img = Image.fromarray(imgaug_np[0, ...])
            # img.save('6.png')
        img = self.transform(img)
        return img


class RandomResizedCrop_Flip(torch.nn.Module):
    """Crop the given image to random size and aspect ratio.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        scale (tuple of float): range of size of the origin size cropped
        ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped.
        interpolation (int): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, p=0.5):
        super().__init__()
        # import pdb;pdb.set_trace()
        self.size = T._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.p = p

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img_hr, img_lr):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        # import pdb;pdb.set_trace()
        # i, j, h, w = self.get_params(img_hr, self.scale, self.ratio)
        # img_hr = F.resized_crop(img_hr, i, j, h, w, self.size, self.interpolation)
        # img_lr = F.resized_crop(img_lr, i, j, h, w, self.size, self.interpolation)
        if torch.rand(1) < self.p:
            img_hr = F.hflip(img_hr)
            img_lr = F.hflip(img_lr)
        return img_hr,img_lr

    def __repr__(self):
        interpolate_str = T._pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class ResizeNormalize(object):
  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation
    self.toTensor = transforms.ToTensor()

  def __call__(self, img):
    img = img.resize(self.size, self.interpolation)
    img = self.toTensor(img)
    img.sub_(0.5).div_(0.5)
    return img


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

def build(image_set, args):
    root = Path(args.data_root)
    if image_set == 'train':
        dataset_names = args.train_dataset.split(':')
    elif image_set == 'val':
        dataset_names = args.val_dataset.split(':')
    
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'jinshan_image_train1':
            data_root = root / 'scan' / 'train'
        elif dataset_name == 'jinshan_image_test1':
            data_root = root / 'scan' / 'test'
        elif dataset_name == 'jinshan_image_train2':
            data_root = root / 'cut' / 'train'
        elif dataset_name == 'jinshan_image_train3':
            data_root = root / 'deblur' / 'train_data_01'
        elif dataset_name == 'jinshan_image_train4':
            data_root = root / 'cut2' 
        elif dataset_name == 'jinshan_image_test2':
            data_root = root / 'cut' / 'test'
        elif dataset_name == 'jinshan_image_test3':
            data_root = root / 'test_demo'
        elif dataset_name == 'textzoom_train_1':
            data_root = root / 'train1'
        elif dataset_name == 'textzoom_train_2':
            data_root = root / 'train2'
        elif dataset_name == 'textzoom_test_easy':
            data_root = root / 'test' / 'easy'
        elif dataset_name == 'textzoom_test_medium':
            data_root = root / 'test' / 'medium'
        elif dataset_name == 'textzoom_test_hard':
            data_root = root / 'test' / 'hard'
        # elif dataset_name == 'scutens_train':
        #     data_root = root / 'SCUT-ENS' / 'train' /home/pci/disk1/lcy/unitext/Whole_im/dataset/deblur_test/train_data_01
        # elif dataset_name == 'scutens_test':
        #     data_root = root / 'SCUT-ENS' / 'test'
        else:
            raise NotImplementedError 
        
        # transforms = make_erase_transform(image_set, args)
        dataset = lmdbDataset(data_root, img_w = 128, img_h = 32, voc_type='upper', max_len=100, phase=image_set)
        datasets.append(dataset)
    
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset