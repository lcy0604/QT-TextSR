# QT-TextSR

This repository is the implementation of "QT-TextSR: Enhancing scene text image super-resolution via efficient interaction with text recognition using a Query-aware Transformer", which is published on Neurocomputing 2024. [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231224020125)

The training and inference codes are available. 

## Environment
My environment can be refered as follows:
- Python 3.8.11
- PyTorch 1.8.0
- Polygon
- shapely
- skimage

## Datasets

We used TextZoom as our training and testing benchmark. You can download the data here. [data](https://drive.google.com/drive/folders/1Tx8mBVFGYIfniflgf_jaqJiv8gFVgN1J?usp=sharing)

After downloading the dataset, you can directly place the folders as

```bash
data/
--textzoom
----train1
----train2
----test
...
```

## Training 

Create an new directory (```./pretrained/```) and place the pretrain weights, including the MAE-pretrained weights and the VGG parameters.  All of them are available at [here](https://drive.google.com/drive/folders/1jYytUaiJK-9qwIM3MWm6qNzoBTFj-ORl?usp=sharing). You can also retrain the ViT-base using MAE by yourself.

``` bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=8949 --use_env \
    main.py \
    --train_dataset textzoom_train_1:textzoom_train_2 \
    --val_dataset textzoom_test_easy \
    --data_root ./data/textzoom/ \
    --output_dir ./checkpoint_vit_finetune/ \
    --backbone vit \
    --decoder deconv \
    --batch_size 48 \
    --lr 0.0001 \
    --num_workers 8 \
    --code_dir . \
    --epochs 300 \
    --save_interval 10 \
    --warmup_epochs 5 \
    --dataset_file sr_lmdb \
    --rotate_max_angle 10 \
    --rotate_prob 0.0 \
    --crop_min_ratio 0.8 \
    --crop_max_ratio 1.0 \
    --crop_prob 1.0 \
    --pixel_embed_dim 768 \
    --train  
```

## Testing

For generating the SR results, the commond is as follows:

``` bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=8954 --use_env \
    main.py \
    --train_dataset textzoom_test_easy \
    --val_dataset textzoom_test_medium \
    --data_root ./data/textzoom/ \
    --output_dir ./checkpoint_vit_common/ \
    --backbone vit \
    --decoder deconv \
    --batch_size 1 \
    --lr 0.0001 \
    --num_workers 4 \
    --code_dir . \
    --epochs 300 \
    --save_interval 10 \
    --warmup_epochs 10 \
    --dataset_file sr_lmdb \
    --rotate_max_angle 10 \
    --rotate_prob 0.0 \
    --crop_min_ratio 0.8 \
    --crop_max_ratio 1.0 \
    --crop_prob 1.0 \
    --pixel_embed_dim 768 \
    --eval \
    --resume /your/checkpoint/
```

## Acknowledge

The repository is benefit a lot from [DETR](https://github.com/facebookresearch/detr). Thanks a lot for their excellent work.

## Citation
If you find our method or dataset useful for your reserach, please cite:
```
@article{LIU2025129241,
title = {QT-TextSR: Enhancing scene text image super-resolution via efficient interaction with text recognition using a Query-aware Transformer},
journal = {Neurocomputing},
volume = {620},
pages = {129241},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.129241},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224020125},
author = {Chongyu Liu and Qing Jiang and Dezhi Peng and Yuxin Kong and Jiaixin Zhang and Longfei Xiong and Jiwei Duan and Cheng Sun and Lianwen Jin},
keywords = {Scene text image super-resolution, Vision-language interaction, Task-specific query, Transformer},
}
```
