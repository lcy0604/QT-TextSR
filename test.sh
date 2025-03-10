# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=8956 --use_env \
#     main.py \
#     --train_dataset tooth \
#     --val_dataset tooth \
#     --data_root ./../seg_tooth/ \
#     --output_dir ./checkpoint_tooth/ \
#     --backbone resnet50 \
#     --pretrained_backbone ./pretrained/resnet50-19c8e357.pth  \
#     --decoder deconv \
#     --batch_size 1 \
#     --lr 0.0005 \
#     --num_workers 0 \
#     --code_dir . \
#     --epochs 300 \
#     --save_interval 5 \
#     --warmup_epochs 10 \
#     --dataset_file erase \
#     --rotate_max_angle 10 \
#     --rotate_prob 0.3 \
#     --crop_min_ratio 0.7 \
#     --crop_max_ratio 1.0 \
#     --crop_prob 1.0 \
#     --pixel_embed_dim 512 \
#     --eval \
#     --resume /home/l/storage2/cy/SR_jinshan/whole_img/Swin-Erase/checkpoint_tooth/checkpoint0129.pth
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=8954 --use_env \
    main.py \
    --train_dataset textzoom_test_easy \
    --val_dataset textzoom_test_medium \
    --data_root ./data/textzoom/ \
    --output_dir ./checkpoint_vit_common/ \
    --backbone vit \
    --pretrained_backbone ./pretrained/resnet50-19c8e357.pth  \
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
    --resume /home/pci/disk2/lcy/detr_sr/checkpoint_vit_finetune/checkpoint0299.pth