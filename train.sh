# python train.py \--root /root/mediazen/speech2anim/essentials_1443 --patience 5 --num_classes 16 --lr 0.0001 --gpu 2
python train.py \
    --root /root/mediazen/speech2anim/essentials_1443 \
    --epoch 1000 \
    --patience 30 \
    --num_classes 16 \
    --lr 0.0001 \
    --gpu 2 \
    --name ds_full_expand_G_resume \
    --checkpoint_dir /shared/youngkim/mediazen/ckpt \
    --pretrained /shared/youngkim/mediazen/ckpt/ds_full_expand_G-03:19:48:10/last.ckpt \
    # --fast_dev_run False