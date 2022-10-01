# python train.py \--root /root/mediazen/speech2anim/essentials_1443 --patience 5 --num_classes 16 --lr 0.0001 --gpu 2
python train.py \
    --root /root/mediazen/speech2anim/essentials_1443 \
    --patience 30 \
    --num_classes 16 \
    --lr 0.0001 \
    --gpu 2 \
    --name add_D_resume_e11 \
    --checkpoint_dir /shared/youngkim/mediazen/ckpt \
    --pretrained /shared/youngkim/mediazen/ckpt/add_D-29:11:42:01/last.ckpt \
    # --fast_dev_run True