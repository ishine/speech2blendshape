python run.py \
    --data_dir /shared/youngkim/mediazen/preprocessed/column16 \
    --batch_size 32 \
    --num_classes 16 \
    --epoch 1000 \
    --patience 30 \
    --lr 0.0001 \
    --gpu 2 \
    --name deepspeech_fc_finetune_expand_G_full_dataset \
    --checkpoint_dir /shared/youngkim/mediazen/ckpt \
    # --fast_dev_run True \
    # --debug True \
    # --pretrained /shared/youngkim/mediazen/ckpt/ds_full_expand_G_resume-04:16:16:25/last.ckpt \