
import os
from datetime import datetime
import argparse
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.datamodule import FaceDataModule
from src.model import S2BModel

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    pl.seed_everything(1234)
    
    dt_string = datetime.now().strftime("%d:%H:%M:%S")
    # wandb.init(project='mediazen', entity='normalkim', config=args, name=f'{args.name}-dt_string')
    wandb_logger = WandbLogger(name=f'{args.name}-{dt_string}', project='mediazen', entity='normalkim')

    data_loader = FaceDataModule(os.path.join(args.root, 'audio_ggongggong.pt')
                                ,os.path.join(args.root, 'shape_ggongggong.pt'), 8, 64)
    print(f'Loaded data from {args.root}')
    
    print(args.pretrained)
    if args.pretrained:
        model = S2BModel.load_from_checkpoint(args.pretrained)
    else:
        model = S2BModel(
            lr=args.lr,
            fc1_dim=1024,
            fc2_dim=1024,
            num_classes=args.num_classes,
            lambda_G=100)
    early_stop = EarlyStopping(monitor='v_loss_G_MSE', patience=args.patience)
    checkpoint_callback = ModelCheckpoint(dirpath=f'{args.checkpoint_dir}/{args.name}-{dt_string}', save_top_k=2, monitor="v_loss_G_MSE", save_last=True)
    
    if args.debug:
        trainer = pl.Trainer(
            accelerator="cpu", 
            max_epochs=1, 
            limit_train_batches=0.01, limit_val_batches=0.01,
            log_every_n_steps=1,
            fast_dev_run=args.fast_dev_run,
            )
    else:
        trainer = pl.Trainer(
            accelerator="gpu", 
            devices=[args.gpu], 
            max_epochs=args.epoch, 
            callbacks=[
                early_stop,
                checkpoint_callback], 
            logger=wandb_logger,
            fast_dev_run=args.fast_dev_run,
            # limit_train_batches=0.01, limit_val_batches=0.01,
            log_every_n_steps=5
            )
    trainer.fit(model, data_loader)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train Ggomyang model')
    parser.add_argument('--root', help='Root directory to load data from', type=str, default='essentials')
    parser.add_argument('--num_classes', help='The number of Blendshapes to predict', type=int)
    parser.add_argument('--epoch', help='Max epoch', type=int, default=100)
    parser.add_argument('--patience', help='Epochs to keep training before early stopping', type=int, default=15)
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.01)
    parser.add_argument('--gpu', help='GPU', type=int, default=0)
    parser.add_argument('--checkpoint_dir', help='directory to save checkpoint', type=str, default='ckpt')
    parser.add_argument('--name', help='Name of experiment', type=str, default='s2b')
    parser.add_argument('--pretrained', help='Pretrained path if resume', type=str, default=None)
    parser.add_argument('--fast_dev_run', help='If true, trainer runs 5 batch', type=bool, default=False)
    parser.add_argument('--debug', help='If true, trainer runs in cpu to use pdb', type=bool, default=False)


    
    args = parser.parse_args()

    main(args)