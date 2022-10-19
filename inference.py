import os
import argparse
import torch
import csv

from torch.cuda.amp import autocast
from tqdm import tqdm
import pytorch_lightning as pl

from src.datasets.datamodule import FaceDataModule
from speech2blendshape.src.models.pl_model import S2BModel


def main(args):
    pl.seed_everything(1234)

    data_loader = FaceDataModule(os.path.join(args.root,'audio_ggongggong.pt')
                                , os.path.join(args.root,'shape_ggongggong.pt'), 8, 64)

    if args.model_path and not args.model_version:
        model_path = args.model_path
    elif args.model_version and not args.model_path:
        model_root = os.path.join('lightning_logs', f'version_{args.model_version}', 'checkpoints')
        print(model_root)
        model_path = os.path.join(model_root, os.listdir(model_root)[-1])
    else:
        raise ValueError('Provide one of model_version or model_path, not both')

    model = S2BModel.load_from_checkpoint(model_path)
    model.save_name = args.name
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[args.gpu], 
        logger=False)
    trainer.test(model, data_loader)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Inference and output blendshape files')
    parser.add_argument('--root', help='Root directory to load data from', type=str, default='essentials')
    parser.add_argument('--model_path', help='Path of saved model you want to inference', default=None) #, default='checkpoint/ggomggom_v68_epoch-40.ckpt')
    parser.add_argument('--model_version', help='lightning_logs version of saved model you want to inference', default=None)
    parser.add_argument('--gpu', help='GPU', type=int, default=0)
    parser.add_argument('--name', help='Name of experiment', type=str, default='s2b')


    args = parser.parse_args()

    main(args)