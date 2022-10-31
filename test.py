import os
from datetime import datetime
import argparse
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

from src.datasets.new_datamodule import FaceDataModule
from src.models.pl_model import S2BModel

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(version_base='1.1', config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    dm = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    model = model.load_from_checkpoint(cfg.path.pretrained)
    
    if cfg.debug:
        trainer = pl.Trainer(
            accelerator="cpu", 
            fast_dev_run=cfg.trainer.fast_dev_run,
            logger=False
            )
    else:
        trainer = pl.Trainer(
            accelerator="gpu", 
            devices=[cfg.trainer.devices], 
            fast_dev_run=cfg.trainer.fast_dev_run,
            precision=cfg.trainer.precision,
            logger=False
            )

    print(OmegaConf.to_yaml(cfg))
    trainer.test(model, dm)

if __name__=="__main__":
    main()