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
from omegaconf import DictConfig

from src.datasets.new_datamodule import FaceDataModule
from speech2blendshape.src.models.pl_model import S2BModel

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    if not cfg.resume:
        dt_string = datetime.now().strftime("%d:%H:%M:%S")
        cfg.name = f'{cfg.name}-{dt_string}'
    
    wandb_logger = WandbLogger(
        name=cfg.name, 
        project=cfg.project, 
        entity=cfg.entity,
        settings=wandb.Settings(code_dir='./src')
        )

    dm = FaceDataModule(
        cfg.path.data_dir, 
        cfg.datamodule.batch_size, 
        cfg.datamodule.num_workers, 
        cfg.seed,
        cfg.datamodule.blendshape_columns
        )
    
    if cfg.resume:
        model = S2BModel.load_from_checkpoint(cfg.path.pretrained)
    else:
        model = hydra.utils.instantiate(cfg.model)
    
    early_stop = EarlyStopping(
        monitor=cfg.trainer.es_monitor, 
        patience=cfg.trainer.es_patience
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{cfg.path.checkpoint_dir}/{cfg.name}', 
        save_top_k=cfg.trainer.save_top_k, 
        monitor=cfg.trainer.monitor, 
        save_last=cfg.trainer.save_last
        )
    
    if cfg.debug:
        trainer = pl.Trainer(
            accelerator="cpu", 
            max_epochs=1, 
            limit_train_batches=0.01, limit_val_batches=0.01,
            log_every_n_steps=1,
            fast_dev_run=cfg.trainer.fast_dev_run,
            )
    else:
        trainer = pl.Trainer(
            accelerator="gpu", 
            devices=[cfg.trainer.devices], 
            max_epochs=cfg.trainer.epoch, 
            callbacks=[
                early_stop,
                checkpoint_callback], 
            logger=wandb_logger,
            fast_dev_run=cfg.trainer.fast_dev_run,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            precision=cfg.trainer.precision,
            )

    print(cfg.pretty())
    
    trainer.fit(model, dm)
    trainer.test(model, dm)

if __name__=="__main__":
    main()