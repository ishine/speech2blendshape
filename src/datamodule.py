
from random import shuffle
import pytorch_lightning as pl

import torch

from src import dataset


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, audio_blob_path, shape_blob_path, batch_size, num_workers=128):
        super().__init__(FaceDataModule)

        self.audio_blob_path = audio_blob_path
        self.shape_blob_path = shape_blob_path

        self.batch_size = batch_size
        self.num_workers = num_workers

    
    def prepare_data(self):
        self.sample_rate, self.indices, self.audio_data, self.audio_lengths = torch.load(self.audio_blob_path)
        self.timecodes, self.blendshape_count, self.blendshape_columns, self.shape_data, self.shape_lengths = torch.load(self.shape_blob_path)
        

    def setup(self, stage=None):
        full_dataset = dataset.FaceDataset(self.audio_data, self.audio_lengths, self.shape_data, self.shape_lengths, (self.indices, self.timecodes) if stage == 'test' else None)
        # self.full_ds = full_dataset

        ratio = len(full_dataset) // 20
        train_len = 14 * ratio
        validation_len = 3 * ratio
        test_len = len(full_dataset) - train_len - validation_len

        self.train_dataset, self.validation_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, [train_len, validation_len, test_len])
        
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)
        # return torch.utils.data.DataLoader(self.full_ds, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)
    

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)


    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)
