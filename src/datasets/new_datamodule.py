import os
from random import shuffle
import re

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, random_split

from src import dataset
from src.datasets.new_dataset import FaceDataset


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__(FaceDataModule)
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.blendshape_columns = [
            'JawForward',
            'JawOpen',
            'MouthClose',
            'MouthFunnel',
            'MouthPucker',
            'MouthDimpleLeft',
            'MouthDimpleRight',
            'MouthStretchLeft',
            'MouthStretchRight',
            'MouthRollLower',
            'MouthRollUpper',
            'MouthShrugLower',
            'MouthShrugUpper',
            'MouthPressLeft',
            'MouthPressRight',
            'CheekPuff'
        ]

    
    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            print(f"No Data in {self.data_dir}")
        

    def setup(self, stage=None):

        data_paths = [d.path for d in os.scandir(self.data_dir)]
        file_names = [os.path.basename(d) for d in data_paths]
        sentence_nums = [re.sub(r'[^0-9]', '', d.split('_')[2]) for d in file_names]
        speaker_names = [re.sub(r'[0-9]+', '', d.split('_')[2]) for d in file_names]

        test_sentences = [5, 11, 18, 147, 183]
        train_valid_data_paths = [d for i, d in enumerate(data_paths) if int(sentence_nums[i]) not in test_sentences]
        test_data_paths = [d for i, d in enumerate(data_paths) if int(sentence_nums[i]) in test_sentences]

        if stage in (None, "fit"):

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=self.seed)
            indices = list(range(len(train_valid_data_paths)))
            train_valid_sentence_nums = [re.sub(r'[^0-9]', '', os.path.basename(d).split('_')[2]) for d in train_valid_data_paths]
            train_index, valid_index = next(iter(sss.split(indices, train_valid_sentence_nums)))

            train_valid_data_paths = np.array(train_valid_data_paths)

            train_data_paths = train_valid_data_paths[train_index]
            valid_data_paths = train_valid_data_paths[valid_index]

            self.train_dataset = FaceDataset(train_data_paths, stage)
            self.validation_dataset = FaceDataset(valid_data_paths, stage)

        if stage in (None, 'test'):
            self.test_dataset = FaceDataset(test_data_paths, stage)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True)
    

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=self.validation_dataset.collate_fn,
            pin_memory=True)


    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True)
