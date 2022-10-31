import os
from random import shuffle
import re

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
import torch
from torch.utils.data import DataLoader, random_split

from src.datasets.new_dataset import FaceDataset, GGongGGongDataset, WavDataset, PredictDataset


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, seed, blendshape_columns):
        super().__init__(FaceDataModule)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.blendshape_columns = blendshape_columns

    
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


class PredictDataModule(pl.LightningDataModule):
    def __init__(self, data_path, num_workers, seed, blendshape_columns):
        super().__init__(PredictDataModule)
        self.data_path = data_path
        self.num_workers = num_workers
        self.seed = seed
        self.blendshape_columns = blendshape_columns

    
    def prepare_data(self):
        if not os.path.exists(self.data_path):
            print(f"No Data in {self.data_path}")
        
        
    def setup(self, stage=None):
        self.test_dataset = PredictDataset(self.data_path)


    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=1, 
            num_workers=self.num_workers, 
            pin_memory=True)


class GGongGGongDataModule(pl.LightningDataModule):
    def __init__(self, base_dir, batch_size, num_workers, seed, blendshape_columns, speakers=None):
        super().__init__(GGongGGongDataModule)

        self.audio_blob_path = os.path.join(base_dir, 'preprocessed/ggongggong2/audio_ggongggong.pt')
        self.shape_blob_path = os.path.join(base_dir, 'preprocessed/ggongggong2/shape_ggongggong.pt')

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.blendshape_columns = blendshape_columns
        self.speakers = speakers

    
    def prepare_data(self):
        self.sample_rate, self.indices, self.audio_data, self.audio_lengths = torch.load(self.audio_blob_path)
        self.timecodes, self.blendshape_count, blendshape_columns, self.shape_data, self.shape_lengths, self.f_names = torch.load(self.shape_blob_path)
        assert self.blendshape_columns == blendshape_columns
        self.data = list(zip(self.audio_data, self.audio_lengths, self.shape_data, self.shape_lengths, self.indices, self.timecodes, self.f_names))

    def setup(self, stage=None):
        
        sentence_nums = [re.sub(r'[^0-9]', '', d.split('_')[2]) for d in self.f_names]
        speaker_names = [re.sub(r'[0-9]+', '', d.split('_')[2]) for d in self.f_names]

        test_sentences = [5, 11, 18, 147, 183]
        train_valid_data_names = [d for i, d in enumerate(self.f_names) if int(sentence_nums[i]) not in test_sentences if speaker_names[i] in self.speakers]
        test_data_names = [d for i, d in enumerate(self.f_names) if int(sentence_nums[i]) in test_sentences]

        sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=1234)
        indices = list(range(len(train_valid_data_names)))
        train_valid_sentence_nums = [re.sub(r'[^0-9]', '', os.path.basename(d).split('_')[2]) for d in train_valid_data_names]
        train_index, valid_index = next(iter(sss.split(indices, train_valid_sentence_nums)))

        train_valid_data_names = np.array(train_valid_data_names)

        train_data_names = train_valid_data_names[train_index]
        valid_data_names = train_valid_data_names[valid_index]

        train_file_indices = [int(d.split('_')[0]) for d in train_data_names]
        valid_file_indices = [int(d.split('_')[0]) for d in valid_data_names]
        test_file_indices = [int(d.split('_')[0]) for d in test_data_names]

        train_data = [xx[:4] for xx in self.data if xx[4].item() in train_file_indices]
        valid_data = [xx[:4] for xx in self.data if xx[4].item() in valid_file_indices]
        test_data = [xx for xx in self.data if xx[4].item() in test_file_indices]

        if stage in (None, 'fit'):
            self.train_dataset = GGongGGongDataset(train_data)
            self.valid_dataset = GGongGGongDataset(valid_data)
        if stage in (None, 'test'):
            self.test_dataset = GGongGGongDataset(test_data)        
    

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, shuffle=True)
    

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)



class TestGGongGGongDataModule(pl.LightningDataModule):
    def __init__(self, base_dir, batch_size, num_workers, seed, blendshape_columns):
        super().__init__(TestGGongGGongDataModule)

        self.audio_blob_path = os.path.join(base_dir, 'preprocessed/test/ggongggong/audio_ggongggong.pt')
        self.shape_blob_path = os.path.join(base_dir, 'preprocessed/test/ggongggong/shape_ggongggong.pt')

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.blendshape_columns = blendshape_columns
    
    def prepare_data(self):
        self.sample_rate, self.indices, self.audio_data, self.audio_lengths = torch.load(self.audio_blob_path)
        self.timecodes, self.blendshape_count, blendshape_columns, self.shape_data, self.shape_lengths, self.f_names = torch.load(self.shape_blob_path)
        assert self.blendshape_columns == blendshape_columns
        self.data = list(zip(self.audio_data, self.audio_lengths, self.shape_data, self.shape_lengths, self.indices, self.timecodes, self.f_names))

    def setup(self, stage=None):
        if stage in (None, 'test'):
            self.test_dataset = GGongGGongDataset(self.data)        
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)



class WavDataModule(pl.LightningDataModule):
    def __init__(self, base_dir, batch_size, num_workers, seed, blendshape_columns, speakers=None):
        super().__init__(WavDataModule)

        self.wav_blob_path = os.path.join(base_dir, 'preprocessed/ggongggong2/w2v2_960h_base_ggongggong.pt')
        self.shape_blob_path = os.path.join(base_dir, 'preprocessed/ggongggong2/shape_ggongggong.pt')
        self.wav_dir = os.path.join(base_dir, 'preprocessed/wav')

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.blendshape_columns = blendshape_columns
        self.speakers = speakers

    
    def prepare_data(self):
        self.indices, self.audio_data, self.audio_lengths = torch.load(self.wav_blob_path)
        self.timecodes, self.blendshape_count, blendshape_columns, self.shape_data, self.shape_lengths, self.f_names = torch.load(self.shape_blob_path)
        assert self.blendshape_columns == blendshape_columns
        self.data = list(zip(self.audio_data, self.audio_lengths, self.shape_data, self.shape_lengths, self.indices, self.timecodes, self.f_names))

    def setup(self, stage=None):
        
        sentence_nums = [re.sub(r'[^0-9]', '', d.split('_')[2]) for d in self.f_names]
        speaker_names = [re.sub(r'[0-9]+', '', d.split('_')[2]) for d in self.f_names]

        test_sentences = [5, 11, 18, 147, 183]
        train_valid_data_names = [d for i, d in enumerate(self.f_names) if int(sentence_nums[i]) not in test_sentences if speaker_names[i] in self.speakers]
        test_data_names = [d for i, d in enumerate(self.f_names) if int(sentence_nums[i]) in test_sentences]

        sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=1234)
        indices = list(range(len(train_valid_data_names)))
        train_valid_sentence_nums = [re.sub(r'[^0-9]', '', os.path.basename(d).split('_')[2]) for d in train_valid_data_names]
        train_index, valid_index = next(iter(sss.split(indices, train_valid_sentence_nums)))

        train_valid_data_names = np.array(train_valid_data_names)

        train_data_names = train_valid_data_names[train_index]
        valid_data_names = train_valid_data_names[valid_index]

        train_file_indices = [int(d.split('_')[0]) for d in train_data_names]
        valid_file_indices = [int(d.split('_')[0]) for d in valid_data_names]
        test_file_indices = [int(d.split('_')[0]) for d in test_data_names]

        train_data = [xx[:4] for xx in self.data if xx[4].item() in train_file_indices]
        valid_data = [xx[:4] for xx in self.data if xx[4].item() in valid_file_indices]
        test_data = [xx for xx in self.data if xx[4].item() in test_file_indices]

        if stage in (None, 'fit'):
            self.train_dataset = WavDataset(train_data)
            self.valid_dataset = WavDataset(valid_data)
        if stage in (None, 'test'):
            self.test_dataset = WavDataset(test_data)        
    

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, shuffle=True)
    

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False)
