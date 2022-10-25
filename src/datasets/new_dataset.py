import os
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, stage):
        self.data_paths = data_paths
        self.len = len(self.data_paths)
        self.stage = stage
    
    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        '''
        data.keys() 
            idx, 
            spectrogram, 
            spectrogram_length, 
            speaker, 
            timecode, 
            blendshape_tensor, 
            blendshape_length,
        '''
        
        data = torch.load(self.data_paths[idx])
        if self.stage == 'test':
            data['f_name'] = os.path.basename(self.data_paths[idx]).rstrip('.pt')

        return data


    def collate_fn(self, batch):
        batch_spectrogram = [x['spectrogram'] for x in batch]
        batch_spectrogram_padded = pad_sequence(batch_spectrogram, batch_first=True)
        
        batch_blendshape = [x['blendshape_tensor'] for x in batch]
        batch_blendshape_padded = pad_sequence(batch_blendshape, batch_first=True)

        ret = [
            batch_spectrogram_padded.contiguous(),
            torch.tensor([x['spectrogram_length'] for x in batch]),
            batch_blendshape_padded.contiguous(),
            torch.tensor([x['blendshape_length'] for x in batch]),
        ]
        if self.stage == 'test':
            batch_timecode = [x['timecode'] for x in batch]
            batch_timecode_padded = pad_sequence(batch_timecode, batch_first=True)
            ret.append(torch.tensor([x['idx'] for x in batch]))
            ret.append(batch_timecode_padded.contiguous())
            ret.append([x['f_name'] for x in batch])

        return ret


class GGongGGongDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

    
    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        return self.data[idx]


class WavDataset(torch.utils.data.Dataset):
    def __init__(self, data, wav_dir):
        self.data = data
        self.wav_dir = wav_dir
        self.len = len(self.data)
    
    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        f_name = self.data[idx][2]
        audio_tensor, sample_rate = torchaudio.load(os.path.join(self.wav_dir, f'{f_name}.wav'))

        return audio_tensor, *self.data[idx]