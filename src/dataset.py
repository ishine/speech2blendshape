
import torch

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, audio_data, audio_lengths, shape_data, shape_lengths, indices_and_timecodes=None):
        if indices_and_timecodes:
            self.data = list(zip(audio_data, audio_lengths, shape_data, shape_lengths, *indices_and_timecodes))
        else:
            self.data = list(zip(audio_data, audio_lengths, shape_data, shape_lengths))

        self.len = len(self.data)

    
    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        return self.data[idx]