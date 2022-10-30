import os, multiprocessing.pool
import ffmpeg, torchaudio
import pandas as pd
import numpy as np
import torch, librosa
import math
import matplotlib.pyplot as plt
import librosa
from torch.optim.lr_scheduler import _LRScheduler



def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def plot_result(blendshape):
    fig, axs = plt.subplots(1, 1)
    axs.set_title("Face Animation")
    axs.set_ylabel("Blendshape")
    axs.set_xlabel("frame")
    im = axs.imshow(blendshape, origin="lower", aspect="auto", filternorm=False, interpolation='none')
    fig.colorbar(im, ax=axs)
    plt.show(block=False)



class CosineAnnealingWarmUpRestarts(_LRScheduler):
    '''
    https://gaussian37.github.io/dl-pytorch-lr_scheduler/
    '''
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



def audio_preprocessing(wav):
    HERTZ = 50
    audio_tensor, sample_rate = torchaudio.load(wav)
    squeezed_audio_tensor = audio_tensor.squeeze()
    squeezed_audio_ndarray = squeezed_audio_tensor.numpy()

    n_fft = int(sample_rate / HERTZ)
    hop_length = int(sample_rate / (HERTZ * 2))
    D = librosa.stft(squeezed_audio_ndarray, 
                        n_fft=n_fft, 
                        win_length=n_fft, 
                        hop_length=hop_length, 
                        window='hamming')
    spectrogram, phase = librosa.magphase(D)

    # S = log(S+1)
    log_spectrogram = np.log1p(spectrogram)
    mean, stdev = log_spectrogram.mean(), log_spectrogram.std()
    normalized_spectrogram = (log_spectrogram - mean) / stdev
    normalized_spectrogram_tensor = torch.FloatTensor(normalized_spectrogram)

    return normalized_spectrogram_tensor.T
