
import sys, argparse, os, multiprocessing.pool
from tqdm import tqdm

import ffmpeg, torchaudio
import pandas as pd
import numpy as np
import torch, librosa


class Preprocessor:
    HERTZ = 50

    def __init__(self, source_dir, target_dir, ignored_columns, start_idx=0, end_idx=None):
        self.source_dir = source_dir
        self.start_idx = start_idx
        recording_directories = sorted([d.path for d in os.scandir(self.source_dir) if d.is_dir()])
        if end_idx is None:
            self.recording_directories = recording_directories[start_idx:]
        else:
            self.recording_directories = recording_directories[start_idx:end_idx]
        self.target_dir = target_dir
        self.ignored_columns = ignored_columns

        if not os.path.exists(self.target_dir):
            os.mkdir(self.target_dir)

        self.wav_dir = os.path.join(self.target_dir, 'wav')
        if not os.path.exists(self.wav_dir):
            os.mkdir(self.wav_dir)

        self.essentials_dir = os.path.join(self.target_dir, 'essentials')
        if not os.path.exists(self.essentials_dir):
            os.mkdir(self.essentials_dir)


    def save_audio(self, source_media, wav_name):
        target_audio = os.path.join(self.wav_dir, f'{wav_name}.wav')

        try:
            (ffmpeg
                .input(source_media, vn=None)
                .output(filename=target_audio,
                        ac=1, 
                        acodec='pcm_s16le', 
                        ar='16k', 
                        loglevel='quiet', 
                        nostats=None)
                .run(overwrite_output=True))

        except:
            print(f'ffmpeg on {source_media} failed')

        print(f'{wav_name}.wav saved.')

        return target_audio


    def audio_preprocessing(self, wav):
        audio_tensor, sample_rate = torchaudio.load(wav)
        squeezed_audio_tensor = audio_tensor.squeeze()
        squeezed_audio_ndarray = squeezed_audio_tensor.numpy()

        n_fft = int(sample_rate / self.HERTZ)
        hop_length = int(sample_rate / (self.HERTZ * 2))
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

        return normalized_spectrogram_tensor.T, sample_rate

    def blendshape_preprocessing(self, source_shape):
        df = pd.read_csv(source_shape)
        shape = {key: value.tolist() for key, value in df.to_dict('series').items()
                        if key not in self.ignored_columns}

        return shape


    def get_data(self, recording_directory):
        recording_files = os.listdir(recording_directory)

        for recording_file in recording_files:
            if recording_file.endswith('.mov'):
                media = os.path.join(recording_directory, recording_file)
            elif recording_file.endswith('cal.csv'):
                shape = os.path.join(recording_directory, recording_file)
    
        try:
            media
        except:
            print(f'Directory {recording_directory} does not contain video file')
        try:
            shape
        except:
            print(f'Directory {recording_directory} does not contain calibrated Blendshape csv file')

        return media, shape

    def save_essentials(self, spec, sample_rate, blendshape, pt_name):
        target_essentials = os.path.join(self.essentials_dir, f'{pt_name}.pt')
        essentials = (spec, sample_rate, blendshape)
        torch.save(essentials, target_essentials)
        print(f'{pt_name}.pt saved.')

    def preprocess(self):
        # if self.threads > 1:
        #     with multiprocessing.pool.Pool(processes=self.threads) as pool:
        #         self.data = pool.starmap(self.sample_dispatcher, zip(self.recording_directories, range(len(self.recording_directories))))

        # elif self.threads == 1:
        for recording_directory, count in zip(self.recording_directories, range(len(self.recording_directories))):
            count += self.start_idx
            print(f"Processing No.{count} - {os.path.basename(recording_directory)}")
            count_with_name = f"{count}_{os.path.basename(recording_directory)}"
            mov_path, source_shape = self.get_data(recording_directory)
            wav_audio_path = self.save_audio(mov_path, count_with_name)
            spec, sample_rate = self.audio_preprocessing(wav_audio_path)
            blendshape = self.blendshape_preprocessing(source_shape)
            self.save_essentials(spec, sample_rate, blendshape, count_with_name)

    def sample_dispatcher(self, recording_directory, count):
        count += self.start_idx
        print(f"Processing No.{count} - {os.path.basename(recording_directory)}")
        count_with_name = f"{count}_{os.path.basename(recording_directory)}"
        mov_path, source_shape = self.get_data(recording_directory)
        wav_audio_path = self.save_audio(mov_path, count_with_name)
        spec, sample_rate = self.audio_preprocessing(wav_audio_path)
        blendshape = self.blendshape_preprocessing(source_shape)
        return spec, sample_rate, blendshape, count_with_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--source', help='Path to source directory', required=True)
    parser.add_argument('--target', help='Path to target directory', required=True)
    parser.add_argument('--start_idx', help='Start index of source', default=0)
    parser.add_argument('--threads', help='threads', default=1)
    parser.add_argument('--ignore', nargs='*', help='List of ignored columns', default=[])

    args = parser.parse_args()

    preprocessor = Preprocessor(
        source_dir=args.source,
        target_dir=args.target, 
        ignored_columns=args.ignore,
        start_idx=int(args.start_idx))
    preprocessor.preprocess()
