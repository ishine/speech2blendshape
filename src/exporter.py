import os, multiprocessing, functools

import pandas as pd
import numpy as np
import scipy.signal


class PpujikPpujik:
    def ttukttakttukttak_migglemiggle(order, high_cut_frequency):
        return functools.partial(PpujikPpujik.migglemiggle, order=order, high_cut_frequency=high_cut_frequency)

    def migglemiggle(x, order, high_cut_frequency):
        return scipy.signal.sosfilt(scipy.signal.butter(order, high_cut_frequency, fs=60, output='sos'), x, axis=0)

    def ttukttakttukttak_manjilmanjil(window_size):
        return functools.partial(PpujikPpujik.manjilmanjil, window_size=window_size)

    def manjilmanjil(x, window_size):
        kernel = np.ones(window_size) / window_size
        return np.convolve(x, kernel, mode='same')

    def ssemssem(x):
        return x


    def __init__(self, target_dir, column_filter=ssemssem):
        self.target_dir = target_dir
        self.column_filter = column_filter

        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)


    def save_to_csv(self, length, f_name, timecode, column, prediction):
        recovered_timecode =  [f'{(s := str(time.item()))[:-9]}:{s[-9:-7]}:{s[-7:-5]}:{s[-5:-3]}.{s[-3:]}' for time in timecode]
        timecode_index = pd.Index(recovered_timecode, name='Timecode')

        blendshape_count = np.expand_dims(np.full(len(timecode), len(column)), axis=1)
        filtered_prediction = np.apply_along_axis(self.column_filter, 0, prediction)
        recovered_content = np.hstack([blendshape_count, filtered_prediction])

        recovered_column = ['BlendShapeCount', *column]

        df = pd.DataFrame(recovered_content, index=timecode_index, columns=recovered_column)
        chopped_df = df[:length.item()]
        chopped_df.to_csv(os.path.join(self.target_dir, f'{f_name}_prediction.csv'))


    def batch_save_to_csvs(self, lengths, f_names, timecodes, columns, predictions, threads=32):
        packed_data = zip(lengths, f_names, timecodes, [columns]*len(f_names), predictions)
        # for packed_datum in packed_data:
        #     self.save_to_csv(*packed_datum)
        with multiprocessing.Pool(processes=threads) as pool:
            pool.map(self.sample_dispatcher, packed_data)
    
    def sample_dispatcher(self, packed_datum):
        self.save_to_csv(*packed_datum)
    