import random

import numpy as np
import pickle
from tqdm import tqdm
import os
from scipy.io import loadmat
import pandas as pd
import biosppy as bio
from ecgdetectors import Detectors
class Dataloader():
    def ImportFile(self, path):

        file_list = []

        # root, directories, files
        for r, d, f in os.walk(path):
            for file in f:
                if '.mat' in file:
                    file_dir = os.path.join(r, file)
                    file_list.append(file_dir)

        file_list = sorted(file_list)

        signals = []
        for file in tqdm(file_list):
            sig = list(loadmat(file).values())[0][0] / 1000
            signals.append(sig)

        # import reference
        refer_path = os.path.join(path, 'REFERENCE.csv')
        reference = np.array(pd.read_csv(refer_path, header=None))
        label = reference[:, 1]
        label[label == 'N'] = 0  # Normal
        label[label == 'A'] = 1  # Afib
        label[label == 'O'] = 2  # Other
        label[label == '~'] = 3  # Noise

        dataset = list(zip(label, signals))

        return dataset

    def WindowSelection(self, signal, win_size=9000, StartPoint=0):

        sig_len = len(signal)
        if sig_len < win_size:
            extended_signal = list(signal) + list(signal) + list(signal) + list(signal)
            return np.array(extended_signal[StartPoint:win_size])

        else:
            start_point = StartPoint
            end_point = start_point + win_size
            select_signal_win = signal[start_point:end_point]
            return select_signal_win

    def prepare_input_data(self, dataset):
        Signals=[]
        labels=[]

        for lb, sig in tqdm(dataset):
            if (lb == 0) or (lb == 1):

                windowed_signal = self.WindowSelection(signal=sig, win_size=9000, StartPoint=0)
                #filtered, _, _ = bio.signals.tools.filter_signal(signal=windowed_signals,
                #                                                  ftype='FIR',
                #                                                  band='bandpass',
                #                                                  order=int(0.3*300.0),
                #                                                  frequency=[3, 45],
                #                                                  sampling_rate=300)

                Signals.append(windowed_signal)
                labels.append(lb)


        train_data = list(zip(labels, Signals))
        n_lbs = []
        n_signals = []
        a_lbs = []
        a_signals = []
        for lb,sig in train_data:
            if lb == 0:
                n_lbs.append(lb)
                n_signals.append(np.array(sig))
            if lb == 1:
                a_lbs.append(lb)
                a_signals.append(np.array(sig))
        n_dataset = list(zip(n_lbs,n_signals))
        random.seed(24)
        sampled_n_data = random.sample(n_dataset, len(a_lbs))
        a_dataset = list(zip(a_lbs,a_signals))

        X_train=[]
        y_train = []
        for lb,sig in sampled_n_data:
            X_train.append(sig)
            y_train.append(lb)
        for lb,sig in a_dataset:
            X_train.append(sig)
            y_train.append(lb)
        x_train=np.array(X_train)
        y_train=np.array(y_train)

        return x_train,y_train
