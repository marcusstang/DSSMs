import os
import random
import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils import data

class SpeechSequencesFull(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """

    def __init__(self, file_list, sequence_len, STFT_dict, shuffle):

        super().__init__()

        # STFT parameters
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']

        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.shuffle = shuffle

        self.compute_len()

    def compute_len(self):

        self.valid_seq_list = []

        for wavfile in self.file_list:

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')

            # remove beginning and ending silence
            if self.trim:
                _, (ind_beg, ind_end) = librosa.effects.trim(x, top_db=30)
            else:
                ind_beg = 0
                ind_end = len(x)

            # Check valid wav files
            seq_length = (self.sequence_len - 1) * self.hop
            file_length = ind_end - ind_beg
            n_seq = (1 + int(file_length / self.hop)) // self.sequence_len
            for i in range(n_seq):
                seq_start = i * seq_length + ind_beg
                seq_end = (i + 1) * seq_length + ind_beg
                seq_info = (wavfile, seq_start, seq_end)
                self.valid_seq_list.append(seq_info)

        if self.shuffle:
            random.shuffle(self.valid_seq_list)

    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_seq_list)

    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """

        # Read wav files
        wavfile, seq_start, seq_end = self.valid_seq_list[index]
        x, fs_x = sf.read(wavfile)

        # Sequence tailor
        x = x[seq_start:seq_end]

        # Normalize sequence
        x = x / np.max(np.abs(x))

        # STFT transformation
        audio_spec = torch.stft(torch.from_numpy(x), n_fft=self.nfft, hop_length=self.hop,
                                win_length=self.wlen, window=self.win,
                                center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

        # Square of magnitude
        sample = (torch.real(audio_spec[:, :]) ** 2 + torch.imag(audio_spec[:, :]) ** 2).float()

        return sample