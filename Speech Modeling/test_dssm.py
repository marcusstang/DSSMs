import os
import datetime

from matplotlib import colors, ticker
from tqdm import tqdm
import soundfile as sf
from eval_metric import EvalMetrics, compute_median
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import SpeechSequencesFull
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
#from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    config_enumerate,
)
from pyro.optim import (
    DCTAdam,
    ClippedAdam,
)

from misc.gen_data import *
from models.utils import *
from models.VRNN import *
from models.FSSM import *
from models.ArFSSM import *
from models.RSSM import *
from models.SRNN import *

def test():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    ret_dir = './results/ret'
    if not os.path.isdir(ret_dir):
        os.makedirs(ret_dir)

    torch.manual_seed(42)
    np.random.seed(torch.initial_seed() % (2**32 - 1))

    wlen_sec = 64e-3
    hop_percent = 0.25
    fs =  16e3
    zp_percent = 0
    trim = True
    seq_len = 50
    batch_size = 100

    wlen = wlen_sec * fs
    wlen = np.int64(np.power(2, np.ceil(np.log2(wlen))))  # pwoer of 2
    hop = np.int64(hop_percent * wlen)
    nfft = wlen + zp_percent * wlen
    win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)

    STFT_dict = {}
    STFT_dict['fs'] = fs
    STFT_dict['wlen'] = wlen
    STFT_dict['hop'] = hop
    STFT_dict['nfft'] = nfft
    STFT_dict['win'] = win
    STFT_dict['trim'] = trim

    data_suffix = "flac"

    test_file_list = librosa.util.find_files('./data/train', ext=data_suffix)
    '''test_dataset = SpeechSequencesFull(file_list=test_file_list, sequence_len=seq_len,
                                        STFT_dict=STFT_dict, shuffle=False)
    train_num = test_dataset.__len__()
    print('Training samples: {}'.format(train_num))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)'''

    srnn = ArFSSM(
        dim_x=513,
        dim_z=16,
        dim_h1=16,
        dim_h2=128,
        dim_dx=256,
        dim_dz=32,
        dim_g=128,
        num_layers=1,
        rnn_dropout_rate=0.0,
    )

    ## testing phase
    save_file = os.path.join('./results/arfssm_5/arfssm' + str(1000) + '.pt')
    #save_file = os.path.join('./results/fssm_1/best.pt')
    srnn.load_state_dict(torch.load(save_file, map_location='cpu'))
    eval_metrics = EvalMetrics(metric='all')

    list_rmse = []
    list_sisdr = []
    list_pesq = []
    list_estoi = []

    for audio_file in tqdm(test_file_list):

        root, file = os.path.split(audio_file)
        filename, _ = os.path.splitext(file)
        recon_audio = os.path.join(ret_dir, 'recon_{}.wav'.format(filename))
        orig_audio = os.path.join(ret_dir, 'orig_{}.wav'.format(filename))

        x, fs_x = sf.read(audio_file)
        if trim:
            x, _ = librosa.effects.trim(x, top_db=30)

        scale = np.max(np.abs(x))  # normalized by Max(|x|)
        x = x / scale
        # STFT
        X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)

        data_orig = np.abs(X) ** 2  # (x_dim, seq_len)
        data_orig = torch.from_numpy(data_orig.astype(np.float32))
        data_orig = data_orig.permute(1, 0).unsqueeze(0)  # (x_dim, seq_len) => (1, seq_len, x_dim)
        with torch.no_grad():
            x_pred, _ = reconstruct_arfssm_new(srnn, data_orig, guide=5, num_MC=1000)
            data_recon = x_pred[0]
            data_recon = data_recon.T

        #data_recon = data_recon.to('cpu').detach().squeeze().permute(1, 0).numpy()

        X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
        x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)

        scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x)))) * 0.9

        sf.write(recon_audio, scale_norm * x_recon, fs_x)
        sf.write(orig_audio, scale_norm * x, fs_x)

        '''fig_dir = './results/fig/arfssm_3'
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        plt.figure(figsize=(8, 4))
        norm = colors.Normalize(vmin=-150, vmax=-20)  # 设置colorbar显示的最大最小值
        plt.specgram(scale_norm * x, NFFT=nfft, Fs=fs_x, window=win, noverlap=hop, scale='dB', norm=norm)
        #spec, _, _, _ = plt.specgram(scale_norm * x, NFFT=nfft, Fs=fs_x, window=win, noverlap=hop, scale='dB', norm=norm)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        cb.set_label("dB")
        plt.savefig(os.path.join(fig_dir,'org_{}.png'.format(filename)))
        plt.close()
        plt.figure(figsize=(8, 4))
        plt.specgram(scale_norm * x_recon, NFFT=nfft, Fs=fs_x, window=win, noverlap=hop, scale='dB', norm=norm)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        cb.set_label("dB")
        plt.savefig(os.path.join(fig_dir,'recon_{}.png'.format(filename)))
        plt.close()'''

        rmse, sisdr, pesq, estoi = eval_metrics.eval(audio_est=recon_audio, audio_ref=orig_audio)
        list_rmse.append(rmse)
        list_sisdr.append(sisdr)
        list_pesq.append(pesq)
        list_estoi.append(estoi)

    np_rmse = np.array(list_rmse)
    np_sisdr = np.array(list_sisdr)
    np_pesq = np.array(list_pesq)
    np_estoi = np.array(list_estoi)

    print('Re-synthesis finished')
    print('RMSE: {:.4f}'.format(np.mean(np_rmse)))
    print('SI-SDR: {:.4f}'.format(np.mean(np_sisdr)))
    print('PESQ: {:.4f}'.format(np.mean(np_pesq)))
    print('ESTOI: {:.4f}'.format(np.mean(np_estoi)))

    rmse_median, rmse_ci = compute_median(np_rmse)
    sisdr_median, sisdr_ci = compute_median(np_sisdr)
    pesq_median, pesq_ci = compute_median(np_pesq)
    estoi_median, estoi_ci = compute_median(np_estoi)

    print("Median evaluation")
    print('median rmse score: {:.4f} +/- {:.4f}'.format(rmse_median, rmse_ci))
    print('median sisdr score: {:.4f} +/- {:.4f}'.format(sisdr_median, sisdr_ci))
    print('median pesq score: {:.4f} +/- {:.4f}'.format(pesq_median, pesq_ci))
    print('median estoi score: {:.4f} +/- {:.4f}'.format(estoi_median, estoi_ci))

def main():
    test()
    print(666)

if __name__ == '__main__':
    main()