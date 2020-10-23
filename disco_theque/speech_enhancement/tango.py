# tango.py: two-step DANSE [1] where the VAD can be replaced by a TF-mask. The TF-mask can be oracle or predicted
# by a neural network.
#
# [1] Bertrand, Alexander, and Marc Moonen. 
#     "Distributed adaptive node-specific signal estimation in fully connected sensor networks—Part I:
#     Sequential node updating." IEEE Transactions on Signal Processing 58.10 (2010): 5277-5291.
#     ftp://ftp.esat.kuleuven.be/pub/SISTA/abertran/code/WOLA_DANSE1.m
import os
import argparse
import pickle
import soundfile as sf
import librosa as lb
import numpy as np
import torch
from code_utils.metrics import fw_snr, fw_sd
from code_utils.sigproc_utils import vad_oracle_batch
from code_utils.se_utils.internal_formulas import intern_filter
from disco_theque.dnn.utils import tf_mask
from disco_theque.speech_enhancement.utils import prepare_data, plot_conf
from disco_theque.dnn.models.heymann import build_heymann
from disco_theque.dnn.models.crnn import build_crnn
from mir_eval.separation import bss_eval_sources as bss
from pystoi.stoi import stoi
import copy
import ipdb

# %% Parameters
N_FFT = 512
N_HOP = 256
nb_ch = np.array([4, 4, 4, 4])
nb_nodes = len(nb_ch)
ref_mics = [0, 0, 0, 0]       # Reference microphone at each node

WIN_LEN = 21
PRED_FRAME = 'mid'
MASK_Z = 'local'
SNR_RANGE = [[0, 6]]
path_to_dataset='../../../dataset'


def get_dset(rir):
    """Given  `rir`, return 'test' or 'train' corresponding to dataset"""
    assert 0 < rir < 12001, "rir ID should be between 1 and 12000"
    out = 'train' if rir < 11001 else 'test'
    return out


def get_directory_name(snr_range):
    """From `snr_range`, return the name of the directory in form of e.g. 0-6 or 3-6_5-15"""
    dir_name = '_'.join(['{}-{}'.format(str(snr_range[k][0]), str(snr_range[k][1]))
                      for k in range(len(snr_range))])  # e.g. '0-6_5-15' or '0-6'
    return dir_name


def get_input_signals(i_rir, scenario='living', noise='ssn', snr_range=None):
    """
     Load the time signals corresponding to `i_rir` of the dataset characterized by input arguments.
     Mix them into a mixture at a random SNR (following uniformed distribution over snr_range).
    Args:
        scenario (str): 'living', 'random', 'meeting' ['living']
        noise (str): noise to load ('ssn', 'it', 'fs') ['ssn']
        snr_range (list[int]): list of [min_snr, max_snr], one per noise source.

    Returns:
        y (list[list[np.ndarray]]): convolved noisy signals. Two-layered list (node, channel)
        s (list[list[np.ndarray]]): convolved target signals. Two-layered list (node, channel)
        n (list[list[np.ndarray]]): convolved noise signals. Two-layered list (node, channel)
        s_dry (np.ndarray): dry target signal.
        n_dry (np.ndarray): dry noise signal.
        fs (int): sampling frequency
        snrs_used (np.ndarray): snr used when mixing the convolved signals (loaded only, should have been saved
                           (during the database generation)
    """
    dset = get_dset(i_rir)
    path_to_set = os.path.join(path_to_dataset, 'disco', scenario, dset)

    if snr_range is None:
        snr_range = [[0, 6]]
    dirry = get_directory_name(snr_range)
    snrs_used = np.load(os.path.join(path_to_set, 'log', 'snrs', 'dry', dirry, '') +
                        '{}_{}.npy'.format(str(i_rir), noise), allow_pickle=True)[0]

    # Preallocate arrays
    y = [[] for _ in range(nb_nodes)]
    s = [[] for _ in range(nb_nodes)]
    n = [[] for _ in range(nb_nodes)]
    # get convolved signals
    tar_root = os.path.join(path_to_set, 'wav_processed', dirry, 'target', '')
    noi_root = os.path.join(path_to_set, 'wav_processed', dirry, 'noise', '')
    mix_root = os.path.join(path_to_set, 'wav_processed', dirry, 'mixture', '')
    ii_ch = 0
    for i_nod in range(nb_nodes):
        for i_ch in range(nb_ch[i_nod]):
            ii_ch += 1
            tar_sig, fs = sf.read(tar_root + '{}_Ch-{}.wav'.format(str(i_rir), str(ii_ch)),
                                  dtype='float32')
            noi_sig = sf.read(noi_root + '{}_{}_Ch-{}.wav'.format(str(i_rir), noise, str(ii_ch)),
                              dtype='float32')[0]
            mix_sig = sf.read(mix_root + '{}_{}_Ch-{}.wav'.format(str(i_rir), noise, str(ii_ch)),
                              dtype='float32')[0]
            s[i_nod].append(tar_sig)
            n[i_nod].append(noi_sig)
            y[i_nod].append(mix_sig)

    dry_target = os.path.join(path_to_set, 'wav_original/dry/target/') + str(i_rir) + '_S-1.wav'
    dry_noise = os.path.join(path_to_set, 'wav_original/dry/noise/') + str(i_rir) + '_S-2_' + noise + '.wav'
    s_dry = sf.read(dry_target, dtype='float32')[0]
    n_dry = sf.read(dry_noise, dtype='float32')[0]
    n_dry *= 10 ** (-snrs_used / 20)

    return y, s, n, s_dry, n_dry, fs, snrs_used


def load_models(types, models, nodes_nbs):
    """
    Load two instances of a class and their weights
    Args;
        types (list[str, None]): list saying 'crnn' or None whether a model should be used to predict the masks or not.
        models (list[str, None]) names of saved state dict for the models to load
    Returns:
        list[nn.Module, None] List of the models or of None whether models should be used to predict the masks or not.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = []
    for i_step in range(len(types)):
        if models[i_step] is not None:
            model, _ = build_crnn((int(nodes_nbs[i_step]), 21, 257), (32, 64, 64), (3, 3, 3), (1, 1, 1),
                                  [(1, 4), (1, 4), (1, 4)],
                                  (None, None, None),
                                  [256], 'GRU',
                                  257,
                                  conv_padding=[(0, 1), (0, 1), (0, 1)])
            saved_weights = torch.load(models[i_step], map_location=torch.device(device))
            model.load_state_dict(saved_weights['model_state_dict'])
        else:
            model = None
        out.append(model)

    return out


def concatenate_signals(y, z, k, m=1):
    """
    Concatenate the mixture with the compressed signals.
    Args:
        y (np.ndarray):   Local microphone signals
        z (np.ndarray):   Compressed signals
        k (int):   Node number
        m (np.ndarray):   Mask of arrival node
    Returns:
        microphone signals stacked with z coming from other nodes
    """
    return np.concatenate((y[k],
                           m * np.array(z)[:k],
                           m * np.array(z)[k + 1:]), axis=0)


def get_z_for_mask(z_s, z_n, k, nb_nodes=nb_nodes, z_sigs='zs_hat'):
    """Return the channels to stack at the input of the NN.
    Args:
        z_s (np.ndarray): compressed signal of the target estimation
        z_n (np.ndarray): compressed signal of the noise estimation
        k (int): Local node (where NN is to predict the mask)
        nb_nodes (int): number of nodes in the microphone array
        z_sigs (str or list[str]): compressed signals to consider to predict the mask.
    """
    if z_sigs in ['zs_hat', 'zn_hat']:
        z_in = z_s if z_sigs == 'zs_hat' else z_n
        ch_of_z = list(range(nb_nodes))
        ch_of_z.pop(k)  # Do not take local z to compute the mask
        z_out = np.array(z_in)[ch_of_z, :, :]
    else:
        z_in = np.concatenate((z_s, z_n), axis=0)
        z_out = 1 * z_in
        # Re-order the zs_zn as in training configuration
        for i in range(z_in.shape[0]):
            if i % 2 == 0:
                z_out[i, :, :] = z_in[int(i/2), :, :]
            else:
                z_out[i, :, :] = z_in[int(1/2*(z_in.shape[0] - 1 + i)), :, :]
        ch_of_z = list(range(2 * nb_nodes))
        ch_of_z.pop(2 * k)  # Do not take local zs to compute the mask
        ch_of_z.pop(2 * k)  # Do not take local zn to compute the mask (channel right after zs)
        z_out = z_out[ch_of_z, :, :]

    return z_out


def get_mask(y, ss, sn, sz=None, mask_type='irm1', mod=None, ts=None, **kwargs):
    """
    Get the mask to apply on y in order to estimate s.
    Args:
        y (np.ndarray): input mixture STFT
        ss (np.ndarray): STFT speech
        sn (np.ndarray): STFT noise
        ss (np.ndarray): time speech
        sn (np.ndarray): time noise
        sz (np.ndarray): STFT of compressed signals coming from the other nodes
        mask_type (str, list[str]): Mask type(s)
        mod (torch.nn.Module): model (with weights)
        ts (np.ndarray): Speech time signal (to compute VAD)
    
    Returns:
        Estimated mask by the neural network or oracle mask (TF-mask or VAD whose binary values are spread
        across frequencies)
    """
    if mask_type[:-1] in ['irm', 'ibm', 'iam']:
        m = tf_mask(ss, sn, type=mask_type)
    elif 'rnn' in mask_type:        # Can actually be 'crnn' or 'rnn'
        mod.eval()
        three_d_tensor = (mask_type == 'crnn')
        lost_frames = int(kwargs['win_len'] - mod.get_loss_frames('last')[-1][-1])  # Only useful for CRNN, 'last'
        y_inp = prepare_data(y, three_d_tensor, z_data=sz, frames_lost=lost_frames, **kwargs)
        m_stack = mod(y_inp).detach().numpy()
        m = reshape_mask(m_stack, kwargs['frame_to_pred'])
    elif mask_type == 'ivad':
        n_freq = np.shape(ss)[0]
        m = np.zeros(np.shape(ss))
        vad = vad_oracle_batch(ts, win_len=N_FFT, win_hop=N_HOP)
        vad = vad[::N_HOP]
        m[:, :len(vad)] = np.tile(vad, (n_freq, 1))
    else:
        raise ValueError('Unknown value for `mask_type`')

    return m


def reshape_mask(mask, output_frame='last'):
    """ Reshape the output of the model into a 2D mask"""
    if output_frame == 'last':
        output_mask = mask[:, -1, :]
    elif output_frame == 'mid':
        win_len = np.shape(mask)[1]
        output_mask = mask[:, int(np.floor(win_len / 2)):int(np.ceil(win_len / 2)), :]
    elif output_frame == 'all':
        raise NotImplementedError('This case was not implemented yet')
    else:
        raise ValueError(":param output_frame: should be either 'last', 'all' or 'mid'")

    return np.squeeze(output_mask).T


def save_conf(i_rir, save_path, title=None, scene='living', case='test'):
    """ Save the 2D representation of the spatial configuration."""
    infos = np.load(os.path.join(path_to_dataset, 'suma/', scene, case, 'log', 'infos', '') + str(i_rir) + '.npy',
                    allow_pickle=True)[()]
    f = plot_conf(infos, mics_per_node=nb_ch, return_fig=True)
    f.gca().set_title(title)
    f.savefig(os.path.join(save_path, '') + str(i_rir) + '.png')


def offline_tango(y, s, n, vads='irm1', mods=None, mask_for_z=MASK_Z, z_sigs='zs_hat'):
    """
    Tango is a kind of DANSE [1] in two steps. The fist step estimates at every node a compressed signal z which is sent
    to all other nodes. In a second step, with the local sensor signals and the compressed signal zs coming from the
    other nodes, a MWF (or SDW-MWF) is applied to estimate the target signal.

    Args:
        y (np.ndarray, list[np.ndarray]): input multichannel mixture: n_nodes x n_channels x time. Probably a list if
                                          nodes don't have the same number of signals.
        s (np.ndarray): clean speech signal; same shape as y
        n (np.ndarray): noise signal; same shape as y
        vads (list[str]): List str whether to use 'vad', ideal mask ('irmX', 'ibmX', 'iamX') or
                          predicted masks ('crnn')]
        mods (list[nn.Module, None]): Pytorch models if the mask should be predicted. None otherwise if mask should be
                                      oracle.
        mask_for_z (str): which mask to apply on the compressed signals ['local']
        z_sigs (str or list[str]): compressed signals to consider to predict the mask ['zs_hat']

    Returns:
       yf_stft (np.ndarray): Filtered signal
       sf_stft (np.ndarray): Result of the filtering on the clear target signal
       nf_stft (np.ndarray): Result of the filtering on the clear noise signal
       z_stft_y (np.ndarray): Compressed target signal (estimation)
       z_stft_y (np.ndarray): Result of the first filtering step on clear target signal
       z_stft_n (np.ndarray): Result of the first filtering step on clear noise signal
       zn_stft (np.ndarray): Compressed noise signal (estimation)
       masks_z (np.ndarray): masks used for the first filtering step
       mask_w (np.ndarray): maskes used for the second filtering step.

    """
    length_signal = len(y[0][0])
    n_nodes = len(y)
    n_channels_per_node = [np.shape(sig)[0] for sig in y]
    n_signals_per_node = [n_channels_per_node[k] + n_nodes - 1 for k in range(n_nodes)]
    n_freq = int(N_FFT / 2 + 1)
    n_frames = 3 + int(np.floor((length_signal - N_FFT) / N_HOP))
    y_stft = [[] for _ in range(n_nodes)]        # Mixture STFT matrix (n_nodes x n_channels x n_freq x n_time)
    s_stft = [[] for _ in range(n_nodes)]        # Mixture STFT matrix (n_nodes x n_channels x n_freq x n_time)
    n_stft = [[] for _ in range(n_nodes)]        # Mixture STFT matrix (n_nodes x n_channels x n_freq x n_time)
    s_hat_stft_z = [[] for _ in range(n_nodes)]       # Speech STFT matrix (n_nodes x n_channels x n_freq x n_time)
    n_hat_stft_z = [[] for _ in range(n_nodes)]       # Noise STFT matrix (n_nodes x n_channels x n_freq x n_time)
    s_hat_stft_w = [[] for _ in range(n_nodes)]       # Speech STFT matrix (n_nodes x n_channels x n_freq x n_time)
    n_hat_stft_w = [[] for _ in range(n_nodes)]       # Noise STFT matrix (n_nodes x n_channels x n_freq x n_time)
    r_ss_loc = [[np.zeros((n_channels_per_node[k], n_channels_per_node[k]), 'complex64') for _ in range(n_freq)]
                for k in range(n_nodes)]
    r_nn_loc = [[np.zeros((n_channels_per_node[k], n_channels_per_node[k]), 'complex64') for _ in range(n_freq)]
                for k in range(n_nodes)]
    r_ss_glo = [[np.zeros((n_signals_per_node[k], n_signals_per_node[k]), 'complex64') for _ in range(n_freq)]
                for k in range(n_nodes)]
    r_nn_glo = [[np.zeros((n_signals_per_node[k], n_signals_per_node[k]), 'complex64') for _ in range(n_freq)]
                for k in range(n_nodes)]

    z_stft_y = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Compressed signal
    zn_stft = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Compressed signal
    z_stft_s = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Speech part of z
    z_stft_n = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Noise part of z
    z_gevd_s = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Noise part of z
    z_gevd_n = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Noise part of z
    yf_stft = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Estimated STFT of desired signal
    sf_stft = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Speech part of z after filtering
    nf_stft = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Noise part of z after filtering
    s_gevd = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # New reference, as if GEVD on S
    n_gevd = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # New reference, as if GEVD on N
    in_to_filter_y = [[] for _ in range(n_nodes)]
    in_to_filter_s = [[] for _ in range(n_nodes)]
    in_to_filter_n = [[] for _ in range(n_nodes)]
    in_gevd_s = [[] for _ in range(n_nodes)]
    in_gevd_n = [[] for _ in range(n_nodes)]
    in_to_phi_s = [[] for _ in range(n_nodes)]
    in_to_phi_n = [[] for _ in range(n_nodes)]

    masks_z = [[] for _ in range(n_nodes)]
    mask_w = [[] for _ in range(n_nodes)]

    for i_nod in range(n_nodes):
        # Activity detector: either a mask or a VAD. If VAD, convert into a mask-like STFT matrix.
        # mask_z will be erased and re-used but not mask_w (we need all masks at the second step)
        # Process all signals individually
        for i_ch in range(n_channels_per_node[i_nod]):
            y_ = np.array(y[i_nod])[i_ch, :]
            s_ = np.array(s[i_nod])[i_ch, :]
            n_ = np.array(n[i_nod])[i_ch, :]
            # STFT
            y_stft[i_nod].append(lb.core.stft(y_, n_fft=N_FFT, hop_length=N_HOP, center=True))
            s_stft[i_nod].append(lb.core.stft(s_, n_fft=N_FFT, hop_length=N_HOP, center=True))
            n_stft[i_nod].append(lb.core.stft(n_, n_fft=N_FFT, hop_length=N_HOP, center=True))
            if i_ch == ref_mics[i_nod]:
                mask_z = get_mask(y_stft[i_nod][i_ch], s_stft[i_nod][i_ch], n_stft[i_nod][i_ch],
                                  mask_type=vads[0], mod=mods[0], win_len=WIN_LEN, win_hop=1, frame_to_pred=PRED_FRAME,
                                  ts=s_)
                masks_z[i_nod] = mask_z
            if 'use_oracle_' in mask_for_z:
                s_hat_stft_z[i_nod].append(s_stft[i_nod][i_ch])             # s_hat: speech estimation
                n_hat_stft_z[i_nod].append(n_stft[i_nod][i_ch])             # n_hat: noise estimation
            else:
                s_hat_stft_z[i_nod].append(mask_z * y_stft[i_nod][i_ch])             # For statistics estimation
                n_hat_stft_z[i_nod].append((1 - mask_z) * y_stft[i_nod][i_ch])

        y_stft[i_nod] = np.array(y_stft[i_nod])
        s_stft[i_nod] = np.array(s_stft[i_nod])
        n_stft[i_nod] = np.array(n_stft[i_nod])
        s_hat_stft_z[i_nod] = np.array(s_hat_stft_z[i_nod])
        n_hat_stft_z[i_nod] = np.array(n_hat_stft_z[i_nod])

        # Local spatial covariance matrices -- computed from the whole signal
        for f in range(n_freq):
            phi_s_f = [[] for it in range(n_frames)]    # Covariance matrix at every frame
            phi_n_f = [[] for it in range(n_frames)]    # Covariance matrix at every frame
            for t in range(n_frames):
                phi_s_f[t] = np.outer(s_hat_stft_z[i_nod][:, f, t], np.conjugate(s_hat_stft_z[i_nod][:, f, t]).T)
                phi_n_f[t] = np.outer(n_hat_stft_z[i_nod][:, f, t], np.conjugate(n_hat_stft_z[i_nod][:, f, t]).T)
            r_ss_loc[i_nod][f] = np.mean(np.array(phi_s_f), axis=0)
            r_nn_loc[i_nod][f] = np.mean(np.array(phi_n_f), axis=0)

            # Local filter (called external filter in DANSE)
            w_loc, (t1_z, _) = intern_filter(r_ss_loc[i_nod][f], r_nn_loc[i_nod][f],
                                             mu=1, type='gevd', rank=1)
            for t in range(n_frames):
                z_stft_y[i_nod][f, t] = np.inner(np.conjugate(w_loc), y_stft[i_nod][:, f, t])
                z_stft_s[i_nod][f, t] = np.inner(np.conjugate(w_loc), s_stft[i_nod][:, f, t])
                z_stft_n[i_nod][f, t] = np.inner(np.conjugate(w_loc), n_stft[i_nod][:, f, t])
                z_gevd_s[i_nod][f, t] = np.inner(t1_z, s_stft[i_nod][:, f, t])      # Reference signal sent
                z_gevd_n[i_nod][f, t] = np.inner(t1_z, n_stft[i_nod][:, f, t])
        # Noise estimation in compressed signal
        zn_stft[i_nod] = (y_stft[i_nod][ref_mics[i_nod]] - z_stft_y[i_nod])

    # Compute speech and noise components at both nodes before second loop
    z_for_rs, z_for_rn = copy.deepcopy(z_stft_y), copy.deepcopy(z_stft_y)
    for i_nod in range(n_nodes):
        # Append z of nodes [j]_{j!=k} to input signals of node k
        in_to_filter_y[i_nod] = concatenate_signals(y_stft, z_stft_y, i_nod)
        in_to_filter_s[i_nod] = concatenate_signals(s_stft, z_stft_s, i_nod)
        in_to_filter_n[i_nod] = concatenate_signals(n_stft, z_stft_n, i_nod)
        in_gevd_s[i_nod] = concatenate_signals(s_stft, z_gevd_s, i_nod)
        in_gevd_n[i_nod] = concatenate_signals(n_stft, z_gevd_n, i_nod)
        z_for_mask = get_z_for_mask(z_stft_y, zn_stft, i_nod, n_nodes, z_sigs)
        if vads[1] == 'crnn' and mods[1] is None:   # Take the same predicted mask as at first step.
            mask_w[i_nod] = masks_z[i_nod]
        else:
            mask_w[i_nod] = get_mask(y_stft[i_nod][0], s_stft[i_nod][0], n_stft[i_nod][0],
                                     sz=z_for_mask,
                                     mask_type=vads[1], mod=mods[1], win_len=WIN_LEN, win_hop=1,
                                     frame_to_pred=PRED_FRAME, ts=s[i_nod][0])
        # Apply the mask if necessary
        if mask_for_z == 'distant':
            z_for_rs[i_nod] *= mask_w[i_nod]
            z_for_rn[i_nod] *= (1 - mask_w[i_nod])
        elif mask_for_z == 'compressed':
            mask_comp = get_mask(z_stft_y[i_nod], z_stft_s[i_nod], z_stft_n[i_nod], mask_type=vads[0],
                                 mod=mods[0])
            z_for_rs[i_nod] *= mask_comp
            z_for_rn[i_nod] *= (1 - mask_comp)
        elif mask_for_z == 'use_oracle_refs':
            z_for_rs[i_nod] = np.array(s_stft)[i_nod, ref_mics[i_nod], :, :]
            z_for_rn[i_nod] = np.array(n_stft)[i_nod, ref_mics[i_nod], :, :]
        elif mask_for_z == 'use_oracle_zs':
            z_for_rs[i_nod] = z_stft_s[i_nod]
            z_for_rn[i_nod] = z_stft_n[i_nod]

    for i_nod in range(n_nodes):
        for i_ch in range(n_channels_per_node[i_nod]):
            s_hat_stft_w[i_nod].append(mask_w[i_nod] * y_stft[i_nod][i_ch])  # s_hat: speech estimation
            n_hat_stft_w[i_nod].append((1 - mask_w[i_nod]) * y_stft[i_nod][i_ch])  # n_hat: noise estimation

        if mask_for_z == 'local':
            ms = mask_w[i_nod]
            mn = (1 - mask_w[i_nod])
        elif mask_for_z is None:
            ms = 1
            mn = 1
            z_for_rn = zn_stft  # Erase previous copy
        elif mask_for_z == 'use_oracle_sigs':
            z_for_rs = s_stft
            z_for_rn = n_stft
            ms = 1
            mn = 1
        else:   # 'previous' -- already computed in previous loop
            ms, mn = 1, 1

        in_to_phi_s[i_nod] = concatenate_signals(s_hat_stft_w, z_for_rs, i_nod, ms)
        in_to_phi_n[i_nod] = concatenate_signals(n_hat_stft_w, z_for_rn, i_nod, mn)
        for f in range(n_freq):
            phi_sz_f = [np.zeros((n_signals_per_node[i_nod], n_signals_per_node[i_nod])) for _ in range(n_frames)]
            phi_nz_f = [np.zeros((n_signals_per_node[i_nod], n_signals_per_node[i_nod])) for _ in range(n_frames)]
            for t in range(n_frames):
                phi_sz_f[t] = np.outer(in_to_phi_s[i_nod][:, f, t], np.conjugate(in_to_phi_s[i_nod][:, f, t]).T)
                phi_nz_f[t] = np.outer(in_to_phi_n[i_nod][:, f, t], np.conjugate(in_to_phi_n[i_nod][:, f, t]).T)
            r_ss_glo[i_nod][f] = np.mean(np.array(phi_sz_f), axis=0)
            r_nn_glo[i_nod][f] = np.mean(np.array(phi_nz_f), axis=0)

            # Global filter (called internal filter in DANSE)
            w_glo, (t1_w, sort_index) = intern_filter(r_ss_glo[i_nod][f], r_nn_glo[i_nod][f],
                                                      mu=1, type='gevd', rank=1)
            for t in range(n_frames):
                yf_stft[i_nod][f, t] = np.inner(np.conjugate(w_glo), in_to_filter_y[i_nod][:, f, t])
                sf_stft[i_nod][f, t] = np.inner(np.conjugate(w_glo), in_to_filter_s[i_nod][:, f, t])
                nf_stft[i_nod][f, t] = np.inner(np.conjugate(w_glo), in_to_filter_n[i_nod][:, f, t])
                s_gevd[i_nod][f, t] = np.inner(t1_w, in_gevd_s[i_nod][:, f, t])
                n_gevd[i_nod][f, t] = np.inner(t1_w, in_gevd_n[i_nod][:, f, t])

            if f == 0:
                w_glo_m = abs(w_glo)
            else:
                w_glo_m += abs(w_glo)

    return yf_stft, sf_stft, nf_stft, z_stft_y, z_stft_s, z_stft_n, zn_stft, masks_z, mask_w


def main(vad_types, save_dir, i_rir, noise, scenario='living', mask_z=MASK_Z, z_sigs='zs_hat', models=[None, None]):
    """
    Main function: load signals, filter them and estimate the performance in terms of BSS metrics.
    Args:
        vad_types (list[str]): use VAD, IRM or predicted masks
        save_dir (str): name of the subfolder where results are saved
        i_rir (int): RIR id of the sample to filter
        noise (str): noise of mixture
        scenario (str): spatial configuration ['living']
        mask_z (str): which mask to apply on the compreseed signals ['local']
        z_sigs (str or list[str]): which compressed signals to use to predict the mask at the second filtering step
                                   ['zs_hat']
        models (list[str, None]): Use oracle mask (None) or predicted ones ('crnn') for each step. 
    """
    # Ensure filtering has not yet been done
    dset = get_dset(i_rir)
    save_dir_root = os.path.join('results', scenario, dset, save_dir, '')
    if os.path.isfile(save_dir_root + 'OIM/results_mwf_' + str(i_rir) + '_' + noise + '.p'):
        print('Conf {} with {} noise already processed'.format(str(i_rir), noise))
        return

    snr_range = SNR_RANGE
    dirry = get_directory_name(snr_range)

    os.makedirs(os.path.join(save_dir_root, 'WAV', str(i_rir)), exist_ok=True)
    os.makedirs(os.path.join(save_dir_root, 'STFT', 'z', 'raw', dirry), exist_ok=True)
    os.makedirs(os.path.join(save_dir_root, 'OIM'), exist_ok=True)
    os.makedirs(os.path.join(save_dir_root, 'FIG'), exist_ok=True)
    os.makedirs(os.path.join(save_dir_root, 'MASK', str(i_rir)), exist_ok=True)

    y, s, n, s_dry, n_dry, fs, rnd_snrs = get_input_signals(i_rir, scenario, noise, snr_range)

    # Load trained models if necessary
    nodes_nbs = [1, nb_nodes] if z_sigs in ['zs_hat', 'zn_hat'] else [1, 1 + 2 * (nb_nodes - 1)]
    models = load_models(vad_types, models, nodes_nbs)
    print("Start algorithm " + str(i_rir))
    sh, s_filtered, n_filtered, z_sh, z_s, z_n, z_nh, mask_sc, mask_mc = offline_tango(y, s, n, vad_types,
                                                                                       mods=models,
                                                                                       mask_for_z=mask_z,
                                                                                       z_sigs=z_sigs)
    print("End algorithm")

    # Quantify filtering
    sdr, sar, sir = np.zeros(nb_nodes), np.zeros(nb_nodes), np.zeros(nb_nodes)
    sdr_dry, sar_dry, sir_dry = np.zeros(nb_nodes), np.zeros(nb_nodes), np.zeros(nb_nodes)
    sdr_in_dry, sir_in_dry, sar_in_dry = np.zeros(nb_nodes), np.zeros(nb_nodes), np.zeros(nb_nodes)
    sdr_dry_z, sar_dry_z, sir_dry_z = np.zeros(nb_nodes), np.zeros(nb_nodes), np.zeros(nb_nodes)
    sdrz, sarz, sirz = np.zeros(nb_nodes), np.zeros(nb_nodes), np.zeros(nb_nodes)
    fw_snr_out = np.zeros(nb_nodes)
    f_sd_mean, f_sd_mean_dry = np.zeros(nb_nodes), np.zeros(nb_nodes)
    f_sd_mean_z, f_sd_mean_dry_z = np.zeros(nb_nodes), np.zeros(nb_nodes)
    fw_snr_out_z = np.zeros(nb_nodes)
    sdr_in_cnv, sir_in_cnv = np.zeros(nb_nodes), np.zeros(nb_nodes)
    delta_stoi, delta_stoi_z, = np.zeros(nb_nodes), np.zeros(nb_nodes)
    delta_stoi_dry = np.zeros(nb_nodes)
    delta_stoi_z_dry = np.zeros(nb_nodes)
    fw_snr_in_dry, fw_snr_in_cnv = np.zeros(nb_nodes), np.zeros(nb_nodes)

    # Preallocate variables
    sh_t = [[] for i in range(nb_nodes)]    # Estimation of the speech -- output
    szh_t = [[] for i in range(nb_nodes)]    # Estimation of the speech -- output
    sf_t = [[] for i in range(nb_nodes)]    # Effect of the filtering on the speech
    nf_t = [[] for i in range(nb_nodes)]    # Effect of the filtering on the noise
    szf_t = [[] for i in range(nb_nodes)]    # Effect of the filtering on the speech part of z
    nzf_t = [[] for i in range(nb_nodes)]    # Effect of the filtering on the noise part of z

    for i_node in range(nb_nodes):
        # Time signals
        sh_t[i_node] = lb.core.istft(sh[i_node],
                                     hop_length=N_HOP, win_length=N_FFT, center=True, length=len(s[0][0]))
        szh_t[i_node] = lb.core.istft(z_sh[i_node],
                                      hop_length=N_HOP, win_length=N_FFT, center=True, length=len(s[0][0]))
        sf_t[i_node] = lb.core.istft(s_filtered[i_node],
                                     hop_length=N_HOP, win_length=N_FFT, center=True, length=len(s[0][0]))
        nf_t[i_node] = lb.core.istft(n_filtered[i_node],
                                     hop_length=N_HOP, win_length=N_FFT, center=True, length=len(s[0][0]))
        szf_t[i_node] = lb.core.istft(z_s[i_node],
                                      hop_length=N_HOP, win_length=N_FFT, center=True, length=len(s[0][0]))
        nzf_t[i_node] = lb.core.istft(z_n[i_node],
                                      hop_length=N_HOP, win_length=N_FFT, center=True, length=len(s[0][0]))

        # OIMs
        min_len = np.min((len(y[0][0]), len(sh_t[0]), len(s_dry), len(n_dry)))
        # Dry sources
        refs_dry = np.vstack((s_dry[fs:min_len], n_dry[fs:min_len]))
        # GEVD Source
        refs = np.vstack((s[i_node][0][fs:min_len], n[i_node][0][fs:min_len]))
        ests = np.vstack((sh_t[i_node][fs:min_len], y[i_node][0][fs:min_len] - sh_t[i_node][fs:min_len]))
        ests_z = np.vstack((szh_t[i_node][fs:min_len], y[i_node][0][fs:min_len] - szh_t[i_node][fs:min_len]))
        ests_i = np.vstack((y[i_node][0][fs:min_len], y[i_node][0][fs:min_len] - sh_t[i_node][fs:min_len]))

        # OIM values
        bss_dry = bss(refs_dry, ests, compute_permutation=False)
        sdr_dry[i_node], sir_dry[i_node], sar_dry[i_node] = bss_dry[0][0], bss_dry[1][0],\
                                                            bss_dry[2][0]
        bss_dry_z = bss(refs_dry, ests_z, compute_permutation=False)
        sdr_dry_z[i_node], sir_dry_z[i_node], sar_dry_z[i_node] = bss_dry_z[0][0], bss_dry_z[1][0],\
                                                                  bss_dry_z[2][0]
        bss_in_dry = bss(refs_dry, ests_i, compute_permutation=False)
        sdr_in_dry[i_node], sir_in_dry[i_node], sar_in_dry[i_node] = bss_in_dry[0][0], bss_in_dry[1][0],\
                                                                     bss_in_dry[2][0]

        sdr_, sir_, sar_, _ = bss(refs, ests, compute_permutation=False)
        sdrz_, sirz_, sarz_, _ = bss(refs, ests_z, compute_permutation=False)
        sdri_, siri_, sari_, _ = bss(refs, ests_i, compute_permutation=False)
        sdr[i_node], sir[i_node], sar[i_node] = sdr_[0], sir_[0], sar_[0]
        sdrz[i_node], sirz[i_node], sarz[i_node] = sdrz_[0], sirz_[0], sarz_[0]
        sdr_in_cnv[i_node], sir_in_cnv[i_node] = sdri_[0], siri_[0]

        stoi_in = stoi(s[i_node][0][fs:min_len], y[i_node][0][fs:min_len], fs)
        stoi_in_dry = stoi(s_dry[fs:min_len], y[i_node][0][fs:min_len], fs)
        stoi_out = stoi(s[i_node][0][fs:min_len], sh_t[i_node][fs:min_len], fs)
        stoi_out_dry = stoi(s_dry[fs:min_len], sh_t[i_node][fs:min_len], fs)
        stoi_out_z = stoi(s[i_node][0][fs:min_len], szh_t[i_node][fs:min_len], fs)
        stoi_out_z_dry = stoi(s_dry[fs:min_len], szh_t[i_node][fs:min_len], fs)
        delta_stoi[i_node] = stoi_out - stoi_in
        delta_stoi_dry[i_node] = stoi_out_dry - stoi_in_dry
        delta_stoi_z[i_node] = stoi_out_z - stoi_in
        delta_stoi_z_dry[i_node] = stoi_out_z_dry - stoi_in_dry

        _, fw_snr_out_mean, F = fw_snr(sf_t[i_node][fs:min_len], nf_t[i_node][fs:min_len], fs)      # Freq-weighted SNR
        _, fw_snr_in_mean, _ = fw_snr(s[i_node][0][fs:min_len], n[i_node][0][fs:min_len], fs)
        _, fw_snr_in_dry_mean, _ = fw_snr(s_dry[fs:min_len], n_dry[fs:min_len], fs)
        fw_snr_out[i_node] = fw_snr_out_mean
        _, fw_snr_out_mean_z, F = fw_snr(szf_t[i_node][fs:min_len], nzf_t[i_node][fs:min_len], fs)  # Freq-weighted SNR
        fw_snr_out_z[i_node] = fw_snr_out_mean_z
        fw_snr_in_dry[i_node] = fw_snr_in_dry_mean
        fw_snr_in_cnv[i_node] = fw_snr_in_mean

        # Freq-weighted SD
        _, f_sd_mean[i_node], _ = fw_sd(sf_t[i_node][fs:min_len], s[i_node][0][fs:min_len], fs)
        _, f_sd_mean_dry[i_node], _ = fw_sd(sf_t[i_node][fs:min_len], s_dry[fs:min_len], fs)
        _, f_sd_mean_z[i_node], _ = fw_sd(szf_t[i_node][fs:min_len], s[i_node][0][fs:min_len], fs)
        _, f_sd_mean_dry_z[i_node], _ = fw_sd(szf_t[i_node][fs:min_len], s_dry[fs:min_len], fs)

        # Save results
        sf.write(save_dir_root + 'WAV/' + str(i_rir) + '/in_mix-' + noise + '_Node-' + str(i_node + 1) + '.wav',
                 y[i_node][0], fs)
        sf.write(save_dir_root + 'WAV/' + str(i_rir) + '/out_mix-' + noise + '_Node-' + str(i_node + 1) + '.wav',
                 sh_t[i_node], fs)
        sf.write(save_dir_root + 'WAV/' + str(i_rir) + '/mid_z-' + noise + '_Node-' + str(i_node + 1) + '.wav',
                 szh_t[i_node], fs)
        sf.write(save_dir_root + 'WAV/' + str(i_rir) + '/in_noi-' + noise + '_Node-' + str(i_node + 1) + '.wav',
                 n[i_node][0], fs)
        sf.write(save_dir_root + 'WAV/' + str(i_rir) + '/out_noi-' + noise + '_Node-' + str(i_node + 1) + '.wav',
                 nf_t[i_node], fs)
        sf.write(save_dir_root + 'WAV/' + str(i_rir) + '/in_tar-' + noise + '_Node-' + str(i_node + 1) + '.wav',
                 s[i_node][0], fs)
        sf.write(save_dir_root + 'WAV/' + str(i_rir) + '/out_tar-' + noise + '_Node-' + str(i_node + 1) + '.wav',
                 sf_t[i_node], fs)
        np.save(save_dir_root + 'MASK/' + str(i_rir) + '/step1_' + noise + '_Node-' + str(i_node + 1),
                mask_sc[i_node])
        np.save(save_dir_root + 'MASK/' + str(i_rir) + '/step2_' + noise + '_Node-' + str(i_node + 1),
                mask_mc[i_node])
        np.save(os.path.join(save_dir_root, 'STFT', 'z', 'raw', dirry, '') +
                '{}_{}_Node-{}'.format(str(i_rir), noise, str(i_node + 1)), z_sh[i_node])

    results = {'snr_in_raw': rnd_snrs,
               'sar_cnv': sar, 'sir_cnv': sir, 'sdr_cnv': sdr,
               'delta_stoi_cnv': delta_stoi, 'delta_stoi_dry': delta_stoi_dry,
               'snr_out': fw_snr_out, 'snr_in_cnv': fw_snr_in_cnv, 'snr_in_dry': fw_snr_in_dry,
               'fw_sd_cnv': f_sd_mean, 'fw_sd_dry': f_sd_mean_dry,
               'sar_dry': sar_dry, 'sir_dry': sir_dry, 'sdr_dry': sdr_dry,
               'sdr_in_cnv': sdr_in_cnv, 'sir_in_cnv': sir_in_cnv,
               'sdr_in_dry': sdr_in_dry, 'sir_in_dry': sir_in_dry, 'sar_in_dry': sar_in_dry}
    resultsz = {'snr_in_raw': rnd_snrs,
                'sar_cnv': sarz, 'sir_cnv': sirz, 'sdr_cnv': sdrz,
                'delta_stoi': delta_stoi_z, 'delta_stoi_dry': delta_stoi_z_dry,
                'snr_out': fw_snr_out_z, 'snr_in_cnv': fw_snr_in_cnv, 'snr_in_dry': fw_snr_in_dry,
                'fw_sd_cnv': f_sd_mean_z, 'fw_sd_dry': f_sd_mean_dry_z,
                'sar_dry': sar_dry_z, 'sir_dry': sir_dry_z, 'sdr_dry': sdr_dry_z,
                'sdr_in_cnv': sdr_in_cnv, 'sir_in_cnv': sir_in_cnv,
                'sdr_in_dry': sdr_in_dry, 'sir_in_dry': sir_in_dry, 'sar_in_dry': sar_in_dry}

    pickle.dump(results, open(save_dir_root + 'OIM/results_tango_' + str(i_rir) + '_' + noise + '.p', 'wb'))
    pickle.dump(resultsz, open(save_dir_root + 'OIM/results_mwf_' + str(i_rir) + '_' + noise + '.p', 'wb'))

    # Save figure
    save_dir_fig = save_dir_root + 'FIG'
    save_conf(i_rir, save_path=save_dir_fig, title=rnd_snrs, scene=scenario, case=dset)

    print(str(i_rir) + '  done')


if __name__ == '__main__':
    print("\t Beginning of script")
    parser = argparse.ArgumentParser(description='DONSE arguments')
    parser.add_argument('--vad_type', '-vt',
                        type=str,
                        nargs=2)
    parser.add_argument('--sav_dir', '-sd',
                        type=str,
                        help='Dir to save results under')
    parser.add_argument('--rir',
                        type=int,
                        help='RIR of signal to filter')
    parser.add_argument('--scenario', '-scene',
                        type=str,
                        help='Scenario to use',
                        choices=['living', 'meeting', 'random'],
                        default='living')
    parser.add_argument('--noise',
                        type=str,
                        choices=['ssn', 'it', 'fs'],
                        default='fs')
    parser.add_argument('--mask_z', '-mz',
                        type=str,
                        help='Mask to apply on z',
                        choices=['None', 'local', 'distant', 'compressed', 'use_oracle_refs', 'use_oracle_zs'],
                        default='local')
    parser.add_argument('--mods', '-m',
                        type=str,
                        nargs=2,
                        help='Name (path + name) of trained pytorch models',
                        default=['None', 'None'])
    parser.add_argument('--zsigs', '-zs',
                        nargs='+',
                        default=['zs_hat'])

    args = parser.parse_args()
    vad_type = args.vad_type
    sav_dir = args.sav_dir
    mod_sc = None if args.mods[0] == 'None' else args.mods[0]
    mod_mc = None if args.mods[1] == 'None' else args.mods[1]
    models = [mod_sc, mod_mc]
    zsigs = args.zsigs[0] if len(args.zsigs) == 1 else args.zsigs
    i_rir_ = args.rir
    scenario = args.scenario
    mask_z = None if args.mask_z == 'None' else args.mask_z
    noise = args.noise

    print("Start main function")
    main(vad_type, sav_dir, i_rir_, noise, mask_z=mask_z, z_sigs=zsigs, scenario=scenario, models=models)



