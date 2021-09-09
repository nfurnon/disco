""" First step of tango to create and save the compressed signals"""
import os
import argparse
import soundfile as sf
import librosa as lb
import numpy as np
import torch
from disco_theque.sigproc_utils import vad_oracle_batch
from disco_theque.se_utils.internal_formulas import intern_filter
from disco_theque.dnn.utils import tf_mask
from disco_theque.speech_enhancement.utils import prepare_data, plot_conf
from disco_theque.dnn.models.crnn import build_crnn
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
path_to_dataset = '../../../dataset'


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

    return y, s, n


def load_models(weightss):
    """
    Load two instances of a class and their weights
    Args;
        models (list[str, None]) names of saved state dict for the models to load
    Returns:
        list[nn.Module, None] List of the models or of None whether models should be used to predict the masks or not.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nodes_nbs = [1, nb_nodes]
    out = []
    i_step = 0
    if weightss[i_step] is not None:
        model, _ = build_crnn((int(nodes_nbs[i_step]), 21, 257), (32, 64, 64), (3, 3, 3), (1, 1, 1),
                              [(1, 4), (1, 4), (1, 4)],
                              (None, None, None),
                              [256], 'GRU',
                              257,
                              conv_padding=[(0, 1), (0, 1), (0, 1)])
        saved_weights = torch.load(weightss[i_step], map_location=torch.device(device))
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


def get_z_for_mask(z_in, k, nb_nodes=nb_nodes):
    """Return the channels to stack at the input of the NN
    Args:
        z_in (np.ndarray): Compressed signal of the target estimation
        k (int): Local node (where NN is to predict the mask)
        nb_nodes (int): number of nodes in the microphone array
    """
    ch_of_z = list(range(nb_nodes))
    ch_of_z.pop(k)  # Do not take local z to compute the mask
    z_out = np.array(z_in)[ch_of_z, :, :]
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
        mask_type (str, list[str]): Mask type
        mod (torch.nn.Module): model (with weights)
        ts (array): Speech time signal (to compute VAD)

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
    else:
        n_freq = np.shape(ss)[0]
        m = np.zeros(np.shape(ss))
        vad = vad_oracle_batch(ts, win_len=N_FFT, win_hop=N_HOP)
        vad = vad[::N_HOP]
        m[:, :len(vad)] = np.tile(vad, (n_freq, 1))

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


def offline_tango(y, s, n, vads='irm1', mods=None, mask_for_z=MASK_Z):
    """
    Only the first filtering step is operated as we are only interested in the compressed signals.
    Args:
        y (np.ndarray, list[array]): input multichannel mixture: n_nodes x n_channels x time. Probably a list if nodes
                                    don't have the same number of signals.
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
    n_freq = int(N_FFT / 2 + 1)
    n_frames = 3 + int(np.floor((length_signal - N_FFT) / N_HOP))
    y_stft = [[] for _ in range(n_nodes)]        # Mixture STFT matrix (n_nodes x n_channels x n_freq x n_time)
    s_stft = [[] for _ in range(n_nodes)]        # Mixture STFT matrix (n_nodes x n_channels x n_freq x n_time)
    n_stft = [[] for _ in range(n_nodes)]        # Mixture STFT matrix (n_nodes x n_channels x n_freq x n_time)
    s_hat_stft_z = [[] for _ in range(n_nodes)]       # Speech STFT matrix (n_nodes x n_channels x n_freq x n_time)
    n_hat_stft_z = [[] for _ in range(n_nodes)]       # Noise STFT matrix (n_nodes x n_channels x n_freq x n_time)
    r_ss_loc = [[np.zeros((n_channels_per_node[k], n_channels_per_node[k]), 'complex64') for _ in range(n_freq)]
                for k in range(n_nodes)]
    r_nn_loc = [[np.zeros((n_channels_per_node[k], n_channels_per_node[k]), 'complex64') for _ in range(n_freq)]
                for k in range(n_nodes)]

    z_stft_y = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Compressed signal
    zn_stft = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Compressed signal
    z_stft_s = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Speech part of z
    z_stft_n = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Noise part of z
    z_gevd_s = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Noise part of z
    z_gevd_n = [np.zeros((n_freq, n_frames), 'complex64') for _ in range(n_nodes)]    # Noise part of z

    masks_z = [[] for _ in range(n_nodes)]
    filtre, rank = 'gevd', 1

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
                                  mask_type=vads, mod=mods[0], win_len=WIN_LEN, win_hop=1, frame_to_pred=PRED_FRAME,
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
                                             mu=1, type=filtre, rank=rank)
            for t in range(n_frames):
                z_stft_y[i_nod][f, t] = np.inner(np.conjugate(w_loc), y_stft[i_nod][:, f, t])
                z_stft_s[i_nod][f, t] = np.inner(np.conjugate(w_loc), s_stft[i_nod][:, f, t])
                z_stft_n[i_nod][f, t] = np.inner(np.conjugate(w_loc), n_stft[i_nod][:, f, t])
                z_gevd_s[i_nod][f, t] = np.inner(t1_z, s_stft[i_nod][:, f, t])      # Reference signal sent
                z_gevd_n[i_nod][f, t] = np.inner(t1_z, n_stft[i_nod][:, f, t])
        # Noise estimation in compressed signal
        zn_stft[i_nod] = (y_stft[i_nod][0, :] - z_stft_y[i_nod])

    return z_stft_y, z_stft_s, z_stft_n, zn_stft, masks_z


def main(vad_type, save_dir, i_rir, noise, scenario='living', mask_z=MASK_Z, weights_sc=None):

    # Ensure filtering has not yet been done
    dset = get_dset(i_rir)
    save_dir_root = os.path.join(path_to_dataset, 'disco', scenario, dset, 'stft_z', save_dir, '')
    snr_range = SNR_RANGE
    dirry = get_directory_name(snr_range)

    if os.path.isfile(os.path.join(save_dir_root, 'normed', 'abs', dirry, 'zn_hat', '') +
                      '{}_{}_Node-{}'.format(str(i_rir), noise, str(nb_nodes))):
        print('Conf {} with {} noise already processed'.format(str(i_rir), noise))
        return

    os.makedirs(os.path.join(save_dir_root, 'raw', dirry, 'zs_hat'), exist_ok=True)
    os.makedirs(os.path.join(save_dir_root, 'raw', dirry, 'zn_hat'), exist_ok=True)
    os.makedirs(os.path.join(save_dir_root, 'normed', 'abs', dirry, 'zs_hat'), exist_ok=True)
    os.makedirs(os.path.join(save_dir_root, 'normed', 'abs', dirry, 'zn_hat'), exist_ok=True)

    y, s, n = get_input_signals(i_rir, scenario, noise, snr_range)
    # Load statistics to norm the z signals later

    # Load trained models if necessary
    weights = [weights_sc, None]
    models = load_models(vad_type, weights)
    print("Start algorithm " + str(i_rir))
    z_sh, z_s, z_n, z_nh, mask_sc = offline_tango(y, s, n, vad_type,
                                                  mods=models,
                                                  mask_for_z=mask_z)
    print("End algorithm")

    for i_node in range(nb_nodes):
        # Save results
        np.save(os.path.join(save_dir_root, 'raw', dirry, 'zs_hat', '') +
                '{}_{}_Node-{}'.format(str(i_rir), noise, str(i_node + 1)), z_sh[i_node])
        np.save(os.path.join(save_dir_root, 'raw', dirry, 'zn_hat', '') +
                '{}_{}_Node-{}'.format(str(i_rir), noise, str(i_node + 1)), z_nh[i_node])
        np.save(os.path.join(save_dir_root, 'normed', 'abs', dirry, 'zs_hat', '') +
                '{}_{}_Node-{}'.format(str(i_rir), noise, str(i_node + 1)), abs(z_sh[i_node]))
        np.save(os.path.join(save_dir_root, 'normed', 'abs', dirry, 'zn_hat', '') +
                '{}_{}_Node-{}'.format(str(i_rir), noise, str(i_node + 1)), abs(z_nh[i_node]))
    print(str(i_rir) + '  done')


if __name__ == '__main__':
    print("\t Beginning of script")
    parser = argparse.ArgumentParser(description='DONSE arguments')
    parser.add_argument('--vad_type', '-vt',
                        type=str,
                        default='irm1')
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
    parser.add_argument('--mod_sc', '-msc',
                        type=str,
                        help='Name of single-channel model',
                        default='./')

    args = parser.parse_args()
    vad_type = args.vad_type
    sav_dir = args.sav_dir
    mod_sc_ = None if args.mod_sc == 'None' else args.mod_sc
    i_rir_ = args.rir
    scenario = args.scenario
    mask_z = None if args.mask_z == 'None' else args.mask_z
    noise = args.noise

    print("Start main function")
    main(vad_type, sav_dir, i_rir_, noise, mask_z=mask_z, scenario=scenario, weights_sc=mod_sc_)



