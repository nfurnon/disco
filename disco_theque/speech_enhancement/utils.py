import os
import torch
import numpy as np
import librosa as lb
from disco_theque.dnn.utils import move_to_device

stft_min, stft_max = 1e-6, 1e3
fs = 16000
n_hop = 256
frames_lost = 6


def get_frames_to_pad(in_len, output_frames, out_len=None):
    """The aim is to pad the input sequence so that the selected output frame corresponds to the first frame of the
    input sequence.
    :param in_len:              (int) length of input sequence
    :param output_frames:       (str) which frame(s) is (are) selected in output sequence to predict total segment ?
    :param out_len:             (int) length of sequence predicted by NN. If less than in_len, assumed is that it is
                                      centered in in_len (so that frames are symmetrically lost at beginning and end
                                      of the sequence)
    """
    out_len = in_len if out_len is None else out_len
    if output_frames == 'mid':
        frames_to_pad = (int(np.floor(in_len / 2)), int(np.floor(in_len / 2)))
    elif output_frames == 'last':
        selected_frame = (in_len + out_len) // 2
        frames_to_pad = (selected_frame - 1, in_len - selected_frame)
    elif output_frames == 'all':
        frames_to_pad = (0, 0)
    else:
        raise ValueError(":param output_frames: should be 'mid', 'last' or 'all'")

    return frames_to_pad


def normalization(x, norm_type=None, axis=0):
    """
    Apply normalization on input *numpy* tensor x.
    :param x:               (numpy array)  input data to normalize.
    :param norm_type:       (string) among
                                - scale_to_unit_norm: Scale feature of sample norm to 1
                                - scale_to_1: Scale feature of sample vectors so that almost all values are less than 1
                                - center_and_scale: Centre feature or samples to 0 and divide by their variance
    :param axis:            Axis of the normalization
    :param configuration:   Configuration dictionary for normalisation characterisitcs (proper to disco dataset)
    :return:
    """
    if norm_type != 'pcen':
        x = np.clip(abs(x), stft_min, stft_max)
        if norm_type == 'scale_to_unit_norm':
            x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
        elif norm_type == 'scale_to_1':
            x_norm = np.quantile(x, 0.99, axis=axis, keepdims=True)
        elif norm_type == 'center_and_scale':
            x = x - np.mean(x, axis=axis, keepdims=True)
            x_norm = np.std(x, axis=axis, keepdims=True)
        else:
            return x
        x = x / x_norm

    else:   # PCEN: need pcen from librosa so do no turn x into torch tensor straight away
        x = np.clip(abs(x), stft_min, stft_max)
        x *= (2 ** 31)
        x = lb.core.pcen(x, sr=fs, hop_length=n_hop)

    return x


def prepare_data(y_data, three_d_tensor,
                 z_data=None, win_len=21, win_hop=1, frame_to_pred='last',
                 norm_type=None, frames_lost=frames_lost):
    """
    Reshape data to form windows that can be processed by model.
    Args:
        y_data (array): y_data; input mixture
        three_d_tensor (bool): 3D tensor ? If yes, z is stacked on channel (first) dimension
        z_data (None, list[array]: can be None in single-channel prediction.
                                   Can be a list in multichannel prediction, in which case the list length is the number
                                   of channels stacked on top of the input mixture. [None]
        win_len (int): length of subwindows  [21]
        win_hop (int): shifting between two adjacent windows [1]
        frame_to_pred (str): Which of the input frames to predict ('mid', 'last', 'all') ['last']
        norm_type (str): Type of normalization [None]
        frames_lost (int): Number of frames lost between input and output of the NN
    Returns:
        (n_samples x n_ch x n_time x n_freq) with n_samples successive `win_len`-long windows, shifted of `win_hop`,
                                                    and n_ch channels (stacked in the third dimension if three_d_tensor
                                                    is False)
    """
    # Normalize the data
    y_in = normalization(y_data, norm_type=norm_type, axis=1)
    if z_data is not None:
        z_in = [[] for _ in range(len(z_data))]
        for iz, z in enumerate(z_data):
            z_in[iz] = normalization(z, norm_type=norm_type, axis=1)

    frames_to_pad = get_frames_to_pad(win_len, frame_to_pred, out_len=win_len - frames_lost)

    # Pad input at edges to compensate feature maps
    y_in = np.pad(y_in, ((0, 0), frames_to_pad), 'constant', constant_values=0)
    if z_data is not None:
        for iz, z in enumerate(z_in):
            z_in[iz] = np.pad(z, ((0, 0), frames_to_pad), 'constant', constant_values=0)
        z_in = np.array(z_in)

    # Preallocate output
    n_samples = int(1 + np.floor((y_data.shape[1] + np.sum(frames_to_pad) - win_len) / win_hop))
    n_freq = y_data.shape[0]
    n_ch = 1
    if z_data is not None:
        n_ch += len(z_data)
    if three_d_tensor:
        y_out = np.zeros((n_samples, n_ch, win_len, n_freq))    # NB: we transpose time and freq dimensions for pytorch
    else:
        y_out = np.zeros((n_samples, win_len, n_ch * n_freq))   # Stack over frequencies

    # Reshape input into output
    for iwin in range(n_samples):
        y_to_feed = y_in[:, iwin * win_hop:iwin * win_hop + win_len].T
        if three_d_tensor:
            y_out[iwin, 0, :, :] = y_to_feed
            if z_data is not None:
                for iz, z in enumerate(z_in):
                    z_to_feed = z[:, iwin * win_hop:iwin * win_hop + win_len].T
                    y_out[iwin, iz + 1, :, :] = z_to_feed
        else:
            y_out[iwin, :, :n_freq] = y_to_feed
            if z_data is not None:
                for iz, z in enumerate(z_in):
                    z_to_feed = z[:, iwin * win_hop:iwin * win_hop + win_len].T
                    y_out[iwin, :, (iz+1)*n_freq:(iz+2)*n_freq] = z_to_feed

    # Move to GPU / Keep on CPU
    y_out = torch.from_numpy(y_out).float()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y_out = y_out.to(device, non_blocking=True)

    return y_out


def plot_conf(infos, mics_per_node=[4, 4, 4, 4], return_fig=False):
    """
    Plot spatial configuration of room whose informations is stored in infos. View from above (no height information).
    Args:
        infos (dict): infos saved during the generation of the database.
        mics_per_node (list, array, tuple): Number of microphones per node.
        return_fig (bool): if `True`, the figure variable is returned.
    """
    # Import here to avoid matplotlib requirement if plotting is not wished.
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.ioff()
    f = plt.figure()
    plt.gca().add_patch(Rectangle((0, 0), infos['room']['length'], infos['room']['width'], fill=False, linewidth=3))
    plt.plot(infos['mics'][0, :], infos['mics'][1, :], 'x')
    plt.plot(infos['sources'][:, 0], infos['sources'][:, 1], 'x')
    plt.gca().axis('equal')
    # Text
    n_nodes = len(mics_per_node)
    n_sources = np.shape(infos['sources'])[0]
    mics_cums = np.cumsum([0] + list(mics_per_node))
    for i_n in range(n_nodes):
        plt.text(1.05 * infos['mics'][0, mics_cums[i_n]], 1.05 * infos['mics'][1, mics_cums[i_n]],
                 'Node ' + str(i_n + 1), fontsize=10)
    for i_s in range(n_sources):
        plt.text(1.05 * infos['sources'][i_s, 0], 1.05 * infos['sources'][i_s, 1],
                 'Source ' + str(i_s + 1), fontsize=10)
    plt.gca().set(xlim=(-1, infos['room']['length'] + 1), ylim=(-1, infos['room']['width'] + 1))
    if return_fig:
        return f
    else:
        f.show()

