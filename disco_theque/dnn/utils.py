import os
import sys
import glob
import time
import torch
import string
import numpy as np
from torch import nn
from code_utils.math_utils import db2lin
from disco_theque.dnn.models.crnn import build_crnn
from disco_theque.dnn.engine.losses import reconstruction_loss


def normalization(x, norm_type=None, axis=0):
    """
    Apply normalization on torch input tensor x.
    Args:
        x (torch tensor):  input data to normalize.
        norm_type (string): among
                         - scale_to_unit_norm: Scale feature of sample norm to 1
                         - scale_to_1: Scale feature of sample vectors so that almost all values are less than 1
                         - center_and_scale: Centre feature or samples to 0 and divide by their variance
        axis (int): Axis of the normalization
    Returns:
        normalized tensor same shape as `x`.
    """
    # In case of norm_over_db
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if norm_type == 'scale_to_unit_norm':
        x_norm = torch.norm(x, dim=axis, keepdim=True)
    elif norm_type == 'scale_to_1':
        x_norm = torch.max(x, dim=axis, keepdim=True)[0]
    elif norm_type == 'center_and_scale':
        x = x - torch.mean(x, dim=axis, keepdim=True)
        x_norm = torch.std(x, dim=axis, keepdim=True)
    else:
        return x

    x = x / x_norm

    return x


def tf_mask(s, n, type='irm1', bin_thr=0):
    """
    Compute the TF mask when `s` and `n` are target and noise STFT respectively.
    Args:
        s (array): target STFT
        n (array): noise STFT
        type (str): in form of 'irmX', 'iamX', 'ibmX' with X an integer, for Wiener, amplitude or 
                    binary masks ['irm1']
        bin_thr (float): Threshold for the binary mask (in dB) [0]
    Returns:
        A TF-mask same shape as `s` and `n`.
    """

    power = int(type[-1])
    if 'irm' in type:
        n_ = np.maximum(abs(n), sys.float_info.epsilon)
        xi = (abs(s) / n_) ** power
        m = xi / (1 + xi)
    elif 'ibm' in type:
        n_ = np.maximum(abs(n), sys.float_info.epsilon)
        xi = (abs(s) / n_) ** power
        m = (xi >= db2lin(bin_thr))
    elif 'iam' in type:
        m = (abs(s) / abs(s + n)) ** power
    else:
        raise ValueError('Unknown mask type. Should be "irmX", "ibmX" or "iamX"')

    return m


def get_input_lists(path_to_data, rirs_to_get, scenes=None,
                    snr_range=None, noise_to_get='ssn', ref_channel=1, z_sigs=None, z_file='oracle'):
    """Return a list of lists containing the names of the data (input channels + labels) to load during training
    Args:
        path_to_data (str): path where the data is saved.
        rirs_to_get (list, np.arange): All the RIR ids to load.
        scenes (str, list[str], None): 'random', 'living' or 'meeting' for spatial configuration to load. 
                                        Several possible (take a list) [random]
        snr_range (list): list of [snr_min, snr_max] for each noise source. [None]
        noise_to_get (str): Code of noise(s) seen during training ['ssn']
        ref_channel (int): Index of the reference microphone in the node (from 1 to n_mics_per_node[i_node])
        z_sigs (str, list[str], None): Compressed signals seen by NN during training.
                                       'zs_hat', 'zn_hat', ['zs_hat', 'zn_hat'] or None. [None] 
        z_file (str): name of the subfolder where the compressed signals are saved. ['oracle']
    Returns:
        A list of lists: [mixture_list[, compressed_lists(s)], label_list]
    """
    scenes = ['random'] if scenes is None else scenes
    scenes = [scenes] if type(scenes) is not list else scenes
    # Common parameters to all scenes
    n_nodes = 4
    y_chs = range(n_nodes)
    z_chs = [] if z_sigs is None else range(n_nodes)
    z_sigs = [z_sigs] if type(z_sigs) is not list else z_sigs
    output_list = [[] for _ in range(n_nodes + len(z_sigs) * len(z_chs) + n_nodes)]  # 4 refs + z + 4 masks

    for rir in rirs_to_get:
        # Chose scene randomly among possible ones
        rnd_scene_id = np.random.randint(len(scenes))
        scene = scenes[rnd_scene_id]
        path_to_data_ = os.path.join(path_to_data, scene, 'train', '')

        snr_range = [[0, 6]] if snr_range is None else snr_range
        snr_range = [snr_range] if len(np.shape(snr_range)) == 1 else snr_range
        snr_range_ = '_'.join(['{}-{}'.format(str(snr_range[k][0]), str(snr_range[k][1]))
                     for k in range(len(snr_range))])
        ## Paths
        path_to_input_y = os.path.join(path_to_data_, 'stft_processed', 'normed', 'abs', snr_range_, 'mixture', '')
        path_to_output = os.path.join(path_to_data_, 'mask_processed', snr_range_, '')

        # Chose noise among possible ones
        if noise_to_get in ['ssn', 'it', 'fs']:
            noises_to_get = [noise_to_get]
        elif noise_to_get == 'noit':
            noises_to_get = ['ssn', 'fs']
        elif noise_to_get == 'all':
            noises_to_get = ['ssn', 'it', 'fs']
        rnd_noise_id = np.random.randint(len(noises_to_get))

        for y_ch in y_chs:
            output_list[y_ch].append(path_to_input_y + '{}_{}_Ch-{}.npy'.format(str(rir),
                                                                                noises_to_get[rnd_noise_id],
                                                                                str(ref_channel + n_nodes * y_ch)))
            output_list[-n_nodes + y_ch].append(path_to_output
                                                + '{}_{}_Ch-{}.npy'.format(str(rir),
                                                                           noises_to_get[rnd_noise_id],
                                                                           str(ref_channel + n_nodes * y_ch)))
        for z_ch in range(len(z_chs)):
            for z_sig in range(len(z_sigs)):
                path_to_input_z = os.path.join(path_to_data_, 'stft_z', z_file, 'normed', 'abs',
                                               snr_range_, z_sigs[z_sig], '')
                output_list[n_nodes + z_ch + z_sig*n_nodes].append(path_to_input_z +
                                                                   '{}_{}_Node-{}.npy'.format(str(rir),
                                                                                              noises_to_get[rnd_noise_id],
                                                                                              str(z_chs[z_ch] + 1)))

    return output_list


def load_architecture(n_ch, win_len=21, **kwargs):
    """Load a model instance"""
    model, opt = building_fn((n_ch, 21, 257), (32, 64, 64), (3, 3, 3), (1, 1, 1),
                             [(1, 4), (1, 4), (1, 4)],
                             (None, None, None),
                             [256], 'GRU',
                             257,
                             conv_padding=[(0, 1), (0, 1), (0, 1)],
                             rnn_dropouts=0.5)
    return model, opt


def load_states(model, optimizer, saved_states):
    """Load state dictionaries stored in `saved_states` into instances of corresponding model and optimizer.
        Args:
            model (pytorch model): instance of pytorch model
            optimizer (pytorch optimizer): instance of pytorch optimizer
            saved_states (dict):    dictionary with pytorch model and optimizer instances under the keys
                                    'model_state_dict' and 'optimizer_state_dict' respectively
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load state dicts into instances
    trained_states = torch.load(saved_states, map_location=device)
    model.load_state_dict(trained_states['model_state_dict'])
    model = model.to(device, non_blocking=True) # Should be done before opt. constr.
    optimizer.load_state_dict(trained_states['optimizer_state_dict'])
    # Return loss values of first training
    train_losses = trained_states['train_loss']
    val_losses = trained_states['val_loss']
    train_losses = np.trim_zeros(train_losses, 'b')
    val_losses = np.trim_zeros(val_losses, 'b')

    return train_losses, val_losses


def get_model_name(model_name):
    """Return random name or already used one depending on whether model has already been trained."""
    if model_name is None:
        all_chars = string.ascii_letters + string.digits
        rnd_string = "".join(all_chars[int(str(time.time())[-4:]) % len(all_chars)] for x in range(4))
    else:
        rnd_string = model_name.split('/')[-1].split('_model')[0] + '_retrain'

    return rnd_string


def get_loss_frames(win_len, part):
    """Select the label frames corresponding to the input (some are lost by the convolutional layers)
    Args:
        win_len (int): length of the input samples (nb of frames)
        part (str, int): 'all', 'mid', 'last' to select all the frames, or only the middle ones or only the last one
    """
    if part == 'all':
        first_frame, last_frame = 0, win_len
    elif part == 'mid':
        first_frame = int(np.ceil(win_len) / 2)
        last_frame = first_frame + 1
    elif part == 'last':
        first_frame = win_len - 1
        last_frame = first_frame + 1
    elif isinstance(part, int):
        first_frame = part
        last_frame = part + 1
    else:
        raise ValueError("Unknown argument value {}. It should be either 'all', 'mid' or 'last'.".format(part))

    return first_frame, last_frame


def get_x_for_loss(x, part, model=None, model_output=1, n_freq=257):
    """
    Input and output of NN do not obviously have the same shape because of convolutional layers. This selects the input frames
    corresponding to the output ones in order to weight the MSE loss with the input STFT.
    Args:
        x (torch tensor): Input tensor
        part (str): 'mid', 'all', 'last' or integer: which frame(s) to take.
        model (torch nn.Module): torch model with a function yelding the first and last frames of train data / output data
        model_output (0/1): if 1:   x is the output of the model and its shape might differ from the shape of the input of
                             the model.
                            if 0: x is ground truth data / train data. We might want to adapt its data to the output of the
                             model.
        n_freq (int): Expected number of features in the last dimension.
    Returns:
        Input frames matching with the output ones.
    """
    if model is None:
        win_len = x.shape[-2]
        ff, lf = get_loss_frames(win_len, part)
    else:
        ff, lf = model.get_loss_frames(part)[model_output]
    # Check channel dimension (MC CRNN)
    if len(x.size()) == 3:
        x_out = torch.squeeze(x[:, ff:lf, :])
    elif len(x.size()) == 4:
        x_out = torch.squeeze(x[:, 0, ff:lf, :])    # First channel is considered for loss
    else:
        raise NotImplementedError('Unexpected case')
    # Check feature dimension (MC RNN)
    if len(x_out.size()) == 2:      # When returning last or mid frame only
        x_out = x_out[:, :n_freq]
    else:
        x_out = x_out[:, :, :n_freq]

    return torch.squeeze(x_out)


def train_one_batch(model, x, y, optimizer, output_frames='last'):
    """
    Perform one step of training (for and backward)
    Args:
        model (torch.nn.MOdule):   Model instance
        x (tensor):       (*un*normed) input of model
        y (tensor):       label corresponding to input `x`
        optimizer (torch.optim):       optimizer (pytorch)
        output_frames (str, int):   one of ['last', 'mid', 'all'] or integer, about which frame(s) should be predicted.
    Returns:
        loss value
    """
    model.train()       # Dropout ON and batchnormalization updated
    est = model(x)
    x_loss = get_x_for_loss(x, output_frames, model=model, model_output=0)      # For rec loss: unnormed input
    y_loss = get_x_for_loss(y, output_frames, model=model, model_output=0)
    est_loss = get_x_for_loss(est, output_frames, model=model, model_output=1)
    loss_value = reconstruction_loss(y_loss, est_loss, x_loss)
    # Only for training:
    optimizer.zero_grad()
    loss_value.backward()
    if optimizer.clip:
        nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
    optimizer.step()
    return loss_value.item()


def eval_one_batch(model, x, y, output_frames='last'):
    """
    Perform one step of evaluation (no dropout nor batchnormalization, gradient is not propagated)
    Args:
        model (torch.nn.MOdule):   Model instance
        x (tensor):       (*un*normed) input of model
        y (tensor):       label corresponding to input `x`
        output_frames (str, int):   one of ['last', 'mid', 'all'] or integer, about which frame(s) should be predicted.
    Returns:
        evaluation value
    """
    model.eval()        # Dropout and batchnormalization "off"
    est = model(x)
    x_loss = get_x_for_loss(x, output_frames, model=model, model_output=0)
    y_loss = get_x_for_loss(y, output_frames, model=model, model_output=0)
    est_loss = get_x_for_loss(est, output_frames, model=model, model_output=1)
    loss_value = reconstruction_loss(y_loss, est_loss, x_loss)
    # loss_value = nn.MSELoss()(y_loss, est_loss)
    return loss_value.item()

