import os
import sys
import glob
import time
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from code_utils.math_utils import db2lin, my_stft
from code_utils.mask_utils import tf_mask
import pyroomacoustics as pra
from code_utils.sigproc_utils import vad_oracle_batch
from database_classes import SignalSetup, MeetitSetup
from mir_eval.separation import bss_eval_sources as bss
import ipdb


# %%
def create_directories(scene, case):
    """
    Create the directories where all the files will be saved
    :param scene:
    :param case:
    :return:
    """
    os.makedirs('/home/nfurnon/dataset/meetit/' + scene + '/' + case + '/log/infos/', exist_ok=True)
    os.makedirs('/home/nfurnon/dataset/meetit/' + scene + '/' + case + '/wav/clean/dry', exist_ok=True)
    os.makedirs('/home/nfurnon/dataset/meetit/' + scene + '/' + case + '/wav/clean/cnv/', exist_ok=True)

    return os.path.join('/home/nfurnon/dataset/meetit/', scene, case, '')


def get_wavs_list(case):
    """
    Return lists of wavs for the target and noise signals, as well as a list of speakers which one can pick from to
    compute a SSN noise.
    :param case:
    :return:
    """
    if case == 'test':
        talkers_list = glob.glob('/home/nfurnon/corpus/LibriSpeech/audio/test-clean/*/*/*.flac')
    elif case == 'val':
        talkers_list = glob.glob('/home/nfurnon/corpus/LibriSpeech/audio/dev-clean/*/*/*.flac')
    elif case == 'train':
        talkers_list = glob.glob('/home/nfurnon/corpus/LibriSpeech/audio/train-clean-360/*/*/*.flac')
    else:
        raise ValueError('`case` should be "test", "val" or "train"')
    talkers_list.sort()

    return talkers_list


def get_value_range(i_rir, n_rirs, vmin=0, vmax=20, n_bins=5):
    """
    According to the RIR index, return a range of values among the ranges that are linearly spaced between `vmin` and
    `vmax`.
    Args:
        i_rir (int): RIR index
        n_rirs (int): Total number of RIRs created
        vmin (float): minimal value
        vmax (float): maximal value
        n_bins (int): Number of bins
    """
    i_bin = i_rir // (n_rirs / n_bins)
    dval = vmax - vmin
    return np.array([vmin + i_bin * dval / n_bins, vmin + (i_bin + 1) * dval / n_bins])


def mix_signals(room):
    """ Simulate the room that has been instantiated. Return mixed signals at microphones, image signals (unmixed ones)
    and the RIR
    :param room:                        pyroomacoustic's instance of room with mics and sources positions and sources
                                        images.
    :return clean_reverbed_signals:     Image signals
            mixed_reberbed_signals:     Mixed signals
            rirs:                       Room impulse response from every source to every microphone
    """
    room.image_source_model(use_libroom=True)
    room.compute_rir()
    clean_reverbed_signals = room.simulate(return_premix=True)

    return clean_reverbed_signals, room.rir


def _get_convolved_vads(x):
    """
    Compute VAD on image signals of the target
    :param  x:      Array of image signals we want to compute the VAD of. Time dimension is the last (2nd) one.
    :return:
    """
    vads = np.zeros(x.shape)
    for i_m in range(vads.shape[0]):
        vads[i_m, :] = vad_oracle_batch(x[i_m, :], thr=0.001)

    return vads


def sir_at_node(s, n):
    """Compute the SIR of the mixture determined by `s` + `n`"""
    sirs = np.zeros(s.shape[0])
    m = s + n
    for i in range(s.shape[0]):
        refs = np.vstack((s[i], n[i]))
        ests = np.vstack((m[i], m[i]))
        sdr, sir, sar, _ = bss(refs, ests, compute_permutation=False)
        sirs[i] = sir[0]

    return np.mean(sirs)


def check_sir_validity(current_sirs, past_sirs, sir_classes, delta_sir=2):
    """Returns False if difference between the nodes is too high or if enough configurations have been already created
    with a similar SIR"""
    # Reject if difference between the nodes is too high
    for i_source in range(len(past_sirs) - 1):
        if np.any((current_sirs - np.roll(current_sirs, i_source + 1)) > delta_sir):
            return False
    # We now (arbitrarily) focus on the first SIR value a see if we already have enough of it
    bin_starts = [int(k.split('-')[0]) for k in sir_classes.keys()]
    bin_stops = [int(k.split('-')[-1]) for k in sir_classes.keys()]
    if current_sirs[0] < bin_starts[0] or current_sirs[0] > bin_stops[-1]:     # SIR is too low or too high
        return False
    bin_index = np.where(current_sirs[0] > bin_starts)[0][-1]
    filled_sirs = plt.hist(past_sirs[:][0], bins=len(sir_classes), range=(bin_starts[0], bin_stops[-1]))[0]
    if filled_sirs[bin_index] > sir_classes['{}-{}'.format(bin_starts[bin_index], bin_stops[bin_index])]:
        return False
    else:
        return True


def simulate_room(room_setup, signal_setup, i_target_file, dset, mics_per_node, past_sirs, n_rirs_per_proc):
    """
    Simulate the RIRs of a room with two or more noise sources.
    We assume that there is only one target source and that all others are noise.
    :param room_setup:
    :param signal_setup:
    :param i_target_file:
    :return:
    """
    # Create signals
    target_file = signal_setup.speakers_list[i_target_file]
    target_source_signal, target_source_vad, fs = signal_setup.get_target_segment(target_file=target_file)
    if target_source_signal is None:
        return "redraw_source_signal"

    # Create room
    room = pra.ShoeBox([room_setup.length, room_setup.width, room_setup.height],
                       fs=fs,
                       max_order=20,
                       absorption=room_setup.alpha)

    # Add microphones
    room.add_microphone_array(pra.MicrophoneArray(room_setup.microphones_positions, room.fs))

    # First source (target source)
    room.add_source(room_setup.source_positions[0], signal=target_source_signal)
    room.source_vad = target_source_vad
    # Other sources
    for noise_source in range(room_setup.n_sources - 1):
        noise_source_signal, noise_vad = signal_setup.get_noise_segment(duration=signal_setup.target_duration)
        room.add_source(room_setup.source_positions[noise_source + 1], signal=noise_source_signal)

    # Reverb signals, mix them
    image_signals, rirs = mix_signals(room)
    # VAD of reverbed signal; better when computed on reverbed signal rather than shifting the source VAD
    target_image_vad = _get_convolved_vads(image_signals[0])
    room.image_vad = target_image_vad

    # Measure SIR
    sirs = np.zeros(len(mics_per_node))
    for noise_source in range(room_setup.n_sources):
        total_mics = np.hstack((np.array([0]), np.cumsum(mics_per_node)))
        local_target = image_signals[noise_source][total_mics[noise_source]:total_mics[noise_source + 1], :]
        noise_sources_ids = list(range(room_setup.n_sources))
        noise_sources_ids.remove(noise_source)  # All *other* sources are considered as noise
        local_noises = sum(image_signals[noise_sources_ids, total_mics[noise_source]:total_mics[noise_source + 1], :],
                           0)  # Sum all other sources
        sirs[noise_source] = sir_at_node(local_target, local_noises)
    # Check SIRs
    n_classes = 4
    bin_level = np.ceil(n_rirs_per_proc / n_classes)
    sir_classes = {'{}-{}'.format(2 + 3 * k, 2 + 3 * (k + 1)): bin_level for k in range(n_classes)}
    sirs_are_valid = check_sir_validity(sirs, sir_classes=sir_classes, past_sirs=past_sirs)
    if not sirs_are_valid:
        return "redraw_room_setup"

    if dset in ['train', 'val']:
        len_max = int((signal_setup.duration_range[-1] + 1) * fs)
        len_to_pad = np.maximum(len_max - image_signals.shape[-1], 0)
        image_signals = np.pad(image_signals, ((0, 0), (0, 0), (0, len_to_pad)), 'constant', constant_values=0)
        image_signals = image_signals[:, :, :len_max]    # truncate if necessary

    return room, image_signals, target_image_vad, sirs


def get_masks(sigs, mics_per_node, room):
    """
    Take STFT of sigs and compute the mask of each source for all nodes.
    ASSUMPTIONS:
        - First mic of each node is the reference mic
    """
    n_sources = room.n_sources
    stfts, masks = [], [[] for _ in range(n_sources)]
    mics_of_nodes = np.hstack((np.array([0]), np.cumsum(mics_per_node)))
    for i_node in range(len(mics_per_node)):
        local_signals = sigs[:, mics_of_nodes[i_node]:mics_of_nodes[i_node + 1], :]
        # Compute the STFTs
        for i_ch in range(mics_per_node[i_node]):
            stfts.append(my_stft(sum(local_signals[:, i_ch, :], 0)))
            # Compute the mask
            for i_source in range(n_sources):
                noise_sources_ids = list(range(n_sources))
                noise_sources_ids.remove(i_source)  # All *other* sources are considered as noise
                target = my_stft(local_signals[i_source, i_ch, :])
                noises = my_stft(sum(local_signals[noise_sources_ids, i_ch, :], 0))
                masks[i_source].append(tf_mask(target, noises, type='irm1'))

    return np.array(stfts), np.array(masks)


def save_data(sources, images, infos, id, path, fs=16000):
    """
    Ca devient un peu cochon ici. We only save the signals we might use. So in meeting scenario, SSN is saved for all
    sources but interferent talker only for second source and freesound only for third source.
    :param sources:     Source images of target and noise
    :param images:      Image signals of all sources (n_sources x n_mics x n_samples)
    :param infos:
    :param path:
    :param id:
    :param fs:
    :return:
    """
    path_wav_clean = os.path.join(path, 'wav', 'clean', '')
    path_log = os.path.join(path, 'log', 'infos', '')

    # Save source signals
    for i_s in range(len(sources)):
        path_source = os.path.join(path_wav_clean, 'dry', "")
        sf.write(path_source + str(id) + '_S-' + str(i_s + 1) + '.wav', sources[i_s].signal, fs)
    # Save the images of clean signals
    for i_s in range(len(images)):
        path_image = os.path.join(path_wav_clean, 'cnv', "")
        for i_ch in range(len(images[i_s])):
            sf.write(path_image + str(id) + '_S-' + str(i_s + 1) + '_Ch-' + str(i_ch + 1) + '.wav',
                     images[i_s][i_ch], fs)

    # Save infos
    np.save(path_log + str(id), infos, allow_pickle=True)


if __name__ == "__main__":
    # %% JOB PROPERTIES
    dset = sys.argv[1]
    if dset == 'train':
        rir_start = 1
    elif dset == 'val':
        rir_start = 10001
    else:
        rir_start = 11001
    scenario = sys.argv[2]
    i_rir = int(sys.argv[3])
    time.sleep((i_rir - 1)/55)
    n_rirs_per_process = int(sys.argv[4])
    n_prl_processes = int(sys.argv[5])
    rir_stop = int(i_rir + n_rirs_per_process - 1)
    n_sources = int(sys.argv[6])     # NB There will be as many nodes as sources

    root_out = create_directories(scenario, dset)       # Create dirs and return root to wav/

    # Instantiate the room and signal classes
    l_range, w_range, h_range, beta_range = [3, 9], [3, 7], [2.5, 3], [0.3, 0.6]
    n_sensors_per_node = [4 for _ in range(n_sources)]
    d_mw, d_mn, d_nn, d_rnd_mics = 0.5, 0.05, 0.5, 1
    d_ss, d_sn, d_sw = 0.5, 0.5, 0.5
    z_range_m = [0.7, 0.8]  # Table
    z_range_s = [1.15, 1.30]  # Sitting people
    rmin, rmax = 0.3, 2.5
    r_range = (rmin, rmax)
    d_nt_range, d_st_range = (0.05, 0.20), (0, 0.50)

    room_setup = MeetitSetup(l_range=l_range, w_range=w_range, h_range=h_range, beta_range=beta_range,
                             n_sensors_per_node=n_sensors_per_node,
                             d_mw=d_mw, d_mn=d_mn, d_nn=d_nn, z_range_m=z_range_m,
                             d_rnd_mics=d_rnd_mics,
                             n_sources=n_sources, d_ss=d_ss, d_sn=d_sn, d_sw=d_sw, z_range_s=z_range_s,
                             r_range=r_range,
                             d_nt_range=d_nt_range, d_st_range=d_st_range)

    # %% SIGNAL PARAMETERS
    duration_range = (7, 10)
    var_tar = db2lin(-23)
    snr_dry_range = np.array([[0, 0]])    # No influence in this script.
    snr_cnv_range = (-10, 15)
    min_delta_snr = 0

    speakers_lists = get_wavs_list(dset)   # Same (ordered) lists for all // processes

    # Which SIR bin should be filled with this rir_start ?
    past_sirs = [[] for _ in range(n_sources)]
    # %% CREATE SIGNALS FOR ALL REQUIRED IDs -- SAVE THEM
    while i_rir <= rir_stop:
        if not os.path.isfile(os.path.join(root_out, 'log', 'infos', '') + str(i_rir) + '.npy'):
            signal_setup = SignalSetup(speakers_lists, duration_range, var_tar, snr_dry_range,
                                       snr_cnv_range, min_delta_snr)
            # Create a room configuration with sources and microphones placed in it
            room_setup.create_room_setup()
            # Convolve the target signal i_file in the room and mix it with SSN noise
            i_file = np.random.randint(len(signal_setup.speakers_list))
            function_output = simulate_room(room_setup,
                                            signal_setup,
                                            i_file,
                                            dset,
                                            n_sensors_per_node,
                                            past_sirs,
                                            n_rirs_per_process)
            if function_output == "redraw_source_signal":
                i_file = np.random.randint(len(signal_setup.speakers_list))
                continue
            elif function_output == "redraw_room_setup":
                continue
            # This is reached only if simulate_room was successful
            room, images, image_target_vad, sirs = function_output
            for k in range(n_sources):
                past_sirs[k].append(sirs[k])

            infos = {'rirs': room.rir,
                     'sources_files': signal_setup.speakers_files,
                     'mics': room_setup.microphones_positions,
                     'room': {'length': room_setup.length, 'width': room_setup.width, 'height': room_setup.height,
                              'alpha': room_setup.alpha, 'table_radius': room_setup.table_radius},
                     'sources': room_setup.source_positions,
                     'sirs': sirs}

            save_data(room.sources, images, infos, i_rir, root_out)
        else:
            print(str(i_rir) + " already done")
        i_rir += 1

