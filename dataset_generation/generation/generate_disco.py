import os
import sys
import glob
import numpy as np
np.random.seed(20)
from code_utils.math_utils import db2lin
import pyroomacoustics as pra
from code_utils.db_utils import increase_to_snr
from code_utils.metrics import fw_snr
from code_utils.sigproc_utils import vad_oracle_batch
from database_classes import SignalSetup, RandomRoomSetup, MeetingRoomSetup, LivingRoomSetup
import ipdb


# %%
def create_directories(scene, case):
    """
    Create the directories where all the files will be saved
    :param scene:
    :param case:
    :return:
    """
    os.makedirs('../../../../dataset/disco/' + scene + '/' + case + '/LOG/', exist_ok=True)
    os.makedirs('../../../../dataset/disco/' + scene + '/' + case + '/WAV/', exist_ok=True)
    os.makedirs('../../../../dataset/disco/' + scene + '/' + case + '/WAV/dry', exist_ok=True)
    os.makedirs('../../../../dataset/disco/' + scene + '/' + case + '/WAV/dry/target/', exist_ok=True)
    os.makedirs('../../../../dataset/disco/' + scene + '/' + case + '/WAV/dry/noise/', exist_ok=True)
    os.makedirs('../../../../dataset/disco/' + scene + '/' + case + '/WAV/cnv', exist_ok=True)
    os.makedirs('../../../../dataset/disco/' + scene + '/' + case + '/WAV/cnv/target/', exist_ok=True)
    os.makedirs('../../../../dataset/disco/' + scene + '/' + case + '/WAV/cnv/noise/', exist_ok=True)

    return os.path.join('../../../../dataset/disco/', scene, case, '')


def get_wavs_list(case, scene):
    """
    Return lists of wavs for the target and noise signals, as well as a list of speakers which one can pick from to
    compute a SSN noise.
    :param case:
    :return:
    """
    fs_list = glob.glob('../../../../dataset/freesound/data/' + case + '/*/*.wav')
    if case == 'test':
        target_list = glob.glob('../../../../corpus/LibriSpeech/audio/test-clean/*/*/*.flac')
        talkers_list = glob.glob('../../../../corpus/LibriSpeech/audio/train-clean-360/[89]**/*/*.flac')
    elif case == 'train':
        target_list = glob.glob('../../../../corpus/LibriSpeech/audio/train-clean-100/*/*/*.flac')
        talkers_list = glob.glob('../../../../corpus/LibriSpeech/audio/train-clean-360/[1234567]**/*/*.flac')
    else:
        raise ValueError('`case` should be "test" or "train"')
    target_list.sort()
    np.random.shuffle(target_list)       # Force to shuffle identically for all parallel processes (so set numpy.seed)
    talkers_list.sort()
    np.random.shuffle(talkers_list)
    fs_list.sort()
    if scene == 'meeting':      # Interferent talker is desired
        noises_dict = {'interferent_talker': talkers_list, 'freesound': fs_list}
    else:
        noises_dict = {'freesound': fs_list}

    return target_list, talkers_list, noises_dict


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
    mixed_reverbed_signals = room.mic_array.signals

    return clean_reverbed_signals, mixed_reverbed_signals, room.rir


def get_image_statistics(x):
    """
    Pre-generation function. Compute variance of signals x and their extrema
    :param x:
    :return:
    """
    x_var = np.var(x, axis=-1)
    x_max = np.max(x, axis=-1)

    return x_var, x_max


def _delay_vads(x, h):
    """
    Delay a signal x by the peak index of the transfer function h
    :param x:
    :param h:       RIRs (n_mic x n_sources x time)
    :return:
    """
    n_mics = np.shape(h)[0]
    n_samp = np.maximum(np.max([len(h_) for h_ in np.array(h)[:, 0]]), np.max([len(h_) for h_ in np.array(h)[:, 1]]))
    len_x_out = len(x) + n_samp - 1
    if len_x_out % 2 == 1:      # See pra.room.py line 1066
        len_x_out += 1
    xs_delayed = np.zeros((n_mics, len_x_out))
    for i_m in range(n_mics):
        peak_id = np.argmax(np.array(h)[:, 0][i_m])
        xs_delayed[i_m, :len(x) + len(np.array(h)[:, 0][i_m]) - 1] = np.pad(x,
                                                                            (peak_id, len(np.array(h)[:, 0][i_m])
                                                                                      - 1 - peak_id),
                                                                            'constant', constant_values=0)

    return xs_delayed


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


def get_target_vads(x_cnv, x_vad, h):
    """
    Compute VAD of image target signals with two methods. Output of each method are stacked along last axis.
    :param x_cnv:       Image signals (n_mic x time)
    :param x_vad:       VAD of source signal
    :param h:           RIRs (n_mics x time)
    :return:
    """
    target_image_vads = _delay_vads(x_vad, h)
    target_cnv_vads = _get_convolved_vads(x_cnv)
    target_vads = np.stack((target_image_vads, target_cnv_vads), axis=2)

    return target_vads


def reverb_other_noises(room, signal_setup, dset='train'):
    """
    Reverb other types of noises with the RIR already created of room. This enables to have for one source position
    several signals.
    We assume that the first source is the target source and that all other sources are noise sources.
    :param room:            Instance of pra.room with RIRs and source signals
    :param signal_setup:    Instance of SignalSetup with noise types and the function to get them
    :return:
    """
    h = room.rir
    n_rir = np.shape(h)[0]
    n_noise_sources = np.shape(h)[1] - 1    # First RIRs are target -> Do not count them
    n_noi = len(signal_setup.noises_dict.keys())

    target_duration = len(room.sources[0].signal) / room.fs
    if dset == 'train':
        len_max = int((signal_setup.duration_range[-1] + 1) * room.fs)
    else:
        len_max = len(room.image_vad[0])
    # Preallocate reverbed signals; See pra.room to get length of reverbed signals
    # n_samp = np.max([np.max([len(h_) for h_ in np.array(h)[:, j]]) for j in range(n_noise_sources + 1)])
    # len_x_out = len(room.sources[0].signal) + n_samp - 1
    # if len_x_out % 2 == 1:      # See pra.room.py line 1066
    #     len_x_out += 1

    source_noises = np.zeros((n_noise_sources, n_noi, len(room.sources[0].signal)))
    reverbed_noises = np.zeros((n_noise_sources, n_noi, n_rir, len_max))
    noise_files, noise_starts = [[] for _ in range(n_sources)], np.zeros((n_noise_sources, n_noi))

    # Convolve all noises with all RIRs
    for i_sou in range(n_noise_sources):
        for i_noi, noise_name in enumerate(signal_setup.noises_dict.keys()):
            noise_segment_ok = False
            while not noise_segment_ok:     # Some noises are so narrow-band that SNR is biased
                n, n_file, n_file_start, vad_noise, fs = signal_setup.get_noise_segment(noise_name, target_duration)
                n = increase_to_snr(room.sources[0].signal, n, signal_setup.source_snr[i_sou],
                                    weight=True, vad_tar=room.source_vad, vad_noi=vad_noise, fs=fs)
                snr_check = fw_snr(room.sources[0].signal, n, fs,
                                   vad_tar=room.source_vad, vad_noi=vad_noise, clipping=True)[1]
                noise_segment_ok = (abs(snr_check - signal_setup.source_snr[i_sou]) < 1)
            source_noises[i_sou, i_noi, :len(n)] = n
            for i_rir in range(n_rir):
                n_reverbed = np.convolve(n, room.rir[i_rir][i_sou + 1])
                n_reverbed = n_reverbed[:len_max]
                reverbed_noises[i_sou, i_noi, i_rir, :len(n_reverbed)] = n_reverbed
            noise_files[i_sou].append(n_file)
            noise_starts[i_sou, i_noi] = n_file_start

    return source_noises, reverbed_noises, noise_files, noise_starts


def snr_at_mics(s, n, mics_per_node, fs=16000, vad_s=None, vad_n=None):
    """
    Compute the SNR at each microphone and return them. Also return the minimal difference and the maximal difference
    between all nodes.
    SNR at one node is computed as the mean of the SNRs at the microphones of the node.
    :param s:               Array of target image signals in shape n_mics x time
    :param n:               Array of noise image signals in shape n_mics x time
    :param fs:              Sampling frequency of signals s and n.
    :param mics_per_node:   Number of mics per node
    :param vad_s            VAD of signal s. If mentioned, it will be applied on s before computing its variance
    :param vad_n            VAD of signal n. If mentioned, it will be applied on n before computing its variance

    :return:
        - snrs:             SNRs at all mics
        - node_snrs:        SNR at all nodes (as mean of SNRs of mics of the node)
        - delta_snr_min:    Minimal difference between all nodes
    """
    n_mic = s.shape[0]
    total_mics = np.hstack((np.array([0]), np.cumsum(mics_per_node)))
    n_nodes = len(mics_per_node)
    n_pairs = int(n_nodes * (n_nodes - 1) / 2)      # Number of possible pairs of nodes: sum(k) or (k among N) with k=2
    snrs = np.zeros(n_mic)
    nodes_snr = np.zeros(n_nodes)
    delta_snrs = np.zeros(n_pairs)

    if vad_s is None:
        vad_s = np.tile(None, n_mic)
    if vad_n is None:
        vad_n = np.tile(None, n_mic)

    # SNRs at all microphones
    for i_mic in range(n_mic):
        snrs[i_mic] = fw_snr(s[i_mic], n[i_mic], fs=fs, vad_tar=vad_s[i_mic], vad_noi=vad_n[i_mic])[1]
    # SNRs at all nodes
    for i_nod in range(n_nodes):
        nodes_snr[i_nod] = np.mean(snrs[total_mics[i_nod]:total_mics[i_nod + 1]])
    # SNR differences between all nodes
    i_pair = 0
    for i in range(n_nodes - 1):
        delta_snrs[i_pair:i_pair + n_nodes - i - 1] = nodes_snr[i] - nodes_snr[i + 1]
        i_pair += n_nodes - i - 1       # Too lazy to compute this as function of i

    return snrs, nodes_snr, np.min(abs(delta_snrs))


def simulate_room(room_setup, signal_setup, noise_types, i_target_file, dset):
    """
    Simulate the RIRs of a room with two or more noise sources.
    We assume that there is only one target source and that all others are noise.
    :param room_setup:
    :param signal_setup:
    :param noise_types:     Noise types; one per noise source
    :param i_target_file:
    :return:
    """
    if len(noise_types) != room_setup.n_sources - 1 or len(noise_types) != len(signal_setup.source_snr):
        raise ValueError('The number of noise types should be equal to the number of noise sources')

    # Create signals
    target_file = signal_setup.target_list[i_target_file]
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
    # Second source (first noise source)
    for i_source, noise_type in enumerate(noise_types):
        noise_source_signal, _, _, noise_vad, _ = signal_setup.get_noise_segment(n_type=noise_type,
                                                                                 duration=signal_setup.target_duration)
        noise_source_signal = increase_to_snr(target_source_signal, noise_source_signal,
                                              signal_setup.source_snr[i_source],
                                              weight=True, vad_tar=target_source_vad, vad_noi=noise_vad, fs=fs)
        room.add_source(room_setup.source_positions[i_source + 1], signal=noise_source_signal)

    # Reverb signals, mix them
    image_signals, noisy_reverbed_signals, rirs = mix_signals(room)
    # VAD of reverbed signal; better when computed on reverbed signal rather than shifting the source VAD
    target_image_vad = _get_convolved_vads(image_signals[0])
    room.image_vad = target_image_vad

    snr_images, snr_nodes, snr_diff = snr_at_mics(image_signals[0], image_signals[1], n_sensors_per_node, fs,
                                                  vad_s=target_image_vad)

    snrs_are_valid = (np.all(signal_setup.snr_cnv_range[0] < snr_nodes)
                      and np.all(snr_nodes < signal_setup.snr_cnv_range[1])
                      and signal_setup.min_delta_snr < snr_diff)
    if snrs_are_valid:
        if dset == 'train':
            len_max = int((signal_setup.duration_range[-1] + 1) * fs)
            len_to_pad = np.maximum(len_max - image_signals.shape[-1], 0)
            image_signals = np.pad(image_signals, ((0, 0), (0, 0), (0, len_to_pad)), 'constant', constant_values=0)
            image_signals = image_signals[:, :, :len_max]    # truncate if necessary
        return room, image_signals, target_image_vad, snr_images
    else:
        return "redraw_room_setup"


def save_data(sources, images, noises, infos, id, path, fs=16000):
    """
    Ca devient un peu cochon ici. We only save the signals we might use. So in meeting scenario, SSN is saved for all
    sources but interferent talker only for second source and freesound only for third source.
    :param sources:     Source images of target and noise (noise is only SSN)
    :param images:      Image signals of target and noise (noise is SSN)
    :param noises:      Concatenation of source and image noises that are not SSN
    :param infos:
    :param path:
    :param id:
    :param fs:
    :return:
    """
    import soundfile as sf

    path_wav = os.path.join(path, 'WAV', '')
    path_log = os.path.join(path, 'LOG', '')

    # Save source signals
    if 'meeting/' in path:
        dirs = ['target', 'noise', 'noise']
        sig_names = ['', '_ssn', '_ssn', '_it', '_fs']
    else:
        dirs = ['target', 'noise']
        sig_names = ['', '_ssn', '_fs']
    # Save source target and SSN (3 sources)
    for i_s in range(len(sources)):
        path_source = os.path.join(path_wav, 'dry', dirs[i_s], "")
        sf.write(path_source + str(id) + '_S-' + str(i_s + 1) + sig_names[i_s] + '.wav', sources[i_s].signal, fs)
    # Save the other noises source signals
    for i_sn in range(len(noises[0])):
        sf.write(path_source + str(id) + '_S-' + str(i_sn + 2) + sig_names[i_sn + len(sources)] + '.wav',
                 noises[0][i_sn][i_sn], fs)
    # Save the images of target and SSN
    for i_s in range(len(images)):
        path_image = os.path.join(path_wav, 'cnv', dirs[i_s], "")
        for i_ch in range(len(images[i_s])):
            sf.write(path_image + str(id) + '_S-' + str(i_s + 1) + sig_names[i_s] + '_Ch-' + str(i_ch + 1) + '.wav',
                     images[i_s][i_ch], fs)
    # Save the other noise images
    for i_s in range(len(noises[1])):
        for i_ch in range(len(noises[1][i_s][i_s])):
            sf.write(path_image + str(id) + '_S-' + str(i_s + 2) + sig_names[i_s + len(sources)] +
                     '_Ch-' + str(i_ch + 1) + '.wav',
                     noises[1][i_s][i_s][i_ch], fs)

    # Save infos
    np.save(path_log + str(id), infos, allow_pickle=True)


if __name__ == "__main__":
    # %% DEFAULT ROOM PARAMETERS
    l_range, w_range, h_range, beta_range = [3, 8], [3, 5], [2.5, 3], [0.3, 0.6]
    n_sensors_per_node = [4, 4, 4, 4]  # Should be constant in all nodes, because of pra.add_microphone_array(micArray)
    d_mw, d_mn, d_nn, d_rnd_mics = 0.5, 0.05, 0.5, 1
    n_sources = 3
    d_ss, d_sn, d_sw = 0.5, 0.5, 0.15
    z_range_m = [0.7, 2]
    z_range_s = [1.20, 2]

    r_range, d_nt_range, d_st_range, phi_ss_range = (0.5, 1), (0.05, 0.20), (0, 0.50), (np.pi / 8, 15 * np.pi / 8)
    # %% JOB PROPERTIES
    dset = sys.argv[1]
    if dset == 'train':
        rir_start = 1
    else:
        rir_start = 11001
    scenario = sys.argv[2]
    i_rir = int(sys.argv[3])
    n_rirs_per_process = int(sys.argv[4])
    rir_stop = int(i_rir + n_rirs_per_process - 1)
    i_file = (i_rir - rir_start) * 2     # Pick different talker for every RIR -- leave margin for cases +=1

    root_out = create_directories(scenario, dset)       # Create dirs and return root to WAV/

    # Instantiate the room and signal classes
    if scenario == 'meeting':
        d_mw = 0.5                      # Minimal distance of all mics to the closest wall
        n_sources = 3
        z_range_m = [0.7, 0.8]          # Table
        z_range_s = [1.15, 1.30]        # Sitting people
        room_setup = MeetingRoomSetup(l_range=l_range, w_range=w_range, h_range=h_range, beta_range=beta_range,
                                      n_sensors_per_node=n_sensors_per_node,
                                      d_mw=d_mw, d_mn=d_mn, d_nn=d_nn, z_range_m=z_range_m,
                                      d_rnd_mics=d_rnd_mics,
                                      n_sources=n_sources, d_ss=d_ss, d_sn=d_sn, d_sw=d_sw, z_range_s=z_range_s,
                                      r_range=r_range,
                                      d_nt_range=d_nt_range, d_st_range=d_st_range,
                                      phi_ss_range=phi_ss_range)
    elif scenario == 'living':
        d_mw = 0.5                  # Maximal distance of mics to closest wall (mics are close to wall in LivingRoom)
        n_sources = 2
        z_range_m = [0.7, 0.95]     # Table basse - buffet
        z_range_s = [1.20, 1.90]    # Personne assise - grande personne debout
        room_setup = LivingRoomSetup(l_range=l_range, w_range=w_range, h_range=h_range, beta_range=beta_range,
                                     n_sensors_per_node=n_sensors_per_node,
                                     d_mw=d_mw, d_mn=d_mn, d_nn=d_nn, z_range_m=z_range_m,
                                     d_rnd_mics=d_rnd_mics,
                                     n_sources=n_sources, d_ss=d_ss, d_sn=d_sn, d_sw=d_sw, z_range_s=z_range_s)
    else:
        room_setup = RandomRoomSetup(l_range=l_range, w_range=w_range, h_range=h_range, beta_range=beta_range,
                                     n_sensors_per_node=n_sensors_per_node,
                                     d_mw=d_mw, d_mn=d_mn, d_nn=d_nn, z_range_m=z_range_m,
                                     d_rnd_mics=d_rnd_mics,
                                     n_sources=n_sources, d_ss=d_ss, d_sn=d_sn, d_sw=d_sw, z_range_s=z_range_s)

    # %% SIGNAL PARAMETERS
    duration_range = (5, 10)
    var_tar = db2lin(-23)
    if scenario == 'meeting':                           # Two noise sources in meeting
        snr_dry_range = np.array([[-3, 6],                   # First noise source is interferent talker
                                 [0, 10]])                   # Second noise source in far field noise
        noise_types = ['SSN', 'SSN']
    elif scenario == 'living':                          # Single noise source
        snr_dry_range = np.array([-6, 6])[np.newaxis, :]    # For compatibility in SignalSetup.get_random_source_snr()
        noise_types = ['SSN']
    else:
        snr_dry_range = np.tile((-6, 6), (n_sources - 1, 1))
        noise_types = np.tile('SSN', n_sources - 1)
    snr_cnv_range = (-10, 15)
    min_delta_snr = 0

    target_list, talkers_list, noise_dict = get_wavs_list(dset, scenario)
    signal_setup = SignalSetup(target_list, talkers_list, noise_dict, duration_range, var_tar, snr_dry_range,
                               snr_cnv_range, min_delta_snr)

    # %% CREATE SIGNALS FOR ALL REQUIRED IDs -- SAVE THEM
    while i_rir <= rir_stop:
        if not os.path.isfile(os.path.join(root_out, 'LOG', '') + str(i_rir) + '.npy'):
            # Create a room configuration with sources and microphones placed in it
            room_setup.create_room_setup()
            # Convolve the target signal i_file in the room and mix it with SSN noise
            function_output = simulate_room(room_setup, signal_setup, noise_types, i_file, dset)
            if function_output == "redraw_source_signal":
                i_file += 1
                continue
            elif function_output == "redraw_room_setup":
                continue
            # This is reached only if simulate_room was successful
            room, images, image_target_vad, snr_images = function_output
            # Simulate the same room but with the other noise types at each noise source
            noise_sources, noise_images, noise_files, noise_starts = reverb_other_noises(room, signal_setup, dset)
            noises = [noise_sources, noise_images]

            if scenario == 'living':
                infos = {'rirs': room.rir,
                         'target': signal_setup.target_list[i_file].split('/')[-1].strip('.flac'),
                         'noise': {'freesound': {'file': noise_files[0][0], 'start': noise_starts[0][0]}},
                         'mics': room_setup.microphones_positions,
                         'room': {'length': room_setup.length, 'width': room_setup.width, 'height': room_setup.height,
                                  'alpha': room_setup.alpha},
                         'sources': room_setup.source_positions}
            else:
                infos = {'rirs': room.rir,
                         'target': signal_setup.target_list[i_file].split('/')[-1].strip('.flac'),
                         'noise': {'freesound': {'file': noise_files[1][1], 'start': noise_starts[1][1]},
                                   'interferent_talker': {'file': noise_files[0][0], 'start': noise_starts[0][0]}},
                         'mics': room_setup.microphones_positions,
                         'room': {'length': room_setup.length, 'width': room_setup.width, 'height': room_setup.height,
                                  'alpha': room_setup.alpha},
                         'sources': room_setup.source_positions}

            save_data(room.sources, images, noises, infos, i_rir, root_out)
            i_file += 1
            i_rir += 1

        else:
            i_rir += 1
