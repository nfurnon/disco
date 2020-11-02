import os
import glob
import argparse
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
from disco_theque.math_utils import db2lin
from disco_theque.sigproc_utils import vad_oracle_batch, fw_snr, increase_to_snr
from disco_theque.dataset_utils.room_setups import RandomRoomSetup, MeetingRoomSetup, LivingRoomSetup
from disco_theque.dataset_utils.signal_setups import SpeechAndNoiseSetup


# %%
def create_directories(output_path, scene, case):
    """
    Create the directories where all the files will be saved
    Args:
        str: root directory where the (sub-)folders will be created
        str: living/meeting/random   Name of the first subdirectory relative to the scenario
        str: test/train/val
    """
    os.makedirs(os.path.join(output_path, scene, case, 'log', 'infos'), exist_ok=True)
    os.makedirs(os.path.join(output_path, scene, case, 'wav_original'), exist_ok=True)
    os.makedirs(os.path.join(output_path, scene, case, 'wav_original', 'dry'), exist_ok=True)
    os.makedirs(os.path.join(output_path, scene, case, 'wav_original', 'dry', 'target'), exist_ok=True)
    os.makedirs(os.path.join(output_path, scene, case, 'wav_original', 'dry', 'noise'), exist_ok=True)
    os.makedirs(os.path.join(output_path, scene, case, 'wav_original', 'cnv'), exist_ok=True)
    os.makedirs(os.path.join(output_path, scene, case, 'wav_original', 'cnv', 'target'), exist_ok=True)
    os.makedirs(os.path.join(output_path, scene, case, 'wav_original', 'cnv', 'noise'), exist_ok=True)


def get_wavs_list(path_to_freesound, path_to_librispeech, case, scene):
    """
    Return lists of wavs for the target and noise signals, as well as a list of speakers which one can pick from to
    compute a SSN noise.

    .. note:: If this script is to be run in parallel processes, one has to make sure that the same target file will not
    be selected by two processes. That's why the random seed is set at the header of the script, and the shuffling here
    shuffles the list in the exact same way for all processes. However, the noises are randomly selected, so this seed
    should be later changed, which is done in the `__main__` function.

    Args:
        str: path of freesound folder (before train/test/val)
        str: path of LibriSpeech folder (before train-clean-360)
        str: train/test/val
        str: living/random/meeting

    Returns:
        list:   list of the LibriSpeech target speech files
        list:   list of the LibriSpeech interferent speech files
        dict:   dictionnary of {noise_type: noise_list}
    """
    #TODO: remove this if
    if os.path.isfile('tmp/lists/talker_list.npy'):
        talkers_list = np.load('tmp/lists/talker_list.npy')
        target_list = np.load('tmp/lists/target_list.npy')
    else:
        if case == 'test':
            target_list = glob.glob(os.path.join(path_to_librispeech, 'test-clean', '') + '*/*/*.flac')
            talkers_list = glob.glob(os.path.join(path_to_librispeech, 'train-clean-360', '') + '[89]**/*/*.flac')
        elif case == 'train':
            target_list = glob.glob(os.path.join(path_to_librispeech, 'train-clean-100', '') + '*/*/*.flac')
            talkers_list = glob.glob(os.path.join(path_to_librispeech, 'train-clean-360', '') + '[1234567]**/*/*.flac')
        else:
            raise ValueError('`case` should be "test" or "train"')
        target_list.sort()
        np.random.shuffle(target_list)       # Force to shuffle identically for all parallel processes (so set numpy.seed)
        talkers_list.sort()
        np.random.shuffle(talkers_list)

        np.save('tmp/lists/target_list', target_list)
        np.save('tmp/lists/talker_list', talkers_list)
    fs_list = glob.glob(os.path.join(path_to_freesound, case) + '/*/*.wav')
    fs_list.sort()
    if scene == 'living':   # One noise source; one extra noise type (additionnally to SSN)

        noises_dict = {'freesound': fs_list}
    else:      # Interferent talker is desired
        noises_dict = {'interferent_talker': talkers_list, 'freesound': fs_list}

    return target_list, talkers_list, noises_dict


def mix_signals(room):
    """ Simulate the room that has been instantiated. Return mixed signals at microphones, image signals (unmixed ones)
    and the RIR
    Args:
        pra instance:   pyroomacoustic's instance of room with mics and sources positions and sources images.
    Returns:
        np.ndarray: Image signals
        np.ndarray: Mixed signals
        np.ndarray: Room impulse response from every source to every microphone
    """
    room.image_source_model(use_libroom=True)
    room.compute_rir()
    clean_reverbed_signals = room.simulate(return_premix=True)
    mixed_reverbed_signals = room.mic_array.signals

    return clean_reverbed_signals, mixed_reverbed_signals, room.rir


def get_convolved_vads(x):
    """
    Compute VAD on image signals of the target.
    Args:
        np.ndarray: Array of image signals we want to compute the VAD of. Time dimension is the last (2nd) one.

    Returns:
        np.ndarray: Array of VADs. One per input row.
    """
    vads = np.zeros(x.shape)
    for i_m in range(vads.shape[0]):
        vads[i_m, :] = vad_oracle_batch(x[i_m, :], thr=0.001)

    return vads


def reverb_other_noises(room, signal_setup, dset='train'):
    """
    Reverb other types of noises with the RIR already created of room. This enables to have for one source position
    several signals.
    We assume that the first source is the target source and that the second is the noise source.
    Args:
        pra instance:   Instance of pra.room with RIRs and source signals
        signal_setup:   Instance of SignalSetup with noise types and the function to get them

    Returns:
        np.ndarray: dry noises (n_noise_sources x noise_types x time)
        np.ndarray: reverberated noises (n_noise_sources x noise_types x n_mic x time)
        list:       list of noise files that were picked (n_noise_sources x noise_types)
        list:       list of the index corresponding to the starting sample in the file that was picked
    """
    h = room.rir
    n_rir = np.shape(h)[0]
    n_noi = len(signal_setup.noises_dict.keys())

    target_duration = len(room.sources[0].signal) / room.fs
    if dset in ['train', 'val']:
        len_max = int((signal_setup.duration_range[-1] + 1) * room.fs)
    else:
        len_max = len(room.image_vad[0])

    source_noises = np.zeros((n_noi, len(room.sources[0].signal)))
    reverbed_noises = np.zeros((n_noi, n_rir, len_max))
    noise_files, noise_starts = [], np.zeros(n_noi)

    # Convolve all noises with all RIRs
    for i_noi, noise_name in enumerate(signal_setup.noises_dict.keys()):
        # Select the noise
        noise_segment_ok = False
        while not noise_segment_ok:     # Some noises are so narrow-band that SNR is biased
            n, n_file, n_file_start, vad_noise, fs = signal_setup.get_signal(noise_name, target_duration)
            n = increase_to_snr(room.sources[0].signal, n, signal_setup.source_snr[0],
                                weight=True, vad_tar=room.source_vad, vad_noi=vad_noise, fs=fs)
            snr_check = fw_snr(room.sources[0].signal, n, fs,
                               vad_tar=room.source_vad, vad_noi=vad_noise, clipping=True)[1]
            noise_segment_ok = (abs(snr_check - signal_setup.source_snr[0]) < 1)    # Check deltaSNR < 1 dB
        source_noises[i_noi, :len(n)] = n
        # Convolve the noise signal
        for i_rir in range(n_rir):
            n_reverbed = np.convolve(n, room.rir[i_rir][1])
            n_reverbed = n_reverbed[:len_max]
            reverbed_noises[i_noi, i_rir, :len(n_reverbed)] = n_reverbed
        noise_files.append(n_file)
        noise_starts[i_noi] = n_file_start

    return source_noises, reverbed_noises, noise_files, noise_starts


def snr_at_mics(s, n, mics_per_node, fs=16000, vad_s=None, vad_n=None):
    """
    Compute the SNR at each microphone and return them. Also return the minimal difference and the maximal difference
    between all nodes.
    SNR at one node is computed as the mean of the SNRs at the microphones of the node.
    Args:
        np.ndarray: Array of target image signals in shape n_mics x time
        np.ndarray: Array of noise image signals in shape n_mics x time
        int:        Sampling frequency of signals s and n.
        np.ndarray: Number of mics per node (1D)
        np.ndarray: VAD of signal s. If mentioned, it will be applied on s before computing its variance
        np.ndarray: VAD of signal n. If mentioned, it will be applied on n before computing its variance

    Returns:
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
    # SNRs at all nodes (mean over the mics of the node)
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
    Simulate the RIRs of a room with two sources (one target source and one noise source).
    Several noise types (SSN, freesound, interferent talker) can be played by the noise source. In this function, we
    only simulate the SSN noise, and reverberate the other noises in `reverb_other_noises`.
    Args:
        room_setup (pra instance): Pyroomacoustics instance of the room to simulate.
        signal_setup: Class instance of SignalSetup
        noise_types (list): List of the noise types (one per noise source)
        i_target_file (int): Index of the target file to load
        dset (str): train/val/test

    Returns:
        Either  - "redraw_source_signal": no adequate target signal could be found
                - "redraw_room_setup":  no adequate room configuration could be found
                - room instance, reverberated signals, target VADs, SNRs
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
    noise_source_signal, _, _, noise_vad, _ = signal_setup.get_signal(n_type="SSN",
                                                                      duration=signal_setup.target_duration)
    noise_source_signal = increase_to_snr(target_source_signal, noise_source_signal,
                                          signal_setup.source_snr[0],
                                          weight=True, vad_tar=target_source_vad, vad_noi=noise_vad, fs=fs)
    room.add_source(room_setup.source_positions[1], signal=noise_source_signal)

    # Reverberate signals, mix them
    image_signals, noisy_reverbed_signals, rirs = mix_signals(room)
    # VAD of reverbed signal; better when computed on reverberated signal rather than shifting the source VAD
    target_image_vad = get_convolved_vads(image_signals[0])
    room.image_vad = target_image_vad

    snr_images, snr_nodes, snr_diff = snr_at_mics(image_signals[0], image_signals[1], n_sensors_per_node, fs,
                                                  vad_s=target_image_vad)

    snrs_are_valid = (np.all(signal_setup.snr_cnv_range[0] < snr_nodes)
                      and np.all(snr_nodes < signal_setup.snr_cnv_range[1])
                      and signal_setup.min_delta_snr < snr_diff)
    if snrs_are_valid:
        if dset == 'train':     # Pad train signals to make them fit in a common np.array (easier to handle)
            len_max = int((signal_setup.duration_range[-1] + 1) * fs)
            len_to_pad = np.maximum(len_max - image_signals.shape[-1], 0)
            image_signals = np.pad(image_signals, ((0, 0), (0, 0), (0, len_to_pad)), 'constant', constant_values=0)
            image_signals = image_signals[:, :, :len_max]    # truncate if necessary
        return room, image_signals, target_image_vad, snr_images
    else:
        return "redraw_room_setup"


def save_data(sources, images, noises, infos, id, path, fs=16000):
    """
    We only save the signals we might use. So in meeting scenario, SSN is saved for all
    sources but interferent talker only for second source and freesound only for third source.
    Args:
        np.ndarray: Source images of target and noise (noise is only SSN)
        np.ndarray: Image signals of target and noise (noise is SSN)
        np.ndarray: Concatenation of source and image noises that are not SSN
        dict:       Dictionary of informations to save (room and signal setups)
        str:        Root path where data is saved
        int:        id of RIR
        int:        Sampling frequency  [16000]
    """
    path_wav = os.path.join(path, 'wav_original', '')
    path_log = os.path.join(path, 'log', 'infos', '')

    # Save source signals
    dirs = ['target', 'noise']
    sig_names = ['', '_ssn', '_it', '_fs']
    # Save source target and SSN (2 sources)
    for i_s in range(len(sources)):
        path_source = os.path.join(path_wav, 'dry', dirs[i_s], "")
        sf.write(path_source + str(id) + '_S-' + str(i_s + 1) + sig_names[i_s] + '.wav', sources[i_s].signal, fs)
    # Save the other noises source signals (one noise source, two extra noises: interferent speaker and freesound)
    for i_sn in range(len(noises[0])):
        sf.write(path_source + str(id) + '_S-2' + sig_names[i_sn + len(sources)] + '.wav',
                 noises[0][i_sn, :], fs)
    # Save the images of target and SSN
    for i_s in range(len(images)):
        path_image = os.path.join(path_wav, 'cnv', dirs[i_s], "")
        for i_ch in range(len(images[i_s])):
            sf.write(path_image + str(id) + '_S-' + str(i_s + 1) + sig_names[i_s] + '_Ch-' + str(i_ch + 1) + '.wav',
                     images[i_s][i_ch], fs)
    # Save the other noise images
    for i_s in range(len(noises[1])):
        for i_ch in range(noises[1][i_s].shape[0]):
            sf.write(path_image + str(id) + '_S-2' + sig_names[i_s + len(sources)] +
                     '_Ch-' + str(i_ch + 1) + '.wav',
                     noises[1][i_s, i_ch, :], fs)

    # Save infos
    np.save(path_log + str(id), infos, allow_pickle=True)


if __name__ == "__main__":
    np.random.seed(30)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset',
                        help='train/test dataset',
                        choices=['train', 'test'],
                        type=str,
                        default='test')
    parser.add_argument('--scenario',
                        help='meeting/living/random scenario',
                        choices=['random', 'living', 'meeting'],
                        type=str,
                        default='random')
    parser.add_argument('--rirs', '-r',
                        help='ID of the first RIR to create and number of RIRs to create.',
                        nargs=2,
                        type=int,
                        default=[1, 1])
    parser.add_argument('--dir_out', '-d',
                        help='directory where data will be saved',
                        type=str,
                        default='../../dataset/disco/')
    args = parser.parse_args()
    dset = args.dset
    scenario = args.scenario
    i_rir, n_rir = args.rir_id
    root_dir = args.dir_out

    path_to_freesound = '../../dataset/freesound/data/'
    path_to_librispeech = '../../dataset/LibriSpeech/'

    # %% DEFAULT ROOM PARAMETERS
    l_range, w_range, h_range, beta_range = (3, 8), (3, 5), (2.5, 3), (0.3, 0.6)
    n_sensors_per_node = [4, 4, 4, 4]  # Should be constant in all nodes, because of pra.add_microphone_array(micArray)
    d_mw, d_mn, d_nn, d_rnd_mics = 0.5, 0.05, 0.5, 1
    n_sources = 2
    d_ss, d_sn, d_sw = 0.5, 0.5, 0.5
    z_range_m = (0.7, 2)
    z_range_s = (1.20, 2)

    r_range, d_nt_range, d_st_range, phi_ss_range = (0.5, 1), (0.05, 0.20), (0, 0.50), (np.pi / 8, 15 * np.pi / 8)
    # %% JOB PROPERTIES
    rir_stop = int(i_rir + n_rir - 1)
    i_file = (i_rir - 1) * 2  # Pick different talker for every RIR -- leave margin for cases +=1

    create_directories(root_dir, scenario, dset)       # Create dirs and return root to wav/

    # Instantiate the room and signal classes
    if scenario == 'meeting':
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
        z_range_m = [0.7, 0.95]     # coffe table - dresser
        z_range_s = [1.20, 2]    # Sitting person - tall standing person
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
    noise_types = ['SSN']
    snr_dry_range = np.array([[0, 0]])
    snr_cnv_range = (-10, 15)
    min_delta_snr = 0

    target_list, talkers_list, noise_dict = get_wavs_list(path_to_freesound, path_to_librispeech, dset, scenario)
    np.random.seed(i_file)  # Don't pick the same noise for different RIR. Do it after speech lists are equally shuffled
    signal_setup = SpeechAndNoiseSetup(target_list, talkers_list, noise_dict, duration_range, var_tar, snr_dry_range,
                                       snr_cnv_range, min_delta_snr)

    # %% CREATE SIGNALS FOR ALL REQUIRED IDs -- SAVE THEM
    save_path = os.path.join(root_dir, scenario, dset)
    while i_rir <= rir_stop:
        print('Simulate room {}'.format(str(i_rir)))
        # Create a room configuration with sources and microphones placed in it
        room_setup.create_room_setup()
        # Convolve the target signal i_file in the room and mix it with SSN noise
        function_output = simulate_room(room_setup, signal_setup, noise_types, i_file, dset)
        if function_output == "redraw_source_signal":
            i_file += 1
            print('\t Redraw a target source signal')
            continue
        elif function_output == "redraw_room_setup":
            print('\t Redraw a room configuration')
            continue
        # This is reached only if simulate_room was successful
        room, images, image_target_vad, snr_images = function_output
        # Simulate the same room but with the other noise types at each noise source
        noise_sources, noise_images, noise_files, noise_starts = reverb_other_noises(room, signal_setup, dset)
        noises = [noise_sources, noise_images]

        # Save the data
        infos = {'rirs': room.rir,
                 'target': signal_setup.target_list[i_file].split('/')[-1].strip('.flac'),
                 'noise': {'ssn': None,
                           'interferent_talker': {'file': noise_files[0], 'start': noise_starts[0]},
                           'freesound': {'file': noise_files[1], 'start': noise_starts[1]}},
                 'mics': room_setup.microphones_positions,
                 'room': {'length': room_setup.length, 'width': room_setup.width, 'height': room_setup.height,
                          'alpha': room_setup.alpha},
                 'sources': room_setup.source_positions}

        save_data(room.sources, images, noises, infos, i_rir, save_path)
        i_file += 1
        i_rir += 1
