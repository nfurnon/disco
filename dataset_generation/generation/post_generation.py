import os
import numpy as np
import sys
from suma_classes import PostProcessor


def mix_sigs(rir_start, rir_nb, scene, noise_type, snr_range):
    path_dataset = '/home/nfurnon/dataset/suma/'
    SAVE_TAR=True
    pp = PostProcessor(rir_start, rir_nb, scene, noise_type, snr_range, path_dataset, save_target=SAVE_TAR)
    pp.post_process()


def norm_mix(rir_start, rir_nb, scene, noise_type, snr_range, norm_type):
    path_dataset = '/home/nfurnon/dataset/disco/'
    pp = PostProcessor(rir_start, rir_nb, scene, noise_type, snr_range, path_dataset, norm_type=norm_type)
    pp.norm_db()


def get_max_amp(rir_start, rir_nb, scene, noise_type, snr_range):
    path_dataset = '/home/nfurnon/dataset/disco/'
    pp = PostProcessor(rir_start, rir_nb, scene, noise_type, snr_range, path_dataset)
    local_amp = pp.get_max_amp()

    return local_amp


def get_sir(rir_start, rir_nb, scene, noise_type, snr_range):
    """Return convolved SIRs at all nodes"""
    path_dataset = '/home/nfurnon/dataset/suma/'
    pp = PostProcessor(rir_start, rir_nb, scene, noise_type, snr_range, path_dataset)
    pp.get_sir()


if __name__ == '__main__':
    rir_start, nb_rir, scene, noise_type, norm_type = sys.argv[1:]
    snr_range = [0, 6]
    mix_sigs(int(rir_start), int(nb_rir), scene, noise_type, snr_range)
    # get_sir(int(rir_start), int(nb_rir), scene, noise_type, snr_range)
    # norm_mix(int(rir_start), int(nb_rir), scene, noise_type, snr_range, norm_type)
    # os.makedirs('local_amps', exist_ok=True)
    # la = get_max_amp(int(rir_start), int(nb_rir), scene, noise_type, snr_range)
    # np.save('local_amps/{}-{}_{}_{}'.format(str(rir_start), str(int(rir_start) + int(nb_rir) - 1), scene, noise_type),
    #         la)
