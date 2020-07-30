import argparse
from suma_classes import PostProcessor

path_dataset = '/home/nfurnon/dataset/suma/'
snr_range = [0, 6]
SAVE_TAR = True


def mix_sigs(rir_start, rir_nb, scene, noise_type):
    pp = PostProcessor(rir_start, rir_nb, scene, noise_type, snr_range, path_dataset, save_target=SAVE_TAR)
    pp.post_process()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Parameters to parallelize the post processing of the database generation "
                                     "(mixing signals)")
    parser.add_argument("--rirs", "-r",
                        help="First RIR id and RIR number: two arguments expected",
                        nargs=2,
                        type=int,
                        default=[1, 1])
    parser.add_argument("--scene", "-s",
                        help="Spatial configuration",
                        type=str,
                        choices=['random', 'living', 'meeting'],
                        default="random")
    parser.add_argument("--noise", "-n",
                        help="Type of noise",
                        type=str,
                        choices=['ssn', 'fs', 'it'],
                        default='ssn')

    args = parser.parse_args()
    rir_start, nb_rir = args.rirs
    scene = args.scene
    noise = args.noise
    mix_sigs(int(rir_start), int(nb_rir), scene, noise)
