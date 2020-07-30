import argparse
from disco_theque.dataset_utils.post_generator import PostGenerator

path_dataset = 'tmp/'
snr_range = [0, 6]
SAVE_TAR = True


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
    pp = PostGenerator(rir_start, nb_rir, scene, noise, snr_range, path_dataset, save_target=SAVE_TAR)
    pp.post_process()
