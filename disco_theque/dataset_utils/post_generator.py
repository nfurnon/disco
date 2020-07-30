import os
import glob
import numpy as np
import soundfile as sf
from disco_theque.sigproc_utils import tf_mask
from disco_theque.sigproc_utils import fw_snr
import librosa as lb


class PostGenerator:
    """
    Given the created dataset, post process it, which means:
     - mix target + noise at (random) SNR
     - create STFT
     - compute corresponding mask
     - Save the STFT of the mixture and the mask
    """
    def __init__(self, rir_start, nb_rir, scene, noise, snr_range, path_to_dataset,
                 n_fft=512, n_hop=256, mask_type='irm1', save_target=True, n_samples=None):
        """
        Args:
            rir_start (int): ID of the first RIR to process
            nb_rir (int): Number of RIRs to process
            scene (str): Spatial configuration: 'random'/'living'/'meeting'
            noise (str): Noise to mix
            snr_range (list, tuple): SNR range of the desired mixture
            path_to_dataset (str): root folder to dataset
            n_fft (int): FFT length of STFT [512]
            n_hop (int): FFT hop size of STFT [256]
            mask_type (str): Type of ideal TF-mask ('irmX', 'iamX', 'ibmX') [irm1]
            save_target (bool): Save the target signal ? (This signal is the same for all noises)
            n_samples (list, array): In this order, number of samples for the train, val, test datasets.
        """
        self.rir_start = rir_start
        self.nb_rir = nb_rir
        self.case = self.get_dset()
        self.save_target = save_target  # Useless to overwrite the same signal
        self.scene = scene
        n_noises = 1
        self.noise = noise
        self.snr_range = np.array(snr_range)
        self.snr_out = np.zeros((nb_rir, n_noises))
        self.path_dataset = path_to_dataset
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.mask_type = mask_type
        if n_samples is not None:
            n_train, n_val, n_test = n_samples
        else:
            n_train, n_val, n_test = 10000, 1000, 1000
        self.n_samples = np.cumsum([n_train, n_val, n_test])
        self.snr_dir = self.get_directory_name()
        # HARD-CODED PARAMETERS
        self.fs = 16000
        self.ch_per_node = [4, 4, 4, 4]
        self.n_ch = sum(self.ch_per_node)
        self.n_nodes = len(self.ch_per_node)

    def get_dset(self):
        """ Return train, val or test depending on RIR we are considering"""
        assert 0 < self.rir_start < self.n_samples[-1], "rir should be between 1 and {}".format(str(self.n_samples[-1]))
        assert self.rir_start + self.nb_rir < self.n_samples[np.where(self.rir_start < self.n_samples)[0][0]], \
                                                                          "First and last RIRs do not belong to" \
                                                                          "the same set."
        return ['train', 'val', 'test'][np.where(self.rir_start < self.n_samples)[0][0]]

    def get_directory_name(self):
        """From snr_range in self, return the name of the directory in form of e.g. 0-6"""
        return '{}-{}'.format(str(self.snr_range[0]), str(self.snr_range[1]))

    def post_process(self):
        """Mix signals at random SNR in self.snr_range; take STFT; compute mask; save data."""
        path_out = os.path.join(self.path_dataset, self.scene, self.case)
        os.makedirs(os.path.join(path_out, 'log', 'snrs', 'dry', self.snr_dir), exist_ok=True)
        for rir in range(self.rir_start, self.rir_start + self.nb_rir):
            if not os.path.isfile(os.path.join(path_out, 'log', 'snrs', 'dry', self.snr_dir, '') +
                                  '{}_{}.npy'.format(str(rir), self.noise)):
                target_list, noise_list = self.get_sig_lists(rir)
                tars, nois, mixs, snr_out = self.mix_sigs(target_list, noise_list)
                self.snr_out[rir - self.rir_start, :] = snr_out
                tars_stft, nois_stft, mixs_stft = self.array_stft(tars), self.array_stft(nois), self.array_stft(mixs)
                masks = self.get_mask(tars_stft, nois_stft)
                self.save_data(tars, nois, mixs, tars_stft, nois_stft, mixs_stft, masks, rir)
            else:
                print('{} already processed'.format(str(rir)))

    def get_sig_lists(self, rir):
        """Get the lists of the data to load (all channels of RIR id)"""
        tar_list = glob.glob(os.path.join(self.path_dataset, self.scene, self.case,
                                          'wav_original', 'cnv', 'target', '') + str(rir) + '_S-1_Ch-*.wav')
        tar_list.sort(key=lambda x: int(x.split('_Ch-')[-1].split('.wav')[0]))
        noi_list = glob.glob(os.path.join(self.path_dataset, self.scene, self.case,
                                          'wav_original', 'cnv', 'noise', '') +
                             str(rir) + '_S-2_' + self.noise + '_Ch-*.wav')
        noi_list.sort(key=lambda x: int(x.split('_Ch-')[-1].split('.wav')[0]))
        noi_list = [noi_list]

        return tar_list, noi_list

    def mix_sigs(self, tar_list, noi_list):
        """Load the signals in input lists, mix at random SNR in self.snr_range"""
        tar_segs, noi_segs, mix_segs = [], [], []
        # Pick random SNR once for all channels and for all noises
        snr = self.snr_range[0] + (self.snr_range[1] - self.snr_range[0]) * np.random.random()
        for ch in range(1, self.n_ch + 1):
            tar_seg, _ = sf.read(tar_list[ch - 1], dtype='float32')
            noi_seg = np.zeros(len(tar_seg))
            for i_noise in range(len(noi_list)):
                noi, _ = sf.read(noi_list[i_noise][ch - 1], dtype='float32')
                noi_seg[:len(noi)] += noi * 10 ** (- snr / 20)

            tar_segs.append(tar_seg)
            noi_segs.append(noi_seg)
            mix_segs.append(tar_seg + noi_seg)

        return np.array(tar_segs), np.array(noi_segs), np.array(mix_segs), snr

    def array_stft(self, array):
        """Compute STFT of each signal in input array"""
        out = []
        for i in range(array.shape[0]):
            out.append(lb.core.stft(array[i], n_fft=self.n_fft, hop_length=self.n_hop, center=True))

        return np.array(out)

    def get_mask(self, s, n):
        """Return one tf-mask per line in s and n"""
        ms = []
        for i in range(np.shape(s)[0]):
            ms.append(tf_mask(s[i], n[i], type=self.mask_type))

        return np.array(ms)

    def save_data(self, s, n, m, ss, ns, ms, masks, rir):
        """Save the data"""
        # Create the destination folders
        path_out = os.path.join(self.path_dataset, self.scene, self.case)
        for folder in [os.path.join('stft_processed', 'raw', ''), 'wav_processed']:
            for subfolder in ['target', 'noise', 'mixture']:
                os.makedirs(os.path.join(path_out, folder, self.snr_dir, subfolder), exist_ok=True)
        os.makedirs(os.path.join(path_out, 'stft_processed', 'normed', 'abs', self.snr_dir, 'mixture'), exist_ok=True)
        os.makedirs(os.path.join(path_out, 'mask_processed', self.snr_dir), exist_ok=True)
        # Save the data
        for i in range(s.shape[0]):
            if self.save_target:
                sf.write(os.path.join(path_out, 'wav_processed', self.snr_dir, 'target', '') +
                         '{}_Ch-{}.wav'.format(str(rir), str(i + 1)), s[i], self.fs)
            sf.write(os.path.join(path_out, 'wav_processed', self.snr_dir, 'noise', '') +
                     '{}_{}_Ch-{}.wav'.format(str(rir), self.noise, str(i + 1)), n[i], self.fs)
            sf.write(os.path.join(path_out, 'wav_processed', self.snr_dir, 'mixture', '') +
                     '{}_{}_Ch-{}.wav'.format(str(rir), self.noise, str(i + 1)), m[i], self.fs)
            # STFTs
            if self.save_target:
                np.save(os.path.join(path_out, 'stft_processed', 'raw', self.snr_dir, 'target', '') +
                        '{}_Ch-{}'.format(str(rir), str(i + 1)), ss[i])
            np.save(os.path.join(path_out, 'stft_processed', 'raw', self.snr_dir, 'noise', '') +
                    '{}_{}_Ch-{}'.format(str(rir), self.noise, str(i + 1)), ns[i])
            np.save(os.path.join(path_out, 'stft_processed', 'raw', self.snr_dir, 'mixture', '') +
                    '{}_{}_Ch-{}'.format(str(rir), self.noise, str(i + 1)), ms[i])
            np.save(os.path.join(path_out, 'stft_processed', 'normed', 'abs', self.snr_dir, 'mixture', '') +
                    '{}_{}_Ch-{}'.format(str(rir), self.noise, str(i + 1)), abs(ms[i]))
            # Mask
            np.save(os.path.join(path_out, 'mask_processed', self.snr_dir, '') +
                    '{}_{}_Ch-{}'.format(str(rir), self.noise, str(i + 1)), masks[i])
        # SNR
        np.save(os.path.join(path_out, 'log', 'snrs', 'dry', self.snr_dir, '') + '{}_{}'.format(str(rir), self.noise),
                self.snr_out[rir - self.rir_start, :])

