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
    Save the STFT of the mixture and the corresponding mask
    """
    def __init__(self, rir_start, nb_rir, scene, noise, snr_range, path_to_dataset,
                 n_fft=512, n_hop=256, mask_type='irm1', norm_type='scale_to_1', save_target=True):
        self.rir_start = rir_start
        self.nb_rir = nb_rir
        self.case = self.get_dset()
        self.save_target = save_target  # Useless to overwrite the same signal
        self.scene = scene
        n_noises = 1
        self.noise = noise
        self.snr_range = np.array(snr_range)
        if self.snr_range.ndim == 1:
            self.snr_range = self.snr_range[np.newaxis, :]
        self.snr_out = np.zeros((nb_rir, n_noises))
        self.path_dataset = path_to_dataset
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.mask_type = mask_type
        # Second go
        self.norm = norm_type
        self.stats = self.get_db_stats()

    def get_dset(self):
        """ Return train or test depending on RIR we are considering"""
        assert 0 < self.rir_start < 12001, "rir ID should be between 1 and 12000"
        assert self.rir_start + self.nb_rir <= 12001, "Expected number of RIR is too high (max RIR ID is 12000)"
        assert not self.rir_start < 11001 < self.rir_start + self.nb_rir, "First and last RIR id are not part of the " \
                                                                          "same set"
        out = 'train' if self.rir_start < 11001 else 'test'
        return out

    def post_process(self):
        dirry = '_'.join(['{}-{}'.format(str(self.snr_range[k][0]), str(self.snr_range[k][1]))
                          for k in range(len(self.snr_range))])  # e.g. '0-6_5-15' or '0-6'
        path_out = os.path.join(self.path_dataset, self.scene, self.case)
        os.makedirs(os.path.join(path_out, 'log/snrs/dry', dirry), exist_ok=True)
        for rir in range(self.rir_start, self.rir_start + self.nb_rir):
            if not os.path.isfile(os.path.join(path_out, 'log/snrs/dry', dirry, '') +
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
        # Pick random SNR once for all channels
        snrs = np.zeros(len(noi_list))
        for i_noise in range(len(noi_list)):
            snrs[i_noise] = self.snr_range[i_noise][0] + \
                            (self.snr_range[i_noise][1] - self.snr_range[i_noise][0]) * np.random.random()
        for ch in range(1, 17):
            tar_seg, _ = sf.read(tar_list[ch - 1], dtype='float32')
            noi_seg = np.zeros(len(tar_seg))
            for i_noise in range(len(noi_list)):
                noi, _ = sf.read(noi_list[i_noise][ch - 1], dtype='float32')
                noi_seg[:len(noi)] += noi * 10 ** (- snrs[i_noise] / 20)

            tar_segs.append(tar_seg)
            noi_segs.append(noi_seg)
            mix_segs.append(tar_seg + noi_seg)

        return np.array(tar_segs), np.array(noi_segs), np.array(mix_segs), snrs

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
        fs = 16000
        # Create the destination folders
        dirry = '_'.join(['{}-{}'.format(str(self.snr_range[k][0]), str(self.snr_range[k][1]))
                        for k in range(len(self.snr_range))])   # e.g. '0-6_5-15' or '0-6'
        path_out = os.path.join(self.path_dataset, self.scene, self.case)
        for folder in ['stft_processed/raw/', 'wav_processed']:
            for subfolder in ['target', 'noise', 'mixture']:
                os.makedirs(os.path.join(path_out, folder, dirry, subfolder), exist_ok=True)
        os.makedirs(os.path.join(path_out, 'stft_processed', 'normed', 'abs', dirry, 'mixture'), exist_ok=True)
        os.makedirs(os.path.join(path_out, 'mask_processed', dirry), exist_ok=True)
        # Save the data
        for i in range(s.shape[0]):
            if self.save_target:
                sf.write(os.path.join(path_out, 'wav_processed', dirry, 'target', '') +
                         '{}_Ch-{}.wav'.format(str(rir), str(i + 1)), s[i], fs)
            sf.write(os.path.join(path_out, 'wav_processed', dirry, 'noise', '') +
                     '{}_{}_Ch-{}.wav'.format(str(rir), self.noise, str(i + 1)), n[i], fs)
            sf.write(os.path.join(path_out, 'wav_processed', dirry, 'mixture', '') +
                     '{}_{}_Ch-{}.wav'.format(str(rir), self.noise, str(i + 1)), m[i], fs)
            # STFTs
            if self.save_target:
                np.save(os.path.join(path_out, 'stft_processed', 'raw', dirry, 'target', '') +
                        '{}_Ch-{}'.format(str(rir), str(i + 1)), ss[i])
            np.save(os.path.join(path_out, 'stft_processed', 'raw', dirry, 'noise', '') +
                    '{}_{}_Ch-{}'.format(str(rir), self.noise, str(i + 1)), ns[i])
            np.save(os.path.join(path_out, 'stft_processed', 'raw', dirry, 'mixture', '') +
                    '{}_{}_Ch-{}'.format(str(rir), self.noise, str(i + 1)), ms[i])
            np.save(os.path.join(path_out, 'stft_processed', 'normed', 'abs', dirry, 'mixture', '') +
                    '{}_{}_Ch-{}'.format(str(rir), self.noise, str(i + 1)), abs(ms[i]))
            # Mask
            np.save(os.path.join(path_out, 'mask_processed', dirry, '') +
                    '{}_{}_Ch-{}'.format(str(rir), self.noise, str(i + 1)), masks[i])
        # SNR
        np.save(os.path.join(path_out, 'log/snrs/dry', dirry, '') + '{}_{}'.format(str(rir), self.noise),
                self.snr_out[rir - self.rir_start, :])

    def get_directory_name(self):
        """From snr_range in self, return the name of the directory in form of 0-6 or 3-6_5-15"""
        dirry = '_'.join(['{}-{}'.format(str(self.snr_range[k][0]), str(self.snr_range[k][1]))
                          for k in range(len(self.snr_range))])  # e.g. '0-6_5-15' or '0-6'
        return dirry

    def get_sir(self):
        """Loop over rir_start to rir_end and run get_node_sir"""
        dirry = self.get_directory_name()
        path_out = os.path.join(self.path_dataset, self.scene, self.case, 'log', 'snrs', 'cnv', dirry, '')
        for rir in range(self.rir_start, self.rir_start + self.nb_rir):
            if not os.path.isfile(path_out + '{}_{}.npy'.format(str(rir), self.noise)):
                self.get_node_sir(rir)
            else:
                print('{} already done'.format(str(rir)))

    def get_node_sir(self, rir):
        """Get SIR at each node"""
        dirry = self.get_directory_name()
        path_wav = os.path.join(self.path_dataset, self.scene, self.case, 'wav_processed', dirry, '')
        sirs = np.zeros(4)
        for i_nod in range(4):
            ch_snrs = np.zeros(4)
            for i_ch in range(4):
                ch = i_nod * 4 + i_ch + 1
                s, fs = sf.read(os.path.join(path_wav, 'target', '')
                            + '{}_Ch-{}.wav'.format(str(rir), str(ch)))
                n = sf.read(os.path.join(path_wav, 'noise', '')
                            + '{}_{}_Ch-{}.wav'.format(str(rir), self.noise, str(ch)))[0]
                siri = fw_snr(s, n, fs)
                ch_snrs[i_ch] = siri[1]
            sirs[i_nod] = np.mean(ch_snrs)
        # Save the SIRs
        np.save(os.path.join(self.path_dataset, self.scene, self.case, 'log', 'snrs', 'cnv', dirry, '') +
                '{}_{}'.format(str(rir), self.noise), sirs)
