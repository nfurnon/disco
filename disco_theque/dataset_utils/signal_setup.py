import numpy as np
import soundfile as sf
from disco_theque.sigproc_utils import vad_oracle_batch, noise_from_signal, stack_talkers


class SignalSetup:
    """
    Class of setup for what deals with the signals: SNR, corpus they are taken from, duration.
    It is based on a list-logic, that is to say that the WAV files are picked among the limited possibilities of a list.
    This is because the database generation is expected to be run on parallel processes, so we want to avoid taking two
    times the same WAV file.
    """

    def __init__(self, target_list, talkers_list, noises_dict, duration_range, var_tar, snr_dry_range, snr_cnv_range,
                 min_delta_snr):
        self.target_list = target_list  # List where we pick target signal from
        self.ssn_list = talkers_list
        self.noises_dict = noises_dict  # Lists where we pick noise signal from
        self.duration_range = duration_range  # min_dur, max_dur of signals (signals > min are padded to max)
        self.target_duration = None  # Duration of target signal -- determined in get_target_segment
        self.var_tar = var_tar  # Normalized variance of all target signals
        self.snr_dry_range = snr_dry_range  # SNR range of dry signals (at loudspeakers)
        self.snr_cnv_range = snr_cnv_range  # SNR range of convolved signals (at microphones)
        self.min_delta_snr = min_delta_snr  # Maximum difference of SNRs between nodes
        self.source_snr = np.zeros(np.shape(snr_dry_range)[0])

    def get_target_segment(self, target_file):
        """
        Return source signals (one noise, one target)
        :param target_file:     name of an audio file
        :return:                If target_file is long enough, the reshaped signal; if too short, None
        """
        min_duration, max_duration = self.duration_range[0], self.duration_range[1]
        signal, fs = sf.read(target_file)
        signal = signal[:int(max_duration * fs)]
        signal -= np.mean(signal)       # Some librispeech files are not zero-meaned
        sig_duration = len(signal) / fs

        if sig_duration < min_duration:
            ssignal = None
            vsignal = None
        else:
            # VAD
            vad_signal = vad_oracle_batch(signal, thr=0.001)
            # Normalize the segment and add one second of silence at the beginning
            signal *= np.sqrt(self.var_tar / np.var(signal[vad_signal == 1, ]))
            # Update VAD (because of energy, no linear process and VAD is different)
            vad_signal = vad_oracle_batch(signal, thr=0.001)
            ssignal = np.concatenate((np.zeros(fs), signal))
            vsignal = np.concatenate((np.zeros(fs), vad_signal))

        self.target_duration = sig_duration + 1

        return ssignal, vsignal, fs

    def get_noise_segment(self, n_type, duration):
        fs = 16000
        n_types = [nm for nm in self.noises_dict.keys()]
        if n_type.lower() in n_types:
            n, fs, n_file, n_file_start = self._read_random_signal(n_type, duration)
            if n_type == 'inteferent_talker':
                noise_vad = vad_oracle_batch(n, thr=0.001)
            else:
                noise_vad = None

        elif n_type == 'SSN':
            tlk_tot, _, _ = stack_talkers(self.ssn_list, duration, None, nb_tlk=5)
            ssn_dry = noise_from_signal(tlk_tot)  # SSN; length is longer than tar_dry
            n = ssn_dry[:int(duration * fs)]
            n_file = None
            n_file_start = None
            noise_vad = None
        else:
            raise ValueError('Unknown noise type')

        return n, n_file, n_file_start, noise_vad, fs

    def _read_random_signal(self, n_type, duration):
        """
        Read a random duration of a random signal among signals listed in self.noise_dict[n_type]
        :param n_type:      Type of the noise ("chime", "freesound", ...)
        :param duration:    Duration of the noise
        :return:
        """
        assert (duration > 0), "Duration should be strictly positive"
        noise_list = self.noises_dict[n_type]
        sig_duration, n_trials = 0, 0
        max_trials = np.maximum(100, 2 * len(noise_list))
        while sig_duration < duration and n_trials < max_trials:
            rnd_file = np.random.randint(0, len(noise_list))
            sig_duration = sf.info(noise_list[rnd_file]).duration
            n_trials += 1
        if n_trials == max_trials:
            raise ValueError("Failed to find a file lasting more that {} s. "
                             "Please choose a shorter duration".format(duration))
        else:
            sig, fs = sf.read(noise_list[rnd_file])
            rnd_start = np.int((len(sig)) * np.random.rand())   # We start anywhere in the signal. If the end is reached
            sig_rolled = np.roll(sig, len(sig) - rnd_start)     # before the duration, we roll the beginning of the
            y = sig_rolled[:np.int(duration * fs)]              # signal at the end.
            y -= np.mean(y)

        return y, fs, noise_list[rnd_file], rnd_start

    def get_random_dry_snr(self):
        """
        :return:    Random SNR values uniformly picked between self.snr_dry_range[0] and self.snr_dry_range[1]
        """
        n_sources = np.shape(self.snr_dry_range)[0]
        for i_source in range(n_sources):
            alea = np.random.rand()  # Just for the nexline to fit in
            self.source_snr[i_source] = self.snr_dry_range[i_source][0] \
                                        + (self.snr_dry_range[i_source][1] - self.snr_dry_range[i_source][0]) * alea
        return self.source_snr

