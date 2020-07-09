import re
import os
import scipy
import numpy as np
import soundfile as sf
from acoustics.signal import OctaveBand
from disco_theque.math_utils import next_pow_2, lin2db, db2lin


#%% VAD
def vad_oracle_batch(x_, win_len=512, win_hop=256, thr=0.001, rat=2):
    """Estimates voice activity segments based on signal power.

    The function determines whether an audio window contains speech by comparing its power to a threshold.
    More precisely, a window of audio samples is labeled as containing speech if the number of samples with an
    instantaneous power larger than `thr` is larger than the number of samples in the window divided by `rat`. With
    `rat` set to 2, it means that more than half the samples in the window have a instantaneous power larger than `thr`.

    The output is a vector of 0s and 1s of the same length as `x_`. A 0 indicates the absence of speech while a 1
    indicates its presence.

    .. note::

        This method should work reasonably well for clean speech but will
        perform poorly on noisy data.

    Oracle voice activity detector; speech is detected in batch mode, i.e. one binary decision is taken for a batch
    of N points.

    Args:
      x_: Audio waveform
      win_len: Length of the window (Default: 512)
      win_hop: Hop size of windows (Default: 256)
      thr: Threshold value (Default: 0.001)
      rat: Ratio of N (Default: 2)

    Returns:
        np.ndarray: Estimated voice activity

    """
    x = x_ - np.mean(x_)
    x2 = abs(x ** 2)

    thr_ = thr * np.quantile(x2, 0.99)      # Avoid determining threshold out of outliners
    vad_o = np.zeros(len(x2))
    # Buffer
    for n in np.arange(int(np.ceil((len(x2) - win_len) / win_hop + 1))):
        x2_win = x2[n * win_hop:np.minimum(n * win_hop + win_len, len(x2))]
        x2_win_va = 1 * (x2_win > thr_)
        nb_va = sum(x2_win_va)
        N_ = len(x2_win)        # Last window has probably less samples than all other ones
        if nb_va >= np.int(N_ / rat):
            vad_o[n * win_hop:np.minimum(n * win_hop + win_len, len(x2))] = 1
    return vad_o


#%% Filterbanks
def third_octave_filterbank(F, fs, order=8):
    """Computes coefficients of a third-octave filter bank.

    Row `i` of the returned arrays contain the coefficients for the `i`-th filter

    .. warning::
    
        Suboptimal minimalist function

    Args:
      F (list[int]): Center frequencies
      fs (int): Sampling frequency
      order: Order of the Butterworth filters (Default: 8)

    Returns:
        tuple: numerator coefficients, denominator coefficients

    """
    N = len(F)
    b = np.zeros((N, np.int(2 * order + 1)))
    a = np.zeros((N, np.int(2 * order + 1)))
    for i in np.arange(N):
        ob = OctaveBand(center=F[i], fraction=3)
        b[i, :], a[i, :] = scipy.signal.butter(order, np.array([ob.lower.item(), ob.upper.item()]) * 2 / fs,
                                               btype='bandpass', output='ba')

    return b, a


#%% Metric
def fw_snr(s, n, fs, vad_tar=None, vad_noi=None, clipping=1, db=True):
    """Computes the SNR weighted by the speech intelligibility.

    Args:
        s: Target speech
        n: Noise speech
        fs: Sampling frequency
        vad_tar:  (Default: None)
        vad_noi:  (Default: None)
        clipping: If (Default: 1)
        db: bool (Default: True)

    Returns:
        tuple:
            * fqwt_snr: Frequency weighted SNR
            * fw_snr_mean: Average fw_SNR
            * F: Center frequencies

    """
    # Band importance function
    r = 2**(1/6)
    if fs/2 > 4500:
        I = np.array([83, 95, 150, 289, 440, 578, 653, 711, 818, 844, 882, 898, 868, 844, 771, 527, 364, 185])*1e-4
        F = np.array([160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
                      8000])
        f2 = F*r
        N = np.sum(f2 < fs/2)
        F = F[:N]
        I = I[:N]
    else:
        I = np.array([128, 320, 320, 447, 447, 639, 639, 767, 959, 1182, 1214, 1086, 1086, 757])*1e-4
        F = np.array([200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000])
        f2 = F*r
        N = np.sum(f2 < fs/2)
        F = F[:N]
        I = I[:N]
        
    snr_var = np.zeros((N,))
    s_p = np.zeros((N,))
    n_p = np.zeros((N,))
    s_f, n_f = [], []
    # Calculation of the SNR values in the different bands
    b, a = third_octave_filterbank(F, fs, order=4)
    for i in np.arange(N):
        s_f.append(scipy.signal.lfilter(b[i, :], a[i, :], s, axis=0))
        n_f.append(scipy.signal.lfilter(b[i, :], a[i, :], n, axis=0))
        if vad_tar is None:
            s_p[i] = lin2db(np.var(s_f[i][s_f[i] != 0]))
        else:
            s_p[i] = lin2db(np.var(s_f[i][vad_tar != 0]))

        if vad_noi is None:
            n_p[i] = lin2db(np.var(n_f[i][n_f[i] != 0]))
        else:
            n_p[i] = lin2db(np.var(n_f[i][vad_noi != 0]))

        snr_var[i] = s_p[i] - n_p[i]
        
        if clipping:
            snr_var[i] = np.minimum(np.maximum(-15, snr_var[i]), 25)

    fqwt_snr = I/np.sum(I)*snr_var
    fw_snr_mean = np.sum(fqwt_snr)

    # Convert if necessary in dB
    fqwt_snr = db * fqwt_snr + (not db) * db2lin(fqwt_snr)
    fw_snr_mean = db * fw_snr_mean + (not db) * db2lin(fw_snr_mean)

    return fqwt_snr, fw_snr_mean, F


def increase_to_snr(x, n, snr_out, vad_tar=None, vad_noi=None, weight=False, fs=None):
    """Changing the *noise* level to ensure a SNR of snr_out dB.
    Arguments:
        - x         Target signal
        - n         Noise signal
        - snr_out   Desired output SNR (in dB)
        - vad       VAD of noise to compute level on non silent parts. Should be the same length as n.
        - weight    (bool) If True, weight SNR in freq domain
        -fs         (float) Sampling frequency. Only required if weight=True
    Output:
        - n_ where n_ level is such that var(x) - var(n) = 10**(snr/10)

    TO DO: transform this in function(s) depending on RIR and SNR and noise signal
    """
    if weight:
        _, snr_0, _ = fw_snr(x, n, fs, vad_tar=vad_tar, vad_noi=vad_noi)
        n_ = n * 10 ** ((snr_0 - snr_out) / 20)
    else:
        if vad_tar is not None:
            var_x = np.var(x[vad_tar != 0])
        else:
            var_x = np.var(x[x != 0])

        if vad_noi is not None:
            var_n = np.var(n[vad_noi != 0])
        else:
            var_n = np.var(n[n != 0])
        n_ = n * np.sqrt(10 ** (-snr_out / 10) * var_x / var_n)
    return n_


def stack_talkers(tlk_list, dur_min, speaker, nb_tlk=5):
    """Stacks talkers from tlk_list until dur_min is reached and number of
    speakers exceeds nb_tlk
    Arguments:
        - tlk_list       list of flac/wav files to pick talkers in $
        - dur_min       Minimal duration *in seconds* of the signal
    """
    i_tlk = 0
    tlk_tot = np.array([])
    str_files = str()
    fs = 16000

    while len(tlk_tot) < int(dur_min * fs) or i_tlk < nb_tlk:  # At least 5 talkers' speech shape
        rnd_tmp = np.random.randint(0, len(tlk_list))  # random talker we are going to pick
        spk_tmp = re.split('/', tlk_list[rnd_tmp])[-1].split('-')[0]
        if spk_tmp != speaker:  # Don't take same speaker for SSN
            tlk_tmp, fs = sf.read(tlk_list[rnd_tmp])
            tlk_tot = np.hstack((tlk_tot, tlk_tmp))
            i_tlk += 1
            str_files = str_files + os.path.basename(tlk_list[rnd_tmp])[:-5] + '\n'

    return tlk_tot, fs, str_files


def noise_from_signal(x):
    """Create a noise with same spectrum as the input signal.

    Parameters
    ----------
    x : array_like
        Input signal.

    Returns
    -------
    ndarray
        Noise signal.

    Copyright https://github.com/achabotl/pambox/blob/develop/pambox/distort.py
    """
    x = np.asarray(x)
    n_x = x.shape[-1]
    n_fft = next_pow_2(n_x)
    X = np.fft.rfft(x, next_pow_2(n_fft))
    # Randomize phase.
    noise_mag = np.abs(X) * np.exp(
        2 * np.pi * 1j * np.random.random(X.shape[-1]))
    noise = np.real(np.fft.irfft(noise_mag, n_fft))
    out = noise[:n_x]

    return out

