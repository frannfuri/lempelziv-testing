import numpy as np
import matplotlib.pyplot as plt
from lzc_comp import KC_complexity_lempelziv_count
from main_binarization_methods import numpy_to_str_sequence, median_bin
from scipy.signal import chirp
from features_comp import sliding_windowing

def gen_sinusoid(f, Fs, t_len):
    x = np.arange(0, t_len, 1/Fs)
    y = np.sin(2*np.pi*f*x)
    return x, y

def gen_AM_sin(fc, Ac, fm, m_id, Fs, t_len):
    x = np.arange(0, t_len, 1 / Fs)
    y = Ac * (1 + m_id*np.sin(2*np.pi*fm*x)) * np.sin(2*np.pi*fc*x)
    return x, y

def gen_FM_sin(fc, fm, m_id, Fs, t_len):
    x = np.arange(0, t_len, 1 / Fs)
    y = np.sin(2*np.pi*fc*x + m_id*np.sin(2*np.pi*fm*x))
    return x, y

def gen_noisy_sin(f, Fs, t_len, noiseSigma=0.1, noiseAmp=2):
    x = np.arange(0, t_len, 1 / Fs)
    sinusoid = np.sin(2 * np.pi * f * x)
    noise = noiseAmp * np.random.normal(0, noiseSigma, len(x))
    # Averaged spectral density (we use sampling_rate // 2 points for Nyquist)
    cleanPS = np.sum(np.abs(np.fft.fft(sinusoid, Fs // 2) / len(x)) ** 2)
    noisePS = np.sum(np.abs(np.fft.fft(noise, Fs // 2) / len(x)) ** 2)
    # 10 instead of 20 because we're using power instead of RMS amplitude
    measuredSNR = 10 * np.log10(cleanPS / noisePS)
    y = noise + sinusoid
    return x, y, measuredSNR

def gen_white_noise(Fs, t_len, mean=0, std=1):
    x = np.arange(0, t_len, 1 / Fs)
    y = np.random.normal(mean, std, len(x))
    return x, y

def gen_four_components(f, Fs, t_len):
    x = np.arange(0, t_len, 1 / Fs)
    len_x_seg = int(len(x) / 4)
    y = np.zeros(len(x))
    odd=1
    mult=2
    for i in range(4):
        k = i * mult + odd
        yh = (1/k) * np.sin(2 * np.pi * k * f * x[-len_x_seg*(4-i):])
        y = y + np.concatenate((np.zeros(len_x_seg*i),yh))
    return x,y

def gen_increm_snr_sin(f, Fs, t_len):
    x = np.arange(0, t_len, 1 / Fs)
    len_x_seg = int(len(x)/4)
    sinusoid = 2*np.sin(2 * np.pi * f * x)
    n1 = 2 * np.random.normal(0, 0.1, len_x_seg*4)
    n2 = np.concatenate((np.zeros(len_x_seg), 2 * np.random.normal(0, 0.2, len_x_seg*3)))
    n3 = np.concatenate((np.zeros(2*len_x_seg), 2 * np.random.normal(0, 0.3, len_x_seg*2)))
    n4 = np.concatenate((np.zeros(3*len_x_seg), 2 * np.random.normal(0, 0.31, len_x_seg)))
    y = sinusoid + n1 +n2+n3+n4
    return x, y

def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def variable_bandwidth_noise(Fs, t_len):
    x = np.arange(0, t_len, 1/Fs)
    y1 = band_limited_noise(0.1, 3, samples=int(len(x)/4), samplerate=Fs)
    y2 = 0.8*band_limited_noise(0.1, 7, samples=int(len(x)/4), samplerate=Fs)
    y3 = 0.5*band_limited_noise(0.1, 15, samples=int(len(x)/4), samplerate=Fs)
    y4 = 0.3*band_limited_noise(0.1, 30, samples=int(len(x)/4), samplerate=Fs)
    y = np.concatenate((y1,y2,y3,y4))
    return x, y


if __name__ == '__main__':
    Fs = 125
    # FIRST TEST
    x_sin, y_sin = gen_sinusoid(2, Fs,t_len=5)
    lzc0 = KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(y_sin)))
    x_am_sin, y_am_sin = gen_AM_sin(fc=10, Ac=2, fm=1, m_id=0.5, Fs=Fs, t_len=5)
    lzc1 = KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(y_am_sin)))
    x_fm_sin, y_fm_sin = gen_FM_sin(fc=10, fm=1, m_id=3, Fs=Fs, t_len=5)
    lzc2 = KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(y_fm_sin)))
    x_noisy_sin, y_noisy_sin, snr = gen_noisy_sin(2, Fs, 5)
    lzc3 = KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(y_noisy_sin)))
    x_white_noise, y_white_noise = gen_white_noise(Fs, 5)
    lzc4 = KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(y_white_noise)))
    fig, axs = plt.subplots(5,1, figsize=(15,10))
    axs[0].set_title('Sinusoid', fontsize=9)
    axs[0].plot(x_sin, y_sin, label='LZC: {:.4f}'.format(lzc0), linewidth=0.6)
    axs[0].legend(loc='upper right')
    axs[0].tick_params(axis='both', labelsize=8)
    axs[1].set_title('Amp. modulated Sinusoid', fontsize=9)
    axs[1].plot(x_am_sin, y_am_sin, label='LZC: {:.4f}'.format(lzc1), linewidth=0.6)
    axs[1].legend(loc='upper right')
    axs[1].tick_params(axis='both', labelsize=8)
    axs[2].set_title('Freq. modulated Sinusoid', fontsize=9)
    axs[2].plot(x_fm_sin, y_fm_sin, label='LZC: {:.4f}'.format(lzc2), linewidth=0.6)
    axs[2].legend(loc='upper right')
    axs[2].tick_params(axis='both', labelsize=8)
    axs[3].set_title('Noisy Sinusoid (SNR={:.4f})'.format(snr), fontsize=9)
    axs[3].plot(x_noisy_sin, y_noisy_sin, label='LZC: {:.4f}'.format(lzc3), linewidth=0.6)
    axs[3].legend(loc='upper right')
    axs[3].tick_params(axis='both', labelsize=8)
    axs[4].set_title('White Noise', fontsize=9)
    axs[4].plot(x_white_noise, y_white_noise, label='LZC: {:.4f}'.format(lzc4), linewidth=0.6)
    axs[4].legend(loc='upper right')
    axs[4].set_xlabel('t [s]', fontsize=10)
    axs[4].xaxis.set_label_coords(0.5, -0.2)
    axs[4].tick_params(axis='both', labelsize=8)
    plt.tight_layout()

    # SECOND TEST
    t_len=40
    x_chirp = np.arange(0, t_len, 1/Fs)
    y_chirp = chirp(x_chirp, 0.5, 40, 5)
    x_four_comp , y_four_comp = gen_four_components(0.4, Fs, t_len)
    y_am_chirp = 2 * (1 + 0.3*np.sin(2*np.pi*0.05*x_chirp)) * y_chirp
    x_increm_snr, y_increm_snr = gen_increm_snr_sin(0.8, Fs, 40)
    y_increm_power_noise = np.random.normal(0, 1, len(x_chirp)) + 0.5*np.concatenate((np.zeros(int(len(x_chirp)/4)), np.random.normal(0,1,int(len(x_chirp)/4)*3)))\
            + 1.2*np.concatenate((np.zeros(2*int(len(x_chirp)/4)), np.random.normal(0,1,int(len(x_chirp)/4)*2))) + 1.9*np.concatenate((np.zeros(int(len(x_chirp)/4)*3), np.random.normal(0,1,int(len(x_chirp)/4))))
    x_variable_bw_noise, y_variable_bw_noise = variable_bandwidth_noise(Fs, t_len)

    fig2, axs2 = plt.subplots(6,1, figsize= (15,10))
    axs2[0].plot(x_chirp, y_chirp, label = 'LZC vs signal freq.', linewidth=0.6)
    y_chirp_windows = sliding_windowing(y_chirp,10,0.9,Fs)
    axs2[0].legend(loc='upper right')
    axs2[0].tick_params(axis='both', labelsize=8)
    axs2[1].plot(x_chirp, y_four_comp, label='LZC vs numb. of freq. components', linewidth=0.6)
    y_four_comp_windows = sliding_windowing(y_four_comp, 10, 0.9, Fs)
    axs2[1].legend(loc='upper right')
    axs2[1].tick_params(axis='both', labelsize=8)
    axs2[2].plot(x_chirp, y_am_chirp, label='LZC vs amplitude', linewidth=0.6)
    y_am_chirp_windows = sliding_windowing(y_am_chirp, 10, 0.9, Fs)
    axs2[2].legend(loc='upper right')
    axs2[2].tick_params(axis='both', labelsize=8)
    axs2[3].plot(x_chirp, y_increm_snr, label='LZC vs SNR', linewidth=0.6)
    y_increm_snr_windows = sliding_windowing(y_increm_snr, 10, 0.9, Fs)
    axs2[3].legend(loc='upper right')
    axs2[3].tick_params(axis='both', labelsize=8)
    axs2[4].plot(x_chirp, y_increm_power_noise, label='LZC vs noise power', linewidth=0.6)
    y_increm_power_noise_windows = sliding_windowing(y_increm_power_noise, 10, 0.9, Fs)
    axs2[4].legend(loc='upper right')
    axs2[4].tick_params(axis='both', labelsize=8)
    axs2[5].plot(x_chirp, y_variable_bw_noise, label='LZC vs noise bandwidth', linewidth=0.6)
    y_variable_bw_noise_windows = sliding_windowing(y_variable_bw_noise, 10, 0.9, Fs)
    axs2[5].legend(loc='upper right')
    axs2[5].set_xlabel('t [s]', fontsize=10)
    axs2[5].xaxis.set_label_coords(0.5, -0.2)
    axs2[5].tick_params(axis='both', labelsize=8)
    plt.tight_layout()

    fig3, axs3 = plt.subplots(2,3, figsize=(18,18))
    y_chirp_lzcs = []
    y_four_comp_lzcs = []
    y_am_chirp_lzcs = []
    y_increm_snr_lzcs = []
    y_increm_power_noise_lzcs = []
    y_variable_bw_noise_lzcs = []
    for w_id in range(0, len(y_chirp_windows)):
        y_chirp_lzcs.append(KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(y_chirp_windows[w_id]))))
        y_four_comp_lzcs.append(KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(y_four_comp_windows[w_id]))))
        y_am_chirp_lzcs.append(KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(y_am_chirp_windows[w_id]))))
        y_increm_power_noise_lzcs.append(KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(y_increm_power_noise_windows[w_id]))))
        y_increm_snr_lzcs.append(KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(y_increm_snr_windows[w_id]))))
        y_variable_bw_noise_lzcs.append(KC_complexity_lempelziv_count(numpy_to_str_sequence(median_bin(y_variable_bw_noise_windows[w_id]))))
    fig3.suptitle('LZC evolution')
    axs3[0,0].plot(y_chirp_lzcs, '-o', label='LZC vs freq.', linewidth=0.8, markersize=2)
    axs3[0,0].legend(loc='upper right', fontsize=7)
    axs3[0,0].set_ylim((0.1, 0.2))
    axs3[0,1].plot(y_four_comp_lzcs, '-o', label='LZC vs numb. of freq. components', linewidth=0.8, markersize=2)
    axs3[0, 1].legend(loc='upper right', fontsize=7)
    axs3[0, 1].set_ylim((0, 0.2))
    axs3[0, 2].plot(y_am_chirp_lzcs, '-o', label='LZC vs amplitude', linewidth=0.8, markersize=2)
    axs3[0, 2].legend(loc='upper right', fontsize=7)
    axs3[0, 2].set_ylim((0.1, 0.2))
    axs3[1, 0].plot(y_increm_snr_lzcs, '-o', label='LZC vs SNR', linewidth=0.8, markersize=2)
    axs3[1, 0].legend(loc='upper right', fontsize=7)
    axs3[1, 0].set_ylim((0, 0.8))
    axs3[1, 1].plot(y_increm_power_noise_lzcs, '-o', label='LZC vs power noise', linewidth=0.8, markersize=2)
    axs3[1, 1].legend(loc='upper right', fontsize=7)
    axs3[1, 1].set_ylim((0, 2))
    axs3[1, 2].plot(y_variable_bw_noise_lzcs, '-o', label='LZC vs noise bandwidth', linewidth=0.8, markersize=2)
    axs3[1, 2].legend(loc='upper right', fontsize=7)
    axs3[1, 2].set_ylim((0.1, 0.6))
    axs3[1,1].set_xlabel('window in time', fontsize=8)
    axs3[1, 2].set_xlabel('window in time', fontsize=8)
    axs3[1, 0].set_xlabel('window in time', fontsize=8)
    plt.show()
    a = 0