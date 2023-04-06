import numpy as np

def sliding_windowing(complete_record, w_len, overlap, fs):
    ''' Function that receive a signal, and returns sliding windows of that signal.
    :param w_len: in SECONDS.
    :param overlap: a decimal number between 0 and 1.
    '''
    w_len = int(w_len*fs)
    N = len(complete_record)
    step = int(w_len - overlap*w_len)
    num_w = int(np.floor((N-overlap*w_len)/step))
    signal_windows = np.zeros((num_w, w_len))
    for i in range(num_w):
        signal_windows[i,:] = complete_record[i*step:i*step+w_len]
    return signal_windows

def sliding_conc_samples_of_windows(complete_record, w_len, conc_sample_len,
                                    conc_sample_overlap, fs):
    assert w_len < conc_sample_len
    w_len = int(w_len*fs)
    conc_sample_len = int(conc_sample_len*fs)
    conc_sample_overlap = int(conc_sample_overlap*fs)
    N = len(complete_record)
    conc_sample_step = conc_sample_len - conc_sample_overlap
    num_conc_samples = int(np.floor(((N + (conc_sample_len - 1)) - conc_sample_overlap) / conc_sample_step))
    num_windows_per_conc_samples = int(np.floor(conc_sample_len / w_len))

    # dim -->  ( numb_conc_samples,  num_windows_per_conc_samples, len_of_each_window )
    all_windows = np.zeros((num_conc_samples, num_windows_per_conc_samples, w_len))

    # padding final of the signal with the beginning of the signal
    complete_record = np.concatenate((complete_record, complete_record[:(conc_sample_len-1)]), axis=0)
    for i in range(num_conc_samples):
        id_i = i * conc_sample_step
        id_f = conc_sample_len + i*conc_sample_step
        conc_sample_i = complete_record[id_i : id_f]
        for j in range(num_windows_per_conc_samples):
            all_windows[i, j, :] = conc_sample_i[j*w_len:(j+1)*w_len]
    return all_windows

