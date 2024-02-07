import numpy as np
from scipy.signal import find_peaks, correlate
from sklearn.decomposition import PCA, FastICA
from scipy.signal import firwin, lfilter, iirnotch, filtfilt, butter

def interpolate_nans(signal):
    '''
    function that:
    - interpolates over NaN values in a given signal
    - prints a warning if signal contains only NaNs

    parameters:
    - signal: A 2D numpy array representing the ECG signal with leads in rows

    Returns:
    - the signal with NaNs interpolated
    '''
    
    for i in range(signal.shape[0]): 
        lead_signal = signal[i, :]

        if np.isnan(lead_signal).all():
            print(f"Warning: Lead {i} contains only NaNs.")
            continue

        if np.isnan(lead_signal).any():
            nan_indices = np.isnan(lead_signal)
            valid_indices = ~nan_indices
            valid_signal = lead_signal[valid_indices]
            signal[i, :] = np.interp(
                np.arange(len(lead_signal)),
                valid_indices.nonzero()[0],
                valid_signal
            )
    return signal


def filter_signal(signal, cutoff, notch, fs, numtaps):
    '''
    function that filters signal with 
    - high-pass FIR filter to prevent baseline wandering
    - notch filter to cancell the powerline interferance
    '''
    if np.isnan(signal).any():
        print("Warning: NaN values detected in the input signal before filtering.")

    nyq = 0.5 * fs
    fir_coeff = firwin(numtaps, cutoff/nyq, pass_zero=False)
    filtered_signal = lfilter(fir_coeff, 1.0, signal)
    b, a = iirnotch(notch/nyq, 30)
    filtered_signal = filtfilt(b, a, filtered_signal)

    if np.isnan(filtered_signal).any():
        print("Warning: NaN values detected in the filtered signal after filtering.")


    return filtered_signal, fir_coeff


def qrs_detector(signal, fs, min_distance):
    '''
    function that performs QRS detection in multichannel ecg signal
    
    parameters:
    - signal: multichannel ecg signal from one patient
    - fs: corresponding sampling rate
    - min_distance [s]: minimal distance between R peaks based on HR
      (different for adult and fetus)

    returns:
    - enhanced signal
    - positions of QRS complexes
    - template of the QRS complex
    '''
    # Step 1: Enhance QRS
    normalized_signal = signal / np.std(signal, axis=1, keepdims=True)
    pca = PCA(n_components=1)
    pca_signal = pca.fit_transform(normalized_signal.T).flatten()

    # Step 2: Detect peaks to create multiple QRS candidates
    peak_indices, _ = find_peaks(pca_signal, distance=fs*min_distance)
    qrs_candidates = [pca_signal[idx - int(0.1*fs): idx + int(0.1*fs)] for idx in peak_indices if idx - int(0.1*fs) > 0 and idx + int(0.1*fs) < len(pca_signal)]
    
    # Step 3: Cluster or average QRS candidates to create a template
    # For simplicity, here we'll just average the candidates to create the template
    qrs_template = np.mean(qrs_candidates, axis=0)

    # Step 4: Detect QRS peaks using the template
    correlation = correlate(pca_signal, qrs_template, mode='same')
    qrs_peaks, _ = find_peaks(correlation, distance=fs*min_distance)
    
    return qrs_peaks, pca_signal, qrs_template


def average_complex(signal, qrs_peaks, fs, N=10, adult = True):
    '''
    returns:
    - average MECG complex over N complexes
    - calculated HR

    '''
    if adult == True:
        left = int(0.25 * fs)  # 0.25 seconds before the QRS
        right = int(0.45 * fs)  # 0.45 seconds after the QRS
    else:
        left = int(0.15 * fs)  # 0.15 seconds before the QRS
        right = int(0.30 * fs)  # 0.30 seconds after the QRS


    complexes = []

    for peak in qrs_peaks[1:N+1]:
        if peak - left >= 0 and peak + right <= len(signal):
            qrs_complex = signal[peak - left:peak + right]
            complexes.append(qrs_complex)

    mu = np.mean(complexes, axis=0)

    return mu

def windowing(mu, fs):
    '''
    function that takes average MECG complex that lasts 0.7s and windows it to create:
    - mu_p - average P wave form 0 to 0.20s
    - mu_qrs - average QRS complex from 0.20s to 0.30s
    - mu_r -average T wave from 0.30s to 0.70s 

    returns:
    - martix M that assembles P wave, T wave and QRS complex vectors
    '''

    p_qrs = int(0.20*fs)
    qrs_t = int(0.3*fs)

    mu_p = mu[:p_qrs]
    mu_qrs = mu[p_qrs:qrs_t]
    mu_t = mu[qrs_t:]

    # Assemble the matrix M
    M = np.zeros((len(mu), 3))
    M[:len(mu_p), 0] = mu_p 
    M[p_qrs:qrs_t, 1] = mu_qrs 
    M[qrs_t:, 2] = mu_t 

    return M


def cancel_mecg(channel_signal, qrs_peaks, M, fs):
    '''
    Apply MECG cancellation to the entire signal for each channel.
    '''

    fecg_signal = np.copy(channel_signal)
    
    for peak in qrs_peaks:
        
        left = int(0.25 * fs) 
        right = int(0.45 * fs)      

        if peak - left >= 0 and peak + right <= len(channel_signal):

            m = channel_signal[peak-left:peak+right] #extrect MECG complex

            a = np.linalg.pinv(M) @ m #scalin vector

            m_hat = M @ a.reshape(-1, 1) #MECG complex estimate
            
            fecg_signal[peak-left:peak+right] -= m_hat.ravel()
    
    return fecg_signal
    
def calculate_fhr_trace(qrs_peaks, fs):
    rr_intervals = np.diff(qrs_peaks) / fs
    hr = 60 / rr_intervals
    return hr

def calculate_accuracy(qrs_peaks, fqrs_locations, epsilon_ms=50, fs=2000):
    epsilon_samples = int((epsilon_ms / 1000) * fs)  # Convert epsilon from ms to samples

    correct_count = 0

    for qrs_peak in qrs_peaks:
        for fqrs_location in fqrs_locations:
            if abs(qrs_peak - fqrs_location) <= epsilon_samples:
                correct_count += 1
                break  # Break inner loop if a match is found

    accuracy = (correct_count / len(qrs_peaks)) * 100

    return accuracy




