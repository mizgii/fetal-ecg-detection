import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

from other_functions import interpolate_nans, filter_signal, qrs_detector
from other_functions import average_complex, windowing, cancel_mecg
from other_functions import calculate_fhr_trace, calculate_accuracy

from ploty import plot_filter_characteristics, plot_signals_comparision
from ploty import save_plots_original_signals, save_plots_filtered_signals
from ploty import plot_qrs_detection, plot_qrs_template



def load_ecg_dataset(dataset_path, numtaps = 1001, new_fs = 2000, plot_sample = False):
    records = []
    fqrs_locations =[]
    for filename in os.listdir(dataset_path):
        if filename.endswith('.hea'):  # HEA file defines the header for each record
            record_name = os.path.splitext(filename)[0]
            record_path = os.path.join(dataset_path, record_name)
            record = wfdb.rdrecord(record_path)
            fs = record.fs #all signals have the same fs of 1000
            signal = record.p_signal.T #transform for more intuitive structure with .T

            interpolated_signal = interpolate_nans(signal)

            filtered_signal, fir_coeff = filter_signal(interpolated_signal, 3, 50, fs, numtaps)

            num_samples = int(len(filtered_signal[0]) * (new_fs / fs)) # SRI - sampling-rate increaser.
            upsampled_signal = resample(filtered_signal, num_samples, axis=1)

            records.append(upsampled_signal) 

            fqrs_annotations = wfdb.rdann(record_path, 'fqrs')
            upscaled_fqrs_locations = [location * (new_fs // fs) for location in fqrs_annotations.sample]
            fqrs_locations.append(upscaled_fqrs_locations)

    #ploting a sample comparison of unfiltered and filtered signal for last processed record
    if plot_sample == True:
        plot_filter_characteristics(fir_coeff, fs, numtaps, (0,50))
        plot_signals_comparision(signal, filtered_signal, fs)
            
    return records, new_fs, fqrs_locations

def mecq_canceller(record, fs, N):
    '''
    function that performs MECG cancellation for a given record
    '''
    fecg_signals = []
    qrs_peaks, _, __ = qrs_detector(record, fs, 0.6)
    for channel_signal in record:
        mu = average_complex(channel_signal, qrs_peaks, fs, N, True)
        M = windowing(mu, fs)
        fecg = cancel_mecg(channel_signal, qrs_peaks, M, fs)
        fecg_signals.append(fecg)
      
    return fecg_signals



#path to dataset
data_path = r'path\to\data'
data_path = r'C:\Users\mizgi\Desktop\gowno\studia\erasmus\bsp\project\set-a\set-a'

#variables
numtaps = 1001
fs_new = 2000
N = 10 #num of averaged MECG for cancellation
N_av = 100 #not 150, signals too short
min_distance_adult = 0.6 #HR100
min_distance_fetus = 0.34 #HR175
sample_idx = 12
epsilon_ms = 70


records, fs, fqrs_locations = load_ecg_dataset(data_path, numtaps, fs_new, False)




#-----------------------------------------------------------------
#-----------------------------------------------------------------
#--------EXAMPLE IMPLEMENTATION FOR A SAMPLE SIGNAL BELOW---------
#-----------------------------------------------------------------
#-----------------------------------------------------------------

# for creating plots for presentation I mostly use a sample record
# so here is an example of the algorithm aplied to one sample recording
sample_record = records[sample_idx]

#plot mecg qrs detection
qrs_peaks, pca_signal, qrs_template = qrs_detector(sample_record, fs, min_distance_adult)
plot_qrs_detection(sample_record, pca_signal, qrs_peaks, fs)
plot_qrs_template(qrs_template, fs)

#plot fecg qrs detection
sample_fecg = mecq_canceller(sample_record, fs, N)
qrs_peaks, pca_signal, qrs_template = qrs_detector(sample_fecg, fs, min_distance_fetus)
plot_qrs_detection(sample_fecg, pca_signal, qrs_peaks, fs)
plot_qrs_template(qrs_template, fs)
  
# ploting of fecg complex templates
fig, axs = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
for i in range(4):
    fecg_complex = average_complex(sample_fecg[i], qrs_peaks, fs, N_av, False)
    t = np.arange(len(fecg_complex)) / fs
    axs[i].plot(t, fecg_complex, 'k')
    axs[i].axis('off')
plt.show()

# ploting fhr trace
fhr = calculate_fhr_trace(qrs_peaks, fs)
fig=plt.figure(figsize=(10,6), dpi= 100)
t = np.arange(len(fhr)) /2
plt.plot(t, fhr, 'k.')
plt.ylabel('FHR (bpm)')
plt.xlabel('Time [s]')
plt.show()

# ploting piece of record with detected and actual fecg complexes
fqrs_loc = fqrs_locations[sample_idx]
fig=plt.figure(figsize=(10,6), dpi= 100)
plt.plot(sample_fecg[3][:qrs_peaks[20]], 'k', label='Filtered Signal')
plt.plot(qrs_peaks[:20] , sample_fecg[3][qrs_peaks[:20]], 'mx', label='Detected FECG complexes')
plt.plot(fqrs_loc[:20], sample_fecg[3][fqrs_loc[:20]], 'cx', label='Original FECG complexes')
plt.xlabel('Time [samples]')
plt.legend(loc='upper left')
plt.show()


def accuracy_across_dataset(records, fs):
    qrs_peaks_global = []

    for i, record in enumerate(records):
        record_f = mecq_canceller(record, fs, N)
        qrs_peaks, pca_signal, qrs_template = qrs_detector(record_f, fs, min_distance_fetus)
        qrs_peaks_global.append(qrs_peaks)

    epsilon_values = range(10, 151, 10)  # From 10ms to 150ms, with a step of 10ms
    mean_accuracies = []

    for epsilon in epsilon_values:
        accuracies = [calculate_accuracy(qrs_peaks_global[i], fqrs_locations[i], epsilon_ms=epsilon, fs=2000) for i in range(len(qrs_peaks_global))]
        mean_accuracies.append(np.mean(accuracies))

    plt.plot(epsilon_values, mean_accuracies,'k', marker='o')
    plt.xlabel('Epsilon (ms)')
    plt.ylabel('Mean Accuracy (%)')
    plt.title('Mean Accuracy vs. Epsilon')
    plt.grid(True)
    plt.show()

    return mean_accuracies

mean_accuracies = accuracy_across_dataset(records, fs)