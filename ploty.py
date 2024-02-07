import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

def plot_filter_characteristics(b, fs, numtaps, f_lim = None):
    '''
    function that plots characteristics of FIR filter used for baseline wandering
    '''
    if f_lim == None:
        f_lim = (0, fs/2)

    f = np.linspace(0, fs/2, numtaps)

    # Calculate frequency response
    w, h = freqz(b, worN=f, fs=fs)
    amplitude_response = 20 * np.log10(abs(h))
    phase_response = np.degrees(np.unwrap(np.angle(h)))
    #phase in degrees, not radians

    # Plotting
    fig, axs = plt.subplots(1,2, figsize=(10,6))

    # Amplitude response
    axs[0].plot(w, amplitude_response, 'k')
    axs[0].set_title('Amplitude Response')
    axs[0].set_ylabel('Amplitude [dB]')
    axs[0].set_xlabel('Frequency [Hz]')
    axs[0].set_xlim(f_lim)
    axs[0].grid(True)

    # Phase response
    axs[1].plot(w, phase_response, 'k')
    axs[1].set_title('Phase Response')
    axs[1].set_ylabel('Phase [radians]')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_xlim(f_lim)
    axs[1].grid(True)


    plt.tight_layout()
    plt.show()

def plot_signals_comparision(signal, filtered_signal, fs):
    '''
    function that plots the comparison of a signal
    before and after filtering
    '''
    plt.figure(figsize=(12, 6))
    start = 20000
    stop = start + 10000
    t = np.arange(signal[0, start:stop].size) / fs
    plt.plot(t, signal[0, start:stop], 'm', label='Original Signal')
    plt.plot(t, filtered_signal[0, start:stop], 'k', label='Filtered Signal')
    plt.title('Comparison of Unfiltered and Filtered Signal for Patient a01')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude [μV]')
    plt.xlim((0,10))
    plt.legend()
    plt.grid()
    plt.show()


def plot_qrs_detection(original_signals, enhanced_signal, qrs_peaks, fs):
    '''
    function that plots the qrs detection in a signal
    '''
    start_sample = int(3 * fs)
    end_sample = int(6 * fs)

    fig, axes = plt.subplots(len(original_signals) + 1, 1, figsize=(6,8), sharex=True)
    time_vector = np.arange(start_sample, end_sample) / fs
    qrs_peaks_in_range = qrs_peaks[(qrs_peaks >= start_sample) & (qrs_peaks < end_sample)]

    for i, lead_signal in enumerate(original_signals):
        axes[i].plot(time_vector, lead_signal[start_sample:end_sample], 'k', label=f'Lead {i+1}')
        axes[i].axis('off')

    axes[-1].plot(time_vector, enhanced_signal[start_sample:end_sample], 'k', label='Enhanced Signal')
    axes[-1].plot(qrs_peaks_in_range / fs, enhanced_signal[qrs_peaks_in_range], 'mx', label='Detected QRS Peaks')
    axes[-1].axis('off')

    plt.xlabel('Time (seconds)')
    plt.tight_layout()
    plt.show()

def plot_qrs_template(template, fs):
    '''
    function that plots the template of the qrs complex for a given signal
    '''
    plt.figure(figsize=(6, 4))
    t = np.arange(len(template)) / fs
    plt.plot(t, template, 'k')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.show()



def save_plots_original_signals(ecg_directory, end_path):
    print('Preparing plots for original signals')
    #list of unique patient IDs based on the .hea files
    patient_ids = [f.split('.')[0] for f in os.listdir(ecg_directory) if f.endswith('.hea')]

    for patient_id in patient_ids:

        record_path = os.path.join(ecg_directory, patient_id)
        record = wfdb.rdrecord(record_path)
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f'Original signal for patient {patient_id}')

        for i in range(4):
            axs[i].plot(record.p_signal[:, i], 'k')
            axs[i].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_file_path = os.path.join(end_path, f'Patient_{patient_id}.png')
        plt.savefig(output_file_path)
        plt.close(fig)

def save_plots_filtered_signals(records, end_path):
    print('Preparing plots for filtered signals')
    for n in range(len(records)):
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex='all')
        fig.suptitle(f'Filtered signal for patient a{n+1:02d}')
        for i in range(4):
            axs[i].plot(records[n][i], 'k')
            axs[i].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(end_path, f'Filtered_Patient_a{n+1:02d}.png'))
        plt.close(fig)



