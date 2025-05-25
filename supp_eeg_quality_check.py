import os
import mne
import copy
import numpy as np
import pandas as pd
import mne_connectivity
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, coherence, windows
from mne.time_frequency import tfr_morlet


def plot_eeg_psd(processed_eeg_df):
    averaged_eeg = processed_eeg_df.groupby(['teamID', 'human_ai', 'role'])['eeg'].apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index(name='eeg')
    sfreq = 256  # Sampling frequency

    for ha in ['human', 'ai']:
        for role in ['Yaw', 'Pitch']:
            eeg_data = np.array(list(averaged_eeg[(averaged_eeg.human_ai == ha) & (averaged_eeg.role == role)].eeg))
            eeg_data = eeg_data[:, :, :1024]  # shape (17386, 20, 1024)

            # Create an MNE Raw object from the EEG data
            info = mne.create_info(ch_names=channel_locs.ch_names, sfreq=sfreq, ch_types='eeg', verbose=False)
            epochs = mne.EpochsArray(eeg_data, info)
            montage = mne.channels.make_dig_montage(ch_pos=channel_locs.get_positions()['ch_pos'], coord_frame='head')
            epochs.set_montage(montage, verbose=False)

            epochs.compute_psd(fmin=0.5, fmax=30, method='welch',tmin=-2).plot(average=True)
            plt.savefig(f"plots/eeg_plots/eeg_psd_{ha}_{role}.png", dpi=300)
            plt.close()

def plot_time_frequency(processed_eeg_df):
    freqs = np.linspace(0.5, 30, 20)        # frequencies of interest
    n_cycles = freqs / 2                 # number of cycles per frequency
    averaged_eeg = processed_eeg_df.groupby(['teamID', 'human_ai', 'role'])['eeg'].apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index(name='eeg')
    sfreq = 256  # Sampling frequency
    plt.rc('font', size=20) # Sets the default font size to 20
    for ha in ['human', 'ai']:
        for role in ['Yaw', 'Pitch']:
            eeg_data = np.array(list(averaged_eeg[(averaged_eeg.human_ai == ha) & (averaged_eeg.role == role)].eeg))
            eeg_data = eeg_data[:, :, :1024]*1e6  # shape (17386, 20, 1024)

            # Create an MNE Raw object from the EEG data
            info = mne.create_info(ch_names=channel_locs.ch_names, sfreq=sfreq, ch_types='eeg', verbose=False)
            epochs = mne.EpochsArray(eeg_data, info)
            montage = mne.channels.make_dig_montage(ch_pos=channel_locs.get_positions()['ch_pos'], coord_frame='head')
            epochs.set_montage(montage, verbose=False)

            power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False)  # average=False = retain single epochs
            avg_power = power.data.mean(axis=1).mean(axis=0)  # shape: (n_freqs, n_times)

            plt.figure(figsize=(18, 8))
            plt.imshow(avg_power, aspect='auto', origin='lower',
                       extent=[-2, 2, freqs[0], freqs[-1]],
                       cmap='RdBu_r', vmin=-50, vmax=250)
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.colorbar(label="Power")
            plt.tight_layout()
            plt.savefig(f"plots/eeg_plots/eeg_tf_{ha}_{role}.png", dpi=300)
            plt.close()

    
if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    processed_eeg_df = pd.read_pickle('../data/eeg_epochs_both_human_and_alice.pkl')
    channel_locs = mne.channels.read_custom_montage('chan_locs.sfp')
    plot_eeg_psd(processed_eeg_df)
    plot_time_frequency(processed_eeg_df)

