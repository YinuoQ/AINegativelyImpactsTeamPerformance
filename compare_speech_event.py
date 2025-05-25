import os
import glob
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.patches import Ellipse
from scipy.stats import sem, ttest_ind, f_oneway


def read_files():
    pd.set_option('display.max_columns', None)
    # read all human speech event file
    all_human_speech_df = pd.read_pickle('../data/all_human/epoched_data/epoched_raw_speech.pkl')
    # read human ai speech event file
    human_ai_speech_df = pd.read_pickle('../data/human_ai/epoched_data/epoched_raw_speech.pkl')
    return all_human_speech_df, human_ai_speech_df

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def get_number_of_speech_event(input_arr):
    """
    Counts the number of continuous chunks of non-zero numbers in a list.

    Parameters:
    - input_arr: array of numbers

    Returns:
    - int: number of continuous chunks of non-zero numbers
    """
    continuous_chunks = 0
    in_chunk = False
    if len(input_arr) == 2:
        input_arr = input_arr[0]
    # Iterate through the list to count continuous chunks
    for value in input_arr:
        # If we find a non-zero value and we are not already in a chunk, it starts a new chunk
        if value != 0 and not in_chunk:
            continuous_chunks += 1
            in_chunk = True
        # If the current value is zero, it means we are not in a chunk anymore
        elif value == 0:
            in_chunk = False
    return continuous_chunks  

def get_duration_of_speech_event(input_arr):
        return np.sum(input_arr)/60


def plot_ellipse(a, b, a_err, b_err, color, marker, line_style, alpha=0.5, err_color='k', zorder=1):
    plt.errorbar(a, b, xerr=a_err, yerr=b_err, fmt=marker, color=err_color, ls=line_style, ms=max([int(a_err*5), 2]), zorder=zorder+2, linewidth=1, alpha=alpha)
    ellipse = Ellipse(xy=(a, b), width=a_err*2, height=b_err*2,color=color, alpha=alpha, ec=None, zorder=zorder)
    plt.gca().add_patch(ellipse)


def simple_vad(audio, threshold=0.01, close_gap=0.001, sample_rate=11025):
    """
    Simple voice activity detection with additional feature to merge close speech segments.

    :param audio: Audio signal array
    :param threshold: Amplitude threshold for speech detection
    :param close_gap: Time gap in seconds to merge close speech segments
    :param sample_rate: Sample rate of the audio
    :return: Speech detection array
    """
    # Compute the absolute amplitude and detect initial speech frames
    amplitude = audio
    speech_frames = np.where(amplitude > threshold, 1, 0)

    # Number of samples corresponding to the close_gap
    close_gap_samples = int(close_gap * sample_rate)

    # Iterate through detected speech frames and merge close segments
    i = 0
    while i < len(speech_frames):
        if speech_frames[i] == 1:
            j = i + 1
            # Find the end of the current speech segment
            while j < len(speech_frames) and speech_frames[j] == 1:
                j += 1
            # Look ahead to see if the next speech segment is close
            next_speech = j
            while next_speech < len(speech_frames) and next_speech - j <= close_gap_samples:
                if speech_frames[next_speech] == 1:
                    # If close enough, mark all in-between frames as speech
                    speech_frames[i:next_speech+1] = 1
                    break
                next_speech += 1
            # Move to the next segment
            i = next_speech
        else:
            i += 1

    return speech_frames

def change_sample_rate(data):
    resampled_data = signal.resample(data, 240)
    return resampled_data

def filter_short_speech(speech):
    # speech_idx = np.where(speech) == 1
    # non_speech_idx = np.where(speech) == 0
    speech_start_or_end_idx = np.where(np.diff(speech)!= 0)[0]
    each_pair_len = np.diff(speech_start_or_end_idx)
    short_epoch_starts = speech_start_or_end_idx[np.where(each_pair_len < 220)]
    short_epoch_ends = speech_start_or_end_idx[np.where(each_pair_len < 220)[0]+1]
    for i in zip(short_epoch_starts, short_epoch_ends):
        speech[i[0]:i[1]] = [speech[i[0]]]*np.diff(i)[0]
    return speech

def down_sample_speech_event(speech_event):
    if sum(speech_event) == 0:
        # no speech in this epoch
        down_sampled_speech_event = np.zeros(240)
    elif sum(speech_event) == 11025*4:
        # talk thorugh whole epoch
        down_sampled_speech_event = np.ones(240)
    else:

        # talk some time during the epoch
        speech_event_new = filter_short_speech(speech_event)
        resampled_data = change_sample_rate(speech_event_new)
        resampled_data[resampled_data < 0.2] = 0
        resampled_data[resampled_data != 0] = 1
        down_sampled_speech_event = resampled_data
    
    return down_sampled_speech_event


def process_speech_event_from_raw_speech_epochs(speech_df):
    speech_event_lst = []
    # Parameters for the band-pass filter
    lowcut = 300  # Low frequency edge of the band
    highcut = 3400 # High frequency edge of the band
    sr = 11025

    if speech_df.human_ai.iloc[0] == 'human':
        valid_speech_df = speech_df.drop(speech_df[(speech_df.teamID == 'T2') & (speech_df.sessionID == 'S2') & (speech_df.role == 'Thrust')].index)
        valid_speech_df = valid_speech_df.drop(valid_speech_df[(valid_speech_df.teamID == 'T4') & (valid_speech_df.sessionID == 'S1') & (valid_speech_df.role == 'Pitch')].index)
        valid_speech_df = valid_speech_df.drop(valid_speech_df[(valid_speech_df.teamID == 'T7') & (valid_speech_df.sessionID == 'S1') & (valid_speech_df.role == 'Pitch')].index)
        valid_speech_df = valid_speech_df.drop(valid_speech_df[(valid_speech_df.teamID == 'T8') & (valid_speech_df.sessionID == 'S3') & (valid_speech_df.role == 'Pitch')].index)
        valid_speech_df = valid_speech_df.drop(valid_speech_df[(valid_speech_df.teamID == 'T8') & (valid_speech_df.sessionID == 'S3') & (valid_speech_df.role == 'Thrust')].index)
        valid_speech_df = valid_speech_df.reset_index(drop=True)
    elif speech_df.human_ai.iloc[0] == 'ai':
        valid_speech_df = speech_df.drop(speech_df[(speech_df.teamID == 'T3') & (speech_df.sessionID == 'S2') & (speech_df.role == 'Yaw')].index)
        valid_speech_df = valid_speech_df.drop(valid_speech_df[(valid_speech_df.teamID == 'T6') & (valid_speech_df.sessionID == 'S2') & (valid_speech_df.role == 'Pitch')].index)
        valid_speech_df = valid_speech_df.reset_index(drop=True)
        
    for i in tqdm(range(len(valid_speech_df))):
        if valid_speech_df.iloc[i].communication != 'Word' and valid_speech_df.iloc[i].communication != 'Free':
            speech_event_lst.append(None)
            continue
        
        else:
            temp_raw_speech = valid_speech_df.iloc[i].speech
            if temp_raw_speech is None or len(temp_raw_speech) == 0:
                speech_event_lst.append(None)
                continue
            # Apply the band-pass filter
            if valid_speech_df.human_ai.iloc[i] == 'ai' and valid_speech_df.iloc[i].role == 'Thrust':
                filtered_audio = bandpass_filter(temp_raw_speech, 100, 3500, sr, order=6)
                speech_event = simple_vad(filtered_audio, threshold=200, close_gap=0.2, sample_rate=11025)

            else:
                filtered_audio = bandpass_filter(temp_raw_speech, lowcut, highcut, sr, order=6)
                speech_event = simple_vad(filtered_audio, threshold=80, close_gap=0.2, sample_rate=11025)
            down_sampled_speech_event = down_sample_speech_event(speech_event)
            speech_event_lst.append(down_sampled_speech_event)
    valid_speech_df['speech_event'] = speech_event_lst

    return valid_speech_df


def plot_communication_across_all_conditions(speech_event_df):
    a = speech_event_df.groupby(['human_ai', 'teamID']).apply(lambda x: x.speech_frequency.mean())
    b = speech_event_df.groupby(['human_ai', 'teamID']).apply(lambda x: x.speech_duration.mean())

    freq_dur_df = pd.DataFrame(np.array(list(a.keys())), columns=a.keys().names)
    freq_dur_df['freq'] = a.values
    freq_dur_df['dura'] = b.values
    plt.figure(figsize=(12,6))

    a = freq_dur_df[freq_dur_df.human_ai == 'human'].freq.mean()
    b = freq_dur_df[freq_dur_df.human_ai == 'human'].dura.mean()
    a_err = np.nanstd(freq_dur_df[freq_dur_df.human_ai == 'human'].freq) / np.sqrt(len(freq_dur_df[freq_dur_df.human_ai == 'human'].freq))
    b_err = np.nanstd(freq_dur_df[freq_dur_df.human_ai == 'human'].dura) / np.sqrt(len(freq_dur_df[freq_dur_df.human_ai == 'human'].dura))
    color = '#F78474'
    plot_ellipse(a, b, a_err, b_err, color, 'o', '-', alpha=1)

    a = freq_dur_df[freq_dur_df.human_ai == 'ai'].freq.mean()
    b = freq_dur_df[freq_dur_df.human_ai == 'ai'].dura.mean()
    a_err = np.nanstd(freq_dur_df[freq_dur_df.human_ai == 'ai'].freq) / np.sqrt(len(freq_dur_df[freq_dur_df.human_ai == 'ai'].freq))
    b_err = np.nanstd(freq_dur_df[freq_dur_df.human_ai == 'ai'].dura) / np.sqrt(len(freq_dur_df[freq_dur_df.human_ai == 'ai'].dura))
    color = '#57A0D3'
    plot_ellipse(a, b, a_err, b_err, color, 'o', '-', alpha=1)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(['Human-only', 'Human-AI'], fontsize=12)
    plt.xlim(0.25,0.75)
    plt.ylim(0.05,0.45)
    plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    plt.yticks([0.1, 0.2, 0.3, 0.4])
    plt.savefig('plots/comm_all.png', dpi=300)
    print('frequency')
    print(ttest_ind(freq_dur_df[freq_dur_df.human_ai == 'human'].freq, freq_dur_df[freq_dur_df.human_ai == 'ai'].freq))
    print('duration')
    print(ttest_ind(freq_dur_df[freq_dur_df.human_ai == 'human'].dura, freq_dur_df[freq_dur_df.human_ai == 'ai'].dura))

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    # get speech event from raw speech files and save as speech event for each epoch.
    # running the section below takes some time
    # run the section below if you don't have "speech_epochs_both_human_and_alice.pkl" in data folder
    
    ##############################################################################################################
    '''
    # epoched speech event analzie
    all_human_speech_df, human_ai_speech_df = read_files()

    all_human_speech_df['human_ai'] = 'human'
    human_ai_speech_df['human_ai'] = 'ai'

    valid_speech_human = process_speech_event_from_raw_speech_epochs(all_human_speech_df)
    valid_speech_ai = process_speech_event_from_raw_speech_epochs(human_ai_speech_df)
    speech_event_df = pd.concat([valid_speech_human.dropna(), valid_speech_ai.dropna()]).reset_index(drop=True)
    speech_event_df = speech_event_df.drop(columns='speech')
    
    speech_event_df['speech_frequency'] = 0
    speech_event_df['speech_duration'] = 0
    for i in tqdm(range(len(speech_event_df))):
        speech_event_df.iat[i,-2] = get_number_of_speech_event(speech_event_df.iloc[i].speech_event)
        speech_event_df.iat[i,-1] = get_duration_of_speech_event(speech_event_df.iloc[i].speech_event)
    
    speech_event_df.to_pickle('../data/speech_epochs_both_human_and_alice.pkl')
    '''
    ##############################################################################################################

    speech_event_df = pd.read_pickle('../data/speech_epochs_both_human_and_alice.pkl')
    plot_communication_across_all_conditions(speech_event_df)

