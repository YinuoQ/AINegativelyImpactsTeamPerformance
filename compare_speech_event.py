import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from statistical_test import t_test_two_groups


def read_speech_files():
    """
    Load raw speech data from pickle files for both human-only and human-AI teams.
    
    Returns:
        tuple: A tuple containing:
            - all_human_speech_df (pd.DataFrame): Raw speech data from human-only teams
            - human_ai_speech_df (pd.DataFrame): Raw speech data from human-AI teams
    """
    # Configure pandas to display all columns for better debugging
    pd.set_option('display.max_columns', None)
    
    # Load epoched raw speech data
    all_human_speech_df = pd.read_pickle('../physiological_behavioral_results/data/all_human/epoched_data/epoched_raw_speech.pkl')
    human_ai_speech_df = pd.read_pickle('../physiological_behavioral_results/data/human_ai/epoched_data/epoched_raw_speech.pkl')
    
    return all_human_speech_df, human_ai_speech_df


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to audio data to isolate speech frequencies.
    
    This function removes frequencies outside the typical speech range to improve
    voice activity detection accuracy.
    
    Parameters:
        data (np.array): Input audio signal
        lowcut (float): Low frequency cutoff in Hz
        highcut (float): High frequency cutoff in Hz
        fs (int): Sampling frequency in Hz
        order (int): Filter order (default: 5)
        
    Returns:
        np.array: Filtered audio signal
    """
    # Calculate normalized frequencies
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design and apply Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    
    return y


def get_number_of_speech_events(input_arr):
    """
    Count the number of continuous chunks of speech events in an array.
    
    This function identifies sequences of consecutive speech activity periods
    and counts how many such sequences exist in the recording.
    
    Parameters:
        input_arr (array-like): Array of speech activity indicators (0=no speech, 1=speech)
        
    Returns:
        int: Number of continuous speech event chunks
        
    Example:
        [0, 1, 1, 0, 1, 0, 1, 1, 1] -> 3 chunks: [1,1], [1], [1,1,1]
    """
    continuous_chunks = 0
    in_chunk = False
    
    # Handle special case where input_arr is nested
    if len(input_arr) == 2:
        input_arr = input_arr[0]
    
    # Iterate through the array to identify and count continuous speech chunks
    for value in input_arr:
        # Start of a new chunk: speech detected when not already in a chunk
        if value != 0 and not in_chunk:
            continuous_chunks += 1
            in_chunk = True
        # End of current chunk: no speech detected
        elif value == 0:
            in_chunk = False
    
    return continuous_chunks


def get_duration_of_speech_events(input_arr):
    """
    Calculate the total duration of speech events in an array.
    
    Parameters:
        input_arr (array-like): Array of speech activity indicators
        
    Returns:
        float: Total speech duration in seconds (assuming 60 samples per second)
    """
    return np.sum(input_arr) / 60


def simple_voice_activity_detection(audio, threshold=0.01, close_gap=0.001, sample_rate=11025):
    """
    Simple voice activity detection with gap closing to merge nearby speech segments.
    
    This function detects speech activity based on amplitude threshold and merges
    speech segments that are separated by short gaps to reduce fragmentation.
    
    Parameters:
        audio (np.array): Audio signal array
        threshold (float): Amplitude threshold for speech detection
        close_gap (float): Time gap in seconds to merge close speech segments
        sample_rate (int): Sample rate of the audio
        
    Returns:
        np.array: Binary array indicating speech activity (1=speech, 0=silence)
    """
    # Detect initial speech frames based on amplitude threshold
    amplitude = audio
    speech_frames = np.where(amplitude > threshold, 1, 0)

    # Calculate number of samples corresponding to the close_gap
    close_gap_samples = int(close_gap * sample_rate)

    # Iterate through detected speech frames and merge close segments
    i = 0
    while i < len(speech_frames):
        if speech_frames[i] == 1:
            j = i + 1
            # Find the end of the current speech segment
            while j < len(speech_frames) and speech_frames[j] == 1:
                j += 1
            
            # Look ahead to see if the next speech segment is close enough to merge
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


def resample_speech_data(data):
    """
    Resample speech data to 240 samples to match epoch structure.
    
    Parameters:
        data (np.array): Input speech data
        
    Returns:
        np.array: Resampled data with 240 samples
    """
    resampled_data = signal.resample(data, 240)
    return resampled_data


def filter_short_speech_segments(speech):
    """
    Filter out very short speech or silence segments to reduce noise.
    
    This function identifies segments shorter than 220 samples and fills them
    with the surrounding context to create more stable speech detection.
    
    Parameters:
        speech (np.array): Binary speech activity array
        
    Returns:
        np.array: Filtered speech activity array
    """
    # Find indices where speech activity changes
    speech_change_indices = np.where(np.diff(speech) != 0)[0]
    
    # Calculate length of each segment
    segment_lengths = np.diff(speech_change_indices)
    
    # Identify short segments (< 220 samples)
    short_segment_mask = segment_lengths < 220
    short_segment_starts = speech_change_indices[short_segment_mask]
    short_segment_ends = speech_change_indices[np.where(short_segment_mask)[0] + 1]
    
    # Fill short segments with surrounding context
    for start, end in zip(short_segment_starts, short_segment_ends):
        speech[start:end] = speech[start]
    
    return speech


def downsample_speech_events(speech_event):
    """
    Downsample speech events from original sampling rate to 240 samples per epoch.
    
    This function handles different cases: no speech, continuous speech, or
    intermittent speech during the epoch.
    
    Parameters:
        speech_event (np.array): High-resolution speech activity array
        
    Returns:
        np.array: Downsampled speech activity array (240 samples)
    """
    if sum(speech_event) == 0:
        # No speech detected in this epoch
        downsampled_speech = np.zeros(240)
    elif sum(speech_event) == 11025 * 4:
        # Continuous speech throughout the entire epoch
        downsampled_speech = np.ones(240)
    else:
        # Intermittent speech during the epoch
        # Filter short segments and downsample
        filtered_speech = filter_short_speech_segments(speech_event)
        resampled_data = resample_speech_data(filtered_speech)
        
        # Binarize the resampled data
        resampled_data[resampled_data < 0.2] = 0
        resampled_data[resampled_data != 0] = 1
        downsampled_speech = resampled_data
    
    return downsampled_speech


def process_speech_events_from_raw_audio(speech_df):
    """
    Process raw speech data to extract speech events for each epoch.
    
    This function applies bandpass filtering, voice activity detection, and
    downsampling to convert raw audio into binary speech event arrays.
    
    Parameters:
        speech_df (pd.DataFrame): DataFrame containing raw speech data
        
    Returns:
        pd.DataFrame: DataFrame with added 'speech_event' column containing processed speech events
    """
    speech_event_list = []
    
    # Audio processing parameters
    lowcut = 300   # Low frequency edge for speech band
    highcut = 3400 # High frequency edge for speech band
    sample_rate = 11025
    
    # Remove problematic recordings based on team type
    if speech_df.human_ai.iloc[0] == 'human':
        # Remove known problematic human team recordings
        problematic_sessions = [
            ('T2', 'S2', 'Thrust'),
            ('T4', 'S1', 'Pitch'),
            ('T7', 'S1', 'Pitch'),
            ('T8', 'S3', 'Pitch'),
            ('T8', 'S3', 'Thrust')
        ]
        
        valid_speech_df = speech_df.copy()
        for team, session, role in problematic_sessions:
            mask = (valid_speech_df.teamID == team) & \
                   (valid_speech_df.sessionID == session) & \
                   (valid_speech_df.role == role)
            valid_speech_df = valid_speech_df.drop(valid_speech_df[mask].index)
        
    elif speech_df.human_ai.iloc[0] == 'ai':
        # Remove known problematic AI team recordings
        problematic_sessions = [
            ('T3', 'S2', 'Yaw'),
            ('T6', 'S2', 'Pitch')
        ]
        
        valid_speech_df = speech_df.copy()
        for team, session, role in problematic_sessions:
            mask = (valid_speech_df.teamID == team) & \
                   (valid_speech_df.sessionID == session) & \
                   (valid_speech_df.role == role)
            valid_speech_df = valid_speech_df.drop(valid_speech_df[mask].index)
    
    valid_speech_df = valid_speech_df.reset_index(drop=True)
    
    # Process each speech recording
    for i in tqdm(range(len(valid_speech_df)), desc="Processing speech events"):
        # Skip non-communication conditions
        if valid_speech_df.iloc[i].communication not in ['Word', 'Free']:
            speech_event_list.append(None)
            continue
        
        # Get raw speech data
        raw_speech = valid_speech_df.iloc[i].speech
        if raw_speech is None or len(raw_speech) == 0:
            speech_event_list.append(None)
            continue
        
        # Apply different processing parameters based on team type and role
        if (valid_speech_df.human_ai.iloc[i] == 'ai' and 
            valid_speech_df.iloc[i].role == 'Thrust'):
            # Special parameters for AI thrust pilot
            filtered_audio = bandpass_filter(raw_speech, 100, 3500, sample_rate, order=6)
            speech_events = simple_voice_activity_detection(
                filtered_audio, threshold=200, close_gap=0.2, sample_rate=sample_rate
            )
        else:
            # Standard parameters for other recordings
            filtered_audio = bandpass_filter(raw_speech, lowcut, highcut, sample_rate, order=6)
            speech_events = simple_voice_activity_detection(
                filtered_audio, threshold=80, close_gap=0.2, sample_rate=sample_rate
            )
        
        # Downsample to match epoch structure
        downsampled_events = downsample_speech_events(speech_events)
        speech_event_list.append(downsampled_events)
    
    # Add processed speech events to DataFrame
    valid_speech_df['speech_event'] = speech_event_list
    
    return valid_speech_df


def plot_error_ellipse(x, y, x_err, y_err, color, marker='o', alpha=0.7):
    """
    Plot error ellipse with center point and error bars for 2D data visualization.
    
    Parameters:
        x, y (float): Center coordinates
        x_err, y_err (float): Standard errors in x and y directions
        color (str): Color for the ellipse and marker
        marker (str): Marker style for center point
        alpha (float): Transparency level
    """
    # Plot error bars and center point
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt=marker, 
                color='k', linestyle='-', markersize=max([int(x_err*10), 3]), 
                linewidth=1, alpha=alpha)
    
    # Add error ellipse
    ellipse = Ellipse(xy=(x, y), width=x_err*2, height=y_err*2,
                     color=color, alpha=alpha, edgecolor=None)
    plt.gca().add_patch(ellipse)


def plot_communication_comparison(speech_event_df):
    """
    Create scatter plot comparing speech frequency vs duration between team types.
    
    This function creates a 2D visualization showing the relationship between
    speech frequency and duration, with error ellipses representing variability.
    
    Parameters:
        speech_event_df (pd.DataFrame): DataFrame containing speech event data with
                                      ['human_ai', 'teamID', 'speech_frequency', 'speech_duration']
    """
    # Calculate team-level means for frequency and duration
    team_freq_means = speech_event_df.groupby(['human_ai', 'teamID'])['speech_frequency'].mean()
    team_duration_means = speech_event_df.groupby(['human_ai', 'teamID'])['speech_duration'].mean()

    # Create DataFrame combining frequency and duration data
    freq_dur_df = pd.DataFrame({
        'human_ai': [key[0] for key in team_freq_means.index],
        'teamID': [key[1] for key in team_freq_means.index],
        'freq': team_freq_means.values,
        'duration': team_duration_means.values
    })

    # Create the visualization
    plt.figure(figsize=(8, 6))

    # Plot human-only teams
    human_data = freq_dur_df[freq_dur_df.human_ai == 'human']
    human_freq_mean = human_data.freq.mean()
    human_dur_mean = human_data.duration.mean()
    human_freq_sem = human_data.freq.std() / np.sqrt(len(human_data))
    human_dur_sem = human_data.duration.std() / np.sqrt(len(human_data))
    
    plot_error_ellipse(human_freq_mean, human_dur_mean, human_freq_sem, human_dur_sem, 
                      '#F78474', alpha=1.0)

    # Plot human-AI teams
    ai_data = freq_dur_df[freq_dur_df.human_ai == 'ai']
    ai_freq_mean = ai_data.freq.mean()
    ai_dur_mean = ai_data.duration.mean()
    ai_freq_sem = ai_data.freq.std() / np.sqrt(len(ai_data))
    ai_dur_sem = ai_data.duration.std() / np.sqrt(len(ai_data))
    
    plot_error_ellipse(ai_freq_mean, ai_dur_mean, ai_freq_sem, ai_dur_sem, 
                      '#57A0D3', alpha=1.0)

    # Customize plot appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(['Human-only', 'Human-AI'], fontsize=12, loc='upper right')
    plt.xlim(0.25, 0.75)
    plt.ylim(0.05, 0.45)
    plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    plt.yticks([0.1, 0.2, 0.3, 0.4])
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Duration', fontsize=12)

    # Save plot
    plt.savefig('plots/comm_all.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Perform statistical comparisons
    print("Communication Analysis - Statistical Comparisons:")
    print("=" * 55)
    
    # Compare speech frequency between team types
    t_test_two_groups(human_data.freq, ai_data.freq, 'Human-only', 'Human-AI',
                     comparison_title="Speech Frequency")
    print("=" * 55)
    
    # Compare speech duration between team types
    t_test_two_groups(human_data.duration, ai_data.duration, 'Human-only', 'Human-AI',
                     comparison_title="Speech Duration")


def load_or_process_speech_data():
    """
    Load processed speech data or create it from raw audio if needed.
    
    Returns:
        pd.DataFrame: DataFrame containing processed speech event data
    """
    try:
        # Try to load pre-processed speech data
        print("Loading processed speech event data...")
        speech_event_df = pd.read_pickle('../physiological_behavioral_results/data/speech_epochs_both_human_and_alice.pkl')
        print(f"Loaded {len(speech_event_df)} speech event records")
        return speech_event_df
        
    except FileNotFoundError:
        print("Processed speech data not found. Processing raw audio files...")
        print("This may take several minutes...")
        
        # Load raw speech data
        all_human_speech_df, human_ai_speech_df = read_speech_files()
        
        # Add team type labels
        all_human_speech_df['human_ai'] = 'human'
        human_ai_speech_df['human_ai'] = 'ai'
        
        # Process speech events
        print("Processing human-only team speech...")
        valid_speech_human = process_speech_events_from_raw_audio(all_human_speech_df)
        
        print("Processing human-AI team speech...")
        valid_speech_ai = process_speech_events_from_raw_audio(human_ai_speech_df)
        
        # Combine and clean data
        speech_event_df = pd.concat([
            valid_speech_human.dropna(), 
            valid_speech_ai.dropna()
        ]).reset_index(drop=True)
        
        # Remove raw speech column to save space
        speech_event_df = speech_event_df.drop(columns='speech', errors='ignore')
        
        # Calculate speech metrics
        print("Calculating speech frequency and duration metrics...")
        speech_event_df['speech_frequency'] = 0
        speech_event_df['speech_duration'] = 0
        
        for i in tqdm(range(len(speech_event_df)), desc="Computing speech metrics"):
            if speech_event_df.iloc[i].speech_event is not None:
                speech_event_df.iat[i, -2] = get_number_of_speech_events(
                    speech_event_df.iloc[i].speech_event
                )
                speech_event_df.iat[i, -1] = get_duration_of_speech_events(
                    speech_event_df.iloc[i].speech_event
                )
        
        # Save processed data for future use
        speech_event_df.to_pickle('../physiological_behavioral_results/data/speech_epochs_both_human_and_alice.pkl')
        print("Processed speech data saved for future use")
        
        return speech_event_df


def main():
    """
    Main execution function that orchestrates the speech analysis workflow.
    """
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    
    # Load or process speech event data
    speech_event_df = load_or_process_speech_data()
    
    print("\n" + "="*80)
    
    # Generate communication analysis
    print("Analyzing communication patterns...")
    plot_communication_comparison(speech_event_df)
    

if __name__ == '__main__':
    main()