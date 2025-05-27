import mne
import copy
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import mne_connectivity
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM 
from scipy.stats import f_oneway, ttest_ind, ttest_rel
from scipy.signal import hilbert, butter, filtfilt, coherence, windows
from statistical_test import t_test_two_groups, repeated_measure_ANOVA


def baseline_TI(preprocessed_eeg_df, fs=256, duration_sec=2):
    """
    Compute baseline Total Interdependence (TI) over the first duration_sec seconds for each team/session pair.
    
    This function calculates TI values during the baseline period to establish a reference point
    for comparing task-related neural synchronization changes.
    
    Parameters:
        preprocessed_eeg_df (pd.DataFrame): DataFrame with columns ['teamID', 'sessionID', 'role', 'processed_eeg']
        fs (int): Sampling frequency in Hz (default: 256)
        duration_sec (int): Time window for baseline calculation in seconds (default: 2)

    Returns:
        pd.DataFrame: DataFrame with columns ['teamID', 'sessionID', 'baseline_TI']
    """
    baseline_records = []
    n_samples = int(fs * duration_sec)

    # Group by team and session to process each pair
    grouped = preprocessed_eeg_df.groupby(['teamID', 'sessionID'])

    for (teamID, sessionID), group in tqdm(grouped, desc="Computing baseline TI for teams/sessions"):
        # Need exactly 2 participants (Pitch and Yaw roles)
        if len(group) != 2:
            continue
        
        # Extract EEG data for both participants during baseline period
        eeg1 = np.array(group.iloc[0]['processed_eeg'])[:, :n_samples]
        eeg2 = np.array(group.iloc[1]['processed_eeg'])[:, :n_samples]

        # Skip if insufficient data
        if eeg1.shape[1] < n_samples or eeg2.shape[1] < n_samples:
            continue

        # Compute TI between the two participants
        ti_value = compute_channel_total_interdependence(eeg1, eeg2)

        baseline_records.append({
            'teamID': teamID,
            'sessionID': sessionID,
            'baseline_TI': ti_value
        })

    return pd.DataFrame(baseline_records)


def compute_baseline_TI(duration_sec=2):
    """
    Load preprocessed EEG data and compute baseline TI for both human-AI and human-only teams.
    
    Parameters:
        duration_sec (int): Duration in seconds for baseline calculation
        
    Returns:
        pd.DataFrame: Combined baseline TI data with human_ai labels
    """
    # Load and process human-AI team data
    preprocessed_eeg_df = pd.read_pickle('../physiological_behavioral_results/data/human_ai/preprocessed_eeg.pkl')
    human_ai_baseline = baseline_TI(preprocessed_eeg_df, duration_sec=duration_sec)
    
    # Load and process human-only team data
    preprocessed_eeg_df = pd.read_pickle('../physiological_behavioral_results/data/all_human/preprocessed_eeg.pkl')
    all_human_baseline = baseline_TI(preprocessed_eeg_df, duration_sec=duration_sec)
    
    # Add team type labels
    human_ai_baseline['human_ai'] = 'ai'
    all_human_baseline['human_ai'] = 'human'
    
    # Combine datasets
    baseline_ti = pd.concat([all_human_baseline, human_ai_baseline]).reset_index(drop=True)
    
    return baseline_ti


def total_interdependence_broadband(signal_1, signal_2, fs=256, fmin=0.5, fmax=30, nperseg=512):
    """
    Compute Total Interdependence (TI) between two signals over a specified frequency range.
    
    TI quantifies the degree of neural synchronization between two EEG signals by measuring
    coherence across frequency bands. Higher TI values indicate stronger synchronization.

    Parameters:
        signal_1, signal_2 (np.array): EEG signals as 1D numpy arrays
        fs (int): Sampling frequency in Hz (default: 256)
        fmin (float): Minimum frequency to include in Hz (default: 0.5 to exclude slow drift)
        fmax (float): Maximum frequency to include in Hz (default: 30)
        nperseg (int): Segment length for Welch method (default: 512)

    Returns:
        float: Total interdependence value (scalar)
    """
    # Compute coherence between the two signals
    f, C2 = coherence(signal_1, signal_2, fs=fs, nperseg=nperseg, detrend=False)
    
    # Set maximum frequency to Nyquist if not specified
    if fmax is None:
        fmax = fs / 2

    # Apply frequency mask to select desired frequency range
    freq_mask = (f >= fmin) & (f <= fmax)
    C2_selected = C2[freq_mask]
    f_selected = f[freq_mask]

    # Check if sufficient frequency points are available
    M = len(f_selected)
    if M < 2:
        return np.nan

    # Calculate frequency resolution
    delta_f = fs / (2 * (M - 1))

    # Compute Total Interdependence using the formula: TI = -(2/fs) * sum(log(1-C2) * delta_f)
    with np.errstate(divide='ignore', invalid='ignore'):
        TI = -(2 / fs) * np.sum(np.log(1 - C2_selected) * delta_f)

    return TI


def compute_channel_total_interdependence(eeg_data_1, eeg_data_2):
    """
    Compute Total Interdependence averaged across all matching channels between two EEG arrays.
    
    This function calculates TI for each channel pair and returns the average,
    providing a single measure of overall neural synchronization between participants.

    Parameters:
        eeg_data_1 (np.array): EEG array [20 channels, T samples] from participant 1
        eeg_data_2 (np.array): EEG array [20 channels, T samples] from participant 2

    Returns:
        float: Average TI across all matching channel pairs
    """
    ti_values = []

    # Calculate TI for each channel pair
    for ch in range(20):
        eeg1 = eeg_data_1[ch]
        eeg2 = eeg_data_2[ch]

        # Compute TI for this channel pair
        ti = total_interdependence_broadband(eeg1, eeg2)
        ti_values.append(ti)

    # Return average TI across all channels
    return np.nanmean(ti_values)


def compute_ring_TI(eeg_df, baseline_ti):
    """
    Compute TI values for each ring event across teams, sessions, trials, and roles.
    
    This function processes all ring events and calculates both absolute TI and
    TI difference from baseline for each event.

    Parameters:
        eeg_df (pd.DataFrame): EEG data with event information
        baseline_ti (pd.DataFrame): Baseline TI values for normalization
        
    Returns:
        pd.DataFrame: DataFrame with TI values for each ring event
    """
    # Create unique combinations of team, session, trial, and ring events
    team_session_trial_ring = eeg_df[
        ['teamID', 'sessionID', 'trialID', 'ringID', 'human_ai', 'communication', 'difficulty']
    ].drop_duplicates().reset_index(drop=True)
    
    # Initialize columns for TI measurements
    team_session_trial_ring['TI'] = None
    team_session_trial_ring['TI_diff'] = None

    # Process each ring event
    for i in tqdm(range(len(team_session_trial_ring)), leave=False, desc="Computing ring TI"):
        row = team_session_trial_ring.iloc[i]
        
        # Extract EEG data for current event
        temp_eeg = eeg_df.query(
            f"teamID == '{row.teamID}' and "
            f"sessionID == '{row.sessionID}' and "
            f"trialID == {row.trialID} and "
            f"ringID == {row.ringID} and "
            f"human_ai == '{row.human_ai}'"
        ).eeg
        
        # Extract corresponding baseline TI
        temp_baseline = baseline_ti.query(
            f"teamID == '{row.teamID}' and "
            f"sessionID == '{row.sessionID}' and "
            f"human_ai == '{row.human_ai}'"
        ).baseline_TI

        # Compute TI if both participants' EEG data are available
        if len(temp_eeg) == 2:
            eeg_data_1 = temp_eeg.iloc[0]
            eeg_data_2 = temp_eeg.iloc[1]
            
            # Calculate TI (excluding last sample to avoid edge effects)
            ti_val = compute_channel_total_interdependence(eeg_data_1[:, :-1], eeg_data_2[:, :-1])
            
            # Store absolute TI and difference from baseline
            team_session_trial_ring.at[i, 'TI'] = ti_val
            team_session_trial_ring.at[i, 'TI_diff'] = ti_val - temp_baseline.iloc[0]
    
    # Return only rows with valid TI calculations
    return team_session_trial_ring.dropna().reset_index(drop=True)


def plot_TI_across_all_condition(TI_df):
    """
    Create violin plot comparing TI differences between human-only and human-AI teams.
    
    This function visualizes the overall difference in neural synchronization between
    team types and performs statistical comparison.
    
    Parameters:
        TI_df (pd.DataFrame): DataFrame containing TI measurements for all conditions
    """
    human_ai_lst = []
    conditions = ['Human-only', 'Human-AI']
    
    # Calculate team-level mean TI differences for each condition
    for human_ai in ['human', 'ai']:
        team_mean_lst = []
        
        # Get unique team IDs for current condition
        unique_teams = TI_df[TI_df['human_ai'] == human_ai].teamID.unique()
        
        for teamid in unique_teams:
            # Calculate mean TI_diff for this team
            team_data = TI_df[
                (TI_df['human_ai'] == human_ai) & 
                (TI_df['teamID'] == teamid)
            ]['TI_diff'].dropna()
            
            if len(team_data) > 0:
                team_mean_lst.append(np.nanmean(team_data))
        
        human_ai_lst.append(team_mean_lst)

    # Create tidy DataFrame for visualization
    df_plot = pd.concat([
        pd.DataFrame({'Condition': conditions[0], 'TI_diff': human_ai_lst[0]}),
        pd.DataFrame({'Condition': conditions[1], 'TI_diff': human_ai_lst[1]})
    ])
    
    # Set color palette matching your standard
    colors = ['#F78474', '#57A0D3']

    # Create violin plot
    plt.figure(figsize=(4, 4))
    sns.violinplot(
        data=df_plot, 
        x='Condition', 
        y='TI_diff',
        palette=colors, 
        inner='box', 
        saturation=1, 
        linewidth=0.5
    )

    # Customize plot appearance to match your standards
    plt.ylim(0.19, 0.39)
    plt.yticks([0.2, 0.25, 0.3, 0.35])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Perform statistical comparison using standardized function
    print('Overall TI Comparison:')
    t_test_two_groups(human_ai_lst[0], human_ai_lst[1], 'Human-only', 'Human-AI', comparison_title="TI Difference Comparison")

    # Save plot
    plt.savefig('plots/TI_total.png', dpi=300)
    plt.close()

def plot_TI_sess(TI_df):
    """
    Create bar plot showing TI differences across sessions with individual team trajectories.
    
    This function visualizes how neural synchronization changes across sessions for both
    team types and performs statistical comparisons for each session.
    
    Parameters:
        TI_df (pd.DataFrame): DataFrame containing TI measurements across sessions
    """
    human_ai_lst = []
    x_labels = ['S1', 'S2', 'S3']
    x = np.arange(len(x_labels))

    # Calculate session-wise data for each team type
    for human_ai in ['human', 'ai']:
        sess_lst = []
        
        for sess in x_labels:
            team_vals = []
            unique_teams = TI_df[TI_df['human_ai'] == human_ai].teamID.unique()
            
            for teamid in unique_teams:
                # Get TI_diff values for this team and session
                team_session_data = TI_df[
                    (TI_df['human_ai'] == human_ai) & 
                    (TI_df['teamID'] == teamid) & 
                    (TI_df['sessionID'] == sess)
                ]['TI_diff'].dropna().values
                
                if len(team_session_data) > 0:
                    team_vals.append(np.nanmean(team_session_data))
                else:
                    team_vals.append(np.nan)
            
            sess_lst.append(team_vals)
        human_ai_lst.append(sess_lst)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # Create bar plots for group means
    ax.bar(x - 0.2, [np.nanmean(s) for s in human_ai_lst[0]], width=0.4,
           color='#F78474', label='Human-only')
    ax.bar(x + 0.2, [np.nanmean(s) for s in human_ai_lst[1]], width=0.4,
           color='#57A0D3', label='Human-AI')

    # Add individual team trajectories
    for human_ai, color, offset in zip(['human', 'ai'], ['#B03B3B', '#1E5F8A'], [-0.2, 0.2]):
        teamIDs = TI_df[TI_df['human_ai'] == human_ai].teamID.unique()
        
        for teamid in teamIDs:
            team_vals = []
            for sess in x_labels:
                session_data = TI_df[
                    (TI_df['human_ai'] == human_ai) & 
                    (TI_df['teamID'] == teamid) & 
                    (TI_df['sessionID'] == sess)
                ]['TI_diff'].values
                
                team_vals.append(np.nanmean(session_data) if len(session_data) > 0 else np.nan)
            
            # Plot individual team trajectory
            ax.plot(x + offset, team_vals, color=color, alpha=0.5, 
                   marker='o', linewidth=0.8, markersize=3, markeredgewidth=0.0)

    # Customize plot appearance
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("TI_diff")
    ax.set_ylim(0.18, 0.34)
    ax.set_yticks([0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()

    # Perform statistical comparisons for each session
    print("\nSession-wise TI Comparisons:")
    print("=" * 40)
    
    for i, session in enumerate(x_labels):
        t_test_two_groups(human_ai_lst[0][i], human_ai_lst[1][i], 'Human-only', 'Human-AI', comparison_title=f"Session {session}")
        print("=" * 40)

    # Save plot
    plt.savefig('plots/TI_session.png', dpi=300)
    plt.close()
  
    repeated_measure_ANOVA(human_ai_lst[0], ["S1", "S2", "S3"], 'SessionID', 'TI')
    repeated_measure_ANOVA(human_ai_lst[1], ["S1", "S2", "S3"], 'SessionID', 'TI')

def main():
    """
    Main execution function that orchestrates the EEG analysis workflow.
    """
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    have_baseline_TI = True
    have_TI = True

    # Load processed EEG data
    processed_eeg_df = pd.read_pickle('../physiological_behavioral_results/data/eeg_epochs_both_human_and_alice.pkl')
    
    # Load channel locations (though not used in current analysis)
    channel_locs = mne.channels.read_custom_montage('chan_locs.sfp')

    if not have_baseline_TI:
        # Compute baseline TI with 6.5 second window
        baseline_ti = compute_baseline_TI(duration_sec=6.5)
    else:
        baseline_ti = pd.read_pickle("../physiological_behavioral_results/data/baseline_TI.pkl")
    if not have_TI:
        # Compute TI for all ring events
        TI_df = compute_ring_TI(processed_eeg_df, baseline_ti)
    else:
        TI_df = pd.read_pickle("../physiological_behavioral_results/data/TI.pkl")
    # Generate visualizations and statistical analyses
    plot_TI_across_all_condition(TI_df)
    print("\n" + "=" * 60)
    plot_TI_sess(TI_df)


if __name__ == '__main__':
    main()