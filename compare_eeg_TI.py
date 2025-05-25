import os
import mne
import copy
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import mne_connectivity
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM 
from scipy.stats import f_oneway, ttest_ind, ttest_rel
from scipy.signal import hilbert, butter, filtfilt, coherence, windows



def baseline_TI(preprocessed_eeg_df, fs=256, duration_sec=2):
    """
    Compute baseline TI over the first `duration_sec` seconds for each team/session pair.
    
    Parameters:
    - preprocessed_eeg_df: DataFrame with columns ['teamID', 'sessionID', 'role', 'processed_eeg']
    - fs: Sampling frequency
    - duration_sec: Time window for baseline (e.g., 4 seconds)

    Returns:
    - DataFrame with ['teamID', 'sessionID', 'baseline_TI']
    """
    baseline_records = []
    n_samples = int(fs * duration_sec)

    grouped = preprocessed_eeg_df.groupby(['teamID', 'sessionID'])

    for (teamID, sessionID), group in tqdm(grouped, desc="Teams/Sessions"):
        if len(group) != 2:
            continue  # need both Pitch and Yaw

        eeg1 = np.array(group.iloc[0]['processed_eeg'])[:, :n_samples]
        eeg2 = np.array(group.iloc[1]['processed_eeg'])[:, :n_samples]

        if eeg1.shape[1] < n_samples or eeg2.shape[1] < n_samples:
            continue

        ti_value = compute_group_total_interdependence(eeg1, eeg2)

        baseline_records.append({
            'teamID': teamID,
            'sessionID': sessionID,
            'baseline_TI': ti_value
        })

    return pd.DataFrame(baseline_records)

def compute_baseline_TI(duration_sec=2):
    preprocessed_eeg_df = pd.read_pickle(os.path.join('../data/human_ai/', 'preprocessed_eeg.pkl' ))
    human_ai_baseline = baseline_TI(preprocessed_eeg_df, duration_sec=duration_sec)
    preprocessed_eeg_df = pd.read_pickle(os.path.join('../data/all_human/', 'preprocessed_eeg.pkl' ))
    all_human_baseline = baseline_TI(preprocessed_eeg_df, duration_sec=duration_sec)
    human_ai_baseline['human_ai'] = 'ai'
    all_human_baseline['human_ai'] = 'human'
    baseline_ti = pd.concat([all_human_baseline, human_ai_baseline]).reset_index(drop=True)
    return baseline_ti


def total_interdependence_broadband(signal_1, signal_2, fs=256, fmin=0.5, fmax=30, nperseg=512):
    """
    Computes the Total Interdependence (TI) between two signals over a wide frequency range.

    Parameters:
    - signal_1, signal_2: EEG signals (1D numpy arrays)
    - fs: sampling frequency
    - fmin: minimum frequency to include (default: 0.5 Hz to exclude slow drift)
    - fmax: maximum frequency (default: Nyquist = fs/2)
    - nperseg: segment length for Welch method

    Returns:
    - TI: total interdependence value (scalar)
    """
    f, C2 = coherence(signal_1, signal_2, fs=fs, nperseg=nperseg, detrend=False)
    
    if fmax is None:
        fmax = fs / 2

    # Frequency mask
    freq_mask = (f >= fmin) & (f <= fmax)
    C2_selected = C2[freq_mask]
    f_selected = f[freq_mask]

    M = len(f_selected)
    if M < 2:
        return np.nan

    delta_f = fs / (2 * (M - 1))

    with np.errstate(divide='ignore', invalid='ignore'):
        TI = -(2 / fs) * np.sum(np.log(1 - C2_selected) * delta_f)

    return TI


def compute_channel_total_interdependence(eeg_data_1, eeg_data_2):
    """
    Compute band-specific TI for all matching channels across two EEG arrays.

    Parameters:
    - eeg_data_1: EEG array [20, T] from subject 1
    - eeg_data_2: EEG array [20, T] from subject 2
    - freqs: tuple (fmin, fmax) defining the frequency band
    - fs: sampling frequency

    Returns:
    - average TI across matching channels
    """
    ti_values = []

    for ch in range(20):
        eeg1 = eeg_data_1[ch]
        eeg2 = eeg_data_2[ch]

        ti = total_interdependence_broadband(eeg1, eeg2)
        ti_values.append(ti)

    return np.nanmean(ti_values)


def compute_ring_TI(eeg_df, baseline_ti):
    """
    Compute TI values for each ring event across teams, sessions, trials, and roles.

    Returns:
    - DataFrame with one row per event and band-wise TI values.
    """
    team_session_trial_ring = eeg_df[['teamID', 'sessionID', 'trialID', 'ringID', 'human_ai', 'communication', 'difficulty']].drop_duplicates().reset_index(drop=True)
    team_session_trial_ring['TI'] = None
    team_session_trial_ring['TI_diff'] = None

    for i in tqdm(range(len(team_session_trial_ring)), leave=False, desc=f"Computing TI"):
        row = team_session_trial_ring.iloc[i]
        temp_eeg = eeg_df.query(
            f"teamID == '{row.teamID}' and "
            f"sessionID == '{row.sessionID}' and "
            f"trialID == {row.trialID} and "
            f"ringID == {row.ringID} and "
            f"human_ai == '{row.human_ai}'"
        ).eeg
        temp_baseline = baseline_ti.query(
            f"teamID == '{row.teamID}' and "
            f"sessionID == '{row.sessionID}' and "
            f"human_ai == '{row.human_ai}'"
        ).baseline_TI

        if len(temp_eeg) == 2:
            eeg_data_1 = temp_eeg.iloc[0]
            eeg_data_2 = temp_eeg.iloc[1]
            ti_val = compute_channel_total_interdependence(eeg_data_1[:,:-1], eeg_data_2[:,:-1])
            team_session_trial_ring.at[i, 'TI'] = ti_val
            team_session_trial_ring.at[i, 'TI_diff'] = ti_val - temp_baseline.iloc[0]
    return team_session_trial_ring.dropna().reset_index(drop=True)

def plot_TI_across_all_condition(TI_df):
    human_ai_lst = []
    conditions = ['Human-only', 'Human-AI']
    for human_ai in ['human', 'ai']:
        team_mean_lst = []
        for teamid in TI_df[(TI_df['human_ai'] == human_ai)].teamID.unique():
            a = np.array(list(TI_df[(TI_df['human_ai'] == human_ai) & (TI_df['teamID'] == teamid)]['TI_diff'].dropna()))
            team_mean_lst.append(np.nanmean(a))
        human_ai_lst.append(team_mean_lst)

    # Create tidy DataFrame for Seaborn
    df_plot = pd.concat([
        pd.DataFrame({'Condition': conditions[0], 'TI_diff': human_ai_lst[0]}),
        pd.DataFrame({'Condition': conditions[1], 'TI_diff': human_ai_lst[1]})
    ])
    colors = ['#F78474', '#57A0D3']

    # Violin plot
    plt.figure(figsize=(4, 4))
    sns.violinplot(data=df_plot, x='Condition', y='TI_diff',
                   palette=colors, inner='box', saturation=1, linewidth=0.5)

    # Customize plot
    plt.ylim(0.19, 0.39)
    plt.yticks([0.2, 0.25, 0.3, 0.35])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    print('all comparision')
    f_statistic, p_value = ttest_ind(np.array(human_ai_lst[0]), np.array(human_ai_lst[1]))
    f_statistic_rounded = round(f_statistic, 3)
    p_value_rounded = round(p_value, 4)        
    print("f=", f_statistic_rounded)
    print("p=", p_value_rounded)

    plt.savefig('plots/eeg_plots/TI_total.png', dpi=300)

def run_repeated_measure_ANOVA(input_lst, columns_lst, var_name, value_name): 
    # Convert to DataFrame: each row = one subject (team), columns = conditions
    df_wide = pd.DataFrame(np.array(input_lst).T, columns=columns_lst)
    df_wide['teamID'] = df_wide.index.astype(str)  # convert index to string for subject ID

    # Convert to long format
    df_long = pd.melt(df_wide, id_vars=['teamID'], value_vars=columns_lst,
                      var_name=var_name, value_name=value_name)
    df_wide = df_long.pivot(index='teamID', columns='difficulty', values=value_name)

    # Drop rows with any NaN
    df_wide_clean = df_wide.dropna()

    # Optionally, convert back to long format
    df_clean = df_wide_clean.reset_index().melt(id_vars='teamID', value_name=value_name, var_name='difficulty')

    # Run repeated measures ANOVA
    anova = AnovaRM(data=df_clean, depvar=value_name, subject='teamID', within=[var_name])
    res = anova.fit()
    print(res)

    if res.summary().tables[0]['Pr > F'].iloc[0] < 0.05:
        # Get data arrays
        cond1 = df_clean[df_clean[var_name] == columns_lst[0]][value_name]
        cond2 = df_clean[df_clean[var_name] == columns_lst[1]][value_name]
        cond3 = df_clean[df_clean[var_name] == columns_lst[2]][value_name]

        # Run paired t-tests
        t1, p1 = ttest_rel(cond1, cond2)
        t2, p2 = ttest_rel(cond1, cond3)
        t3, p3 = ttest_rel(cond2, cond3)

        print(f"{columns_lst[0]} vs {columns_lst[1]}")
        print(t1, p1)
        print(f"{columns_lst[0]} vs {columns_lst[2]}")
        print(t2, p2)
        print(f"{columns_lst[1]} vs {columns_lst[2]}")
        print(t3, p3)

def plot_TI_sess(TI_df):
    human_ai_lst = []
    x_labels = ['S1', 'S2', 'S3']
    x = np.arange(len(x_labels))

    for human_ai in ['human', 'ai']:
        sess_lst = []
        for sess in x_labels:
            team_vals = []
            for teamid in TI_df[TI_df['human_ai'] == human_ai].teamID.unique():
                a = TI_df[(TI_df['human_ai'] == human_ai) & (TI_df['teamID'] == teamid) & (TI_df['sessionID'] == sess)]['TI_diff'].dropna().values
                if len(a) > 0:
                    team_vals.append(np.nanmean(a))
                else:
                    team_vals.append(np.nan)
            sess_lst.append(team_vals)
        human_ai_lst.append(sess_lst)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharey=True)

    ax.bar(x - 0.2, [np.nanmean(s) for s in human_ai_lst[0]], width=0.4,
           color='#F78474', label='Human-only')
    ax.bar(x + 0.2, [np.nanmean(s) for s in human_ai_lst[1]], width=0.4,
           color='#57A0D3', label='Human-AI')

    for human_ai, color, offset in zip(['human', 'ai'], ['#B03B3B', '#1E5F8A'], [-0.2, 0.2]):
        teamIDs = TI_df[TI_df['human_ai'] == human_ai].teamID.unique()
        for teamid in teamIDs:
            team_vals = []
            for sess in x_labels:
                a = TI_df[(TI_df['human_ai'] == human_ai) & (TI_df['teamID'] == teamid) & (TI_df['sessionID'] == sess)]['TI_diff'].values
                team_vals.append(np.nanmean(a) if len(a) > 0 else np.nan)
            ax.plot(x + offset, team_vals, color=color, alpha=0.5, marker='o', linewidth=0.8, mew=0.0)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("TI_diff")
    ax.set_ylim(0.18, 0.34)
    ax.get_yaxis().set_ticks([0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    
    for i in range(len(x_labels)):
        f_statistic, p_value = ttest_ind(human_ai_lst[0][i], human_ai_lst[1][i], nan_policy='omit')
        print(f"Session {x_labels[i]} — f={round(f_statistic,3)}, p={round(p_value,4)}")
        print(f"df={len(human_ai_lst[0][i])+len(human_ai_lst[1][i])-2}")
    plt.savefig('plots/eeg_plots/TI_session.png', dpi=300)



if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    processed_eeg_df = pd.read_pickle('../data/eeg_epochs_both_human_and_alice.pkl')
    channel_locs = mne.channels.read_custom_montage('chan_locs.sfp')

    baseline_ti = compute_baseline_TI(duration_sec=6.5)

    TI_df = compute_ring_TI(processed_eeg_df, baseline_ti)

    plot_TI_across_all_condition(TI_df)
    print("==========================================================")
    plot_TI_sess(TI_df)


