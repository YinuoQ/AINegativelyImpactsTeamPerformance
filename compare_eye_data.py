import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from scipy.stats import sem
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as mc
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM

def read_files():
    pd.set_option('display.max_columns', None)
    all_human_pupil_df = pd.read_pickle('../data/all_human/epoched_data/epoched_pupil.pkl')
    human_ai_pupil_df = pd.read_pickle('../data/human_ai/epoched_data/epoched_pupil.pkl')
    
    all_human_baseline = pd.read_pickle('../data/all_human/pupil_baseline.pkl')
    human_ai_baseline = pd.read_pickle('../data/human_ai/pupil_baseline.pkl')
    
    all_human_blink_df_raw = pd.read_pickle('../data/all_human/trialed_data/trialed_numb_blink.pkl')
    human_ai_blink_df_raw = pd.read_pickle('../data/human_ai/trialed_data/trialed_numb_blink.pkl')
    
    return all_human_pupil_df, human_ai_pupil_df, all_human_baseline, human_ai_baseline, all_human_blink_df_raw, human_ai_blink_df_raw

def compute_percent_change(pupil_df, baseline_df):
    merged = pupil_df.merge(baseline_df, on=['teamID', 'sessionID', 'role'], how='left')
    merged['pupil_percent'] = (merged['pupilSize'] - merged['baseline']) / merged['baseline']
    return merged

def get_pupil_percent_change(all_human_pupil_df, human_ai_pupil_df, all_human_baseline, human_ai_baseline):

    # Compute percent change and label team type
    all_human_pupil_df = compute_percent_change(all_human_pupil_df, all_human_baseline)
    all_human_pupil_df['human_ai'] = 'human'
    
    human_ai_pupil_df = compute_percent_change(human_ai_pupil_df, human_ai_baseline)
    human_ai_pupil_df['human_ai'] = 'ai'

    # Combine both
    pupil_df = pd.concat([all_human_pupil_df, human_ai_pupil_df], ignore_index=True)

    # Remove invalid entries
    valid_mask = pupil_df['pupil_percent'].apply(
        lambda x: isinstance(x, (list, np.ndarray)) and len(x) == 240 and not np.all(np.isnan(x)) and np.all(np.array(x) <= 500)
    )
    pupil_df = pupil_df[valid_mask].reset_index(drop=True)

    return pupil_df

def repeated_anova(input_arr):
    # Step 1: Remove columns with any NaNs
    valid_mask = ~np.isnan(input_arr).any(axis=0)
    cleaned_arr = input_arr[:, valid_mask]

    # Step 2: Prepare DataFrame for AnovaRM
    n_conditions, n_subjects = cleaned_arr.shape
    df = pd.DataFrame({
        'subject': np.repeat(np.arange(n_subjects), n_conditions),
        'condition': np.tile(np.arange(n_conditions), n_subjects),
        'value': cleaned_arr.flatten()
    })
    # Step 3: Run repeated measures ANOVA using AnovaRM
    anova = AnovaRM(data=df, depvar='value', subject='subject', within=['condition'])
    result = anova.fit()

    print(result)

def plot_pupil_percent_changes_all_conditions(pupil_df):
    reala_lst = []
    realb_lst = []
    reala_mean_lst = []
    realb_mean_lst = []

    for team in pupil_df[(pupil_df.human_ai == 'human')].teamID.unique():
        reala = np.array(list(pupil_df[(pupil_df.human_ai == 'human') & (pupil_df.teamID == team)]['pupil_percent']))
        reala_lst.append(reala)
        reala_mean_lst.append(np.mean(reala, axis=0))

    for team in pupil_df[(pupil_df.human_ai == 'ai')].teamID.unique():
        realb = np.array(list(pupil_df[(pupil_df.human_ai == 'ai') & (pupil_df.teamID == team)]['pupil_percent']))
        realb_lst.append(realb)
        realb_mean_lst.append(np.mean(realb, axis=0))

    x = np.linspace(-2, 2, len(reala_mean_lst[0]))  # infer time steps from data
    fig, ax = plt.subplots(1, figsize=(6, 4))
    plt.tight_layout(pad=1)

    # Plot mean lines
    ax.plot(x, np.mean(np.concatenate(reala_lst), axis=0)*100, color="#F78474")
    ax.plot(x, np.mean(np.concatenate(realb_lst), axis=0)*100, color="#57A0D3")
    ax.legend(['Human-only', 'Human-AI'])

    # Plot SEM shading
    aerr = np.std(reala_mean_lst, axis=0)*100 / np.sqrt(len(reala_mean_lst))
    berr = np.std(realb_mean_lst, axis=0)*100 / np.sqrt(len(realb_mean_lst))

    ax.fill_between(x, np.mean(reala_mean_lst, axis=0)*100 - aerr, np.mean(reala_mean_lst, axis=0)*100 + aerr, facecolor="#F78474", alpha=0.5)
    ax.fill_between(x, np.mean(realb_mean_lst, axis=0)*100 - berr, np.mean(realb_mean_lst, axis=0)*100 + berr, facecolor="#57A0D3", alpha=0.5)

    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.vlines(0, -1, 21, color='k', linestyles='--', alpha=0.3, linewidth=0.5)
    ax.set_xlim([-2.1, 2.1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Run t-test for each time point
    reala_mean_arr = np.array(reala_mean_lst) * 100
    realb_mean_arr = np.array(realb_mean_lst) * 100
    t_vals, p_vals = stats.ttest_ind(reala_mean_arr, realb_mean_arr, axis=0, nan_policy='omit')

    # FDR correction
    reject, p_vals_fdr, _, _ = multipletests(p_vals, alpha=0.1, method='fdr_bh')
    # Plot shaded regions for corrected significance
    sig_mask = reject.astype(bool)
    ax.fill_between(x, -1, 21, where=sig_mask, color='gray', alpha=0.1)

    plt.savefig('plots/pupil_size_comparison.png', dpi=300)
    plt.close()

def plot_pupil_size_session(pupil_percent_change_df):
    # Step 1: Flatten the pupil_percent list into mean per trial
    df = pupil_percent_change_df.copy()
    df['pupil_mean'] = df['pupil_percent'].apply(lambda x: x[120:].mean()*100)

    # Step 2: Compute team-level means per (teamID, role, sessionID, human_ai)
    sessions = [1, 2, 3]
    team_avg = df.groupby(['teamID', 'sessionID', 'human_ai'])['pupil_mean'].mean().reset_index()
    human_ais = ['human', 'ai']
    filtered_lst = []
    for ha in human_ais:
        filtered_df = team_avg[
            (team_avg['human_ai'] == ha)
        ].groupby('teamID').filter(lambda x: x['sessionID'].nunique() == 3)

        filtered_lst.append(filtered_df)
        print(f"{ha}")
        anova = AnovaRM(data=filtered_df,
            depvar='pupil_mean',
            subject='teamID',
            within=['sessionID']).fit()
        print(anova.summary())

    filtered_df = pd.concat(filtered_lst).reset_index(drop=True)

    for sess in sessions:
        print(ttest_ind(filtered_df[(filtered_df['sessionID'] == f'S{sess}') 
                                   & (filtered_df['human_ai'] == 'human')].pupil_mean, 
                        filtered_df[(filtered_df['sessionID'] == f'S{sess}') 
                                   & (filtered_df['human_ai'] == 'ai')].pupil_mean))

    # Step 3: Plotting
    x = np.arange(len(sessions))
    width = 0.35
    offset = width/2
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_mean_df = filtered_df.groupby(['human_ai', 'sessionID']).apply(lambda x: np.nanmean(x.pupil_mean)).reset_index(name='pupil_mean')
    
    ax.bar(x - offset, plot_mean_df[plot_mean_df.human_ai == 'human'].pupil_mean, width, label='Human-only', color='#E88A79')
    ax.bar(x + offset, plot_mean_df[plot_mean_df.human_ai == 'ai'].pupil_mean, width, label='Human-AI', color='#699ECE')

    for human_ai, color, offset in zip(['human', 'ai'], ['#B03B3B', '#1E5F8A'], [-width/2, width/2]):
        teamIDs = filtered_df[filtered_df['human_ai'] == human_ai].teamID.unique()
        for teamid in teamIDs:
            team_vals = []
            for sess in sessions:
                a = filtered_df[(filtered_df['human_ai'] == human_ai) & (filtered_df['teamID'] == teamid) & (filtered_df['sessionID'] == f'S{sess}')]['pupil_mean'].values
                team_vals.append(np.nanmean(a) if len(a) > 0 else np.nan)
            ax.plot(x + offset, team_vals, color=color, alpha=0.5, marker='o', linewidth=0.8, mew=0.0)

    ax.set_xticks(x)
    ax.set_ylim([-30, 85])
    ax.set_xticklabels(x)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig('plots/pupil_sess.png', dpi=300)

def plot_number_of_blink_all_conditions(all_human_blink_df_raw, human_ai_blink_df_raw):
    all_human_blink_df = all_human_blink_df_raw.groupby(['teamID']).apply(lambda x: x.trial_blink.mean())
    human_ai_blink_df = human_ai_blink_df_raw.groupby(['teamID']).apply(lambda x: x.trial_blink.mean())
    df_plot = pd.concat([
              pd.DataFrame({'Condition': 'Human-Only', 'BlinkMean': all_human_blink_df}),
              pd.DataFrame({'Condition': 'Human-AI', 'BlinkMean': human_ai_blink_df})
              ])

    # run t-test
    print(ttest_ind(all_human_blink_df, human_ai_blink_df))

    # plot
    colors = ['#F78474', '#57A0D3']
    plt.figure(figsize=(4, 4))
    sns.violinplot(
        data=df_plot, x='Condition', y='BlinkMean', palette=colors,
        inner='box', saturation=1, linewidth=0.5)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim([-0.05,0.55])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    plt.savefig('plots/number_blink_all.png', dpi=300)
    plt.close()

def plot_number_of_blink_session(all_human_blink_df_raw, human_ai_blink_df_raw):
    # plot number of blinks for different session
    # Group by and aggregate blink frequencies
    all_human_blink_df = all_human_blink_df_raw.groupby(['sessionID', 'teamID']).apply(lambda x: np.nanmean(x.blinkFreq))
    human_ai_blink_df = human_ai_blink_df_raw.groupby(['sessionID', 'teamID']).apply(lambda x: np.nanmean(x.blinkFreq))

    # Unstack teamID to get arrays
    human_arr = all_human_blink_df.unstack(level='teamID').values
    ai_arr = human_ai_blink_df.unstack(level='teamID').values

    print('human')
    repeated_anova(human_arr)      
    print('ai')
    repeated_anova(ai_arr)    
  
    all_human_blink_df = all_human_blink_df.reset_index(name='blink_rate')
    human_ai_blink_df = human_ai_blink_df.reset_index(name='blink_rate')
    sessions = ['S1', 'S2', 'S3']
    for sess in sessions:
        print(ttest_ind(all_human_blink_df[(all_human_blink_df['sessionID'] == sess)].blink_rate, 
                        human_ai_blink_df[(human_ai_blink_df['sessionID'] == sess)].blink_rate,
                        nan_policy='omit'))

    # Step 3: Plotting
    x = np.arange(len(sessions))
    width = 0.35
    offset = width/2

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - offset, 
        all_human_blink_df.groupby('sessionID').apply(lambda x:np.nanmean(x.blink_rate)), 
        width, label='Human-only', color='#E88A79')
    ax.bar(x + offset, 
        human_ai_blink_df.groupby('sessionID').apply(lambda x:np.nanmean(x.blink_rate)), 
        width, label='Human-AI', color='#699ECE')
    session_to_x = {s: i for i, s in enumerate(sessions)}

    # Loop over each human-only team
    for team in all_human_blink_df['teamID'].unique():
        team_data = all_human_blink_df[all_human_blink_df['teamID'] == team]
        # Sort by session
        team_data = team_data.sort_values('sessionID')
        y = team_data['blink_rate'].values
        plt.plot([session_to_x[s] - offset for s in team_data['sessionID']], y, marker='o', color='#B03B3B', alpha=0.5, linewidth=0.8, mew=0.0)
   
    # Loop over each human-AI team
    for team in human_ai_blink_df['teamID'].unique():
        team_data = human_ai_blink_df[human_ai_blink_df['teamID'] == team]
        # Sort by session
        team_data = team_data.sort_values('sessionID')
        y = team_data['blink_rate'].values
        plt.plot([session_to_x[s] + offset for s in team_data['sessionID']], y, marker='o', color='#1E5F8A', alpha=0.5, linewidth=0.8, mew=0.0)

    ax.set_xticks(x)
    ax.set_ylim([0.05, 0.55])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])    

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.savefig('plots/number_blink_sess_role.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # epoched pupil analzie
    all_human_pupil_df, human_ai_pupil_df, all_human_baseline, human_ai_baseline, all_human_blink_df_raw, human_ai_blink_df_raw = read_files()

    pupil_percent_change_df = get_pupil_percent_change(all_human_pupil_df, human_ai_pupil_df, all_human_baseline, human_ai_baseline)
    plot_pupil_percent_changes_all_conditions(pupil_percent_change_df)
    plot_pupil_size_session(pupil_percent_change_df)

    plot_number_of_blink_all_conditions(all_human_blink_df_raw, human_ai_blink_df_raw)
    plot_number_of_blink_session(all_human_blink_df_raw, human_ai_blink_df_raw)











