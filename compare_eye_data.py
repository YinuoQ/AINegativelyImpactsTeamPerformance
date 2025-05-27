import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from statistical_test import t_test_two_groups, repeated_measure_ANOVA


def read_files():
    """
    Load pupil dilation and blink data from pickle files for both human-only and human-AI teams.
    
    Returns:
        tuple: A tuple containing:
            - all_human_pupil_df (pd.DataFrame): Epoched pupil data from human-only teams
            - human_ai_pupil_df (pd.DataFrame): Epoched pupil data from human-AI teams
            - all_human_baseline (pd.DataFrame): Baseline pupil data from human-only teams
            - human_ai_baseline (pd.DataFrame): Baseline pupil data from human-AI teams
            - all_human_blink_df_raw (pd.DataFrame): Blink data from human-only teams
            - human_ai_blink_df_raw (pd.DataFrame): Blink data from human-AI teams
    """
    # Configure pandas to display all columns for better debugging
    pd.set_option('display.max_columns', None)
    
    # Load epoched pupil data
    all_human_pupil_df = pd.read_pickle('../physiological_behavioral_results/data/all_human/epoched_data/epoched_pupil.pkl')
    human_ai_pupil_df = pd.read_pickle('../physiological_behavioral_results/data/human_ai/epoched_data/epoched_pupil.pkl')
    
    # Load baseline pupil data
    all_human_baseline = pd.read_pickle('../physiological_behavioral_results/data/all_human/pupil_baseline.pkl')
    human_ai_baseline = pd.read_pickle('../physiological_behavioral_results/data/human_ai/pupil_baseline.pkl')
    
    # Load blink frequency data
    all_human_blink_df_raw = pd.read_pickle('../physiological_behavioral_results/data/all_human/trialed_data/trialed_numb_blink.pkl')
    human_ai_blink_df_raw = pd.read_pickle('../physiological_behavioral_results/data/human_ai/trialed_data/trialed_numb_blink.pkl')
    
    return all_human_pupil_df, human_ai_pupil_df, all_human_baseline, human_ai_baseline, all_human_blink_df_raw, human_ai_blink_df_raw


def compute_percent_change(pupil_df, baseline_df):
    """
    Calculate percent change in pupil size relative to baseline.
    
    This function merges pupil data with baseline measurements and computes
    the relative change as (pupil_size - baseline) / baseline.
    
    Parameters:
        pupil_df (pd.DataFrame): DataFrame containing pupil size measurements
        baseline_df (pd.DataFrame): DataFrame containing baseline pupil measurements
        
    Returns:
        pd.DataFrame: DataFrame with added 'pupil_percent' column containing percent changes
    """
    # Merge pupil data with baseline data on team, session, and role
    merged = pupil_df.merge(baseline_df, on=['teamID', 'sessionID', 'role'], how='left')
    
    # Calculate percent change from baseline
    merged['pupil_percent'] = (merged['pupilSize'] - merged['baseline']) / merged['baseline']
    
    return merged


def get_pupil_percent_change(all_human_pupil_df, human_ai_pupil_df, all_human_baseline, human_ai_baseline):
    """
    Process and combine pupil percent change data for both team types.
    
    This function computes percent changes from baseline for both human-only and human-AI teams,
    adds team type labels, combines the datasets, and filters out invalid data.
    
    Parameters:
        all_human_pupil_df (pd.DataFrame): Human-only team pupil data
        human_ai_pupil_df (pd.DataFrame): Human-AI team pupil data
        all_human_baseline (pd.DataFrame): Human-only team baseline data
        human_ai_baseline (pd.DataFrame): Human-AI team baseline data
        
    Returns:
        pd.DataFrame: Combined and filtered dataset with pupil percent changes
    """
    # Compute percent change and add team type labels
    all_human_pupil_df = compute_percent_change(all_human_pupil_df, all_human_baseline)
    all_human_pupil_df['human_ai'] = 'human'
    
    human_ai_pupil_df = compute_percent_change(human_ai_pupil_df, human_ai_baseline)
    human_ai_pupil_df['human_ai'] = 'ai'

    # Combine both datasets
    pupil_df = pd.concat([all_human_pupil_df, human_ai_pupil_df], ignore_index=True)

    # Filter out invalid entries (must be valid arrays of length 240 with reasonable values)
    valid_mask = pupil_df['pupil_percent'].apply(
        lambda x: isinstance(x, (list, np.ndarray)) and 
                 len(x) == 240 and 
                 not np.all(np.isnan(x)) and 
                 np.all(np.array(x) <= 500)
    )
    pupil_df = pupil_df[valid_mask].reset_index(drop=True)

    return pupil_df


def run_repeated_measures_anova(input_arr):
    """
    Perform repeated measures ANOVA on a 2D array where rows are conditions and columns are subjects.
    
    This function handles NaN values by removing subjects with missing data and
    runs the ANOVA using statsmodels AnovaRM.
    
    Parameters:
        input_arr (np.array): 2D array with shape (n_conditions, n_subjects)
    """
    # Remove columns (subjects) with any NaN values
    valid_mask = ~np.isnan(input_arr).any(axis=0)
    cleaned_arr = input_arr[:, valid_mask]

    # Prepare DataFrame for AnovaRM (long format)
    n_conditions, n_subjects = cleaned_arr.shape
    df = pd.DataFrame({
        'subject': np.repeat(np.arange(n_subjects), n_conditions),
        'condition': np.tile(np.arange(n_conditions), n_subjects),
        'value': cleaned_arr.flatten()
    })
    
    # Run repeated measures ANOVA
    anova = AnovaRM(data=df, depvar='value', subject='subject', within=['condition'])
    result = anova.fit()
    print(result)


def plot_pupil_percent_changes_all_conditions(pupil_df):
    """
    Create time series plot comparing pupil percent changes between human-only and human-AI teams.
    
    This function plots mean pupil responses with SEM shading and performs statistical testing
    at each time point with FDR correction for multiple comparisons.
    
    Parameters:
        pupil_df (pd.DataFrame): DataFrame containing pupil percent change data
    """
    human_data_lst = []
    ai_data_lst = []
    human_team_means = []
    ai_team_means = []

    # Calculate team-level means for human-only teams
    unique_human_teams = pupil_df[pupil_df.human_ai == 'human'].teamID.unique()
    for team in unique_human_teams:
        team_data = np.array(list(pupil_df[
            (pupil_df.human_ai == 'human') & 
            (pupil_df.teamID == team)
        ]['pupil_percent']))
        
        human_data_lst.append(team_data)
        human_team_means.append(np.mean(team_data, axis=0))

    # Calculate team-level means for human-AI teams
    unique_ai_teams = pupil_df[pupil_df.human_ai == 'ai'].teamID.unique()
    for team in unique_ai_teams:
        team_data = np.array(list(pupil_df[
            (pupil_df.human_ai == 'ai') & 
            (pupil_df.teamID == team)
        ]['pupil_percent']))
        
        ai_data_lst.append(team_data)
        ai_team_means.append(np.mean(team_data, axis=0))

    # Create time vector (assuming 4-second epoch with 240 samples)
    x = np.linspace(-2, 2, len(human_team_means[0]))
    
    # Create the plot
    fig, ax = plt.subplots(1, figsize=(6, 4))
    plt.tight_layout(pad=1)

    # Plot mean lines (convert to percentage)
    ax.plot(x, np.mean(np.concatenate(human_data_lst), axis=0) * 100, 
           color="#F78474", label='Human-only', linewidth=2)
    ax.plot(x, np.mean(np.concatenate(ai_data_lst), axis=0) * 100, 
           color="#57A0D3", label='Human-AI', linewidth=2)

    # Calculate and plot SEM shading
    human_sem = np.std(human_team_means, axis=0) * 100 / np.sqrt(len(human_team_means))
    ai_sem = np.std(ai_team_means, axis=0) * 100 / np.sqrt(len(ai_team_means))

    ax.fill_between(x, 
                   np.mean(human_team_means, axis=0) * 100 - human_sem,
                   np.mean(human_team_means, axis=0) * 100 + human_sem,
                   facecolor="#F78474", alpha=0.3)
    ax.fill_between(x,
                   np.mean(ai_team_means, axis=0) * 100 - ai_sem,
                   np.mean(ai_team_means, axis=0) * 100 + ai_sem,
                   facecolor="#57A0D3", alpha=0.3)

    # Customize plot appearance
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.vlines(0, -1, 21, color='k', linestyles='--', alpha=0.3, linewidth=0.5)
    ax.set_xlim([-2.1, 2.1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pupil size change (%)')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Perform statistical testing at each time point
    human_team_means_pct = np.array(human_team_means) * 100
    ai_team_means_pct = np.array(ai_team_means) * 100
    
    t_vals, p_vals = stats.ttest_ind(human_team_means_pct, ai_team_means_pct, 
                                    axis=0, nan_policy='omit')

    # Apply FDR correction for multiple comparisons
    reject, p_vals_fdr, _, _ = multipletests(p_vals, alpha=0.1, method='fdr_bh')
    
    # Highlight significant time periods
    sig_mask = reject.astype(bool)
    ax.fill_between(x, -1, 21, where=sig_mask, color='gray', alpha=0.1, 
                   label='Significant (FDR p<0.1)')

    # Save plot
    plt.savefig('plots/pupil_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print(f"Significant time points: {np.sum(sig_mask)}/{len(sig_mask)} ({100*np.sum(sig_mask)/len(sig_mask):.1f}%)")


def plot_pupil_size_session(pupil_percent_change_df):
    """
    Create bar plot showing pupil size changes across sessions with individual team trajectories.
    
    This function analyzes pupil responses during the post-ring period (second half of epoch)
    and performs repeated measures ANOVA and between-group comparisons.
    
    Parameters:
        pupil_percent_change_df (pd.DataFrame): DataFrame containing pupil percent change data
    """
    # Calculate mean pupil size during post-ring period (second half of epoch)
    df = pupil_percent_change_df.copy()
    df['pupil_mean'] = df['pupil_percent'].apply(lambda x: np.array(x)[120:].mean() * 100)

    # Compute team-level means per session
    team_avg = df.groupby(['teamID', 'sessionID', 'human_ai'])['pupil_mean'].mean().reset_index()
    
    # Filter teams that have data for all 3 sessions
    sessions = ['S1', 'S2', 'S3']
    human_ais = ['human', 'ai']
    filtered_lst = []
    
    for ha in human_ais:
        # Keep only teams with complete session data
        filtered_df = team_avg[
            team_avg['human_ai'] == ha
        ].groupby('teamID').filter(lambda x: x['sessionID'].nunique() == 3)

        filtered_lst.append(filtered_df)
        
        # Run repeated measures ANOVA for this team type
        print(f"\nRepeated Measures ANOVA - {ha.upper()} teams:")
        print("=" * 50)
        anova = AnovaRM(data=filtered_df,
                       depvar='pupil_mean',
                       subject='teamID',
                       within=['sessionID']).fit()
        print(anova.summary())

    # Combine filtered data
    filtered_df = pd.concat(filtered_lst).reset_index(drop=True)

    # Perform between-group comparisons for each session
    print("\nBetween-group comparisons by session:")
    print("=" * 50)
    
    for sess in sessions:
        human_data = filtered_df[
            (filtered_df['sessionID'] == sess) & 
            (filtered_df['human_ai'] == 'human')
        ]['pupil_mean']
        
        ai_data = filtered_df[
            (filtered_df['sessionID'] == sess) & 
            (filtered_df['human_ai'] == 'ai')
        ]['pupil_mean']
        
        t_test_two_groups(human_data, ai_data, 'Human-only', 'Human-AI', 
                         comparison_title=f"Session {sess}")
        print("=" * 50)

    # Create the visualization
    x = np.arange(len(sessions))
    width = 0.35
    offset = width / 2
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Calculate group means for bar plot
    plot_mean_df = filtered_df.groupby(['human_ai', 'sessionID']).agg({
        'pupil_mean': 'mean'
    }).reset_index()
    
    # Create bar plots
    human_means = plot_mean_df[plot_mean_df.human_ai == 'human']['pupil_mean'].values
    ai_means = plot_mean_df[plot_mean_df.human_ai == 'ai']['pupil_mean'].values
    
    ax.bar(x - offset, human_means, width, label='Human-only', color='#F78474')
    ax.bar(x + offset, ai_means, width, label='Human-AI', color='#57A0D3')

    # Add individual team trajectories
    colors_dark = {'human': '#B03B3B', 'ai': '#1E5F8A'}
    offsets = {'human': -offset, 'ai': offset}
    
    for human_ai in ['human', 'ai']:
        unique_teams = filtered_df[filtered_df['human_ai'] == human_ai]['teamID'].unique()
        
        for teamid in unique_teams:
            team_vals = []
            for i, sess in enumerate(sessions):
                team_session_data = filtered_df[
                    (filtered_df['human_ai'] == human_ai) & 
                    (filtered_df['teamID'] == teamid) & 
                    (filtered_df['sessionID'] == sess)
                ]['pupil_mean'].values
                
                team_vals.append(np.nanmean(team_session_data) if len(team_session_data) > 0 else np.nan)
            
            # Plot individual team trajectory
            ax.plot(x + offsets[human_ai], team_vals, 
                   color=colors_dark[human_ai], alpha=0.5, 
                   marker='o', linewidth=0.8, markersize=3, markeredgewidth=0.0)

    # Customize plot appearance
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i+1}' for i in range(len(sessions))])
    ax.set_xlabel('Session')
    ax.set_ylabel('Pupil size change (%)')
    ax.set_ylim([-30, 85])
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save plot
    plt.savefig('plots/pupil_sess.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_number_of_blink_all_conditions(all_human_blink_df_raw, human_ai_blink_df_raw):
    """
    Create violin plot comparing blink rates between human-only and human-AI teams.
    
    This function aggregates blink data by team and creates a visualization
    with statistical comparison.
    
    Parameters:
        all_human_blink_df_raw (pd.DataFrame): Raw blink data from human-only teams
        human_ai_blink_df_raw (pd.DataFrame): Raw blink data from human-AI teams
    """
    # Calculate team-level mean blink rates
    all_human_blink_means = all_human_blink_df_raw.groupby(['teamID'])['trial_blink'].mean()
    human_ai_blink_means = human_ai_blink_df_raw.groupby(['teamID'])['trial_blink'].mean()
    
    # Create DataFrame for visualization
    df_plot = pd.concat([
        pd.DataFrame({'Condition': 'Human-only', 'BlinkMean': all_human_blink_means}),
        pd.DataFrame({'Condition': 'Human-AI', 'BlinkMean': human_ai_blink_means})
    ])

    # Perform statistical comparison
    print("Blink Rate Comparison:")
    print("=" * 30)
    t_test_two_groups(all_human_blink_means, human_ai_blink_means, 
                     'Human-only', 'Human-AI', 
                     comparison_title="Overall Blink Rate")

    # Create violin plot
    colors = ['#F78474', '#57A0D3']
    plt.figure(figsize=(4, 4))
    sns.violinplot(
        data=df_plot, 
        x='Condition', 
        y='BlinkMean', 
        palette=colors,
        inner='box', 
        saturation=1, 
        linewidth=0.5
    )

    # Customize plot appearance
    plt.ylabel('Blink rate')
    plt.ylim([-0.05, 0.55])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save plot
    plt.savefig('plots/blink_rate_all.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_number_of_blink_session(all_human_blink_df_raw, human_ai_blink_df_raw):
    """
    Create bar plot showing blink rates across sessions with individual team trajectories.
    
    This function analyzes session-wise changes in blink frequency and performs
    repeated measures ANOVA and between-group comparisons.
    
    Parameters:
        all_human_blink_df_raw (pd.DataFrame): Raw blink data from human-only teams
        human_ai_blink_df_raw (pd.DataFrame): Raw blink data from human-AI teams
    """
    # Aggregate blink frequencies by session and team
    all_human_blink_df = all_human_blink_df_raw.groupby(['sessionID', 'teamID'])['blinkFreq'].apply(np.nanmean)
    human_ai_blink_df = human_ai_blink_df_raw.groupby(['sessionID', 'teamID'])['blinkFreq'].apply(np.nanmean)

    # Convert to arrays for ANOVA (sessions x teams)
    human_arr = all_human_blink_df.unstack(level='teamID').values
    ai_arr = human_ai_blink_df.unstack(level='teamID').values

    # Run repeated measures ANOVA for each group
    print("Repeated Measures ANOVA - Human-only teams:")
    print("=" * 50)
    run_repeated_measures_anova(human_arr)
    
    print("\nRepeated Measures ANOVA - Human-AI teams:")
    print("=" * 50)    
    run_repeated_measures_anova(ai_arr)

    # Convert back to DataFrame for plotting and comparisons
    all_human_blink_df = all_human_blink_df.reset_index(name='blink_rate')
    human_ai_blink_df = human_ai_blink_df.reset_index(name='blink_rate')
    
    # Perform between-group comparisons for each session
    sessions = ['S1', 'S2', 'S3']
    print(f"\nBetween-group comparisons by session:")
    print("=" * 50)
    
    for sess in sessions:
        human_data = all_human_blink_df[
            all_human_blink_df['sessionID'] == sess
        ]['blink_rate']
        
        ai_data = human_ai_blink_df[
            human_ai_blink_df['sessionID'] == sess
        ]['blink_rate']
        
        t_test_two_groups(human_data, ai_data, 'Human-only', 'Human-AI',
                         comparison_title=f"Session {sess} Blink Rate")
        print("=" * 50)

    # Create the visualization
    x = np.arange(len(sessions))
    width = 0.35
    offset = width / 2

    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Calculate group means and create bar plots
    human_means = all_human_blink_df.groupby('sessionID')['blink_rate'].apply(np.nanmean).values
    ai_means = human_ai_blink_df.groupby('sessionID')['blink_rate'].apply(np.nanmean).values
    
    ax.bar(x - offset, human_means, width, label='Human-only', color='#F78474')
    ax.bar(x + offset, ai_means, width, label='Human-AI', color='#57A0D3')
    
    # Create session-to-position mapping for individual trajectories
    session_to_x = {s: i for i, s in enumerate(sessions)}

    # Plot individual team trajectories for human-only teams
    for team in all_human_blink_df['teamID'].unique():
        team_data = all_human_blink_df[all_human_blink_df['teamID'] == team]
        team_data = team_data.sort_values('sessionID')
        
        x_pos = [session_to_x[s] - offset for s in team_data['sessionID']]
        y_values = team_data['blink_rate'].values
        
        ax.plot(x_pos, y_values, marker='o', color='#B03B3B', 
               alpha=0.5, linewidth=0.8, markersize=3, markeredgewidth=0.0)

    # Plot individual team trajectories for human-AI teams
    for team in human_ai_blink_df['teamID'].unique():
        team_data = human_ai_blink_df[human_ai_blink_df['teamID'] == team]
        team_data = team_data.sort_values('sessionID')
        
        x_pos = [session_to_x[s] + offset for s in team_data['sessionID']]
        y_values = team_data['blink_rate'].values
        
        ax.plot(x_pos, y_values, marker='o', color='#1E5F8A', 
               alpha=0.5, linewidth=0.8, markersize=3, markeredgewidth=0.0)

    # Customize plot appearance
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i+1}' for i in range(len(sessions))])
    ax.set_ylabel('Blink rate')
    ax.set_ylim([0.05, 0.55])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save plot
    plt.savefig('plots/blink_rate_session.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main execution function that orchestrates the pupil and blink analysis workflow.
    """
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    
    # Load all data files
    print("Loading data files...")
    (all_human_pupil_df, human_ai_pupil_df, all_human_baseline, human_ai_baseline, 
     all_human_blink_df_raw, human_ai_blink_df_raw) = read_files()

    # Process pupil data
    print("Processing pupil percent change data...")
    pupil_percent_change_df = get_pupil_percent_change(
        all_human_pupil_df, human_ai_pupil_df, 
        all_human_baseline, human_ai_baseline
    )
    
    # Generate pupil analysis plots
    print("Generating pupil analysis plots...")
    plot_pupil_percent_changes_all_conditions(pupil_percent_change_df)
    plot_pupil_size_session(pupil_percent_change_df)

    # Generate blink analysis plots
    print("Generating blink analysis plots...")
    plot_number_of_blink_all_conditions(all_human_blink_df_raw, human_ai_blink_df_raw)
    plot_number_of_blink_session(all_human_blink_df_raw, human_ai_blink_df_raw)
    


if __name__ == '__main__':
    main()