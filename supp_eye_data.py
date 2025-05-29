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
            - human_ai_blink_df_raw (pd.DataFrame): Raw blink data from human-AI teams
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


# ======================== GENERIC PLOTTING FUNCTIONS ========================

def create_time_series_plot(pupil_df, grouping_vars, condition_labels, 
                           plot_title_prefix, save_filename, 
                           subplot_rows=1, subplot_cols=1, figsize=(12, 4)):
    """
    Generic function to create time series plots for pupil data.
    
    Parameters:
        pupil_df (pd.DataFrame): DataFrame containing pupil percent change data
        grouping_vars (list): List of column names to group by (e.g., ['role', 'difficulty'])
        condition_labels (dict): Dictionary mapping condition values to display labels
        plot_title_prefix (str): Prefix for subplot titles
        save_filename (str): Filename to save the plot
        subplot_rows (int): Number of subplot rows
        subplot_cols (int): Number of subplot columns
        figsize (tuple): Figure size
    """
    colors = {'human': "#F78474", 'ai': "#57A0D3"}
    x = np.linspace(-2, 2, 240)
    
    fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=figsize, sharey=True)
    if subplot_rows == 1 and subplot_cols == 1:
        ax = [ax]
    elif subplot_rows == 1 or subplot_cols == 1:
        ax = ax.flatten()
    else:
        ax = ax.flatten()
    
    plt.tight_layout(pad=2)
    
    # Get unique combinations of grouping variables
    if len(grouping_vars) == 1:
        combinations = [(val,) for val in pupil_df[grouping_vars[0]].unique()]
    else:
        combinations = []
        for i, val1 in enumerate(pupil_df[grouping_vars[0]].unique()):
            for j, val2 in enumerate(pupil_df[grouping_vars[1]].unique()):
                combinations.append((val1, val2))
    
    for idx, combination in enumerate(combinations):
        if idx >= len(ax):
            break
            
        human_team_means = []
        ai_team_means = []
        
        # Create filter condition based on grouping variables
        filter_condition = pd.Series([True] * len(pupil_df), index=pupil_df.index)
        for i, var in enumerate(grouping_vars):
            filter_condition &= (pupil_df[var] == combination[i])
        
        # Process human teams
        human_teams = pupil_df[filter_condition & (pupil_df.human_ai == 'human')].teamID.unique()
        for team in human_teams:
            team_data = np.array(list(pupil_df[
                filter_condition & 
                (pupil_df.human_ai == 'human') & 
                (pupil_df.teamID == team)
            ]['pupil_percent']))
            
            if len(team_data) > 0:
                human_team_means.append(np.mean(team_data, axis=0))
        
        # Process AI teams
        ai_teams = pupil_df[filter_condition & (pupil_df.human_ai == 'ai')].teamID.unique()
        for team in ai_teams:
            team_data = np.array(list(pupil_df[
                filter_condition & 
                (pupil_df.human_ai == 'ai') & 
                (pupil_df.teamID == team)
            ]['pupil_percent']))
            
            if len(team_data) > 0:
                ai_team_means.append(np.mean(team_data, axis=0))
        
        if len(human_team_means) > 0 and len(ai_team_means) > 0:
            human_team_means = np.array(human_team_means) * 100
            ai_team_means = np.array(ai_team_means) * 100
            
            # Plot lines and SEM
            ax[idx].plot(x, np.mean(human_team_means, axis=0), 
                        color=colors['human'], linewidth=2)
            ax[idx].plot(x, np.mean(ai_team_means, axis=0), 
                        color=colors['ai'], linewidth=2)
            
            human_sem = np.std(human_team_means, axis=0) / np.sqrt(len(human_team_means))
            ai_sem = np.std(ai_team_means, axis=0) / np.sqrt(len(ai_team_means))
            
            ax[idx].fill_between(x, 
                                np.mean(human_team_means, axis=0) - human_sem,
                                np.mean(human_team_means, axis=0) + human_sem,
                                facecolor=colors['human'], alpha=0.3)
            ax[idx].fill_between(x,
                                np.mean(ai_team_means, axis=0) - ai_sem,
                                np.mean(ai_team_means, axis=0) + ai_sem,
                                facecolor=colors['ai'], alpha=0.3)
            
            # Statistical testing
            t_vals, p_vals = stats.ttest_ind(human_team_means, ai_team_means, 
                                           axis=0, nan_policy='omit')
            reject, p_vals_fdr, _, _ = multipletests(p_vals, alpha=0.1, method='fdr_bh')
            sig_mask = reject.astype(bool)
            ax[idx].fill_between(x, -2, 22, where=sig_mask, color='gray', alpha=0.1)
        
        # Customize subplot
        title_parts = []
        for i, var in enumerate(grouping_vars):
            value = combination[i]
            if value in condition_labels:
                title_parts.append(condition_labels[value])
            else:
                title_parts.append(str(value))
        
        title = f"{' '.join(title_parts)}" if len(title_parts) > 1 else f"{title_parts[0]} {plot_title_prefix}"
        ax[idx].set_title(title, fontsize=20)
        ax[idx].vlines(0, -2, 30, color='k', linestyles='--', alpha=0.3, linewidth=0.5)
        ax[idx].set_xlim([-2.1, 2.1])
        ax[idx].set_ylim([-2, 30])
        ax[idx].tick_params(axis='x', labelsize=18)
        ax[idx].tick_params(axis='y', labelsize=18)
        ax[idx].spines['top'].set_visible(False)
        ax[idx].spines['right'].set_visible(False)
        
        # Add labels based on position
        if len(grouping_vars) == 2 and subplot_rows == 2:
            # For 2x3 grids
            if idx % subplot_cols == 0:  # Left column
                ax[idx].set_ylabel('Pupil size \nchange (%)', fontsize=20)
            if idx >= subplot_cols:  # Bottom row
                ax[idx].set_xlabel('Time (s)', fontsize=20)
        else:
            # For single row plots
            if idx == 0:
                ax[idx].set_ylabel('Pupil size \nchange (%)', fontsize=20)
                ax[idx].legend(['Human-only', 'Human-AI'], fontsize=18)
            ax[idx].set_xlabel('Time (s)', fontsize=20)
    
    # # Add legend for multi-subplot plots
    # if len(combinations) > 1 and subplot_rows > 1:
    #     ax[0].legend(['Human-only', 'Human-AI'], fontsize=18)
    
    plt.savefig(f'plots/{save_filename}', dpi=300, bbox_inches='tight')
    plt.close()


def create_bar_plot_with_trajectories(data_df, grouping_vars, condition_labels, 
                                     value_col, plot_title_prefix, save_filename, 
                                     ylabel, ylim_range, figsize=(10, 5), 
                                     perform_stats=True):
    """
    Generic function to create bar plots with individual team trajectories.
    
    Parameters:
        data_df (pd.DataFrame): DataFrame containing the data
        grouping_vars (list): List of column names to group by (e.g., ['role', 'difficulty'])
        condition_labels (dict): Dictionary mapping condition values to display labels
                                The order of keys in this dictionary determines the bar order
        value_col (str): Name of the column containing values to plot
        plot_title_prefix (str): Prefix for subplot titles
        save_filename (str): Filename to save the plot
        ylabel (str): Y-axis label
        ylim_range (tuple): Y-axis limits
        figsize (tuple): Figure size
        perform_stats (bool): Whether to perform statistical comparisons
    """
    roles = ['Yaw', 'Pitch']
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    
    for i, role in enumerate(roles):
        role_data = data_df[data_df['role'] == role]
        
        # Get unique values for the non-role grouping variable
        if len(grouping_vars) == 2:
            other_var = [var for var in grouping_vars if var != 'role'][0]
            # Use the order from condition_labels keys, filtering for available data
            available_conditions = set(role_data[other_var].unique())
            conditions = [key for key in condition_labels.keys() if key in available_conditions]
        else:
            other_var = grouping_vars[0]
            # Use the order from condition_labels keys, filtering for available data
            available_conditions = set(role_data[other_var].unique())
            conditions = [key for key in condition_labels.keys() if key in available_conditions]
        
        x = np.arange(len(conditions))
        width = 0.35
        offset = width / 2
        
        # Calculate group means for bar plot
        human_means = []
        ai_means = []
        
        for condition in conditions:
            if len(grouping_vars) == 2 and 'role' in grouping_vars:
                filter_condition = (role_data[other_var] == condition)
            else:
                filter_condition = (role_data[other_var] == condition)
            
            human_data = role_data[filter_condition & (role_data['human_ai'] == 'human')][value_col]
            ai_data = role_data[filter_condition & (role_data['human_ai'] == 'ai')][value_col]
            
            human_means.append(np.nanmean(human_data) if len(human_data) > 0 else 0)
            ai_means.append(np.nanmean(ai_data) if len(ai_data) > 0 else 0)
            
            # Perform statistical comparison
            if perform_stats and len(human_data) > 0 and len(ai_data) > 0:
                condition_label = condition_labels.get(condition, condition)
                print(f"\n{role} - {condition_label}:")
                t_test_two_groups(human_data, ai_data, 'Human-only', 'Human-AI',
                                comparison_title=f"{role} {condition_label} {plot_title_prefix}")
        
        # Create bar plots
        ax[i].bar(x - offset, human_means, width, label='Human-only', color='#F78474')
        ax[i].bar(x + offset, ai_means, width, label='Human-AI', color='#57A0D3')
        
        # Add individual team trajectories
        colors_dark = {'human': '#B03B3B', 'ai': '#1E5F8A'}
        offsets = {'human': -offset, 'ai': offset}
        
        for human_ai in ['human', 'ai']:
            role_subset = role_data[role_data['human_ai'] == human_ai]
            unique_teams = role_subset['teamID'].unique()
            
            for teamid in unique_teams:
                team_vals = []
                for condition in conditions:
                    if len(grouping_vars) == 2 and 'role' in grouping_vars:
                        filter_condition = (role_subset[other_var] == condition) & (role_subset['teamID'] == teamid)
                    else:
                        filter_condition = (role_subset[other_var] == condition) & (role_subset['teamID'] == teamid)
                    
                    team_data = role_subset[filter_condition][value_col].values
                    team_vals.append(np.nanmean(team_data) if len(team_data) > 0 else np.nan)
                
                # Plot individual team trajectory
                ax[i].plot(x + offsets[human_ai], team_vals, 
                          color=colors_dark[human_ai], alpha=0.5, 
                          marker='o', linewidth=0.8, markersize=3, markeredgewidth=0.0)
        
        # Customize subplot
        ax[i].set_title(f'{role} Pilot', fontsize=20)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels([condition_labels.get(c, c) for c in conditions])
        ax[i].set_ylabel(ylabel if i == 0 else '', fontsize=20)
        ax[i].set_ylim(ylim_range)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].tick_params(axis='x', labelsize=18)
        ax[i].tick_params(axis='y', labelsize=18)
        
        if i == 0 and 'difficulty' in grouping_vars:
            ax[i].legend(fontsize=18)
    
    plt.tight_layout()
    plt.savefig(f'plots/{save_filename}', dpi=300, bbox_inches='tight')
    plt.close()

# ======================== PART 1: PUPIL SIZE TIME SERIES ANALYSIS ========================

def plot_pupil_time_series_by_role(pupil_df):
    """Create time series plots comparing pupil percent changes by role."""
    create_time_series_plot(
        pupil_df=pupil_df,
        grouping_vars=['role'],
        condition_labels={'Yaw': 'Yaw Pilot', 'Pitch': 'Pitch Pilot'},
        plot_title_prefix='',
        save_filename='supp_pupil_timeseries_by_role.png',
        subplot_rows=1,
        subplot_cols=2,
        figsize=(12, 3)
    )


def plot_pupil_time_series_by_difficulty(pupil_df):
    """Create time series plots comparing pupil percent changes by difficulty level."""
    create_time_series_plot(
        pupil_df=pupil_df,
        grouping_vars=['role', 'difficulty'],
        condition_labels={'Yaw': 'Yaw', 'Pitch': 'Pitch', 
                         'Easy': 'Easy', 'Medium': 'Intermediate', 'Hard': 'Hard'},
        plot_title_prefix='',
        save_filename='supp_pupil_timeseries_by_difficulty.png',
        subplot_rows=2,
        subplot_cols=3,
        figsize=(15, 6)
    )

def plot_pupil_time_series_by_session(pupil_df):
    """Create time series plots comparing pupil percent changes by session."""
    create_time_series_plot(
        pupil_df=pupil_df,
        grouping_vars=['role', 'sessionID'],
        condition_labels={'Yaw': 'Yaw', 'Pitch': 'Pitch',
                         'S1': 'Session 1', 'S2': 'Session 2', 'S3': 'Session 3'},
        plot_title_prefix='',
        save_filename='supp_pupil_timeseries_by_session.png',
        subplot_rows=2,
        subplot_cols=3,
        figsize=(15, 6)
    )

def plot_pupil_time_series_by_communication(pupil_df):
    """Create time series plots comparing pupil percent changes by communication condition."""
    create_time_series_plot(
        pupil_df=pupil_df,
        grouping_vars=['role', 'communication'],
        condition_labels={'Yaw': 'Yaw', 'Pitch': 'Pitch',
                         'No': 'Incommunicado', 'Word': 'Command Word', 'Free': 'Free Speech'},
        plot_title_prefix='',
        save_filename='supp_pupil_timeseries_by_communication.png',
        subplot_rows=2,
        subplot_cols=3,
        figsize=(15, 6)
    )





# ======================== PART 2: PUPIL SIZE AFTER EVENT ANALYSIS ========================

def plot_pupil_after_event_by_difficulty(pupil_df):
    """Create bar plot showing average pupil size changes after event by difficulty level."""
    # Calculate mean pupil size during post-ring period (second half of epoch)
    df = pupil_df.copy()
    df['pupil_mean'] = df['pupil_percent'].apply(lambda x: np.array(x)[120:].mean() * 100)
    
    # Compute team-level means per (teamID, role, difficulty, human_ai)
    team_avg = df.groupby(['teamID', 'role', 'difficulty', 'human_ai'])['pupil_mean'].mean().reset_index()
    
    create_bar_plot_with_trajectories(
        data_df=team_avg,
        grouping_vars=['role', 'difficulty'],
        condition_labels={'Easy': 'Easy', 'Medium': 'Intermediate', 'Hard': 'Hard'},
        value_col='pupil_mean',
        plot_title_prefix='Difficulty',
        save_filename='supp_pupil_after_event_by_difficulty.png',
        ylabel='Pupil size change (%)',
        ylim_range=(-25, 50),
        figsize=(12, 5)
    )


def plot_pupil_after_event_by_communication(pupil_df):
    """Create bar plot showing average pupil size changes after event by communication condition."""
    # Calculate mean pupil size during post-ring period
    df = pupil_df.copy()
    df['pupil_mean'] = df['pupil_percent'].apply(lambda x: np.array(x)[120:].mean() * 100)
    
    # Compute team-level means
    team_avg = df.groupby(['teamID', 'role', 'communication', 'human_ai'])['pupil_mean'].mean().reset_index()
    
    create_bar_plot_with_trajectories(
        data_df=team_avg,
        grouping_vars=['role', 'communication'],
        condition_labels={'No': 'Incomm-\nunicado', 'Word': 'Command \nword', 'Free': 'Free \nspeech'},
        value_col='pupil_mean',
        plot_title_prefix='Communication',
        save_filename='supp_pupil_after_event_by_communication.png',
        ylabel='Pupil size change (%)',
        ylim_range=(-25, 50),
        figsize=(12, 5)
    )


def plot_pupil_after_event_by_session(pupil_df):
    """Create bar plot showing average pupil size changes after event by session."""
    # Calculate mean pupil size during post-ring period
    df = pupil_df.copy()
    df['pupil_mean'] = df['pupil_percent'].apply(lambda x: np.array(x)[120:].mean() * 100)
    
    # Compute team-level means
    team_avg = df.groupby(['teamID', 'role', 'sessionID', 'human_ai'])['pupil_mean'].mean().reset_index()
    
    create_bar_plot_with_trajectories(
        data_df=team_avg,
        grouping_vars=['role', 'sessionID'],
        condition_labels={'S1': 'Session 1', 'S2': 'Session 2', 'S3': 'Session 3'},
        value_col='pupil_mean',
        plot_title_prefix='Experimental Session',
        save_filename='supp_pupil_after_event_by_session.png',
        ylabel='Pupil size change (%)',
        ylim_range=(-60, 80),
        figsize=(12, 5)
    )

# ======================== PART 3: BLINK RATE ANALYSIS ========================

def plot_blink_rate_by_role_difficulty(all_human_blink_df_raw, human_ai_blink_df_raw):
    """Create bar plot showing blink rates by role and difficulty with individual team trajectories."""
    # Group by and aggregate blink frequencies
    all_human_blink_df = all_human_blink_df_raw.groupby(['role', 'difficulty', 'teamID'])['blinkFreq'].apply(np.nanmean)
    human_ai_blink_df = human_ai_blink_df_raw.groupby(['role', 'difficulty', 'teamID'])['blinkFreq'].apply(np.nanmean)
    
    # Convert to DataFrame for easier manipulation
    all_human_blink_df = all_human_blink_df.reset_index(name='blink_rate')
    all_human_blink_df['human_ai'] = 'human'
    human_ai_blink_df = human_ai_blink_df.reset_index(name='blink_rate')
    human_ai_blink_df['human_ai'] = 'ai'
    
    # Combine datasets
    combined_blink_df = pd.concat([all_human_blink_df, human_ai_blink_df], ignore_index=True)
    
    create_bar_plot_with_trajectories(
        data_df=combined_blink_df,
        grouping_vars=['role', 'difficulty'],
        condition_labels={'Easy': 'Easy', 'Medium': 'Intermediate', 'Hard': 'Hard'},
        value_col='blink_rate',
        plot_title_prefix='Difficulty Blink Rate',
        save_filename='supp_blink_rate_by_role_difficulty.png',
        ylabel='Blink rate',
        ylim_range=(0.05, 0.8),
        figsize=(12, 5)
    )


def plot_blink_rate_by_role_communication(all_human_blink_df_raw, human_ai_blink_df_raw):
    """Create bar plot showing blink rates by role and communication with individual team trajectories."""
    # Group by and aggregate blink frequencies
    all_human_blink_df = all_human_blink_df_raw.groupby(['role', 'communication', 'teamID'])['blinkFreq'].apply(np.nanmean)
    human_ai_blink_df = human_ai_blink_df_raw.groupby(['role', 'communication', 'teamID'])['blinkFreq'].apply(np.nanmean)
    
    # Convert to DataFrame for easier manipulation
    all_human_blink_df = all_human_blink_df.reset_index(name='blink_rate')
    all_human_blink_df['human_ai'] = 'human'
    human_ai_blink_df = human_ai_blink_df.reset_index(name='blink_rate')
    human_ai_blink_df['human_ai'] = 'ai'
    
    # Combine datasets
    combined_blink_df = pd.concat([all_human_blink_df, human_ai_blink_df], ignore_index=True)
    
    create_bar_plot_with_trajectories(
        data_df=combined_blink_df,
        grouping_vars=['role', 'communication'],
        condition_labels={'No': 'Incomm-\nunicado', 'Word': 'Command \nword', 'Free': 'Free \nspeech'},
        value_col='blink_rate',
        plot_title_prefix='Communication Blink Rate',
        save_filename='supp_blink_rate_by_role_communication.png',
        ylabel='Blink rate',
        ylim_range=(0.05, 0.8),
        figsize=(12, 5.5)
    )


def plot_blink_rate_by_role_session(all_human_blink_df_raw, human_ai_blink_df_raw):
    """
    Create bar plot showing blink rates by role and session with individual team trajectories.
    
    This function analyzes session-wise changes in blink frequency and performs
    repeated measures ANOVA and between-group comparisons.
    """
    # Aggregate blink frequencies by session, role and team
    all_human_blink_df = all_human_blink_df_raw.groupby(['role', 'sessionID', 'teamID'])['blinkFreq'].apply(np.nanmean)
    human_ai_blink_df = human_ai_blink_df_raw.groupby(['role', 'sessionID', 'teamID'])['blinkFreq'].apply(np.nanmean)

    # Convert to DataFrame and combine
    all_human_blink_df = all_human_blink_df.reset_index(name='blink_rate')
    all_human_blink_df['human_ai'] = 'human'
    human_ai_blink_df = human_ai_blink_df.reset_index(name='blink_rate')
    human_ai_blink_df['human_ai'] = 'ai'
    
    combined_blink_df = pd.concat([all_human_blink_df, human_ai_blink_df], ignore_index=True)

    # Use the generic bar plot function with custom session handling
    create_bar_plot_with_trajectories(
        data_df=combined_blink_df,
        grouping_vars=['role', 'sessionID'],
        condition_labels={'S1': 'Session 1', 'S2': 'Session 2', 'S3': 'Session 3'},
        value_col='blink_rate',
        plot_title_prefix='Session Blink Rate',
        save_filename='supp_blink_rate_by_role_session.png',
        ylabel='Blink rate',
        ylim_range=(0.05, 0.8),
        figsize=(12, 5)
    )


# ======================== MAIN EXECUTION FUNCTION ========================

def main():
    """
    Main execution function that orchestrates the supplementary eye tracking analysis workflow.
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
    
    # # PART 1: PUPIL SIZE TIME SERIES ANALYSIS
    # print("Generating pupil time series analysis plots...")
    # print("Pupil time series by role...")
    # plot_pupil_time_series_by_role(pupil_percent_change_df)
    
    # print("Pupil time series by difficulty...")
    # plot_pupil_time_series_by_difficulty(pupil_percent_change_df)
    
    print("Pupil time series by session...")
    plot_pupil_time_series_by_session(pupil_percent_change_df)
    
    # print("Pupil time series by communication...")
    # plot_pupil_time_series_by_communication(pupil_percent_change_df)
    

    # # PART 2: PUPIL SIZE AFTER EVENT ANALYSIS
    # print("Generating pupil after event analysis plots...")
    # print("Pupil after event by difficulty...")
    # plot_pupil_after_event_by_difficulty(pupil_percent_change_df)
    
    # print("Pupil after event by communication...")
    # plot_pupil_after_event_by_communication(pupil_percent_change_df)
    
    # print("Pupil after event by session...")
    # plot_pupil_after_event_by_session(pupil_percent_change_df)

    # # PART 3: BLINK RATE ANALYSIS
    # print("Generating blink rate analysis plots...")
    # print("Blink rate by role and difficulty...")
    # plot_blink_rate_by_role_difficulty(all_human_blink_df_raw, human_ai_blink_df_raw)
    
    # print("Blink rate by role and communication...")
    # plot_blink_rate_by_role_communication(all_human_blink_df_raw, human_ai_blink_df_raw)
    
    # print("Blink rate by role and session...")
    # plot_blink_rate_by_role_session(all_human_blink_df_raw, human_ai_blink_df_raw)
    

if __name__ == '__main__':
    main()