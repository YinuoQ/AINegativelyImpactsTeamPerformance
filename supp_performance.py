import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
from statistical_test import t_test_two_groups, repeated_measure_ANOVA
from compare_performance import load_performance_data

def plot_performance_comparison_supp(plot_df, variable, sub_variables):
    """
    Create bar plot comparing performance across different conditions with individual trajectories.
    
    This function creates a grouped bar plot showing mean performance for each condition,
    with individual team trajectories overlaid to show within-team changes.
    
    Parameters:
        plot_df (pd.DataFrame): DataFrame containing performance data
        variable (str): Column name for the grouping variable (e.g., 'difficulty', 'communication')
        sub_variables (list): List of condition names to plot
    """
    # Prepare data structure for plotting
    performance_by_condition = []
    
    for team_type in ['human', 'ai']:
        team_conditions = []
        
        for condition in sub_variables:
            team_performance = []
            unique_teams = plot_df[plot_df['human_ai'] == team_type]['teamID'].unique()
            
            for teamid in unique_teams:
                # Get performance for this team and condition
                team_condition_data = plot_df[
                    (plot_df['human_ai'] == team_type) & 
                    (plot_df[variable] == condition) & 
                    (plot_df['teamID'] == teamid)
                ]['performance']
                
                if len(team_condition_data) > 0:
                    team_performance.append(team_condition_data.iloc[0])
                else:
                    team_performance.append(np.nan)
            
            team_conditions.append(team_performance)
        performance_by_condition.append(team_conditions)

    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    x = np.array(range(len(sub_variables)))
    width = 0.35
    
    # Calculate means for bar plots
    human_means = [np.nanmean(condition) for condition in performance_by_condition[0]]
    ai_means = [np.nanmean(condition) for condition in performance_by_condition[1]]
    
    # Create bar plots
    ax.bar(x - width/2, human_means, width=width, color='#F78474', 
           label='Human-only')
    ax.bar(x + width/2, ai_means, width=width, color='#57A0D3', 
           label='Human-AI')
    
    # Add individual team trajectories
    colors = {'human': '#B03B3B', 'ai': '#1E5F8A'}
    offsets = {'human': -width/2, 'ai': width/2}
    
    for i, (team_type, team_data) in enumerate(zip(['human', 'ai'], performance_by_condition)):
        # Plot individual team trajectories
        for team_idx in range(len(team_data[0])):  # Number of teams
            team_trajectory = [condition[team_idx] for condition in team_data]
            
            # Only plot if team has data for all conditions
            if not any(np.isnan(team_trajectory)):
                ax.plot(x + offsets[team_type], team_trajectory, 
                       color=colors[team_type], alpha=0.4, 
                       marker='o', linewidth=0.8, markersize=3, markeredgewidth=0.0)

    # Customize plot appearance
    ax.set_ylim(0, 740)
    ax.tick_params(axis='y', which='major', labelsize=18)
    ax.set_xticks(x, sub_variables, fontsize=18)
    ax.get_yaxis().set_ticks([0, 200, 400, 600, 800])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Performance', fontsize=24)
    plt.rcParams['font.family'] = 'Helvetica'
    # Save plot
    plt.savefig(f'plots/supp_performance_{variable}.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_session_performance(performance_df):
    """
    Analyze performance across sessions for both team types.
    
    This function aggregates performance by experimental session, performs statistical comparisons
    between team types for each condition, and runs repeated measures ANOVA within each team type.
    
    Parameters:
        performance_df (pd.DataFrame): DataFrame containing performance data
        
    Returns:
        pd.DataFrame: Combined DataFrame with session-level performance data
    """
    # Aggregate performance by team, session, and team type
    human_sess_df = performance_df[performance_df.human_ai == 'human'].groupby(
        ['teamID', 'sessionID', 'human_ai']
    )['performance'].sum().reset_index(name='performance')
    
    ai_sess_df = performance_df[performance_df.human_ai == 'ai'].groupby(
        ['teamID', 'sessionID', 'human_ai']
    )['performance'].sum().reset_index(name='performance')

    # Perform between-group comparisons for each session condition
    print("Performance Comparison by Session:")
    print("=" * 50)
    
    session_ids = ['S1', 'S2', 'S3']
    for sess in session_ids:
        human_data = human_sess_df[
            human_sess_df['sessionID'] == sess
        ]['performance']
        
        ai_data = ai_sess_df[
            ai_sess_df['sessionID'] == sess
        ]['performance']
        
        t_test_two_groups(human_data, ai_data, 'Human-only', 'Human-AI',
                         comparison_title=f"{sess} Session")
        print("=" * 50)

    # Run repeated measures ANOVA for human-only teams
    print("Repeated Measures ANOVA - Human-only teams (Session):")
    print("=" * 60)

    # import IPython
    # IPython.embed()
    # assert False
    # Count the number of sessions per team
    session_counts = human_sess_df.groupby('teamID')['sessionID'].count()

    # Identify teams with at least 3 sessions
    valid_teams = session_counts[session_counts >= 3].index

    # Filter the original DataFrame to keep only those teams
    filtered_df = human_sess_df[human_sess_df['teamID'].isin(valid_teams)].reset_index(drop=True)

    repeated_measure_ANOVA(
        [filtered_df[filtered_df['sessionID'] == sess]['performance'].values 
         for sess in session_ids],
        session_ids,
        'sessionID',
        'performance'
    )

    # Count the number of sessions per team
    session_counts = ai_sess_df.groupby('teamID')['sessionID'].count()

    # Identify teams with at least 3 sessions
    valid_teams = session_counts[session_counts >= 3].index

    # Filter the original DataFrame to keep only those teams
    filtered_df = ai_sess_df[ai_sess_df['teamID'].isin(valid_teams)].reset_index(drop=True)

    # Run repeated measures ANOVA for human-AI teams
    print("\nRepeated Measures ANOVA - Human-AI teams (Session):")
    print("=" * 60)
    repeated_measure_ANOVA(
        [filtered_df[filtered_df['sessionID'] == sess]['performance'].values 
         for sess in session_ids],
        session_ids,
        'sessionID',
        'performance'
    )

    # Combine datasets for plotting
    combined_df = pd.concat([human_sess_df, ai_sess_df], ignore_index=True)
    return combined_df

def main():
    """
    Main execution function that orchestrates the performance analysis workflow.
    """
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    
    # Load performance data
    performance_df = load_performance_data()
    
    # Analyze performance by session 
    print("Analyzing performance by experinemntal sessions...")
    session_plot_df = analyze_session_performance(performance_df)
    plot_performance_comparison_supp(session_plot_df, 'sessionID', ['S1', 'S2', 'S3'])


if __name__ == '__main__':
    main()