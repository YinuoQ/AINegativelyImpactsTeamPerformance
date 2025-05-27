import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM
from statistical_test import t_test_two_groups, repeated_measure_ANOVA


def plot_performance_all(performance_df):
    """
    Create bar plot comparing overall performance between human-only and human-AI teams.
    
    This function aggregates total performance scores by team and visualizes the comparison
    between team types with individual data points overlaid.
    
    Parameters:
        performance_df (pd.DataFrame): DataFrame containing performance data with columns
                                     ['teamID', 'human_ai', 'performance']
    """
    # Calculate total performance per team for each condition
    human_performance = performance_df[performance_df.human_ai == 'human'].groupby('teamID')['performance'].sum()
    ai_performance = performance_df[performance_df.human_ai == 'ai'].groupby('teamID')['performance'].sum()

    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    plt.tight_layout(pad=2)

    # Create bar plots for group means
    ax.bar(0, np.mean(human_performance), color="#F78474", label='Human-only')
    ax.bar(1, np.mean(ai_performance), color="#57A0D3", label='Human-AI')
    
    # Overlay individual team scores
    ax.scatter([0] * len(human_performance), human_performance, c='k', s=5, alpha=0.7)
    ax.scatter([1] * len(ai_performance), ai_performance, c='k', s=5, alpha=0.7)

    # Customize plot appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1700)
    ax.set_yticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600])
    ax.set_xticks([0, 1], ['Human-only', 'Human-AI'], fontsize=10)
    ax.set_ylabel('Performance', fontsize=10)

    # Perform statistical comparison
    print("Overall Performance Comparison:")
    print("=" * 40)
    t_test_two_groups(human_performance, ai_performance, 'Human-only', 'Human-AI',
                     comparison_title="Total Performance")

    # Save plot
    plt.savefig('plots/performance_all.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_difficulty_performance(performance_df):
    """
    Analyze performance across difficulty levels for both team types.
    
    This function aggregates performance by difficulty level, performs statistical comparisons
    between team types for each difficulty, and runs repeated measures ANOVA within each team type.
    
    Parameters:
        performance_df (pd.DataFrame): DataFrame containing performance data
        
    Returns:
        pd.DataFrame: Combined DataFrame with difficulty-level performance data
    """
    # Aggregate performance by team, difficulty, and team type
    human_difficulty_df = performance_df[performance_df.human_ai == 'human'].groupby(
        ['teamID', 'difficulty', 'human_ai']
    )['performance'].sum().reset_index(name='performance')
    
    ai_difficulty_df = performance_df[performance_df.human_ai == 'ai'].groupby(
        ['teamID', 'difficulty', 'human_ai']
    )['performance'].sum().reset_index(name='performance')

    # Perform between-group comparisons for each difficulty level
    print("Performance Comparison by Difficulty Level:")
    print("=" * 50)
    
    difficulty_levels = ['Easy', 'Medium', 'Hard']
    for difficulty in difficulty_levels:
        human_data = human_difficulty_df[
            human_difficulty_df['difficulty'] == difficulty
        ]['performance']
        
        ai_data = ai_difficulty_df[
            ai_difficulty_df['difficulty'] == difficulty
        ]['performance']
        
        t_test_two_groups(human_data, ai_data, 'Human-only', 'Human-AI',
                         comparison_title=f"{difficulty} Difficulty")
        print("=" * 50)

    # Run repeated measures ANOVA for human-only teams
    print("Repeated Measures ANOVA - Human-only teams (Difficulty):")
    print("=" * 60)
    repeated_measure_ANOVA(
        [human_difficulty_df[human_difficulty_df['difficulty'] == diff]['performance'].values 
         for diff in difficulty_levels],
        difficulty_levels,
        'difficulty',
        'performance'
    )

    # Run repeated measures ANOVA for human-AI teams
    print("\nRepeated Measures ANOVA - Human-AI teams (Difficulty):")
    print("=" * 60)
    repeated_measure_ANOVA(
        [ai_difficulty_df[ai_difficulty_df['difficulty'] == diff]['performance'].values 
         for diff in difficulty_levels],
        difficulty_levels,
        'difficulty',
        'performance'
    )

    # Combine datasets for plotting
    combined_df = pd.concat([human_difficulty_df, ai_difficulty_df], ignore_index=True)
    return combined_df


def analyze_communication_performance(performance_df):
    """
    Analyze performance across communication conditions for both team types.
    
    This function aggregates performance by communication type, performs statistical comparisons
    between team types for each condition, and runs repeated measures ANOVA within each team type.
    
    Parameters:
        performance_df (pd.DataFrame): DataFrame containing performance data
        
    Returns:
        pd.DataFrame: Combined DataFrame with communication-level performance data
    """
    # Aggregate performance by team, communication, and team type
    human_comm_df = performance_df[performance_df.human_ai == 'human'].groupby(
        ['teamID', 'communication', 'human_ai']
    )['performance'].sum().reset_index(name='performance')
    
    ai_comm_df = performance_df[performance_df.human_ai == 'ai'].groupby(
        ['teamID', 'communication', 'human_ai']
    )['performance'].sum().reset_index(name='performance')

    # Perform between-group comparisons for each communication condition
    print("Performance Comparison by Communication Condition:")
    print("=" * 50)
    
    communication_types = ['No', 'Word', 'Free']
    for comm_type in communication_types:
        human_data = human_comm_df[
            human_comm_df['communication'] == comm_type
        ]['performance']
        
        ai_data = ai_comm_df[
            ai_comm_df['communication'] == comm_type
        ]['performance']
        
        t_test_two_groups(human_data, ai_data, 'Human-only', 'Human-AI',
                         comparison_title=f"{comm_type} Communication")
        print("=" * 50)

    # Run repeated measures ANOVA for human-only teams
    print("Repeated Measures ANOVA - Human-only teams (Communication):")
    print("=" * 60)
    repeated_measure_ANOVA(
        [human_comm_df[human_comm_df['communication'] == comm]['performance'].values 
         for comm in communication_types],
        communication_types,
        'communication',
        'performance'
    )

    # Run repeated measures ANOVA for human-AI teams
    print("\nRepeated Measures ANOVA - Human-AI teams (Communication):")
    print("=" * 60)
    repeated_measure_ANOVA(
        [ai_comm_df[ai_comm_df['communication'] == comm]['performance'].values 
         for comm in communication_types],
        communication_types,
        'communication',
        'performance'
    )

    # Combine datasets for plotting
    combined_df = pd.concat([human_comm_df, ai_comm_df], ignore_index=True)
    return combined_df


def plot_performance_comparison(plot_df, variable, sub_variables):
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
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.set_xticks(x, sub_variables, fontsize=10)
    ax.get_yaxis().set_ticks([0, 200, 400, 600, 800])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Performance', fontsize=10)

    # Save plot
    plt.savefig(f'plots/performance_{variable}.png', dpi=300, bbox_inches='tight')
    plt.close()


def load_performance_data():
    """
    Load performance data from pickle file.
    
    Returns:
        pd.DataFrame: Performance data with team and condition information
    """
    print("Loading performance data...")
    performance_df = pd.read_pickle("../physiological_behavioral_results/data/performance.pkl")
    print(f"Loaded {len(performance_df)} performance records")
    return performance_df


def main():
    """
    Main execution function that orchestrates the performance analysis workflow.
    """
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    
    # Load performance data
    performance_df = load_performance_data()
    
    # Generate overall performance comparison
    print("Analyzing overall performance...")
    plot_performance_all(performance_df)
    
    print("\n" + "="*80)
    
    # Analyze performance by difficulty level
    print("Analyzing performance by difficulty level...")
    difficulty_plot_df = analyze_difficulty_performance(performance_df)
    plot_performance_comparison(difficulty_plot_df, 'difficulty', ['Easy', 'Medium', 'Hard'])
    
    print("\n" + "="*80)
    
    # Analyze performance by communication condition
    print("Analyzing performance by communication condition...")
    communication_plot_df = analyze_communication_performance(performance_df)
    plot_performance_comparison(communication_plot_df, 'communication', ['No', 'Word', 'Free'])
    

if __name__ == '__main__':
    main()