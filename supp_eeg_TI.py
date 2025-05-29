import mne
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from statistical_test import t_test_two_groups, repeated_measure_ANOVA


def create_TI_bar_plot(TI_df, grouping_column, condition_labels, condition_mapping, 
                       plot_title, save_filename, perform_repeated_anova=False):
    """
    Generic function to create bar plots for TI differences with individual team trajectories.
    
    This function visualizes how neural synchronization changes across different conditions
    for both team types and performs statistical comparisons.
    
    Parameters:
        TI_df (pd.DataFrame): DataFrame containing TI measurements
        grouping_column (str): Column name to group by ('difficulty' or 'communication')
        condition_labels (list): List of condition values to plot
        condition_mapping (dict): Dictionary mapping condition values to display labels
        plot_title (str): Title prefix for statistical comparisons
        save_filename (str): Filename to save the plot
        perform_repeated_anova (bool): Whether to perform repeated measures ANOVA
    """
    human_ai_lst = []
    x = np.arange(len(condition_labels))

    # Calculate condition-wise data for each team type
    for human_ai in ['human', 'ai']:
        condition_lst = []
        
        for condition in condition_labels:
            team_vals = []
            unique_teams = TI_df[TI_df['human_ai'] == human_ai].teamID.unique()
            
            for teamid in unique_teams:
                # Get TI_diff values for this team and condition
                team_condition_data = TI_df[
                    (TI_df['human_ai'] == human_ai) & 
                    (TI_df['teamID'] == teamid) & 
                    (TI_df[grouping_column] == condition)
                ]['TI_diff'].dropna().values
                
                if len(team_condition_data) > 0:
                    team_vals.append(np.nanmean(team_condition_data))
                else:
                    team_vals.append(np.nan)
            
            condition_lst.append(team_vals)
        human_ai_lst.append(condition_lst)
    
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
            for condition in condition_labels:
                condition_data = TI_df[
                    (TI_df['human_ai'] == human_ai) & 
                    (TI_df['teamID'] == teamid) & 
                    (TI_df[grouping_column] == condition)
                ]['TI_diff'].values
                
                team_vals.append(np.nanmean(condition_data) if len(condition_data) > 0 else np.nan)
            
            # Plot individual team trajectory
            ax.plot(x + offset, team_vals, color=color, alpha=0.5, 
                   marker='o', linewidth=0.8, markersize=3, markeredgewidth=0.0)

    # Customize plot appearance
    ax.set_xticks(x)
    ax.set_xticklabels([condition_mapping.get(label, label) for label in condition_labels])
    ax.set_ylabel("Inter-brain synchrony \n(EEG TI)", fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0.18, 0.42)
    ax.set_yticks([0.2, 0.25, 0.3, 0.35, 0.4])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()

    # Perform statistical comparisons for each condition
    print(f"\n{plot_title}-wise TI Comparisons:")
    print("=" * 50)
    
    for i, condition in enumerate(condition_labels):
        condition_display = condition_mapping.get(condition, condition)
        t_test_two_groups(human_ai_lst[0][i], human_ai_lst[1][i], 'Human-only', 'Human-AI', 
                         comparison_title=f"{plot_title} {condition_display}")
        print("=" * 50)

    # Save plot
    plt.savefig(f'plots/{save_filename}', dpi=300)
    plt.close()
    
    # # Optional repeated measures ANOVA
    # if perform_repeated_anova:
    #     print(f"\nRepeated Measures ANOVA for {plot_title}:")
    #     print("-" * 30)
    #     repeated_measure_ANOVA(human_ai_lst[0], condition_labels, grouping_column, 'TI_Human')
    #     repeated_measure_ANOVA(human_ai_lst[1], condition_labels, grouping_column, 'TI_AI')


def plot_TI_by_difficulty(TI_df):
    """
    Create bar plot showing TI differences across difficulty levels with individual team trajectories.
    
    This function visualizes how neural synchronization changes across difficulty levels for both
    team types and performs statistical comparisons for each difficulty level.
    
    Parameters:
        TI_df (pd.DataFrame): DataFrame containing TI measurements across difficulty levels
    """
    create_TI_bar_plot(
        TI_df=TI_df,
        grouping_column='difficulty',
        condition_labels=['Easy', 'Medium', 'Hard'],
        condition_mapping={'Easy': 'Easy', 'Medium': 'Medium', 'Hard': 'Hard'},
        plot_title='Difficulty',
        save_filename='supp_TI_difficulty.png',
        perform_repeated_anova=False
    )


def plot_TI_by_communication(TI_df):
    """
    Create bar plot showing TI differences across communication conditions with individual team trajectories.
    
    This function visualizes how neural synchronization changes across communication conditions for both
    team types and performs statistical comparisons for each communication condition.
    
    Parameters:
        TI_df (pd.DataFrame): DataFrame containing TI measurements across communication conditions
    """
    create_TI_bar_plot(
        TI_df=TI_df,
        grouping_column='communication',
        condition_labels=['No', 'Word', 'Free'],
        condition_mapping={'No': 'Incomm-\nunicado', 'Word': 'Command \nword', 'Free': 'Free \nspeech'},
        plot_title='Communication',
        save_filename='supp_TI_communication.png',
        perform_repeated_anova=False
    )



def main():
    """
    Main execution function that orchestrates the EEG TI analysis workflow.
    
    This function loads the preprocessed TI data and generates comparative visualizations
    across different experimental conditions.
    """
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    
    # Load preprocessed TI data
    print("Loading TI data...")
    TI_df = pd.read_pickle("../physiological_behavioral_results/data/TI.pkl")
    
    # Generate visualizations and statistical analyses
    print("\n" + "=" * 60)
    print("Generating TI analysis by difficulty...")
    plot_TI_by_difficulty(TI_df)
    
    print("\n" + "=" * 60)
    print("Generating TI analysis by communication...")
    plot_TI_by_communication(TI_df)
    



if __name__ == '__main__':
    main()