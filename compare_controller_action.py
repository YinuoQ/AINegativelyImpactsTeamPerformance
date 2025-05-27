import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statistical_test import t_test_two_groups

def read_files():
    """
    Read and load action data from pickle files for both human-only and human-AI teams.
    
    Returns:
        tuple: A tuple containing:
            - all_human_action_df (pd.DataFrame): Action data from human-only teams
            - human_ai_action_df (pd.DataFrame): Action data from human-AI teams
    """
    # Configure pandas to display all columns for better debugging
    pd.set_option('display.max_columns', None)
    
    # Load epoched action data for human-only teams
    all_human_action_df = pd.read_pickle('../physiological_behavioral_results/data/all_human/epoched_data/epoched_action.pkl')
    
    # Load epoched action data for human-AI teams
    human_ai_action_df = pd.read_pickle('../physiological_behavioral_results/data/human_ai/epoched_data/epoched_action.pkl')
    
    return all_human_action_df, human_ai_action_df


def get_number_of_action(input_arr):
    """
    Count the number of continuous chunks of non-zero values in an array.
    
    This function identifies sequences of consecutive non-zero values and counts
    how many such sequences exist. This is useful for analyzing controller input
    patterns where continuous actions represent deliberate control inputs.
    
    Parameters:
        input_arr (array-like): Array of numerical values representing actions
        
    Returns:
        int: Number of continuous chunks of non-zero numbers
        
    Example:
        [0, 1, 1, 0, 2, 0, 3, 3, 3] -> 3 chunks: [1,1], [2], [3,3,3]
    """
    continuous_chunks = 0
    in_chunk = False
    
    # Handle special case where input_arr is nested (length 2 with first element being the actual array)
    if len(input_arr) == 2:
        input_arr = input_arr[0]
    
    # Iterate through the array to identify and count continuous non-zero chunks
    for value in input_arr:
        # Start of a new chunk: non-zero value when not already in a chunk
        if value != 0 and not in_chunk:
            continuous_chunks += 1
            in_chunk = True
        # End of current chunk: zero value resets the chunk state
        elif value == 0:
            in_chunk = False
            
    return continuous_chunks


def plot_compared_actions_all(all_action_df):
    """
    Create violin plot comparing controller actions between human and AI participants across different roles.
    
    This function processes action data to calculate mean number of actions per participant,
    then creates a visualization comparing human vs AI performance across Thrust, Yaw, and Pitch roles.
    Statistical comparisons are performed using independent t-tests.
    
    Parameters:
        all_action_df (pd.DataFrame): Combined dataframe containing action data for all participants
                                    Must contain columns: teamID, role, human_ai, action
    """
    # Create a unique combination of team, role, and human/AI type
    team_role_human = all_action_df[['teamID', 'role', 'human_ai']].drop_duplicates().reset_index(drop=True)
    
    # Initialize actions column to store calculated action counts
    team_role_human['actions'] = 0
    
    # Calculate mean number of actions for each unique team-role-type combination
    for i in range(len(team_role_human)):
        temp_key = team_role_human.iloc[i]
        
        # Filter data for current team, role, and human/AI type
        temp_action_df = list(all_action_df.query(
            f"`teamID` == '{temp_key.teamID}' and `role` == '{temp_key.role}' and human_ai == '{temp_key.human_ai}'"
        )['action'])
        
        # Count actions for each trial/epoch
        number_action_lst = []
        for row in temp_action_df:
            number_action_lst.append(get_number_of_action(row))
        
        # Calculate mean actions (handle case where no actions occurred)
        if np.sum(number_action_lst) == 0:
            numb_action = np.nan
        else:
            numb_action = np.mean(number_action_lst)
            
        team_role_human.at[i, 'actions'] = numb_action
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plt.tight_layout(pad=2)
    
    # Define color palette for human vs AI
    my_pal = {'human': "#F78474", 'ai': "#57A0D3"}
    
    # Create violin plot with box plots inside
    sns.violinplot(
        data=team_role_human, 
        x='role', 
        y='actions', 
        hue='human_ai', 
        inner='box', 
        saturation=1, 
        palette=my_pal, 
        ax=ax, 
        linewidth=0.5,
        order=['Thrust', 'Yaw', 'Pitch']
    )
    
    # Customize plot appearance
    ax.tick_params(axis='y', which='major', labelsize=10)
    
    # Set x-axis labels and ticks
    x = range(0, 3)
    x_labels = ['Thrust', 'Yaw', 'Pitch']
    ax.set_xticks(x, x_labels, fontsize=10)
    ax.set_ylabel('Number of controller actions', fontsize=10)
    
    # Customize legend and limits
    plt.legend(fontsize=15)
    plt.ylim(0, 6)
    
    # Remove top and right spines for cleaner appearance
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Set y-axis ticks
    plt.yticks([0, 2, 4, 6])
    
    # Save the plot
    plt.savefig('plots/actions_all.png', dpi=300)
    plt.close()
    
    # Perform statistical comparisons for each role
    print("Statistical comparisons (Human vs AI):")
    print("=" * 40)
    
    for role in ['Thrust', 'Yaw', 'Pitch']:
        # Extract action data for human participants in current role
        human_actions = team_role_human[
            (team_role_human.role == role) & 
            (team_role_human.human_ai == 'human')
        ].actions
        
        # Extract action data for AI participants in current role
        ai_actions = team_role_human[
            (team_role_human.role == role) & 
            (team_role_human.human_ai == 'ai')
        ].actions
        t_test_two_groups(human_actions, ai_actions, 'Human-only', 'Human-AI', comparison_title=f"{role} Action Differences")
        print("=" * 40)

def main():
    """
    Main execution function that orchestrates the data loading, processing, and visualization.
    """
    # Load data from pickle files
    all_human_action_df, human_ai_action_df = read_files()
    
    # Label human-only team data
    all_human_action_df['human_ai'] = 'human'
    
    # Filter human-AI data to include only specific communication conditions
    # Keep only 'No', 'Word', and 'Free' communication conditions
    human_ai_action_df = human_ai_action_df.loc[
        human_ai_action_df.communication.isin(['No', 'Word', 'Free'])
    ].reset_index(drop=True)
    
    # Label human-AI team data
    human_ai_action_df['human_ai'] = 'ai'
    
    # Combine both datasets for comparative analysis
    all_action_df = pd.concat([all_human_action_df, human_ai_action_df], ignore_index=True)
    
    # Generate comparison plot and statistical analysis
    plot_compared_actions_all(all_action_df)
    

if __name__ == '__main__':
    main()