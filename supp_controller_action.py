import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from statistical_test import t_test_two_groups, repeated_measure_ANOVA
from compare_controller_action import read_files, get_number_of_action


def get_number_of_action_dataframe(all_action_df):
    """
    Calculate the number of controller actions for each trial and add to DataFrame.
    
    This function processes each trial's action data to count the number of discrete
    controller actions and creates a cleaned DataFrame for analysis.
    
    Parameters:
        all_action_df (pd.DataFrame): DataFrame containing action data for all trials
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with action counts added
    """
    # Initialize actions column to store calculated action counts
    all_action_df['actions'] = None
    
    # Calculate number of actions for each trial
    for i in tqdm(range(len(all_action_df)), desc="Computing action counts"):
        numb_action = get_number_of_action(all_action_df.iloc[i].action)
        all_action_df.at[i, 'actions'] = numb_action
    
    # Remove unnecessary columns to create clean analysis DataFrame
    columns_to_drop = ['location', 'ring_time', 'time', 'action', 'action_time']
    numb_action_df = all_action_df.drop(columns=columns_to_drop, errors='ignore')
    
    return numb_action_df



def plot_actions_by_condition_violin(plot_df, variable, sub_variables):
    my_pal = {'human': "#F78474", 'ai': "#57A0D3"}
    data_order = [f"{x}{y}" for x in ['Thrust', 'Yaw', 'Pitch'] for y in sub_variables]
    x_labels = [f"{x} \n {y}" for x in ['Thrust', 'Yaw', 'Pitch'] for y in sub_variables]
   

    fig, ax = plt.subplots(1,1, figsize=(10, 6))
    plt.tight_layout(pad=4)
    plot_df['role_var'] = plot_df['role'] + plot_df[variable]
    sns.violinplot(data=plot_df, x='role_var', y='actions', 
                   hue='human_ai', order=data_order,inner='box', 
                   saturation=2, palette=my_pal, ax=ax, 
                   linewidth=0.5,split=True, width=1.1)
    ax.set_ylim(-2,10)
    x = range(0,9)
    ax.tick_params(axis='y', which='major', labelsize=18)
    ax.set_xticks(x, x_labels, fontsize=18)
    ax.set_ylabel('Number of controller actions', fontsize=20)
    ax.get_legend().remove()
    ax.set(xlabel=None)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(f'plots/supp_actions_{variable}.png', dpi=300)
    plt.close()

def plot_actions_across_difficulty(numb_action_df):
    """
    Create violin plot showing controller actions across roles by difficulty levels.
    """
    diff_df = numb_action_df.groupby(['human_ai', 'role', 'difficulty', 'teamID']).apply(lambda x: np.nanmean(x.actions)).reset_index(name='actions')

    # Define custom order
    human_ai_order = ['human', 'ai']
    difficulty_order = ['Easy', 'Medium', 'Hard']
    role_order = ['Thrust', 'Yaw', 'Pitch']

    # Convert columns to categorical with specified order
    diff_df['human_ai'] = pd.Categorical(diff_df['human_ai'], categories=human_ai_order, ordered=True)
    diff_df['difficulty'] = pd.Categorical(diff_df['difficulty'], categories=difficulty_order, ordered=True)
    diff_df['role'] = pd.Categorical(diff_df['role'], categories=role_order, ordered=True)

    # Sort the dataframe
    diff_df_sorted = diff_df.sort_values(by=['human_ai', 'difficulty', 'role']).reset_index(drop=True)
    for col in ['human_ai', 'difficulty', 'role']:
        diff_df_sorted[col] = diff_df_sorted[col].astype(object)

    plot_actions_by_condition_violin(
        diff_df_sorted, 
        variable='difficulty',
        sub_variables=['Easy', 'Medium', 'Hard'],
    )

    for role in role_order:
        for diff in difficulty_order:
            temp_human_dta = diff_df_sorted.loc[(diff_df_sorted.role == role)
                                              & (diff_df_sorted.difficulty == diff)
                                              & (diff_df_sorted.human_ai == 'human')].actions
            temp_ai_dta = diff_df_sorted.loc[(diff_df_sorted.role == role)
                                              & (diff_df_sorted.difficulty == diff)
                                              & (diff_df_sorted.human_ai == 'ai')].actions
                                              
            t_test_two_groups(temp_human_dta, temp_ai_dta, 'human-only', 'human-AI', f"{role} {diff} human vs ai")


def plot_actions_across_communication(numb_action_df):
    """
    Create violin plot showing controller actions across roles by communication conditions.
    """
    comm_df = numb_action_df.groupby(['human_ai', 'role', 'communication', 'teamID']).apply(lambda x: np.nanmean(x.actions)).reset_index(name='actions')
    
    # Define custom order
    human_ai_order = ['human', 'ai']
    communication_order = ['No', 'Word', 'Free']
    role_order = ['Thrust', 'Yaw', 'Pitch']
    
    # Convert columns to categorical with specified order
    comm_df['human_ai'] = pd.Categorical(comm_df['human_ai'], categories=human_ai_order, ordered=True)
    comm_df['communication'] = pd.Categorical(comm_df['communication'], categories=communication_order, ordered=True)
    comm_df['role'] = pd.Categorical(comm_df['role'], categories=role_order, ordered=True)
    
    # Sort the dataframe
    comm_df_sorted = comm_df.sort_values(by=['human_ai', 'communication', 'role']).reset_index(drop=True)
    
    for col in ['human_ai', 'communication', 'role']:
        comm_df_sorted[col] = comm_df_sorted[col].astype(object)
    
    plot_actions_by_condition_violin(
        comm_df_sorted,
        variable='communication',
        sub_variables=['No', 'Word', 'Free']
    )
    
    for role in role_order:
        for comm in communication_order:
            temp_human_data = comm_df_sorted.loc[(comm_df_sorted.role == role)
                                              & (comm_df_sorted.communication == comm)
                                              & (comm_df_sorted.human_ai == 'human')].actions
            temp_ai_data = comm_df_sorted.loc[(comm_df_sorted.role == role)
                                              & (comm_df_sorted.communication == comm)
                                              & (comm_df_sorted.human_ai == 'ai')].actions
                                              
            t_test_two_groups(temp_human_data, temp_ai_data, 'Human-only', 'Human-AI', f"{role} {comm}")


def plot_actions_across_sessions(numb_action_df):
    """
    Create violin plot showing controller actions across roles by sessions.
    """
    sess_df = numb_action_df.groupby(['human_ai', 'role', 'sessionID', 'teamID']).apply(lambda x: np.nanmean(x.actions)).reset_index(name='actions')
    
    # Define custom order
    human_ai_order = ['human', 'ai']
    session_order = ['S1', 'S2', 'S3']
    role_order = ['Thrust', 'Yaw', 'Pitch']
    
    # Convert columns to categorical with specified order
    sess_df['human_ai'] = pd.Categorical(sess_df['human_ai'], categories=human_ai_order, ordered=True)
    sess_df['sessionID'] = pd.Categorical(sess_df['sessionID'], categories=session_order, ordered=True)
    sess_df['role'] = pd.Categorical(sess_df['role'], categories=role_order, ordered=True)
    
    # Sort the dataframe
    sess_df_sorted = sess_df.sort_values(by=['human_ai', 'sessionID', 'role']).reset_index(drop=True)
    
    for col in ['human_ai', 'sessionID', 'role']:
        sess_df_sorted[col] = sess_df_sorted[col].astype(object)
    
    plot_actions_by_condition_violin(
        sess_df_sorted,
        variable='sessionID',
        sub_variables=['S1', 'S2', 'S3']
    )
    
    for role in role_order:
        for sess in session_order:
            temp_human_data = sess_df_sorted.loc[(sess_df_sorted.role == role)
                                              & (sess_df_sorted.sessionID == sess)
                                              & (sess_df_sorted.human_ai == 'human')].actions
            temp_ai_data = sess_df_sorted.loc[(sess_df_sorted.role == role)
                                              & (sess_df_sorted.sessionID == sess)
                                              & (sess_df_sorted.human_ai == 'ai')].actions
                                              
            t_test_two_groups(temp_human_data, temp_ai_data, 'Human-only', 'Human-AI', f"{role} {sess}")




def main():
    """
    Main execution function that orchestrates the data loading, processing, and visualization.
    """
    print("Loading controller action data...")
    
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
    
    print("Processing action count data...")
    numb_action_df = get_number_of_action_dataframe(all_action_df)
    
    print("\nAnalyzing controller actions by difficulty...")
    plot_actions_across_difficulty(numb_action_df)
    
    print("\nAnalyzing controller actions by session...")
    plot_actions_across_sessions(numb_action_df)
    
    print("\nAnalyzing controller actions by communication...")
    plot_actions_across_communication(numb_action_df)


if __name__ == '__main__':
    main()