import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind, f_oneway

def read_files():
    pd.set_option('display.max_columns', None)
    # read all human action file
    all_human_action_df = pd.read_pickle('../data/all_human/epoched_data/epoched_action.pkl')
    # read human ai action file
    human_ai_action_df = pd.read_pickle('../data/human_ai/epoched_data/epoched_action.pkl')
    return all_human_action_df, human_ai_action_df


def get_number_of_action(input_arr):
    """
    Counts the number of continuous chunks of non-zero numbers in a list.

    Parameters:
    - input_arr: array of numbers

    Returns:
    - int: number of continuous chunks of non-zero numbers
    """
    continuous_chunks = 0
    in_chunk = False
    if len(input_arr) == 2:
        input_arr = input_arr[0]
    # Iterate through the list to count continuous chunks
    for value in input_arr:
        # If we find a non-zero value and we are not already in a chunk, it starts a new chunk
        if value != 0 and not in_chunk:
            continuous_chunks += 1
            in_chunk = True
        # If the current value is zero, it means we are not in a chunk anymore
        elif value == 0:
            in_chunk = False
    return continuous_chunks    

def plot_compaired_actions_all(all_action_df):
    team_role_human = all_action_df[['teamID', 'role', 'human_ai']].drop_duplicates().reset_index(drop=True)
    team_role_human['actions'] = 0

    for i in range(len(team_role_human)):
        temp_key = team_role_human.iloc[i]
        temp_action_df = list(all_action_df.query(f"`teamID` == '{temp_key.teamID}' and `role` == '{temp_key.role}' and human_ai == '{temp_key.human_ai}'")['action'])
        number_action_lst = []
        for row in temp_action_df:
            number_action_lst.append(get_number_of_action(row))
        if np.sum(number_action_lst) == 0:
            numb_action = np.nan
        else:
            numb_action = np.mean(number_action_lst)
        team_role_human.at[i, 'actions'] = numb_action

    fig, ax = plt.subplots(1,1, figsize=(10, 6))
    plt.tight_layout(pad=2)
    my_pal = {'human': "#F78474", 'ai': "#57A0D3"}
    sns.violinplot(data=team_role_human, x='role', y='actions', hue='human_ai', inner='box', saturation=1, palette=my_pal, ax=ax, linewidth=0.5,order=['Thrust', 'Yaw', 'Pitch'])
    ax.tick_params(axis='y', which='major', labelsize=10)
    x = range(0,3)
    x_labels = ['Thrust', 'Yaw', 'Pitch']
    ax.set_xticks(x, x_labels, fontsize=10)
    ax.set_ylabel('Number of controller actions', fontsize=10)
    plt.legend(fontsize=15)
    plt.ylim(0,6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.yticks([0,2,4,6])
    plt.savefig(f'plots/actions_all.png', dpi=300)
    plt.close()

    for role in ['Thrust', 'Yaw', 'Pitch']:
        a = team_role_human[(team_role_human.role == role) & (team_role_human.human_ai == 'human')].actions
        b = team_role_human[(team_role_human.role == role) & (team_role_human.human_ai == 'ai')].actions
        print(role)
        print(ttest_ind(a,b))
    


if __name__ == '__main__':
    all_human_action_df, human_ai_action_df = read_files()
    all_human_action_df['human_ai'] = 'human'
    human_ai_action_df = human_ai_action_df.loc[(human_ai_action_df.communication.isin(['No', 'Word', 'Free']))].reset_index(drop=True)
    human_ai_action_df['human_ai'] = 'ai'
    all_action_df = pd.concat([all_human_action_df, human_ai_action_df], ignore_index=True)
    plot_compaired_actions_all(all_action_df)





