import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistical_test import t_test_two_groups, repeated_measure_ANOVA


def plot_helpfulness_ratings(questionnaire_df):
    """
    Create bar plot showing helpfulness ratings across sessions for human-only vs human-AI teams.
    
    This function analyzes perceived helpfulness of the thrust pilot across sessions,
    comparing human thrust pilots in human-only teams vs AI thrust pilots in human-AI teams.
    Includes error bars (SEM) and performs statistical comparisons.
    
    Parameters:
        questionnaire_df (pd.DataFrame): DataFrame containing questionnaire responses with columns
                                       ['teamID', 'sessionID', 'human_ai', 'helpfulness', 'leader']
    """
    # Filter data to only include helpfulness ratings (remove NaN values)
    helpfulness_df = questionnaire_df[questionnaire_df.helpfulness.notna()].copy()
    
    # Remove leader column for cleaner processing
    helpfulness_df = helpfulness_df.drop(columns='leader', errors='ignore')
    
    # Calculate session-wise means and standard errors for each team type
    sessions = [1, 2, 3]
    x = np.array([0, 1, 2])
    width = 0.35
    
    # Human-only teams statistics
    human_means = helpfulness_df[
        helpfulness_df.human_ai == 'human'
    ].groupby(['sessionID'], sort=False)['helpfulness'].agg(['mean', 'std', 'count'])
    
    human_sem = human_means['std'] / np.sqrt(human_means['count'])
    
    # Human-AI teams statistics  
    ai_means = helpfulness_df[
        helpfulness_df.human_ai == 'ai'
    ].groupby(['sessionID'], sort=False)['helpfulness'].agg(['mean', 'std', 'count'])
    
    ai_sem = ai_means['std'] / np.sqrt(ai_means['count'])
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    plt.tight_layout(pad=2)

    # Create bar plots with error bars
    ax.bar(x - width/2, human_means['mean'], width=width, yerr=human_sem, 
           color='#F78474', label='Human Thrust Pilot')
    ax.bar(x + width/2, ai_means['mean'], width=width, yerr=ai_sem, 
           color='#57A0D3', label='AI Thrust Pilot')

    # Customize plot appearance
    ax.set_ylim(1.5, 5)
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], fontsize=10)
    ax.set_yticks([1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    ax.set_ylabel('Helpfulness Rating', fontsize=10)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save plot
    plt.savefig('plots/helpfulness_thrust.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Perform between-group statistical comparisons for each session
    print("Helpfulness Ratings - Between-Group Comparisons:")
    print("=" * 55)
    
    for i, session in enumerate(sessions):
        human_data = helpfulness_df[
            (helpfulness_df.human_ai == 'human') & 
            (helpfulness_df.sessionID == session)
        ]['helpfulness']
        
        ai_data = helpfulness_df[
            (helpfulness_df.human_ai == 'ai') & 
            (helpfulness_df.sessionID == session)
        ]['helpfulness']
        

        t_test_two_groups(human_data, ai_data, 'Human Pilot', 'AI Pilot',
                         comparison_title=f"Session {i+1} Helpfulness")
        print("=" * 55)

    # Perform within-group repeated measures ANOVA
    print("Repeated Measures ANOVA - Human-only teams (Helpfulness):")
    print("=" * 60)
    human_helpfulness_data = helpfulness_df[
        helpfulness_df.human_ai == 'human'
    ][['teamID', 'sessionID', 'helpfulness']]
    # import IPython
    # IPython.embed()
    # assert False

    repeated_measure_ANOVA(
        [human_helpfulness_data[human_helpfulness_data.sessionID == session]['helpfulness'].values 
         for session in sessions],
        sessions,
        'sessionID',
        'helpfulness'
    )

    print("\nRepeated Measures ANOVA - Human-AI teams (Helpfulness):")
    print("=" * 60)
    ai_helpfulness_data = helpfulness_df[
        helpfulness_df.human_ai == 'ai'
    ][['teamID', 'sessionID', 'helpfulness']]
    
    repeated_measure_ANOVA(
        [ai_helpfulness_data[ai_helpfulness_data.sessionID == session]['helpfulness'].values 
         for session in sessions],
        sessions,
        'sessionID',
        'helpfulness'
    )


def plot_leadership_ratings(questionnaire_df):
    """
    Create bar plot showing leadership ratings across sessions for human-only vs human-AI teams.
    
    This function analyzes perceived leadership of the thrust pilot across sessions,
    comparing human thrust pilots in human-only teams vs AI thrust pilots in human-AI teams.
    Includes error bars (SEM) and performs statistical comparisons.
    
    Parameters:
        questionnaire_df (pd.DataFrame): DataFrame containing questionnaire responses with columns
                                       ['teamID', 'sessionID', 'human_ai', 'helpfulness', 'leader']
    """
    # Filter data to only include leadership ratings (remove NaN values)
    leadership_df = questionnaire_df[questionnaire_df.leader.notna()].copy()
    
    # Remove helpfulness column for cleaner processing
    leadership_df = leadership_df.drop(columns='helpfulness', errors='ignore')
    
    # Calculate session-wise means and standard errors for each team type
    sessions = [1,2,3]
    x = np.array([0, 1, 2])
    width = 0.35
    
    # Human-only teams statistics
    human_means = leadership_df[
        leadership_df.human_ai == 'human'
    ].groupby(['sessionID'], sort=False)['leader'].agg(['mean', 'std', 'count'])
    
    human_sem = human_means['std'] / np.sqrt(human_means['count'])
    
    # Human-AI teams statistics
    ai_means = leadership_df[
        leadership_df.human_ai == 'ai'
    ].groupby(['sessionID'], sort=False)['leader'].agg(['mean', 'std', 'count'])
    
    ai_sem = ai_means['std'] / np.sqrt(ai_means['count'])
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    plt.tight_layout(pad=2)

    # Create bar plots with error bars
    ax.bar(x - width/2, human_means['mean'], width=width, yerr=human_sem, 
           color='#F78474', label='Human Thrust Pilot')
    ax.bar(x + width/2, ai_means['mean'], width=width, yerr=ai_sem, 
           color='#57A0D3', label='AI Thrust Pilot')

    # Customize plot appearance
    ax.set_ylim(1.5, 4.2)
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], fontsize=10)
    ax.set_yticks([1.5, 2, 2.5, 3, 3.5, 4])
    ax.set_ylabel('Leadership Rating', fontsize=10)
    ax.set_xlabel('Session', fontsize=10)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save plot
    plt.savefig('plots/leader_thrust.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Perform between-group statistical comparisons for each session
    print("\nLeadership Ratings - Between-Group Comparisons:")
    print("=" * 55)
    
    for i, session in enumerate(sessions):
        human_data = leadership_df[
            (leadership_df.human_ai == 'human') & 
            (leadership_df.sessionID == session)
        ]['leader']
        
        ai_data = leadership_df[
            (leadership_df.human_ai == 'ai') & 
            (leadership_df.sessionID == session)
        ]['leader']
        
        t_test_two_groups(human_data, ai_data, 'Human Pilot', 'AI Pilot',
                         comparison_title=f"Session {i+1} Leadership")
        print("=" * 55)

    # Perform within-group repeated measures ANOVA
    print("Repeated Measures ANOVA - Human-only teams (Leadership):")
    print("=" * 60)
    human_leadership_data = leadership_df[
        leadership_df.human_ai == 'human'
    ][['teamID', 'sessionID', 'leader']]
    
    repeated_measure_ANOVA(
        [human_leadership_data[human_leadership_data.sessionID == session]['leader'].values 
         for session in sessions],
        sessions,
        'sessionID',
        'leader'
    )

    print("\nRepeated Measures ANOVA - Human-AI teams (Leadership):")
    print("=" * 60)
    ai_leadership_data = leadership_df[
        leadership_df.human_ai == 'ai'
    ][['teamID', 'sessionID', 'leader']]
    
    repeated_measure_ANOVA(
        [ai_leadership_data[ai_leadership_data.sessionID == session]['leader'].values 
         for session in sessions],
        sessions,
        'sessionID',
        'leader'
    )


def load_questionnaire_data():
    """
    Load questionnaire data from CSV file.
    
    Returns:
        pd.DataFrame: Questionnaire data with helpfulness and leadership ratings
    """
    print("Loading questionnaire data...")
    questionnaire_df = pd.read_csv('../physiological_behavioral_results/data/helpfulness_leadership.csv', index_col=0)
    print(f"Loaded {len(questionnaire_df)} questionnaire responses")
    
    # Display basic information about the data
    print(f"Sessions: {sorted(questionnaire_df['sessionID'].unique())}")
    print(f"Team types: {sorted(questionnaire_df['human_ai'].unique())}")
    print("Rating columns available:")
    if 'helpfulness' in questionnaire_df.columns:
        valid_helpfulness = questionnaire_df['helpfulness'].notna().sum()
        print(f"  - Helpfulness: {valid_helpfulness} valid responses")
    if 'leader' in questionnaire_df.columns:
        valid_leadership = questionnaire_df['leader'].notna().sum()
        print(f"  - Leadership: {valid_leadership} valid responses")
    
    return questionnaire_df


def main():
    """
    Main execution function that orchestrates the questionnaire analysis workflow.
    """
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    
    # Load questionnaire data
    questionnaire_df = load_questionnaire_data()
    
    print("\n" + "="*80)
    
    # Analyze helpfulness ratings
    print("Analyzing helpfulness ratings...")
    plot_helpfulness_ratings(questionnaire_df)
    
    print("\n" + "="*80)
    
    # Analyze leadership ratings
    print("Analyzing leadership ratings...")
    plot_leadership_ratings(questionnaire_df)
    


if __name__ == '__main__':
    main()