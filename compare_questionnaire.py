import os
import glob
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM

def repeated_measure_ANOVA(df, value_key):
    # Step 1: Pivot to wide format so each team is one row
    df_wide = df.pivot(index='teamID', columns='sessionID', values=value_key)

    # Step 2: Convert to long format for repeated measures ANOVA
    df_long = df_wide.reset_index().melt(id_vars='teamID', var_name='sessionID', value_name=value_key)

    # Step 3: Make sure sessionID is treated as categorical
    df_long['sessionID'] = df_long['sessionID'].astype(str)

    # Step 4: Run repeated measures ANOVA
    anova = AnovaRM(data=df_long, depvar=value_key, subject='teamID', within=['sessionID'])
    res = anova.fit()

    # Output results
    print(res)

def plot_thrust_helpfulness(questionnaire_df):
    fig, ax = plt.subplots(1,1, figsize=(5, 6))
    x = np.array([0,1,2])
    plt.tight_layout(pad=2)

    helpfulness_df = questionnaire_df[questionnaire_df.helpfulness.notna()].drop(columns='leader')
    width=0.4
    human_role_sess_mean = helpfulness_df[(helpfulness_df.human_ai == 'human')].groupby(['sessionID'], sort=False).apply(lambda x: x.helpfulness.mean())
    human_role_sess_sem = helpfulness_df[(helpfulness_df.human_ai == 'human')].groupby(['sessionID'], sort=False).apply(lambda x: x.helpfulness.std()/np.sqrt(len(x)))
    ai_role_sess_mean = helpfulness_df[(helpfulness_df.human_ai == 'ai')].groupby(['sessionID'], sort=False).apply(lambda x: x.helpfulness.mean())
    ai_role_sess_sem = helpfulness_df[(helpfulness_df.human_ai == 'ai')].groupby(['sessionID'], sort=False).apply(lambda x: x.helpfulness.std()/np.sqrt(len(x)))

    plt.bar(x-0.2, human_role_sess_mean, width=width, yerr=human_role_sess_sem, color='#F78474')
    plt.bar(x+0.2, ai_role_sess_mean, width=width, yerr=ai_role_sess_sem, color='#57A0D3')

    ax.set_ylim(1.5,5)
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.set_xticks(x, ['Session 1','Session 2','Session 3'], fontsize=10)
    ax.get_yaxis().set_ticks([1.5,2,2.5,3,3.5,4,4.5])

    ax.set_ylabel('Helpfulness', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f'plots/helpfulness_thrust.png', dpi=300)
    plt.close()

    for i in range(3):
        ai_helpfulness = helpfulness_df[(helpfulness_df.human_ai == 'ai')].groupby(['sessionID'], sort=False).apply(lambda x: x.helpfulness)[i]
        human_helpfulness = helpfulness_df[(helpfulness_df.human_ai == 'human')].groupby(['sessionID'], sort=False).apply(lambda x: x.helpfulness)[i+1]
        print(ttest_ind(human_helpfulness, ai_helpfulness))

    df = helpfulness_df[(helpfulness_df.human_ai == 'human')][['teamID', 'sessionID', 'helpfulness']]
    print('human repeated measure ANOVA')
    repeated_measure_ANOVA(df, 'helpfulness')

    print('AI repeated measure ANOVA')
    df = helpfulness_df[(helpfulness_df.human_ai == 'ai')][['teamID', 'sessionID', 'helpfulness']]
    repeated_measure_ANOVA(df, 'helpfulness')
 


def plot_thrust_leader(questionnaire_df):
    fig, ax = plt.subplots(1,1, figsize=(5, 6))
    x = np.array([0,1,2])
    plt.tight_layout(pad=2)

    leadership_df = questionnaire_df[questionnaire_df.leader.notna()].drop(columns='helpfulness')
    width=0.4
    human_role_sess_mean = leadership_df[(leadership_df.human_ai == 'human')].groupby(['sessionID'], sort=False).apply(lambda x: x.leader.mean())
    human_role_sess_sem = leadership_df[(leadership_df.human_ai == 'human')].groupby(['sessionID'], sort=False).apply(lambda x: x.leader.std()/np.sqrt(len(x)))
    ai_role_sess_mean = leadership_df[(leadership_df.human_ai == 'ai')].groupby(['sessionID'], sort=False).apply(lambda x: x.leader.mean())
    ai_role_sess_sem = leadership_df[(leadership_df.human_ai == 'ai')].groupby(['sessionID'], sort=False).apply(lambda x: x.leader.std()/np.sqrt(len(x)))

    plt.bar(x-0.2, human_role_sess_mean, width=width, yerr=human_role_sess_sem, color='#F78474', label='Human ThrustPilot')
    plt.bar(x+0.2, ai_role_sess_mean, width=width, yerr=ai_role_sess_sem, color='#57A0D3', label='AI ThrustPilot')

    ax.set_ylim(1.5,4.2)
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.set_xticks(x, ['Session 1','Session 2','Session 3'], fontsize=10)
    ax.get_yaxis().set_ticks([1.5,2,2.5,3,3.5,4])

    ax.set_ylabel('Leadership', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    plt.savefig(f'plots/leader_thrust.png', dpi=300)
    plt.close()

    for i in range(3):
        ai_leader = leadership_df[(leadership_df.human_ai == 'ai')].groupby(['sessionID'], sort=False).apply(lambda x: x.leader)[i]
        human_leader = leadership_df[(leadership_df.human_ai == 'human')].groupby(['sessionID'], sort=False).apply(lambda x: x.leader)[i+1]
        print(ttest_ind(ai_leader, human_leader))

    df = leadership_df[(leadership_df.human_ai == 'human')][['teamID', 'sessionID', 'leader']]
    print('human repeated measure ANOVA')
    repeated_measure_ANOVA(df, 'leader')

    print('AI repeated measure ANOVA')
    df = leadership_df[(leadership_df.human_ai == 'ai')][['teamID', 'sessionID', 'leader']]
    repeated_measure_ANOVA(df, 'leader')
 

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    questionnaire_df = pd.read_csv('../data/helpfulness_leadership.csv', index_col=0)
    plot_thrust_helpfulness(questionnaire_df)
    plot_thrust_leader(questionnaire_df)








    