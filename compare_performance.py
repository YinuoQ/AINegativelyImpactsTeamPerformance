import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import statsmodels.formula.api as smf
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.anova import AnovaRM

def ttest_run(c1, c2, cat1, cat2):
    results = ttest_ind(cat1, cat2)
    df = pd.DataFrame({'categ1': c1,
                       'categ2': c2,
                       'tstat': results.statistic,
                       'pvalue': results.pvalue}, 
                       index = [0])    
    return df
    
def plot_performance_all(performance_df):
    a = performance_df[performance_df.human_ai == 'human'].groupby('teamID').performance.sum()
    b = performance_df[performance_df.human_ai == 'ai'].groupby('teamID').performance.sum()

    fig, ax = plt.subplots(1,1, figsize=(4, 6))
    plt.tight_layout(pad=2)

    ax.bar(0, np.mean(a), color="#F78474")
    ax.bar(1, np.mean(b), color="#57A0D3")
    ax.scatter([0]*18, a, c='k', s=5)
    ax.scatter([1]*12, b, c='k', s=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylim(0,1700)
    ax.set_yticks([0,200,400,600, 800, 1000, 1200, 1400, 1600], [0,200,400,600, 800, 1000, 1200, 1400, 1600])
    ax.set_xticks([0,1], ['All human', 'Human Alice'], fontsize=10)
    plt.savefig(f'plots/performance_all.png', dpi=300)


def get_difficulty_performance(performance_df):
    all_human_df = performance_df[performance_df.human_ai == 'human'].groupby(['teamID', 'difficulty', 'human_ai']).performance.sum().reset_index(name='performance')
    human_ai_df = performance_df[performance_df.human_ai == 'ai'].groupby(['teamID', 'difficulty', 'human_ai']).performance.sum().reset_index(name='performance')
    

    print("Easy")
    print(ttest_run('ai', 'human', human_ai_df.loc[human_ai_df['difficulty']=='Easy'].performance, all_human_df.loc[all_human_df['difficulty']=='Easy'].performance))
    print("Medium")
    print(ttest_run('ai', 'human', human_ai_df.loc[human_ai_df['difficulty']=='Medium'].performance, all_human_df.loc[all_human_df['difficulty']=='Medium'].performance))
    print("Hard")    
    print(ttest_run('ai', 'human', human_ai_df.loc[human_ai_df['difficulty']=='Hard'].performance, all_human_df.loc[all_human_df['difficulty']=='Hard'].performance))

    # Run repeated measures ANOVA
    anova = AnovaRM(
        data=all_human_df,
        depvar='performance',
        subject='teamID',
        within=['difficulty']
    ).fit()
    print("human")
    print(anova)

    # Run repeated measures ANOVA
    anova = AnovaRM(
        data=human_ai_df,
        depvar='performance',
        subject='teamID',
        within=['difficulty']
    ).fit()
    print("AI")
    print(anova)

    combine_df = pd.concat([all_human_df, human_ai_df], ignore_index=True)
    return combine_df

def get_communication_performance(performance_df):
    all_human_df = performance_df[performance_df.human_ai == 'human'].groupby(['teamID', 'communication', 'human_ai']).performance.sum().reset_index(name='performance')
    human_ai_df = performance_df[performance_df.human_ai == 'ai'].groupby(['teamID', 'communication', 'human_ai']).performance.sum().reset_index(name='performance')
    

    combine_df = pd.concat([all_human_df, human_ai_df], ignore_index=True)
    
    print("No")
    print(ttest_run('ai', 'human', human_ai_df.loc[human_ai_df['communication']=='No'].performance, all_human_df.loc[all_human_df['communication']=='No'].performance))
    print("Word")
    print(ttest_run('ai', 'human', human_ai_df.loc[human_ai_df['communication']=='Word'].performance, all_human_df.loc[all_human_df['communication']=='Word'].performance))
    print("Free")    
    print(ttest_run('ai', 'human', human_ai_df.loc[human_ai_df['communication']=='Free'].performance, all_human_df.loc[all_human_df['communication']=='Free'].performance))

    # Run repeated measures ANOVA
    anova = AnovaRM(
        data=all_human_df,
        depvar='performance',
        subject='teamID',
        within=['communication']
    ).fit()
    print("human")
    print(anova)

    # Run repeated measures ANOVA
    anova = AnovaRM(
        data=human_ai_df,
        depvar='performance',
        subject='teamID',
        within=['communication']
    ).fit()
    print("AI")
    print(anova)
    
    return combine_df

def plot_compaired_performance(plot_df, variable, sub_variables):

    fig, ax = plt.subplots(1,1, figsize=(4, 6))
    x = np.array(range(0,3))
    plt.tight_layout(pad=2)
    
    results_lst = []
    for player in ['human', 'ai']:
        sub_lst = []
        for key in sub_variables:
            sub_sub_lst = []
            for teamid in plot_df.query(f"human_ai == '{player}'")['teamID'].unique():
                temp_result = plot_df.query(f"human_ai == '{player}' and `{variable}` == '{key}' and teamID == '{teamid}'")['performance']
                if len(temp_result) > 0:
                    sub_sub_lst.append(temp_result.iloc[0])
                else:
                    sub_sub_lst.append(np.nan)
            sub_lst.append(sub_sub_lst)
        results_lst.append(sub_lst)

    width=0.4
    plt.bar(x-0.2, np.nanmean(np.array(results_lst[0]), axis=1), width=width, color='#F78474', label='Human-only',ecolor='gray')
    plt.bar(x+0.2, np.nanmean(np.array(results_lst[1]), axis=1), width=width, color='#57A0D3', label='Human-AI',ecolor='gray')
    plt.plot(x-0.2, np.array(results_lst[0]), c='r', alpha=0.2, marker='.')
    plt.plot(x+0.2, np.array(results_lst[1]), c='b', alpha=0.2, marker='.')

    ax.set_ylim(0,740)
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.set_xticks(x, sub_variables, fontsize=10)
    ax.set_yticks([0,200,400,600, 800], [0,200,400,600, 800], fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Performance (# ring passed)', fontsize=10)
    plt.savefig(f'plots/performance_{variable}.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    performance_df = pd.read_pickle("../data/performance.pkl")
    plot_performance_all(performance_df)

    difficulty_plot_df = get_difficulty_performance(performance_df)
    variable = 'difficulty'
    sub_variables = ['Easy', 'Medium', 'Hard']
    plot_compaired_performance(difficulty_plot_df, variable, sub_variables)

    communication_plot_df = get_communication_performance(performance_df)
    variable = 'communication'
    sub_variables = ['No', 'Word', 'Free']
    plot_compaired_performance(communication_plot_df, variable, sub_variables)






    