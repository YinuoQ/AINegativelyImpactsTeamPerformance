"""
Comparison Tests Module

This module provides standardized statistical comparison functions with consistent formatting
for comparing pairs of data groups. Maintains the exact output format from your original code.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.anova import AnovaRM 



def t_test_two_groups(group1_data, group2_data, group1_name, group2_name, comparison_title="Statistical Comparison"):
    """
    Perform independent t-test comparing two groups with standardized output format.
    
    This function maintains the exact same output format as your original code,
    including descriptive statistics and formatted t-test results.
    
    Parameters:
        group1_data (array-like): Data for the first group
        group2_data (array-like): Data for the second group
        group1_name (str): Name of the first group (e.g., "Human", "Control")
        group2_name (str): Name of the second group (e.g., "AI", "Treatment")
        comparison_title (str): Title for the comparison section
        
    Returns:
        dict: Dictionary containing test results
    """
    # Print the comparison header
    print(f"{comparison_title}:")
    print("=" * len(comparison_title))
    
    # Clean the data by removing NaN values
    clean_group1 = pd.Series(group1_data).dropna()
    clean_group2 = pd.Series(group2_data).dropna()
    
    # Check if we have sufficient data
    if len(clean_group1) < 2 or len(clean_group2) < 2:
        print(f"ERROR: Insufficient data for comparison (Group 1: {len(clean_group1)}, Group 2: {len(clean_group2)})")
        return {"error": "Insufficient data"}
    
    # Perform independent t-test with NaN policy
    t_stat, p_value = ttest_ind(clean_group1, clean_group2, nan_policy='omit')
    
    # Print results in the exact format from your original code
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  {group1_name} mean: {clean_group1.mean():.2f} (±{clean_group1.std():.2f})")
    print(f"  {group2_name} mean: {clean_group2.mean():.2f} (±{clean_group2.std():.2f})")
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "group1_mean": clean_group1.mean(),
        "group1_std": clean_group1.std(),
        "group2_mean": clean_group2.mean(),
        "group2_std": clean_group2.std(),
        "group1_n": len(clean_group1),
        "group2_n": len(clean_group2)
    }

def repeated_measure_ANOVA(input_lst, columns_lst, var_name, value_name): 
    """
    Perform repeated measures ANOVA with post-hoc pairwise comparisons.
    
    This function converts wide-format data to long format, runs repeated measures ANOVA,
    and performs post-hoc paired t-tests if the main effect is significant.
    
    Parameters:
        input_lst (list): List of data arrays for each condition
        columns_lst (list): List of condition names
        var_name (str): Name of the within-subjects variable
        value_name (str): Name of the dependent variable
    """
    # Convert to DataFrame: each row = one subject (team), columns = conditions
    df_wide = pd.DataFrame(np.array(input_lst).T, columns=columns_lst)
    df_wide['teamID'] = df_wide.index.astype(str)  # Convert index to string for subject ID

    # Convert to long format for ANOVA
    df_long = pd.melt(
        df_wide, 
        id_vars=['teamID'], 
        value_vars=columns_lst,
        var_name=var_name, 
        value_name=value_name
    )
    
    # Pivot back to wide format and clean NaN values
    df_wide = df_long.pivot(index='teamID', columns=var_name, values=value_name)
    df_wide_clean = df_wide.dropna()

    # Convert back to long format for ANOVA
    df_clean = df_wide_clean.reset_index().melt(
        id_vars='teamID', 
        value_name=value_name, 
        var_name=var_name
    )
 
    # Run repeated measures ANOVA
    anova = AnovaRM(data=df_clean, depvar=value_name, subject='teamID', within=[var_name])
    res = anova.fit()
    print(res)

    # Perform post-hoc tests if main effect is significant
    if res.summary().tables[0]['Pr > F'].iloc[0] < 0.05:
        print("\nPost-hoc pairwise comparisons:")
        
        # Get data arrays for each condition
        cond1 = df_clean[df_clean[var_name] == columns_lst[0]][value_name]
        cond2 = df_clean[df_clean[var_name] == columns_lst[1]][value_name]
        cond3 = df_clean[df_clean[var_name] == columns_lst[2]][value_name]

        # Run paired t-tests for all pairwise comparisons
        t1, p1 = ttest_rel(cond1, cond2)
        t2, p2 = ttest_rel(cond1, cond3)
        t3, p3 = ttest_rel(cond2, cond3)

        print(f"{columns_lst[0]} vs {columns_lst[1]}: t = {t1:.3f}, p = {p1:.4f}")
        print(f"{columns_lst[0]} vs {columns_lst[2]}: t = {t2:.3f}, p = {p2:.4f}")
        print(f"{columns_lst[1]} vs {columns_lst[2]}: t = {t3:.3f}, p = {p3:.4f}")


