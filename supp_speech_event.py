import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from statistical_test import t_test_two_groups
from compare_speech_event import load_or_process_speech_data


def plot_error_ellipse(x, y, x_err, y_err, color, marker='o', alpha=0.9, err_color='k', zorder=1):
    """
    Plot error ellipse with center point and error bars for 2D data visualization.
    
    Parameters:
        x, y (float): Center coordinates
        x_err, y_err (float): Standard errors in x and y directions
        color (str): Color for the ellipse and marker
        marker (str): Marker style for center point
        alpha (float): Transparency level
        err_color (str): Color for error bars
        zorder (int): Drawing order for layering
    """
    # Plot error bars and center point
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt=marker, 
                color=err_color, linestyle='-', 
                markersize=max([int(x_err*5), 2]), 
                linewidth=1, alpha=alpha, zorder=zorder+2)
    
    # Add error ellipse
    ellipse = Ellipse(xy=(x, y), width=x_err*2, height=y_err*2,
                     color=color, alpha=alpha, edgecolor=None, zorder=zorder)
    plt.gca().add_patch(ellipse)


def plot_communication_ellipses(speech_event_df, condition_col, condition_values, 
                              colors, filename, title, xlim, ylim, xticks, yticks, 
                              legend_labels, figsize=(10, 6), special_err_colors=None):
    """
    Generic function to create scatter plots with error ellipses for communication analysis.
    
    This function handles the common pattern of creating ellipse plots showing speech frequency
    vs duration across different conditions, with separate ellipses for each team type and condition.
    
    Parameters:
        speech_event_df (pd.DataFrame): DataFrame containing speech event data
        condition_col (str): Column name for the condition to analyze
        condition_values (list): List of condition values to include
        colors (list): List of colors for ellipses (should have 2*len(condition_values) colors)
        filename (str): Output filename (without extension)
        title (str): Plot title
        xlim, ylim (tuple): Axis limits
        xticks, yticks (list): Tick positions
        legend_labels (list): Labels for legend
        figsize (tuple): Figure size (default: (10, 6))
        special_err_colors (dict): Special error bar colors for specific conditions (optional)
        
    Returns:
        pd.DataFrame: Processed frequency-duration data for statistical analysis
    """
    # Calculate team-level means for frequency and duration by condition
    team_freq_means = speech_event_df.groupby(['human_ai', 'teamID', condition_col])['speech_frequency'].mean()
    team_duration_means = speech_event_df.groupby(['human_ai', 'teamID', condition_col])['speech_duration'].mean()

    # Create DataFrame combining frequency and duration data
    freq_dur_df = pd.DataFrame({
        'human_ai': [key[0] for key in team_freq_means.index],
        'teamID': [key[1] for key in team_freq_means.index],
        condition_col: [key[2] for key in team_freq_means.index],
        'freq': team_freq_means.values,
        'duration': team_duration_means.values
    })

    # Create the visualization
    plt.figure(figsize=figsize)
    
    # Plot ellipses for each team type and condition combination
    color_idx = 0
    for human_ai in ['human', 'ai']:
        for zorder, condition in enumerate(condition_values):
            # Filter data for current combination
            data_subset = freq_dur_df[
                (freq_dur_df.human_ai == human_ai) & 
                (freq_dur_df[condition_col] == condition)
            ]
            
            if len(data_subset) > 0:
                # Calculate means and standard errors
                freq_mean = data_subset.freq.mean()
                dur_mean = data_subset.duration.mean()
                freq_sem = data_subset.freq.std() / np.sqrt(len(data_subset))
                dur_sem = data_subset.duration.std() / np.sqrt(len(data_subset))
                
                # Choose error bar color
                if special_err_colors and condition in special_err_colors:
                    err_color = special_err_colors[condition]
                else:
                    err_color = 'k'
                
                # Plot ellipse
                plot_error_ellipse(freq_mean, dur_mean, freq_sem, dur_sem, 
                                 colors[color_idx], alpha=0.9, err_color=err_color, 
                                 zorder=np.abs(zorder-len(condition_values)+1))
            
            color_idx += 1

    # Customize plot appearance
    plt.rcParams['font.family'] = 'Helvetica'
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(legend_labels, ncol=2, fontsize=18)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(xticks, fontsize=18)
    plt.yticks(yticks, fontsize=18)
    plt.xlabel('Frequency', fontsize=20)
    plt.ylabel('Duration', fontsize=20)

    # Save plot
    plt.savefig(f'plots/supp_{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return freq_dur_df


def perform_statistical_comparisons(freq_dur_df, condition_col, condition_values, comparison_name):
    """
    Perform statistical comparisons for speech frequency and duration across conditions.
    
    Parameters:
        freq_dur_df (pd.DataFrame): DataFrame with frequency and duration data
        condition_col (str): Column name for the condition
        condition_values (list): List of condition values
        comparison_name (str): Name for the comparison (used in headers)
    """
    print(f"\nCommunication by {comparison_name} - Statistical Comparisons:")
    print("=" * (45 + len(comparison_name)))
    
    for condition in condition_values:
        human_freq = freq_dur_df[
            (freq_dur_df.human_ai == 'human') & 
            (freq_dur_df[condition_col] == condition)
        ]['freq']
        
        ai_freq = freq_dur_df[
            (freq_dur_df.human_ai == 'ai') & 
            (freq_dur_df[condition_col] == condition)
        ]['freq']
        
        human_dur = freq_dur_df[
            (freq_dur_df.human_ai == 'human') & 
            (freq_dur_df[condition_col] == condition)
        ]['duration']
        
        ai_dur = freq_dur_df[
            (freq_dur_df.human_ai == 'ai') & 
            (freq_dur_df[condition_col] == condition)
        ]['duration']
        
        # Compare frequency
        t_test_two_groups(human_freq, ai_freq, 'Human-only', 'Human-AI',
                         comparison_title=f"{condition} {comparison_name} - Speech Frequency")
        
        # Compare duration
        t_test_two_groups(human_dur, ai_dur, 'Human-only', 'Human-AI',
                         comparison_title=f"{condition} {comparison_name} - Speech Duration")
        
        print("=" * (45 + len(comparison_name)))


def plot_communication_by_difficulty(speech_event_df):
    """
    Create scatter plot comparing speech frequency vs duration across difficulty levels.
    """
    colors = ['#FDE6E6', '#F78474', '#711201', '#DBECFC', '#57A0D3', '#011447']
    legend_labels = ['Human-only Easy', 'Human-only Medium', 'Human-only Hard', 
                    'Human-AI Easy', 'Human-AI Medium', 'Human-AI Hard']
    special_err_colors = {'Hard': 'w'}  # White error bars for dark background
    
    freq_dur_df = plot_communication_ellipses(
        speech_event_df=speech_event_df,
        condition_col='difficulty',
        condition_values=['Easy', 'Medium', 'Hard'],
        colors=colors,
        filename='comm_difficulty',
        title='Communication Patterns by Difficulty Level',
        xlim=(0.15, 1.05),
        ylim=(0, 0.65),
        xticks=[0.2, 0.4, 0.6, 0.8, 1.0],
        yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        legend_labels=legend_labels,
        special_err_colors=special_err_colors
    )
    
    perform_statistical_comparisons(freq_dur_df, 'difficulty', ['Easy', 'Medium', 'Hard'], 'Difficulty')


def plot_communication_by_session(speech_event_df):
    """
    Create scatter plot comparing speech frequency vs duration across sessions.
    """
    colors = ['#FDE6E6', '#F78474', '#711201', '#DBECFC', '#57A0D3', '#011447']
    legend_labels = ['Human-only Session 1', 'Human-only Session 2', 'Human-only Session 3',
                    'Human-AI Session 1', 'Human-AI Session 2', 'Human-AI Session 3']
    special_err_colors = {'S3': 'w'}  # White error bars for dark background
    
    freq_dur_df = plot_communication_ellipses(
        speech_event_df=speech_event_df,
        condition_col='sessionID',
        condition_values=['S1', 'S2', 'S3'],
        colors=colors,
        filename='comm_session',
        title='Communication Patterns by Session',
        xlim=(0.15, 0.85),
        ylim=(0.0, 1.05),
        xticks=[0.2, 0.4, 0.6, 0.8],
        yticks=[0.2, 0.4, 0.6, 0.8, 1.0],
        legend_labels=legend_labels,
        special_err_colors=special_err_colors
    )
    
    perform_statistical_comparisons(freq_dur_df, 'sessionID', ['S1', 'S2', 'S3'], 'Session')


def plot_communication_by_comm_type(speech_event_df):
    """
    Create scatter plot comparing speech frequency vs duration across communication types.
    """
    colors = ['#FDE6E6', '#F78474', '#DBECFC', '#57A0D3']
    legend_labels = ['Human-only CW', 'Human-only FS', 
                    'Human-AI CW', 'Human-AI FS']
    
    freq_dur_df = plot_communication_ellipses(
        speech_event_df=speech_event_df,
        condition_col='communication',
        condition_values=['Word', 'Free'],
        colors=colors,
        filename='comm_communication',
        title='Communication Patterns by Communication Type',
        xlim=(0.15, 0.95),
        ylim=(0.05, 0.65),
        xticks=[0.2, 0.4, 0.6, 0.8],
        yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        legend_labels=legend_labels
    )
    
    perform_statistical_comparisons(freq_dur_df, 'communication', ['Word', 'Free'], 'Communication Type')


def plot_communication_by_role(speech_event_df):
    """
    Create scatter plot comparing speech frequency vs duration across participant roles.
    """
    colors = ['#FDE6E6', '#F78474', '#711201', '#DBECFC', '#57A0D3', '#011447']
    legend_labels = ['Human-only Yaw Pilot', 'Human-only Pitch Pilot', 'Human-only Thrust Pilot',
                    'Human-AI Yaw Pilot', 'Human-AI Pitch Pilot', 'Human-AI Thrust Pilot']
    special_err_colors = {'Thrust': 'w'}  # White error bars for dark background
    
    freq_dur_df = plot_communication_ellipses(
        speech_event_df=speech_event_df,
        condition_col='role',
        condition_values=['Yaw', 'Pitch', 'Thrust'],
        colors=colors,
        filename='comm_role',
        title='Communication Patterns by Participant Role',
        xlim=(0.15, 0.85),
        ylim=(0.05, 0.65),
        xticks=[0.2, 0.4, 0.6, 0.8],
        yticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        legend_labels=legend_labels,
        special_err_colors=special_err_colors
    )
    
    perform_statistical_comparisons(freq_dur_df, 'role', ['Yaw', 'Pitch', 'Thrust'], 'Role')


def main():
    """
    Main execution function that orchestrates the speech analysis workflow.
    """
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    
    # Load or process speech event data
    speech_event_df = load_or_process_speech_data()
    
    print("\n" + "="*80)
    
    print("Generating communication analysis by difficulty...")
    plot_communication_by_difficulty(speech_event_df)
    
    print("\nGenerating communication analysis by session...")
    plot_communication_by_session(speech_event_df)
    
    print("\nGenerating communication analysis by communication type...")
    plot_communication_by_comm_type(speech_event_df)
    
    print("\nGenerating communication analysis by role...")
    plot_communication_by_role(speech_event_df)
    


if __name__ == '__main__':
    main()