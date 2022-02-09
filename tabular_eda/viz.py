from dython.nominal import associations
from typing import Union, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

def plot_var_cor(x: Union[pd.DataFrame, np.ndarray], ax=None, return_values: bool = False, **kwargs) -> Optional[np.ndarray]:
    """
    Given a DataFrame, plot the correlation between columns. Function assumes all numeric continuous data. It masks the top half of the correlation matrix,
    since this holds the same values.
    Decomissioned for use of the dython associations function.
    :param x: Dataframe to plot data from
    :param ax: Axis on which to plot the correlations
    :param return_values: return correlation matrix after plotting
    :param kwargs: Keyword arguments that are passed to `sns.heatmap`.
    :return: If return_values=True, returns correlation matrix of `x` as np.ndarray
    """
    if isinstance(x, pd.DataFrame):
        corr = x.corr().values
    elif isinstance(x, np.ndarray):
        corr = np.corrcoef(x, rowvar=False)
    else:
        raise ValueError('Unknown datatype given. Make sure a Pandas DataFrame or Numpy Array is passed for x.')

    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    if ax is None:
        f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, **kwargs)
    st.pyplot()
    if return_values:
        return corr


def plot_correlation_difference(reference: pd.DataFrame, comparison: pd.DataFrame, plot_diff: bool = True, cat_cols: list = None, annot=False):
    """
    Plot the association matrices for the `reference` dataframe, `comparison` dataframe and plot the difference between them. Has support for continuous and Categorical
    (Male, Female) data types. All Object and Category dtypes are considered to be Categorical columns if `dis_cols` is not passed.
    - Continuous - Continuous: Uses Pearson's correlation coefficient
    - Continuous - Categorical: Uses so called correlation ratio (https://en.wikipedia.org/wiki/Correlation_ratio) for both continuous - categorical and categorical - continuous.
    - Categorical - Categorical: Uses Theil's U, an asymmetric correlation metric for Categorical associations
    :param reference: DataFrame with reference data
    :param comparison: DataFrame with synthetic data
    :param plot_diff: Plot difference if True, else not
    :param cat_cols: List of Categorical columns
    :param boolean annot: Whether to annotate the plot with numbers indicating the associations.
    """
    assert isinstance(reference, pd.DataFrame), f'`reference` parameters must be a Pandas DataFrame'
    assert isinstance(comparison, pd.DataFrame), f'`comparison` parameters must be a Pandas DataFrame'
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    if cat_cols is None:
        cat_cols = reference.select_dtypes(['object', 'category'])
    if plot_diff:
        fig, ax = plt.subplots(1, 3, figsize=(24, 7))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
  
    reference_corr = associations(reference, nominal_columns=cat_cols, plot=False, theil_u=True,
                             mark_columns=True, annot=annot, ax=ax[0], cmap=cmap)['corr']
    comparison_corr = associations(comparison, nominal_columns=cat_cols, plot=False, theil_u=True,
                             mark_columns=True, annot=annot, ax=ax[1], cmap=cmap)['corr']

    if plot_diff:
        diff = abs(reference_corr - comparison_corr)
        sns.set(style="white")
        sns.heatmap(diff, ax=ax[2], cmap=cmap, vmax=.3, square=True, annot=True, center=0,
                    linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')
        
    titles = ['reference', 'comparison', 'Difference'] if plot_diff else ['reference', 'comparison']
    for i, label in enumerate(titles):
        title_font = {'size': '18'}
        ax[i].set_title(label, **title_font)

    plt.tight_layout()

    st.pyplot()

    export_plots(reference, comparison, diff, annot, cat_cols)


def export_plots(reference, comparison, diff, annot, cat_cols):
    """
    Export heatmap plots one by one for the pdf report
    Pass objects from plot_correlation_difference
    """

    assert isinstance(reference, pd.DataFrame), f'`reference` parameters must be a Pandas DataFrame'
    assert isinstance(comparison, pd.DataFrame), f'`comparison` parameters must be a Pandas DataFrame'
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    reference_corr = associations(reference, nominal_columns=cat_cols, plot=False, theil_u=True,
                             mark_columns=True, annot=annot, cmap=cmap)['corr']
    plt.tight_layout()
    plt.savefig('pdf_files/synthetic_data/corr_ref.png',orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0)

    comparison_corr = associations(comparison, nominal_columns=cat_cols, plot=False, theil_u=True,
                             mark_columns=True, annot=annot, cmap=cmap)['corr']
    plt.tight_layout()
    plt.savefig('pdf_files/synthetic_data/corr_comp.png',orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0)

    diff = abs(reference_corr - comparison_corr)
    sns.set(style="white")
    sns.heatmap(diff, cmap=cmap, vmax=.3, square=True, annot=True, center=0,
                linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f', cbar = False)
    plt.tight_layout()
    plt.savefig('pdf_files/synthetic_data/corr_diff.png',orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0) 

def plot_correlation_comparison(evaluators: List, annot=False):
    """
    Plot the correlation differences of multiple TableEvaluator objects.
    :param evaluators: list of TableEvaluator objects
    :param boolean annot: Whether to annotate the plots with numbers.
    """
    nr_plots = len(evaluators) + 1
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, ax = plt.subplots(2, nr_plots, figsize=(4 * nr_plots, 7))
    flat_ax = ax.flatten()
    flat_ax[nr_plots + 1].clear()
    comparison_corr = []
    reference_corr = associations(evaluators[0].reference, nominal_columns=evaluators[0].categorical_columns, plot=False, theil_u=True,
                             mark_columns=True, annot=False, cmap=cmap, cbar=False, ax=flat_ax[0])['corr']
    for i in range(1, nr_plots):
        cbar = True if i % (nr_plots - 1) == 0 else False
        comparison_corr.append(
            associations(evaluators[i - 1].comparison, nominal_columns=evaluators[0].categorical_columns, plot=False, theil_u=True,
                         mark_columns=True, annot=False, cmap=cmap, cbar=cbar, ax=flat_ax[i])['corr']
        )
        if i % (nr_plots - 1) == 0:
            cbar = flat_ax[i].collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)

    for i in range(1, nr_plots):
        cbar = True if i % (nr_plots - 1) == 0 else False
        diff = abs(reference_corr - comparison_corr[i - 1])
        sns.set(style="white")
        az = sns.heatmap(diff, ax=flat_ax[i + nr_plots], cmap=cmap, vmax=.3, square=True, annot=annot, center=0,
                         linewidths=0, cbar=cbar, fmt='.2f')
        if i % (nr_plots - 1) == 0:
            cbar = az.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)
    titles = ['reference'] + [e.name if e.name is not None else idx for idx, e in enumerate(evaluators)]
    for i, label in enumerate(titles):
        flat_ax[i].set_yticklabels([])
        flat_ax[i].set_xticklabels([])
        flat_ax[i + nr_plots].set_yticklabels([])
        flat_ax[i + nr_plots].set_xticklabels([])
        title_font = {'size': '28'}
        flat_ax[i].set_title(label, **title_font)
    plt.tight_layout()
    st.pyplot()

def cdf(data_r, data_f, xlabel: str = 'Values', ylabel: str = 'Cumulative Sum', ax=None):
    """
    Plot continous density function on optionally given ax. If no ax, cdf is plotted and shown.
    :param data_r: Series with reference data
    :param data_f: Series with comparison data
    :param xlabel: Label to put on the x-axis
    :param ylabel: Label to put on the y-axis
    :param ax: The axis to plot on. If ax=None, a new figure is created.
    """
    x1 = np.sort(data_r)
    x2 = np.sort(data_f)
    y = np.arange(1, len(data_r) + 1) / len(data_r)

    ax = ax if ax else plt.subplots()[1]

    axis_font = {'size': '14'}
    ax.set_xlabel(xlabel, **axis_font)
    ax.set_ylabel(ylabel, **axis_font)

    ax.grid()
    ax.plot(x1, y, marker='o', linestyle='none', label='reference', ms=8)
    ax.plot(x2, y, marker='o', linestyle='none', label='comparison', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    # If labels are strings, rotate them vertical
    if isinstance(data_r, pd.Series) and data_r.dtypes == 'object':
        ax.set_xticklabels(data_r.value_counts().sort_index().index, rotation='vertical')

    if ax is None:
        st.pyplot()
    st.pyplot()

def plot_mean_std_comparison(evaluators: List):
    """
    Plot comparison between the means and standard deviations from each evaluator in evaluators.
    :param evaluators: list of TableEvaluator objects that are to be evaluated.
    """
    nr_plots = len(evaluators)
    fig, ax = plt.subplots(2, nr_plots, figsize=(4 * nr_plots, 7))
    flat_ax = ax.flatten()
    for i in range(nr_plots):
        plot_mean_std(evaluators[i].reference, evaluators[i].comparison, ax=ax[:, i])

    titles = [e.name if e is not None else idx for idx, e in enumerate(evaluators)]
    for i, label in enumerate(titles):
        title_font = {'size': '24'}
        flat_ax[i].set_title(label, **title_font)
    plt.tight_layout()
    st.pyplot()

def plot_mean_std(reference: pd.DataFrame, comparison: pd.DataFrame, ax=None):
    """
    Plot the means and standard deviations of each dataset.
    :param reference: DataFrame containing the reference data
    :param comparison: DataFrame containing the comparison data
    :param ax: Axis to plot on. If none, a new figure is made.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle('Absolute Log Mean and standard deviations (SD) of numeric data\n', fontsize=16)
       
    ax[0].grid(True)
    ax[1].grid(True)
    reference = reference._get_numeric_data()
    comparison = comparison._get_numeric_data()
    reference_mean = np.log(np.add(abs(reference.mean()).values, 1e-5))
    comparison_mean = np.log(np.add(abs(comparison.mean()).values, 1e-5))
    min_mean = min(reference_mean) - 1
    max_mean = max(reference_mean) + 1
    line = np.arange(min_mean, max_mean)
    sns.lineplot(x=line, y=line, ax=ax[0])
    sns.scatterplot(x=reference_mean,
                    y=comparison_mean,
                    ax=ax[0])
    ax[0].set_title('Means of reference and comparison data')
    ax[0].set_xlabel('reference data mean (log)')
    ax[0].set_ylabel('comparison data mean (log)')

    
    reference_std = np.log(np.add(reference.std().values, 1e-5))
    comparison_std = np.log(np.add(comparison.std().values, 1e-5))
    min_std = min(reference_std) - 1
    max_std = max(reference_std) + 1
    line = np.arange(min_std, max_std)
    sns.lineplot(x=line, y=line, ax=ax[1])
    sns.scatterplot(x=reference_std,
                    y=comparison_std,
                    ax=ax[1])
    ax[1].set_title('SD of reference and comparison data')
    ax[1].set_xlabel('reference data SD (log)')
    ax[1].set_ylabel('comparison data SD (log)')
    plt.savefig('pdf_files/synthetic_data/mean_std.png',orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0)
    st.pyplot()
    if ax is None:
        st.pyplot()
    