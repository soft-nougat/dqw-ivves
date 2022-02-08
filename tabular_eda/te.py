import copy
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from typing import Tuple, Dict, Union
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, mean_squared_error, jaccard_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from dython.nominal import compute_associations, numerical_encoding
from tabular_eda.viz import *
from tabular_eda.metrics import *
import streamlit as st

def write_text(text):
    
    html_temp = f"""
    <p style = "color:#1F4E79; text_align:left;"> {text} </p>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)


class TableEvaluator:
    """
    Class for evaluating synthetic data. It is given the reference and comparison data and allows the user to easily evaluate data with the `evaluate` method.
    Additional evaluations can be done with the different methods of evaluate and the visual evaluation method.
    """

    def __init__(self, reference: pd.DataFrame, comparison: pd.DataFrame, cat_cols=None, unique_thresh=0, metric='pearsonr',
                 verbose=False, n_samples=None,
                 name: str = None, seed=1337):
        """
        :param reference: reference dataset (pd.DataFrame)
        :param comparison: Synthetic dataset (pd.DataFrame)
        :param unique_thresh: Threshold for automatic evaluation if column is numeric
        :param cat_cols: The columns that are to be evaluated as discrete. If passed, unique_thresh is ignored.
        :param metric: the metric to use for evaluation linear relations. Pearson's r by default, but supports all models in scipy.stats
        :param verbose: Whether to print verbose output
        :param n_samples: Number of samples to evaluate. If none, it will take the minimal length of both datasets and cut the larger one off to make sure they
            are the same length.
        :param name: Name of the TableEvaluator. Used in some plotting functions like `viz.plot_correlation_comparison` to indicate your model.
        """
        self.name = name
        self.unique_thresh = unique_thresh
        self.reference = reference.copy()
        self.comparison = comparison.copy()
        self.comparison_metric = getattr(stats, metric)
        self.verbose = verbose
        self.random_seed = seed

        # Make sure columns and their order are the same.
        if len(reference.columns) == len(comparison.columns):
            comparison = comparison[reference.columns.tolist()]
        assert reference.columns.tolist() == comparison.columns.tolist(), 'Columns in reference and comparison dataframe are not the same'

        if cat_cols is None:
            reference = reference.infer_objects()
            comparison = comparison.infer_objects()
            self.numerical_columns = [column for column in reference._get_numeric_data().columns if
                                      len(reference[column].unique()) > unique_thresh]
            self.categorical_columns = [column for column in reference.columns if column not in self.numerical_columns]
        else:
            self.categorical_columns = cat_cols
            self.numerical_columns = [column for column in reference.columns if column not in cat_cols]

        # Make sure the number of samples is equal in both datasets.
        if n_samples is None:
            self.n_samples = min(len(self.reference), len(self.comparison))
        elif len(comparison) >= n_samples and len(reference) >= n_samples:
            self.n_samples = n_samples
        else:
            raise Exception(f'Make sure n_samples < len(comparison/reference). len(reference): {len(reference)}, len(comparison): {len(comparison)}')

        self.reference = self.reference.sample(self.n_samples)
        self.comparison = self.comparison.sample(self.n_samples)
        assert len(self.reference) == len(self.comparison), f'len(reference) != len(comparison)'

        self.reference.loc[:, self.categorical_columns] = self.reference.loc[:, self.categorical_columns].fillna('[NAN]').astype(
            str)
        self.comparison.loc[:, self.categorical_columns] = self.comparison.loc[:, self.categorical_columns].fillna('[NAN]').astype(
            str)

        self.reference.loc[:, self.numerical_columns] = self.reference.loc[:, self.numerical_columns].fillna(
            self.reference[self.numerical_columns].mean())
        self.comparison.loc[:, self.numerical_columns] = self.comparison.loc[:, self.numerical_columns].fillna(
            self.comparison[self.numerical_columns].mean())

    def plot_mean_std(self):
        """
        Class wrapper function for plotting the mean and std using `viz.plot_mean_std`.
        """
        plot_mean_std(self.reference, self.comparison)


    def plot_cumsums(self, nr_cols=4):
        """
        Plot the cumulative sums for all columns in the reference and comparison dataset. 
        Height of each row scales with the length of the labels. 
        Each plot contains the values of a reference columns and the corresponding comparison column.
        """
        nr_charts = len(self.reference.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not self.reference.select_dtypes(include=['object']).empty:
            lengths = []
            for d in self.reference.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in self.reference[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Cumulative Sums per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(self.reference.columns):
            r = self.reference[col]
            f = self.comparison.iloc[:, self.reference.columns.tolist().index(col)]
            cdf(r, f, col, 'Cumsum', ax=axes[i])
            st.pyplot()
        #plot = plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    def plot_distributions(self, nr_cols=3):
        """
        Plot the distribution plots for all columns in the reference and comparison dataset. Height of each row of plots scales with the length of the labels. Each plot
        contains the values of a reference columns and the corresponding comparison column.
        """
        nr_charts = len(self.reference.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not self.reference.select_dtypes(include=['object']).empty:
            lengths = []
            for d in self.reference.select_dtypes(include=['object']):
                lengths.append(max([len(x.strip()) for x in self.reference[d].unique().tolist()]))
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
        fig.suptitle('Distribution per feature', fontsize=16)
        axes = ax.flatten()
        for i, col in enumerate(self.reference.columns):
            if col not in self.categorical_columns:
                try:
                    sns.distplot(self.reference[col], ax=axes[i], label='reference')
                    sns.distplot(self.comparison[col], ax=axes[i], color='darkorange', label='comparison')
                except RuntimeError:
                    axes[i].clear()
                    sns.distplot(self.reference[col], ax=axes[i], label='reference', kde=False)
                    sns.distplot(self.comparison[col], ax=axes[i], color='darkorange', label='comparison', kde=False)
                axes[i].set_autoscaley_on(True)
                axes[i].legend()
            else:
                reference = self.reference.copy()
                comparison = self.comparison.copy()
                reference['kind'] = 'reference'
                comparison['kind'] = 'comparison'
                concat = pd.concat([comparison, reference])
                palette = sns.color_palette(
                    [(0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
                     (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)])
                x, y, hue = col, "proportion", "kind"
                ax = (concat[x]
                      .groupby(concat[hue])
                      .value_counts(normalize=True)
                      .rename(y)
                      .reset_index()
                      .pipe((sns.barplot, "data"), x=x, y=y, hue=hue, ax=axes[i], saturation=0.8, palette=palette))
                ax.set_xticklabels(axes[i].get_xticklabels(), rotation='vertical')
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.savefig('pdf_files/synthetic_data/distributions.png',orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0)
        st.pyplot()

    def plot_correlation_difference(self, plot_diff=True, **kwargs):
        """
        Plot the association matrices for each table and, if chosen, the difference between them.
        :param plot_diff: whether to plot the difference
        :param kwargs: kwargs for sns.heatmap
        """
        plot_correlation_difference(self.reference, self.comparison, cat_cols=self.categorical_columns, plot_diff=plot_diff,
                                    **kwargs)

    def correlation_distance(self, how: str = 'euclidean') -> float:
        """
        Calculate distance between correlation matrices with certain metric.
        :param how: metric to measure distance. Choose from [``euclidean``, ``mae``, ``rmse``].
        :return: distance between the association matrices in the chosen evaluation metric. Default: Euclidean
        """
        from scipy.spatial.distance import cosine
        if how == 'euclidean':
            distance_func = euclidean_distance
        elif how == 'mae':
            distance_func = mean_absolute_error
        elif how == 'rmse':
            distance_func = rmse
        elif how == 'cosine':
            def custom_cosine(a, b):
                return cosine(a.reshape(-1), b.reshape(-1))

            distance_func = custom_cosine
        else:
            raise ValueError(f'`how` parameter must be in [euclidean, mae, rmse]')

        reference_corr = compute_associations(self.reference, nominal_columns=self.categorical_columns, theil_u=True)
        comparison_corr = compute_associations(self.comparison, nominal_columns=self.categorical_columns, theil_u=True)

        return distance_func(
            reference_corr.values,
            comparison_corr.values
        )

    def plot_pca(self):
        """
        Plot the first two components of a PCA of reference and comparison data.
        """
        reference = numerical_encoding(self.reference, nominal_columns=self.categorical_columns)
        comparison = numerical_encoding(self.comparison, nominal_columns=self.categorical_columns)

        pca_r = PCA(n_components=2)
        pca_f = PCA(n_components=2)

        reference_t = pca_r.fit_transform(reference)
        comparison_t = pca_f.fit_transform(comparison)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('First two components of PCA', fontsize=16)
        sns.scatterplot(ax=ax[0], x=reference_t[:, 0], y=reference_t[:, 1])
        sns.scatterplot(ax=ax[1], x=comparison_t[:, 0], y=comparison_t[:, 1])
        ax[0].set_title('reference data')
        ax[1].set_title('comparison data')

        plt.savefig('pdf_files/synthetic_data/pca.png',orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0)
        st.pyplot()

    def get_copies(self, return_len: bool = False) -> Union[pd.DataFrame, int]:
        """
        Check whether any reference values occur in the comparison data.
        :param return_len: whether to return the length of the copied rows or not.
        :return: Dataframe containing the duplicates if return_len=False, else integer indicating the number of copied rows.
        """
        reference_hashes = self.reference.apply(lambda x: hash(tuple(x)), axis=1)
        comparison_hashes = self.comparison.apply(lambda x: hash(tuple(x)), axis=1)

        dup_idxs = comparison_hashes.isin(reference_hashes.values)
        dup_idxs = dup_idxs[dup_idxs == True].sort_index().index.tolist()

        if self.verbose:
            write_text(f'Nr copied columns: {len(dup_idxs)}')
        copies = self.comparison.loc[dup_idxs, :]

        if return_len:
            return len(copies)
        else:
            return copies

    def get_duplicates(self, return_values: bool = False) -> Tuple[Union[pd.DataFrame, int], Union[pd.DataFrame, int]]:
        """
        Return duplicates within each dataset.
        :param return_values: whether to return the duplicate values in the datasets. If false, the lengths are returned.
        :return: dataframe with duplicates or the length of those dataframes if return_values=False.
        """
        reference_duplicates = self.reference[self.reference.duplicated(keep=False)]
        comparison_duplicates = self.comparison[self.comparison.duplicated(keep=False)]
        if return_values:
            return reference_duplicates, comparison_duplicates
        else:
            return len(reference_duplicates), len(comparison_duplicates)

    def pca_correlation(self, lingress=False):
        """
        Calculate the relation between PCA explained variance values. Due to some very large numbers, in recent implementation the MAPE(log) is used instead of
        regressions like Pearson's r.
        :param lingress: whether to use a linear regression, in this case Pearson's.
        :return: the correlation coefficient if lingress=True, otherwise 1 - MAPE(log(reference), log(comparison))
        """
        self.pca_r = PCA(n_components=5)
        self.pca_f = PCA(n_components=5)

        reference = self.reference
        comparison = self.comparison

        reference = numerical_encoding(reference, nominal_columns=self.categorical_columns)
        comparison = numerical_encoding(comparison, nominal_columns=self.categorical_columns)

        self.pca_r.fit(reference)
        self.pca_f.fit(comparison)
        if self.verbose:
            results = pd.DataFrame({'reference': self.pca_r.explained_variance_, 'comparison': self.pca_f.explained_variance_})
            write_text(f'\nTop 5 PCA components:')
            write_text(results.to_string())

        if lingress:
            corr, p, _ = self.comparison_metric(self.pca_r.explained_variance_, self.pca_f.explained_variance_)
            return corr
        else:
            pca_error = mean_absolute_percentage_error(self.pca_r.explained_variance_, self.pca_f.explained_variance_)
            return 1 - pca_error

    def fit_estimators(self):
        """
        Fit self.r_estimators and self.f_estimators to reference and comparison data, respectively.
        """

        if self.verbose:
            write_text(f'\nFitting reference')
        for i, c in enumerate(self.r_estimators):
            if self.verbose:
                write_text(f'{i + 1}: {type(c).__name__}')
            c.fit(self.reference_x_train, self.reference_y_train)

        if self.verbose:
            write_text(f'\nFitting comparison')
        for i, c in enumerate(self.f_estimators):
            if self.verbose:
                write_text(f'{i + 1}: {type(c).__name__}')
            c.fit(self.comparison_x_train, self.comparison_y_train)

    def score_estimators(self):
        """
        Get F1 scores of self.r_estimators and self.f_estimators on the comparison and reference data, respectively.
        :return: dataframe with the results for each estimator on each data test set.
        """
        if self.target_type == 'class':
            rows = []
            for r_classifier, f_classifier, estimator_name in zip(self.r_estimators, self.f_estimators,
                                                                  self.estimator_names):
                for dataset, target, dataset_name in zip([self.reference_x_test, self.comparison_x_test],
                                                         [self.reference_y_test, self.comparison_y_test], ['reference', 'comparison']):
                    predictions_classifier_reference = r_classifier.predict(dataset)
                    predictions_classifier_comparison = f_classifier.predict(dataset)
                    f1_r = f1_score(target, predictions_classifier_reference, average="micro")
                    f1_f = f1_score(target, predictions_classifier_comparison, average="micro")
                    jac_sim = jaccard_score(predictions_classifier_reference, predictions_classifier_comparison, average='micro')
                    row = {'index': f'{estimator_name}_{dataset_name}_testset', 'f1_reference': f1_r, 'f1_comparison': f1_f,
                           'jaccard_similarity': jac_sim}
                    rows.append(row)
            results = pd.DataFrame(rows).set_index('index')

        elif self.target_type == 'regr':
            r2r = [rmse(self.reference_y_test, clf.predict(self.reference_x_test)) for clf in self.r_estimators]
            f2f = [rmse(self.comparison_y_test, clf.predict(self.comparison_x_test)) for clf in self.f_estimators]

            # Calculate test set accuracies on the other dataset
            r2f = [rmse(self.comparison_y_test, clf.predict(self.comparison_x_test)) for clf in self.r_estimators]
            f2r = [rmse(self.reference_y_test, clf.predict(self.reference_x_test)) for clf in self.f_estimators]
            index = [f'reference_data_{classifier}' for classifier in self.estimator_names] + \
                    [f'comparison_data_{classifier}' for classifier in self.estimator_names]
            results = pd.DataFrame({'reference': r2r + f2r, 'comparison': r2f + f2f}, index=index)
        else:
            raise Exception(f'self.target_type should be either \'class\' or \'regr\', but is {self.target_type}.')
        return results

    def visual_evaluation(self, **kwargs):
        """
        Plot all visual evaluation metrics. Includes plotting the mean and standard deviation, cumulative sums, correlation differences and the PCA transform.
        :param kwargs: any kwargs for matplotlib.

        Edit: retrun plots to pass to st.ss for caching
        """
        mean_std = self.plot_mean_std()
        #self.plot_cumsums() -- doesn't work in streamlit...
        distributions = self.plot_distributions()
        corr_diff = self.plot_correlation_difference(**kwargs)
        pca_plot = self.plot_pca()

        return(mean_std, distributions, corr_diff, pca_plot)

    def statistical_evaluation(self) -> float:
        """
        Calculate the correlation coefficient between the basic properties of self.reference and self.comparison using Spearman's Rho. Spearman's is used because these
        values can differ a lot in magnitude, and Spearman's is more resilient to outliers.
        :return: correlation coefficient
        """
        total_metrics = pd.DataFrame()
        for ds_name in ['reference', 'comparison']:
            ds = getattr(self, ds_name)
            metrics = {}
            # TODO: add discrete columns as factors
            num_ds = ds[self.numerical_columns]

            for idx, value in num_ds.mean().items():
                metrics[f'mean_{idx}'] = value
            for idx, value in num_ds.median().items():
                metrics[f'median_{idx}'] = value
            for idx, value in num_ds.std().items():
                metrics[f'std_{idx}'] = value
            for idx, value in num_ds.var().items():
                metrics[f'variance_{idx}'] = value
            total_metrics[ds_name] = metrics.values()

        total_metrics.index = metrics.keys()
        self.statistical_results = total_metrics
        if self.verbose:
            write_text('\nBasic statistical attributes:')
            write_text(total_metrics.to_string())
        corr, p = stats.spearmanr(total_metrics['reference'], total_metrics['comparison'])
        return corr

    def correlation_correlation(self) -> float:
        """
        Calculate the correlation coefficient between the association matrices of self.reference and self.comparison using self.comparison_metric
        :return: The correlation coefficient
        """
        total_metrics = pd.DataFrame()
        for ds_name in ['reference', 'comparison']:
            ds = getattr(self, ds_name)
            corr_df = compute_associations(ds, nominal_columns=self.categorical_columns, theil_u=True)
            values = corr_df.values
            values = values[~np.eye(values.shape[0], dtype=bool)].reshape(values.shape[0], -1)
            total_metrics[ds_name] = values.flatten()

        self.correlation_correlations = total_metrics
        corr, p = self.comparison_metric(total_metrics['reference'], total_metrics['comparison'])
        if self.verbose:
            write_text('\nColumn correlation between datasets:')
            write_text(total_metrics.to_string())
        return corr

    def convert_numerical(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Special function to convert dataset to a numerical representations while making sure they have identical columns. This is sometimes a problem with
        categorical columns with many values or very unbalanced values
        :return: reference and comparison dataframe with categorical columns one-hot encoded and binary columns factorized.
        """
        reference = numerical_encoding(self.reference, nominal_columns=self.categorical_columns)

        columns = sorted(reference.columns.tolist())
        reference = reference[columns]
        comparison = numerical_encoding(self.comparison, nominal_columns=self.categorical_columns)
        for col in columns:
            if col not in comparison.columns.tolist():
                comparison[col] = 0
        comparison = comparison[columns]
        return reference, comparison

    def estimator_evaluation(self, target_col: str, target_type: str = 'class') -> float:
        """
        Method to do full estimator evaluation, including training. And estimator is either a regressor or a classifier, depending on the task. Two sets are
        created of each of the estimators `S_r` and `S_f`, for the reference and comparison data respectively. `S_f` is trained on ``self.reference`` and `S_r` on
        ``self.comparison``. Then, both are evaluated on their own and the others test set. If target_type is ``regr`` we do a regression on the RMSE scores with
        Pearson's. If target_type is ``class``, we calculate F1 scores and do return ``1 - MAPE(F1_r, F1_f)``.
        :param target_col: which column should be considered the target both both the regression and classification task.
        :param target_type: what kind of task this is. Can be either ``class`` or ``regr``.
        :return: Correlation value or 1 - MAPE
        """
        self.target_col = target_col
        self.target_type = target_type

        # Convert both datasets to numerical representations and split x and  y
        reference_x = numerical_encoding(self.reference.drop([target_col], axis=1), nominal_columns=self.categorical_columns)

        columns = sorted(reference_x.columns.tolist())
        reference_x = reference_x[columns]
        comparison_x = numerical_encoding(self.comparison.drop([target_col], axis=1), nominal_columns=self.categorical_columns)
        for col in columns:
            if col not in comparison_x.columns.tolist():
                comparison_x[col] = 0
        comparison_x = comparison_x[columns]

        assert reference_x.columns.tolist() == comparison_x.columns.tolist(), f'reference and comparison columns are different: \n{reference_x.columns}\n{comparison_x.columns}'

        if self.target_type == 'class':
            # Encode reference and comparison target the same
            reference_y, uniques = pd.factorize(self.reference[target_col])
            mapping = {key: value for value, key in enumerate(uniques)}
            comparison_y = [mapping.get(key) for key in self.comparison[target_col].tolist()]
        elif self.target_type == 'regr':
            reference_y = self.reference[target_col]
            comparison_y = self.comparison[target_col]
        else:
            raise Exception(f'Target Type must be regr or class')

        # For reproducibilty:
        np.random.seed(self.random_seed)

        self.reference_x_train, self.reference_x_test, self.reference_y_train, self.reference_y_test = train_test_split(reference_x, reference_y,
                                                                                                    test_size=0.2)
        self.comparison_x_train, self.comparison_x_test, self.comparison_y_train, self.comparison_y_test = train_test_split(comparison_x, comparison_y,
                                                                                                    test_size=0.2)

        if target_type == 'regr':
            self.estimators = [
                RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42),
                Lasso(random_state=42),
                Ridge(alpha=1.0, random_state=42),
                ElasticNet(random_state=42),
            ]
        elif target_type == 'class':
            self.estimators = [
                LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=500, random_state=42),
                RandomForestClassifier(n_estimators=10, random_state=42),
                DecisionTreeClassifier(random_state=42),
                MLPClassifier([50, 50], solver='adam', activation='relu', learning_rate='adaptive', random_state=42),
            ]
        else:
            raise ValueError(f'target_type must be \'regr\' or \'class\'')

        self.r_estimators = copy.deepcopy(self.estimators)
        self.f_estimators = copy.deepcopy(self.estimators)
        self.estimator_names = [type(clf).__name__ for clf in self.estimators]

        for estimator in self.estimators:
            assert hasattr(estimator, 'fit')
            assert hasattr(estimator, 'score')

        self.fit_estimators()
        self.estimators_scores = self.score_estimators()
        write_text('\nClassifier F1-scores and their Jaccard similarities:') if self.target_type == 'class' \
            else write_text('\nRegressor MSE-scores and their Jaccard similarities:')
        st.dataframe(self.estimators_scores)
       # write_text(self.estimators_scores.to_string())

        if self.target_type == 'regr':
            corr, p = self.comparison_metric(self.estimators_scores['reference'], self.estimators_scores['comparison'])
            return corr
        elif self.target_type == 'class':
            mean = mean_absolute_percentage_error(self.estimators_scores['f1_reference'], self.estimators_scores['f1_comparison'])
            return 1 - mean

    def row_distance(self, n_samples: int = None) -> Tuple[float, float]:
        """
        Calculate mean and standard deviation distances between `self.comparison` and `self.reference`.
        :param n_samples: Number of samples to take for evaluation. Compute time increases exponentially.
        :return: `(mean, std)` of these distances.
        """
        if n_samples is None:
            n_samples = len(self.reference)
        reference = numerical_encoding(self.reference, nominal_columns=self.categorical_columns)
        comparison = numerical_encoding(self.comparison, nominal_columns=self.categorical_columns)

        columns = sorted(reference.columns.tolist())
        reference = reference[columns]

        for col in columns:
            if col not in comparison.columns.tolist():
                comparison[col] = 0
        comparison = comparison[columns]

        for column in reference.columns.tolist():
            if len(reference[column].unique()) > 2:
                reference[column] = (reference[column] - reference[column].mean()) / reference[column].std()
                comparison[column] = (comparison[column] - comparison[column].mean()) / comparison[column].std()
        assert reference.columns.tolist() == comparison.columns.tolist()

        distances = cdist(reference[:n_samples], comparison[:n_samples])
        min_distances = np.min(distances, axis=1)
        min_mean = np.mean(min_distances)
        min_std = np.std(min_distances)
        return min_mean, min_std

    def column_correlations(self):
        """
        Wrapper function around `metrics.column_correlation`.
        :return: Column correlations between ``self.reference`` and ``self.comparison``.
        """
        return column_correlations(self.reference, self.comparison, self.categorical_columns)

    def evaluate(self, target_col: str, target_type: str = 'class', metric: str = None, verbose=None,
                 n_samples_distance: int = 20000) -> Dict:
        """
        Determine correlation between attributes from the reference and comparison dataset using a given metric.
        All metrics from scipy.stats are available.
        :param target_col: column to use for predictions with estimators
        :param target_type: what kind of task to perform on the target_col. Can be either ``class`` for classification or ``regr`` for regression.
        :param metric: overwrites self.metric. Scoring metric for the attributes.
            By default Pearson's r is used. Alternatives include Spearman rho (scipy.stats.spearmanr) or Kendall Tau (scipy.stats.kendalltau).
        :param n_samples_distance: The number of samples to take for the row distance. See documentation of ``tableEvaluator.row_distance`` for details.
        :param verbose: whether to print verbose logging.
        """
        if verbose is not None:
            self.verbose = verbose
        if metric is not None:
            self.comparison_metric = metric

        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        pd.options.display.float_format = '{:,.4f}'.format

        write_text(f'\nCorrelation metric: {self.comparison_metric.__name__}')

        basic_statistical = self.statistical_evaluation()
        correlation_correlation = self.correlation_correlation()
        column_correlation = self.column_correlations()
        estimators = self.estimator_evaluation(target_col=target_col, target_type=target_type)
        pca_variance = self.pca_correlation()
        nearest_neighbor = self.row_distance(n_samples=n_samples_distance)

        miscellaneous = {}
        miscellaneous['Column Correlation Distance RMSE'] = self.correlation_distance(how='rmse')
        miscellaneous['Column Correlation distance MAE'] = self.correlation_distance(how='mae')

        miscellaneous['Duplicate rows between sets (reference/comparison)'] = self.get_duplicates()
        miscellaneous['nearest neighbor mean'] = nearest_neighbor[0]
        miscellaneous['nearest neighbor std'] = nearest_neighbor[1]
        miscellaneous_df = pd.DataFrame({'Result': list(miscellaneous.values())}, index=list(miscellaneous.keys()))
        write_text(f'\nMiscellaneous results:')
        st.dataframe(miscellaneous_df)
        #write_text(miscellaneous_df.to_string())

        all_results = {
            'Basic statistics': basic_statistical,
            'Correlation column correlations': correlation_correlation,
            'Mean Correlation between comparison and reference columns': column_correlation,
            f'{"1 - MAPE Estimator results" if self.target_type == "class" else "Correlation RMSE"}': estimators,
            # '1 - MAPE 5 PCA components': pca_variance,
        }
        total_result = np.mean(list(all_results.values()))
        all_results['Similarity Score'] = total_result
        all_results_df = pd.DataFrame({'Result': list(all_results.values())}, index=list(all_results.keys()))

        write_text(f'\nResults:')
        st.dataframe(all_results_df)
        #write_text(all_results_df.to_string())
        return all_results