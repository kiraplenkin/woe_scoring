from operator import itemgetter
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c

from .functions import calc_iv_dict


class FeatureSelector:
    """
    Initialize a feature selector object with the specified parameters.

    Args:
        selection_type (str): The type of feature selection algorithm to use.
        random_state (int): Random seed for reproducibility.
        class_weight (str): Class weights for imbalanced classification problems.
        cv (int): Number of cross-validation folds to use.
        n_jobs (int): Number of CPU cores to use for parallelization.
        max_vars (int): Maximum number of features to select.
        direction (str): The direction to select features in (forward or backward).
        scoring (str): The scoring metric to use for feature selection.
        l1_exp_scale (int): The exponent used for generating L1 regularization values.
        l1_grid_size (int): The size of the L1 regularization grid to search over.
        iv_threshold (float): The minimum information value threshold for a feature.
    """

    def __init__(
            self, selection_type: str, random_state: int, class_weight: str,
            cv: int, n_jobs: int, max_vars: int, direction: str,
            scoring: str, l1_exp_scale: int, l1_grid_size: int,
            iv_threshold: float
    ):
        self.selection_type = selection_type
        self.random_state = random_state
        self.class_weight = class_weight
        self.cv = cv
        self.n_jobs = n_jobs
        self.max_vars = max_vars
        self.direction = direction
        self.scoring = scoring
        self.l1_exp_scale = l1_exp_scale
        self.l1_grid_size = l1_grid_size
        self.iv_threshold = iv_threshold

        self.selector = self._get_selector(self.selection_type)

    def select(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        return self.selector(data=data, target=target, feature_names=feature_names)

    def _get_selector(self, selection_type) -> callable:
        """Returns an instance of the feature selector based on the selection_type."""
        if selection_type == 'rfe':
            return self._select_by_rfe
        elif selection_type == 'sfs':
            return self._select_by_sfs
        elif selection_type == 'iv':
            return self._select_by_iv
        else:
            raise ValueError(f'Unknown feature selection type: {selection_type}. Should be "rfe", "sfs" or "iv"')

    def _select_by_iv(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[
        str]:
        """
        Selects top features based on Information Value (IV) score.

        Args:
            data (pd.DataFrame): The input dataset.
            target (Union[pd.Series, np.ndarray]): The target variable.
            feature_names (List[str]): A list of feature names to calculate IV for.

        Returns:
            List[str]: A list of top feature names, sorted by IV score in descending order.
        """

        iv_dict_list = [calc_iv_dict(data, target, feature_name) for feature_name in feature_names]
        iv_dict = {}
        for d in iv_dict_list:
            iv_dict |= d

        sorted_iv_dict = dict(sorted(iv_dict.items(), key=itemgetter(1), reverse=True))
        top_features = [feature for feature in sorted_iv_dict if sorted_iv_dict[feature] >= self.iv_threshold]
        return top_features[:self.max_vars]

    def _select_by_sfs(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> \
            List[str]:
        """
        Selects the best features using Sequential Feature Selection (SFS) algorithm.

        Args:
            data (pandas.DataFrame): The input data.
            target (Union[pandas.Series, numpy.ndarray]): The target variable.
            feature_names (List[str]): A list of feature names.

        Returns:
            List[str]: A list of selected feature names.
        """
        selector = SequentialFeatureSelector(
            estimator=LogisticRegression(
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tol=1e-5,
                max_iter=5000,
                penalty="l2",
                warm_start=True,
                C=l1_min_c(data[feature_names], target, loss="log", fit_intercept=True)
            ),
            n_features_to_select=self.max_vars,
            direction=self.direction,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=self.scoring
        )
        selector.fit(data[feature_names], target)
        return np.array(feature_names)[selector.get_support()].tolist()

    def _select_by_rfe(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> \
            List[str]:
        """
        Selects the best features using Recursive Feature Elimination with Cross-Validation (RFE-CV) algorithm.

        :param data: The input data (pandas DataFrame).
        :param target: The target variable (pandas Series or numpy array).
        :param feature_names: The list of feature names to select from (list of strings).
        :return: The list of selected feature names (list of strings).
        """
        selector = RFECV(
            estimator=LogisticRegression(
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tol=1e-5,
                max_iter=5000,
                penalty="l2",
                warm_start=True,
                C=l1_min_c(data[feature_names], target, loss="log", fit_intercept=True)
            ),
            step=1,
            cv=self.cv,
            scoring=self.scoring,
            min_features_to_select=self.max_vars,
            n_jobs=self.n_jobs
        )
        selector.fit(data[feature_names], target)
        return np.array(feature_names)[selector.get_support()].tolist()
