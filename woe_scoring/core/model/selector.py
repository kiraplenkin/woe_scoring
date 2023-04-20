from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

from .functions import (_check_features_gini_threshold, _check_min_pct_group,
                        _get_high_pval_positive_vars)


class FeatureSelector:
    """
    Class for feature selection using various algorithms.
    """
    def __init__(self, selection_type: str, random_state: int, class_weight: str, 
                 cv: int, C: float, n_jobs: int, max_vars: int, direction: str, 
                 scoring: str, min_pct_group: float, gini_threshold: float):
        self.selection_type = selection_type
        self.random_state = random_state
        self.class_weight = class_weight
        self.cv = cv
        self.C = C
        self.n_jobs = n_jobs
        self.max_vars = max_vars
        self.direction = direction
        self.scoring = scoring
        self.min_pct_group = min_pct_group
        self.gini_threshold = gini_threshold

        self.selector = self._get_selector(self.selection_type)

    def select(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        return self.selector(data, target, feature_names)


    def _get_selector(self, selection_type):
        """Returns an instance of the feature selector based on the selection_type."""
        if selection_type == 'rfe':
            return self._select_by_rfe
        elif selection_type == 'sfs':
            return self._select_by_sfs
        elif selection_type == 'iv':
            return self._select_by_iv
        else:
            raise ValueError(f'Unknown feature selection type: {selection_type}')


    def _select_by_iv(self, X, y, feature_names):
        ...

    
    def _select_by_sfs(self, data, target, feature_names):
        selector = SequentialFeatureSelector(
            estimator=LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            n_features_to_select=self.max_vars,
            direction=self.direction,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=self.scoring
        )
        
        return self._feature_selection(
            data, feature_names, target, selector
        )


    def _select_by_rfe(self, data, target, feature_names):
        selector = RFECV(
            estimator=LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            step=1,
            cv=self.cv,
            scoring=self.scoring,
            min_features_to_select=self.max_vars,
            n_jobs=self.n_jobs
        )
        return self._feature_selection(
            data, feature_names, target, selector
        )

    def _feature_selection(self, data, feature_names, target, selector):
        feature_names = _check_min_pct_group(
            data, feature_names=feature_names, min_pct_group=self.min_pct_group
        )
        feature_names = _check_features_gini_threshold(
            data,
            target,
            feature_names=feature_names,
            gini_threshold=self.gini_threshold,
            random_state=self.random_state,
            class_weight=self.class_weight,
            cv=self.cv,
            c=self.C,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        return self._get_feature_vars_from_selector(
            selector, data, target, feature_names
        )


    def _get_feature_vars_from_selector(
    self, selector, data: pd.DataFrame, target: pd.Series, feature_names: List[str]
    ) -> List[str]:
        """
        Returns the feature variables selected by the selector using the given data and target.

        Parameters:
        - selector: SelectorMixin
            The feature selector to use.
        - data: pd.DataFrame
            The dataset to select features from.
        - target: pd.Series
            The target variable to use for feature selection.
        - feature_names: List[str]
            The list of feature variable names to select from.

        Returns:
        - List[str]
            The list of feature variable names selected by the selector.
        """
        selector.fit(data[feature_names], target)
        while bad_vars := _get_high_pval_positive_vars(
            data, target, np.array(feature_names)[selector.get_support()].tolist()
        ):
            feature_names = list(set(feature_names) - set(bad_vars))
            selector.fit(data[feature_names], target)
        return np.array(feature_names)[selector.get_support()].tolist()



    def _select_by_iv(self, data, target, feature_names):
        ...
