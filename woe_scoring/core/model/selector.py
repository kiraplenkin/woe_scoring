from typing import List, Union

import numpy as np
import pandas as pd
from operator import itemgetter
from joblib import Parallel, delayed
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.svm import l1_min_c
from sklearn.linear_model import LogisticRegressionCV
from .functions import _calc_iv_dict

class FeatureSelector:
    """
    Class for feature selection using various algorithms.
    """
    def __init__(self, selection_type: str, random_state: int, class_weight: str, 
                 cv: int, n_jobs: int, max_vars: int, direction: str, 
                 scoring: str, min_pct_group: float, gini_threshold: float, l1_exp_scale: int, l1_grid_size: int, iv_threshold: float):
        self.selection_type = selection_type
        self.random_state = random_state
        self.class_weight = class_weight
        self.cv = cv
        self.n_jobs = n_jobs
        self.max_vars = max_vars
        self.direction = direction
        self.scoring = scoring
        self.min_pct_group = min_pct_group
        self.gini_threshold = gini_threshold
        self.l1_exp_scale = l1_exp_scale
        self.l1_grid_size = l1_grid_size
        self.iv_threshold = iv_threshold

        self.selector = self._get_selector(self.selection_type)

    def select(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        return self.selector(data, target, feature_names)


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


    def _select_by_iv(self, data, target, feature_names) -> List[str]:
        temp_res_dict = Parallel(n_jobs=self.n_jobs)(
            delayed(_calc_iv_dict)(data, target, feature) for feature in feature_names
        )
        res_dict = {}
        for d in temp_res_dict:
            res_dict |= d

        return [feature for feature in dict(sorted(res_dict.items(), key=itemgetter(1), reverse=True)) if
                        res_dict[feature] >= self.iv_threshold][:self.max_vars]

    
    def _select_by_sfs(self, data, target, feature_names) -> List[str]:  
        Cs = l1_min_c(data[feature_names], target, loss="log", fit_intercept=True) * np.logspace(0, self.l1_exp_scale, self.l1_grid_size)
        selector = SequentialFeatureSelector(
            estimator=LogisticRegressionCV(
                Cs=Cs,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                solver='saga',
                tol=1e-5,
                intercept_scaling=10000.0,
                penalty="l1",
                max_iter=1000,
                scoring=self.scoring
            ),
            n_features_to_select=self.max_vars,
            direction=self.direction,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=self.scoring
        )
        selector.fit(data[feature_names], target)
        return np.array(feature_names)[selector.get_support()].tolist()


    def _select_by_rfe(self, data, target, feature_names) -> List[str]:
        Cs = l1_min_c(data[feature_names], target, loss="log", fit_intercept=True) * np.logspace(0, self.l1_exp_scale, self.l1_grid_size)
        selector = RFECV(
            estimator=LogisticRegressionCV(
                Cs=Cs,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                solver='saga',
                tol=1e-5,
                intercept_scaling=10000.0,
                penalty="l1",
                max_iter=1000,
            ),
            step=1,
            cv=self.cv,
            scoring=self.scoring,
            min_features_to_select=self.max_vars,
            n_jobs=self.n_jobs
        )
        selector.fit(data[feature_names], target)
        return np.array(feature_names)[selector.get_support()].tolist()
