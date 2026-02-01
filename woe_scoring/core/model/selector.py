from operator import itemgetter
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .functions import calc_iv_dict


class FeatureSelector:
    """
    Feature selection for predictive models using various algorithms.

    This class provides a unified interface for different feature selection strategies
    including recursive feature elimination (RFE), sequential feature selection (SFS),
    information value (IV), and Variance Inflation Factor (VIF).

    Args:
        selection_type (str): Feature selection algorithm to use: 'rfe', 'sfs', 'iv', or 'vif'.
            - 'rfe': Recursive Feature Elimination with cross-validation
            - 'sfs': Sequential Feature Selection (forward or backward)
            - 'iv': Information Value based selection
            - 'vif': Variance Inflation Factor for removing multicollinearity
        random_state (int): Random seed for reproducibility.
        class_weight (str): Class weights strategy for imbalanced data ('balanced' or None).
        cv (int): Number of cross-validation folds.
        n_jobs (int): Number of CPU cores for parallel processing.
        max_vars (int): Maximum number of features to select.
        direction (str): Direction for sequential selection ('forward' or 'backward').
        scoring (str): Metric for evaluating feature importance (e.g., 'roc_auc').
        l1_exp_scale (int): Exponent scale for L1 regularization grid.
        l1_grid_size (int): Number of points in L1 regularization grid.
        iv_threshold (float): Minimum information value required to keep a feature.
        vif_threshold (float): Maximum VIF value allowed (default 10.0).
    """

    def __init__(
            self, selection_type: str, random_state: int, class_weight: str,
            cv: int, n_jobs: int, max_vars: int, direction: str,
            scoring: str, l1_exp_scale: int, l1_grid_size: int,
            iv_threshold: float, vif_threshold: float = 10.0
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
        self.vif_threshold = vif_threshold

        self.selector = self._get_selector(self.selection_type)

    def select(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        """
        Select features using the configured selection method.

        Args:
            data: Input DataFrame containing all features
            target: Target variable (binary classification)
            feature_names: List of feature names to consider for selection

        Returns:
            List of selected feature names
        """
        # Handle empty feature list
        if not feature_names:
            return []

        return self.selector(data=data, target=target, feature_names=feature_names)

    def _get_selector(self, selection_type) -> callable:
        """
        Returns the appropriate feature selection function based on selection_type.

        Args:
            selection_type: Type of feature selection ('rfe', 'sfs', 'iv', or 'vif')

        Returns:
            Function that implements the selected feature selection method

        Raises:
            ValueError: If an invalid selection_type is provided
        """
        selection_methods = {
            'rfe': self._select_by_rfe,
            'sfs': self._select_by_sfs,
            'iv': self._select_by_iv,
            'vif': self._select_by_vif
        }

        if selection_type in selection_methods:
            return selection_methods[selection_type]

        raise ValueError(
            f'Unknown feature selection type: {selection_type}. '
            f'Should be one of: {", ".join(selection_methods.keys())}'
        )

    def _select_by_vif(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        """
        Selects features by iteratively removing the feature with the highest VIF
        exceeding the threshold.

        Args:
            data: Input dataset containing features
            target: Target variable (not used for VIF but kept for consistency)
            feature_names: List of feature names to consider

        Returns:
            List of selected feature names
        """
        # Handle empty feature list
        if not feature_names:
            return []

        # Make a copy of the list to avoid modifying the original
        selected_features = list(feature_names)
        
        # VIF requires at least two features to detect multicollinearity
        if len(selected_features) < 2:
            return selected_features
        
        while len(selected_features) > 1:
            # Create a dataframe with current features and a constant for intercept
            X = data[selected_features].copy()
            X = X.fillna(0)
            X['const'] = 1
            
            try:
                # Calculate VIF for each feature
                vif_values = [
                    variance_inflation_factor(X.values, i)
                    for i in range(len(selected_features))
                ]
            except Exception:
                # Fallback if matrix is singular or other calculation issues occur
                break
            
            max_vif = max(vif_values)
            
            # If max VIF is below threshold, we are done
            if max_vif < self.vif_threshold:
                break
                
            # Otherwise, remove the feature with max VIF
            idx_to_remove = vif_values.index(max_vif)
            selected_features.pop(idx_to_remove)

        return selected_features[:self.max_vars] if self.max_vars else selected_features


    def _select_by_iv(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        """
        Selects features based on Information Value (IV) score.

        Features are ranked by their IV scores, filtered by the threshold,
        and limited to the maximum number of variables specified.

        Args:
            data: Input dataset containing features
            target: Target variable for binary classification
            feature_names: List of feature names to evaluate

        Returns:
            List of selected feature names, sorted by IV score in descending order
        """
        # Ensure target is numpy array
        target_array = target.values if hasattr(target, 'values') else np.array(target)

        # Calculate IV for each feature efficiently
        iv_values: Dict[str, float] = {}
        for feature_name in feature_names:
            feature_iv = calc_iv_dict(data, target_array, feature_name)
            iv_values.update(feature_iv)

        # Sort features by IV score (descending)
        sorted_features = sorted(iv_values.items(), key=itemgetter(1), reverse=True)

        # Filter by threshold and limit to max_vars
        selected_features = [
            feature for feature, iv_score in sorted_features
            if iv_score >= self.iv_threshold
        ]

        return selected_features[:self.max_vars] if self.max_vars else selected_features

    def _select_by_sfs(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        """
        Selects features using Sequential Feature Selection (SFS).

        SFS adds (forward) or removes (backward) features sequentially to find
        the optimal feature subset that maximizes model performance.

        Args:
            data: Input dataset containing features
            target: Target variable for binary classification
            feature_names: List of feature names to consider

        Returns:
            List of selected feature names
        """
        # Handle empty feature list
        if not feature_names:
            return []

        # Subset the data to only use specified features
        X = data[feature_names]

        # Ensure target is in the right format
        y = target.values if hasattr(target, 'values') else np.array(target)

        # Calculate optimal regularization strength
        C = l1_min_c(X, y, loss="log", fit_intercept=True)

        # Create optimized logistic regression estimator
        estimator = LogisticRegression(
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=1,  # Use 1 here to avoid nested parallelism
            tol=1e-5,
            max_iter=5000,
            penalty="l2",
            warm_start=True,
            solver='liblinear',  # More efficient for this task
            C=C
        )

        # Set up Sequential Feature Selector
        n_features = min(self.max_vars, len(feature_names)) if self.max_vars else 'auto'
        selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_features,
            direction=self.direction,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=self.scoring
        )

        # Fit selector
        selector.fit(X, y)

        # Extract selected feature names
        return list(np.array(feature_names)[selector.get_support()])

    def _select_by_rfe(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        """
        Selects features using Recursive Feature Elimination with Cross-Validation (RFECV).

        RFECV recursively removes features, selecting the optimal feature subset
        that maximizes cross-validated model performance.

        Args:
            data: Input dataset containing features
            target: Target variable for binary classification
            feature_names: List of feature names to consider

        Returns:
            List of selected feature names
        """
        # Handle empty feature list
        if not feature_names:
            return []

        # Subset the data to only use specified features
        X = data[feature_names]

        # Ensure target is in the right format
        y = target.values if hasattr(target, 'values') else np.array(target)

        # Calculate optimal regularization strength
        C = l1_min_c(X, y, loss="log", fit_intercept=True)

        # Create optimized logistic regression estimator
        estimator = LogisticRegression(
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=1,  # Use 1 here to avoid nested parallelism
            tol=1e-5,
            max_iter=5000,
            penalty="l2",
            warm_start=True,
            solver='liblinear',  # More efficient for feature selection
            C=C
        )

        # Set minimum features to select (default to 1 if not specified)
        min_features = self.max_vars if self.max_vars else 1

        # Set up RFECV
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=self.cv,
            scoring=self.scoring,
            min_features_to_select=min_features,
            n_jobs=self.n_jobs,
            importance_getter='coef_'  # Explicitly use coefficients for feature importance
        )

        # Fit selector
        selector.fit(X, y)

        # Extract selected feature names
        return list(np.array(feature_names)[selector.get_support()])
