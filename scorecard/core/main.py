import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from typing import Union, List, Tuple
from .functions import cat_bining
from .functions import num_bining


class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_bins: int = 10,
        min_pcnt_group: float = 0.05,
        verbose: bool = False,
        prefix: str = "WOE_",
        cat_features: List = None,
        special_cols: List = None,
        cat_features_threshold: int = 0,
        safe_original_data: bool = False,
    ):
        """
        Performs the Weight Of Evidence transformation over the input X features using information from y vector.
        :param verbose: boolean flag to add verbose output

        TODO: add n_jobs
        """
        self.max_bins = max_bins
        self.min_pcnt_group = min_pcnt_group
        self.cat_features = cat_features if cat_features else []
        self.special_cols = special_cols if special_cols else []
        self.cat_features_threshold = cat_features_threshold
        self.verbose = verbose
        self.prefix = prefix
        self.safe_original_data = safe_original_data

        self.WOE_IV_dict = []  # self.transformers
        self.feature_names = []
        self.num_features = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Fits the input data
        :param X: data matrix
        :param y: target vector
        :return: self
        """
        if isinstance(X, pd.DataFrame):
            if self.special_cols:
                X = X.drop(self.special_cols, axis=1)
            self.feature_names = X.columns
        elif isinstance(X, np.ndarray):
            self.feature_names = [f"X_{i}" for i in range(X.shape[-1])]
        else:
            raise TypeError("X vector is not np array neither data frame")

        X, y = self._check_inputs(X, y)

        if len(self.cat_features) == 0 and self.cat_features_threshold > 0:
            for i in range(len(self.feature_names)):
                if (
                    type(X[0, i]) == np.dtype("object")
                    or type(X[0, i]) == np.dtype("str")
                    or len(np.unique(X[:, i])) < self.cat_features_threshold
                ):
                    self.cat_features.append(self.feature_names[i])
        if len(self.cat_features) > 0:
            self.num_features = [
                feature
                for feature in self.feature_names
                if feature not in self.cat_features
            ]
            for feature in self.cat_features:
                feature_idx = list(self.feature_names).index(feature)
                self._print(f"Exploring {feature} feature")
                res_dict, missing_position = cat_bining(
                    X=X[:, feature_idx],
                    y=y,
                    min_pcnt_group=self.min_pcnt_group,
                    max_bins=self.max_bins,
                )
                self.WOE_IV_dict.append(
                    {feature: res_dict, "missing_bin": missing_position}
                )
        else:
            self.num_features = self.feature_names

        for feature in self.num_features:
            feature_idx = list(self.feature_names).index(feature)
            self._print(f"Exploring {feature} feature")
            res_dict, missing_position = num_bining(
                X=X[:, feature_idx].astype(float),
                y=y,
                min_pcnt_group=self.min_pcnt_group,
                max_bins=self.max_bins,
            )
            self.WOE_IV_dict.append(
                {feature: res_dict, "missing_bin": missing_position}
            )

        return self

    def transform(self, X: pd.DataFrame):
        """
        Checks and transforms input arrays
        :param X: X data array
        :return: transformed data
        """
        for i, _ in enumerate(self.WOE_IV_dict):
            feature = list(self.WOE_IV_dict[i])[0]
            self._print(f"Preparing {feature} feature")
            new_feature = self.prefix + feature
            for bin_values in self.WOE_IV_dict[i][feature]:
                if feature in self.cat_features:
                    X.loc[
                        np.isin(X[feature], bin_values["bin"]), new_feature
                    ] = bin_values["woe"]
                else:
                    X.loc[
                        np.logical_and(
                            X[feature] >= np.min(bin_values["bin"]),
                            X[feature] < np.max(bin_values["bin"]),
                        ),
                        new_feature,
                    ] = bin_values["woe"]
            if feature in self.cat_features:
                try:
                    X[new_feature].fillna(
                        self.WOE_IV_dict[i][feature][
                            [
                                idx
                                for idx, feature_bins in enumerate(
                                    self.WOE_IV_dict[i][feature]
                                )
                                if "Missing" in feature_bins["bin"]
                            ][0]
                        ]["woe"],
                        inplace=True,
                    )
                except IndexError:
                    pass
            else:
                if self.WOE_IV_dict[i]["missing_bin"] == "first":
                    X[new_feature].fillna(
                        self.WOE_IV_dict[i][feature][0]["woe"], inplace=True
                    )
                if self.WOE_IV_dict[i]["missing_bin"] == "last":
                    X[new_feature].fillna(
                        self.WOE_IV_dict[i][feature][-1]["woe"], inplace=True
                    )
            if not self.safe_original_data:
                del X[feature]

        return X

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ):
        self.fit(X=X, y=y)
        X = self.transform(X=X)

        return X

    def _check_inputs(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check input data
        :param X: data matrix
        :param y: target vector
        :return: X, y
        """
        if type_of_target(y) != "binary":
            raise ValueError("y vector should be binary")

        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            force_all_finite=False,
            dtype=None,
            y_numeric=True,
        )
        return X, y

    def _print(self, msg: str):
        if self.verbose:
            print(msg)
