import json
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_X_y

from .binning.functions import cat_binning, num_binning, refit_woe_dict
from .model.functions import create_model, feature_select, generate_sql, predict_proba, save_reports


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def _check_inputs(
        x: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check input data
    :param x: data matrix
    :param y: target vector
    :return: x, y
    """
    if type_of_target(y) != "binary":
        raise ValueError("y vector should be binary")

    x, y = check_X_y(
        x,
        y,
        accept_sparse=False,
        force_all_finite=False,
        dtype=None,
        y_numeric=True,
    )
    return x, y


class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            max_bins: Union[int, float] = 0.8,
            min_pct_group: float = 0.05,
            verbose: bool = False,
            prefix: str = "WOE_",
            cat_features: List = None,
            special_cols: List = None,
            cat_features_threshold: int = 0,
            diff_woe_threshold: float = 0.05,
            safe_original_data: bool = False,
    ):
        """
        Performs the Weight Of Evidence transformation over the input x features using information from y vector.
        :param verbose: boolean flag to add verbose output

        TODO: add n_jobs
        """
        self.classes_ = None
        self.max_bins = max_bins
        self.min_pct_group = min_pct_group
        self.cat_features = cat_features or []
        self.special_cols = special_cols or []
        self.cat_features_threshold = cat_features_threshold
        self.diff_woe_threshold = diff_woe_threshold
        self.verbose = verbose
        self.prefix = prefix
        self.safe_original_data = safe_original_data

        self.woe_iv_dict = []
        self.feature_names = []
        self.num_features = []

    def fit(self, x: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        """
        Fits the input data
        :param x: data matrix
        :param y: target vector
        """
        if isinstance(x, pd.DataFrame):
            if self.special_cols:
                x = x.drop(self.special_cols, axis=1)
            self.feature_names = x.columns
        elif isinstance(x, np.ndarray):
            self.feature_names = [f"X_{i}" for i in range(x.shape[-1])]
        else:
            raise TypeError("x vector is not np array neither data frame")

        x, y = _check_inputs(x, y)
        self.classes_ = unique_labels(y)

        if len(self.cat_features) == 0 and self.cat_features_threshold > 0:
            for i in range(len(self.feature_names)):
                if (
                        type(x[0, i]) == np.dtype("object")
                        or type(x[0, i]) == np.dtype("str")
                        or len(np.unique(x[:, i])) < self.cat_features_threshold
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
                res_dict, missing_position = cat_binning(
                    x=x[:, feature_idx],
                    y=y,
                    min_pct_group=self.min_pct_group,
                    max_bins=self.max_bins,
                    diff_woe_threshold=self.diff_woe_threshold,
                )
                self.woe_iv_dict.append(
                    {
                        feature: res_dict,
                        "missing_bin": missing_position,
                        "type_feature": "cat",
                    }
                )
        else:
            self.num_features = self.feature_names

        for feature in self.num_features:
            feature_idx = list(self.feature_names).index(feature)
            self._print(f"Exploring {feature} feature")
            res_dict, missing_position = num_binning(
                x=x[:, feature_idx].astype(float),
                y=y,
                min_pct_group=self.min_pct_group,
                max_bins=self.max_bins,
                diff_woe_threshold=self.diff_woe_threshold,
            )
            self.woe_iv_dict.append(
                {
                    feature: res_dict,
                    "missing_bin": missing_position,
                    "type_feature": "num",
                }
            )

    def transform(self, x: pd.DataFrame):
        """
        Transforms input arrays
        :param x: x data array
        :return: transformed data
        """
        for i, _ in enumerate(self.woe_iv_dict):
            feature = list(self.woe_iv_dict[i])[0]
            self._print(f"Transform {feature} feature")
            new_feature = self.prefix + feature
            for bin_values in self.woe_iv_dict[i][feature]:
                if feature in self.cat_features:
                    x.loc[
                        np.isin(x[feature], bin_values["bin"]), new_feature
                    ] = bin_values["woe"]
                else:
                    x.loc[
                        np.logical_and(
                            x[feature] >= np.min(bin_values["bin"]),
                            x[feature] < np.max(bin_values["bin"]),
                        ),
                        new_feature,
                    ] = bin_values["woe"]
            if self.woe_iv_dict[i]["missing_bin"] == "first":
                x[new_feature].fillna(
                    self.woe_iv_dict[i][feature][0]["woe"], inplace=True
                )
            elif self.woe_iv_dict[i]["missing_bin"] == "last":
                x[new_feature].fillna(
                    self.woe_iv_dict[i][feature][-1]["woe"], inplace=True
                )
            elif (
                    self.woe_iv_dict[i][feature][0]["woe"]
                    < self.woe_iv_dict[i][feature][-1]["woe"]
            ):
                x[new_feature].fillna(
                    self.woe_iv_dict[i][feature][0]["woe"], inplace=True
                )
            else:
                x[new_feature].fillna(
                    self.woe_iv_dict[i][feature][-1]["woe"], inplace=True
                )
            if not self.safe_original_data:
                del x[feature]

        return x

    def save(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump(self.woe_iv_dict, file, indent=4, cls=NpEncoder)

    def load(self, path: str) -> None:
        with open(path, "r") as file:
            self.woe_iv_dict = json.load(file)

    def refit(self, x: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:
        if isinstance(x, pd.DataFrame):
            if self.special_cols:
                x = x.drop(self.special_cols, axis=1)
            self.feature_names = x.columns
            self.feature_names = [col.replace("WOE_", "") for col in self.feature_names]
        elif isinstance(x, np.ndarray):
            self.feature_names = [f"X_{i}" for i in range(x.shape[-1])]
        else:
            raise TypeError("x vector is not np array neither data frame")

        x, y = _check_inputs(x, y)

        temp_woe_iv_dict = []

        for i in range(len(self.woe_iv_dict)):
            feature_idx = list(self.feature_names).index(
                list(self.woe_iv_dict[i].keys())[0]
            )
            self._print(f"Refiting {list(self.woe_iv_dict[i].keys())[0]} feature")
            res_dict = refit_woe_dict(
                x=x[:, feature_idx],
                y=y,
                bins=[
                    _bin["bin"]
                    for _bin in self.woe_iv_dict[i][list(self.woe_iv_dict[i].keys())[0]]
                ],
                type_feature=self.woe_iv_dict[i]["type_feature"],
                missing_bin=self.woe_iv_dict[i]["missing_bin"]
            )
            temp_woe_iv_dict.append(
                {
                    list(self.woe_iv_dict[i].keys())[0]: res_dict,
                    "missing_bin": self.woe_iv_dict[i]["missing_bin"],
                    "type_feature": self.woe_iv_dict[i]["type_feature"],
                }
            )
        self.woe_iv_dict = temp_woe_iv_dict
        del temp_woe_iv_dict

    def _print(self, msg: str):
        if self.verbose:
            print(msg)


class CreateModel(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            max_vars: Union[int, float] = 0.8,
            special_cols: List = None,
            unused_cols: List = None,
            n_jobs: int = None,
            gini_threshold: float = 5.0,
            corr_threshold: float = 0.5,
            random_state: int = None,
            class_weight: str = None,
            direction: str = "forward",
            cv: int = 3,
            C: float = None,
            scoring: str = "roc_auc",
    ):

        self.results = None
        self.max_vars = max_vars
        self.special_cols = special_cols or []
        self.unused_cols = unused_cols or []
        self.n_jobs = n_jobs
        self.gini_threshold = gini_threshold
        self.corr_threshold = corr_threshold
        self.random_state = random_state
        self.class_weight = class_weight
        self.direction = direction
        self.cv = cv
        self.C = C
        self.scoring = scoring

        self.feature_names_: List[str] = []
        self.coef_: List[float] = []
        self.intercept_: float = 0.0
        self.model = None

    def fit(self, x: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        if isinstance(x, pd.DataFrame):
            x = x.drop(self.special_cols + self.unused_cols, axis=1)
            self.feature_names_ = x.columns
        elif isinstance(x, np.ndarray):
            self.feature_names_ = [f"X_{i}" for i in range(x.shape[-1])]
        else:
            raise TypeError("x vector is not np array neither data frame")

        if self.C is None:
            self.C = 1.0e4 / x.shape[0]

        self.feature_names_ = feature_select(
            x, y,
            feature_names=self.feature_names_,
            gini_threshold=self.gini_threshold,
            corr_threshold=self.corr_threshold,
            random_state=self.random_state,
            class_weight=self.class_weight,
            max_vars=self.max_vars,
            direction=self.direction,
            cv=self.cv,
            c=self.C,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )

        self.model = create_model(
            x, y,
            feature_names=self.feature_names_
        )

        self.results = pd.read_html(self.model.summary().tables[1].as_html(), header=0, index_col=0)[0].reset_index()
        self.intercept_ = self.results.iloc[0, 1]
        self.coef_ = list(self.results.iloc[1:, 1])
        self.feature_names_ = list(self.results.iloc[1:, 0])

    def save_reports(self, path: str):
        save_reports(self.model, path)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return predict_proba(x, self.model)

    def generate_sql(self, encoder) -> str:
        return generate_sql(
            feature_names=self.feature_names_,
            encoder=encoder,
            results=self.results
        )
