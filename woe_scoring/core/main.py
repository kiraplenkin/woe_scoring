import json
from typing import List, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels

from .binning.functions import cat_processing, find_cat_features, num_processing, prepare_data, refit
from .model.functions import create_model, generate_sql, iv_feature_select, predict_proba, save_reports, save_scorecard_fn, \
    sequential_feature_select


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


class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            max_bins: Union[int, float] = 10,
            min_pct_group: float = 0.05,
            n_jobs: int = 1,
            prefix: str = "WOE_",
            merge_type: str = "chi2",
            cat_features: List = None,
            special_cols: List = None,
            cat_features_threshold: int = 0,
            diff_woe_threshold: float = 0.05,
            safe_original_data: bool = False,
    ):
        """
        Performs the Weight Of Evidence transformation over the input x features using information from y vector.
        """
        self.classes_ = None
        self.max_bins = max_bins
        self.min_pct_group = min_pct_group
        self.cat_features = cat_features or []
        self.special_cols = special_cols or []
        self.cat_features_threshold = cat_features_threshold
        self.diff_woe_threshold = diff_woe_threshold
        self.n_jobs = n_jobs
        self.prefix = prefix
        self.safe_original_data = safe_original_data
        self.merge_type = merge_type

        self.woe_iv_dict = []
        self.feature_names = []
        self.num_features = []

    def fit(self, x: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        """
        Fits the input data
        :param x: data matrix
        :param y: target vector
        """
        x, self.feature_names = prepare_data(x=x, special_cols=self.special_cols)
        self.classes_ = unique_labels(y)

        if len(self.cat_features) == 0 and self.cat_features_threshold > 0:
            self.cat_features = find_cat_features(
                x=x,
                feature_names=self.feature_names,
                cat_features_threshold=self.cat_features_threshold
            )
        if len(self.cat_features) > 0:
            self.num_features = [
                feature
                for feature in self.feature_names
                if feature not in self.cat_features
            ]
            self.woe_iv_dict = Parallel(n_jobs=self.n_jobs)(
                delayed(cat_processing)(
                    x[col],
                    y,
                    self.min_pct_group,
                    self.max_bins,
                    self.diff_woe_threshold
                ) for col in self.cat_features
            )
        else:
            self.num_features = self.feature_names

        num_features_res = Parallel(n_jobs=self.n_jobs)(
            delayed(num_processing)(
                x[col],
                y,
                self.min_pct_group,
                self.max_bins,
                self.diff_woe_threshold,
                self.merge_type
            ) for col in self.num_features
        )

        self.woe_iv_dict += num_features_res

    def transform(self, x: pd.DataFrame):
        """
        Transforms input arrays
        :param x: x data array
        :return: transformed data
        """
        x = x.copy()
        for i, _ in enumerate(self.woe_iv_dict):
            feature = list(self.woe_iv_dict[i])[0]
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
        x, self.feature_names = prepare_data(x=x, special_cols=self.special_cols)
        self.woe_iv_dict = Parallel(n_jobs=self.n_jobs)(
            delayed(refit)(
                x[list(self.woe_iv_dict[i].keys())[0]],
                y.values,
                [_bin["bin"] for _bin in self.woe_iv_dict[i][list(self.woe_iv_dict[i].keys())[0]]],
                self.woe_iv_dict[i]["type_feature"],
                self.woe_iv_dict[i]["missing_bin"]
            ) for i in range(len(self.woe_iv_dict))
        )


class CreateModel(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            selection_method: str = 'iv',
            max_vars: Union[int, float, None] = None,
            special_cols: List = None,
            unused_cols: List = None,
            n_jobs: int = 1,
            gini_threshold: float = 5.0,
            iv_threshold: float = 0.05,
            corr_threshold: float = 0.5,
            random_state: int = None,
            class_weight: str = None,
            direction: str = "forward",
            cv: int = 3,
            C: float = None,
            scoring: str = "roc_auc",
    ):

        self.model_results = None
        self.selection_method = selection_method
        self.max_vars = max_vars
        self.special_cols = special_cols or []
        self.unused_cols = unused_cols or []
        self.n_jobs = n_jobs
        self.gini_threshold = gini_threshold
        self.iv_threshold = iv_threshold
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
        x, self.feature_names_ = prepare_data(x, special_cols=self.special_cols)

        if self.C is None:
            self.C = 1.0e4 / x.shape[0]

        if self.max_vars is not None and self.max_vars < 1:
            self.max_vars = int(len(self.feature_names_) * self.max_vars)

        if self.selection_method == 'iv':
            self.feature_names_ = iv_feature_select(
                x, y,
                feature_names=self.feature_names_,
                iv_threshold=self.iv_threshold,
                max_vars=self.max_vars,
                n_jobs=self.n_jobs,
            )
        elif self.selection_method == 'sequential':
            self.feature_names_ = sequential_feature_select(
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
        else:
            raise NameError("selection_method should be 'iv' or 'sequential'")

        self.model = create_model(
            x, y,
            feature_names=self.feature_names_
        )

        self.model_results = pd.read_html(self.model.summary().tables[1].as_html(), header=0, index_col=0)[0].reset_index()
        self.intercept_ = self.model_results.iloc[0, 1]
        self.coef_ = list(self.model_results.iloc[1:, 1])
        self.feature_names_ = list(self.model_results.iloc[1:, 0])

    def save_reports(self, path: str):
        save_reports(self.model, path)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return predict_proba(x, self.model)

    def generate_sql(self, encoder) -> str:
        return generate_sql(
            feature_names=self.feature_names_,
            encoder=encoder,
            model_results=self.model_results
        )

    def save_scorecard(
            self,
            encoder,
            path: str = '.',
            base_scorecard_points: int = 444,
            odds: int = 10,
            points_to_double_odds: int = 69,
    ):
        save_scorecard_fn(
            feature_names=self.feature_names_,
            encoder=encoder,
            model_results=self.model_results,
            base_scorecard_points=base_scorecard_points,
            odds=odds,
            points_to_double_odds=points_to_double_odds,
            path=path
        )
