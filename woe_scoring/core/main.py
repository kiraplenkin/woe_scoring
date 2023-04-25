import json
from typing import List, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels

from .binning.functions import (cat_processing, find_cat_features,
                                num_processing, prepare_data, refit)
from .model.functions import (_check_correlation_threshold,
                              generate_sql, save_reports, _check_min_pct_group, _check_features_gini_threshold,
                              save_scorecard_fn, _find_bad_features)
from .model.selector import FeatureSelector
from .model.model import Model


class NpEncoder(json.JSONEncoder):
    """Convert NumPy objects to JSON serializable ones."""

    def default(self, obj):
        """Convert a non-serializable object to a serializable one.

        If `obj` is an instance of `np.integer`, this function returns it as a
        Python integer. If `obj` is an instance of `np.floating`, this function
        returns it as a Python float. If `obj` is an instance of `np.ndarray`,
        this function returns its contents as a nested list. Otherwise, this
        function delegates the conversion to the parent class.

        Args:
            obj: An object to be converted to a serializable one.

        Returns:
            A serializable version of `obj`.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            max_bins: Union[int, float] = 10,
            min_pct_group: float = 0.05,
            n_jobs: int = 1,
            prefix: str = "WOE_",
            merge_type: str = "chi2",
            cat_features: List[str] = None,
            special_cols: List[str] = None,
            cat_features_threshold: int = 0,
            diff_woe_threshold: float = 0.05,
            safe_original_data: bool = False,
    ):
        """
        Initializes the class with the given hyperparameters.
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

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> None:
        """
        Fits the input data
        :param data: data matrix
        :param target: target vector
        :return: None
        """
        data, self.feature_names = prepare_data(data=data, special_cols=self.special_cols)
        self.classes_ = unique_labels(target)

        if len(self.cat_features) == 0 and self.cat_features_threshold > 0:
            self.cat_features = find_cat_features(
                data=data,
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
                    data[col],
                    target,
                    self.min_pct_group,
                    self.max_bins,
                    self.diff_woe_threshold
                ) for col in self.cat_features
            )
        else:
            self.num_features = self.feature_names

        num_features_res = Parallel(n_jobs=self.n_jobs)(
            delayed(num_processing)(
                data[col],
                target,
                self.min_pct_group,
                self.max_bins,
                self.diff_woe_threshold,
                self.merge_type
            ) for col in self.num_features
        )

        self.woe_iv_dict += num_features_res


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input data
        :param data: input data frame
        :return: transformed data
        """
        data = data.copy()
        for i, woe_iv in enumerate(self.woe_iv_dict):
            feature = list(woe_iv)[0]
            woe_iv_feature = woe_iv[feature]
            new_feature = self.prefix + feature
            for bin_values in woe_iv_feature:
                if feature in self.cat_features:
                    data.loc[
                        np.isin(data[feature], bin_values["bin"]), new_feature
                    ] = bin_values["woe"]
                else:
                    data.loc[
                        np.logical_and(
                            data[feature] >= np.min(bin_values["bin"]),
                            data[feature] < np.max(bin_values["bin"]),
                        ),
                        new_feature,
                    ] = bin_values["woe"]
            missing_bin = woe_iv["missing_bin"]
            if missing_bin == "first":
                data[new_feature].fillna(woe_iv_feature[0]["woe"], inplace=True)
            elif missing_bin == "last":
                data[new_feature].fillna(woe_iv_feature[-1]["woe"], inplace=True)
            elif woe_iv_feature[0]["woe"] < woe_iv_feature[-1]["woe"]:
                data[new_feature].fillna(woe_iv_feature[0]["woe"], inplace=True)
            else:
                data[new_feature].fillna(woe_iv_feature[-1]["woe"], inplace=True)
            if not self.safe_original_data:
                del data[feature]
        return data



    def save_to_file(self, file_path: str) -> None:
        """
        Save the woe_iv_dict to a JSON file at the specified file path.

        Args:
            file_path (str): The path where the file should be saved.

        Returns:
            None
        """
        with open(file_path, "w") as f:
            json.dump(self.woe_iv_dict, f, indent=4, cls=NpEncoder)


    def load_woe_iv_dict(self, file_path: str) -> None:
        """
        Load a dictionary of WoE and IV values from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            None
        """
        with open(file_path, "r") as json_file:
            self.woe_iv_dict = json.load(json_file)


    def refit(self, x: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:
        x, self.feature_names = prepare_data(x=x, special_cols=self.special_cols)
        self.woe_iv_dict = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(
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
        selection_method: str = 'rfe', # rfe or sfs or iv
        model_type: str = 'sklearn', # 'sklearn' or 'statsmodel'
        max_vars: Union[int, float, None] = None,
        special_cols: List[str] = None,
        unused_cols: List[str] = None,
        n_jobs: int = 1,
        gini_threshold: float = 5.0,
        iv_threshold: float = 0.05,
        corr_threshold: float = 0.5,
        min_pct_group: float = 0.05,
        random_state: int = None,
        class_weight: str = 'balanced',
        direction: str = "forward",
        cv: int = 3,
        l1_exp_scale: int  = 4,
        l1_grid_size: int = 20,
        scoring: str = "roc_auc",
    ):
        self.selection_method = selection_method
        self.model_type = model_type
        self.max_vars = max_vars
        self.special_cols = special_cols or []
        self.unused_cols = unused_cols or []
        self.n_jobs = n_jobs
        self.gini_threshold = gini_threshold
        self.iv_threshold = iv_threshold
        self.corr_threshold = corr_threshold
        self.min_pct_group = min_pct_group
        self.random_state = random_state
        self.class_weight = class_weight
        self.direction = direction
        self.cv = cv
        self.l1_exp_scale = l1_exp_scale
        self.l1_grid_size = l1_grid_size
        self.scoring = scoring

        self.coef_ = None
        self.intercept_ = None
        self.feature_names_ = []
        self.model_score_ = None
        self.pvalues_ = []

        self.model = None


    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> None:
        data, self.feature_names_ = prepare_data(data=data, special_cols=self.special_cols)

        if self.unused_cols:
            self.feature_names_ = [feature for feature in self.feature_names_ if feature not in self.unused_cols]

        if self.max_vars is not None and self.max_vars < 1:
            self.max_vars = int(len(self.feature_names_) * self.max_vars)

        self.feature_names_ = _check_min_pct_group(
            x=data, feature_names=self.feature_names_, min_pct_group=self.min_pct_group
        )

        self.feature_names_ = _check_features_gini_threshold(
            x=data, y=target,
            feature_names=self.feature_names_,
            gini_threshold=self.gini_threshold,
            random_state=self.random_state,
            class_weight=self.class_weight,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs
        )
        
        feature_selector = FeatureSelector(
            selection_type=self.selection_method,
            max_vars=self.max_vars,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            class_weight=self.class_weight,
            direction=self.direction,
            cv=self.cv,
            l1_exp_scale=self.l1_exp_scale,
            l1_grid_size=self.l1_grid_size,
            scoring=self.scoring,
            gini_threshold=self.gini_threshold,
            min_pct_group=self.min_pct_group,
            iv_threshold=self.iv_threshold
        )
        selected_model = Model(
            model_type=self.model_type,
            n_jobs=self.n_jobs,
            l1_exp_scale=self.l1_exp_scale,
            l1_grid_size=self.l1_grid_size,
            cv=self.cv,
            class_weight=self.class_weight,
            random_state=self.random_state,
            scoring=self.scoring
        )

        selected_features = feature_selector.select(data, target, self.feature_names_)
        selected_features = _check_correlation_threshold(
            data, target,
            feature_names=selected_features,
            corr_threshold=self.corr_threshold,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            cv=self.cv,
            scoring=self.scoring,
        )
        self.model = selected_model.get_model(data[selected_features], target)

        while True:
            bad_feature_idx = _find_bad_features(selected_model)
            if len(bad_feature_idx) == 0:
                break
            self.feature_names_ = [feature for feature in self.feature_names_ if feature not in [self.feature_names_[i] for i in bad_feature_idx]]
            selected_features = feature_selector.select(data, target, self.feature_names_)
            selected_features = _check_correlation_threshold(
                data, target,
                feature_names=selected_features,
                corr_threshold=self.corr_threshold,
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
                cv=self.cv,
                scoring=self.scoring,
            )
            self.model = selected_model.get_model(data[selected_features], target)
        
        self.coef_ = selected_model.coef_
        self.intercept_ = selected_model.intercept_
        self.feature_names_ = selected_model.feature_names_
        self.model_score_ = selected_model.model_score_
        self.pvalues_ = selected_model.pvalues_

        return self.model


    def save_reports(self, path: str):
        save_reports(self.model, path)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(data)

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
