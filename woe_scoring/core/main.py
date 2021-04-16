import json
import os
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_X_y, check_array

from .functions import cat_bining, num_bining, refit_WOE_dict


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

        self.WOE_IV_dict = []
        self.feature_names = []
        self.num_features = []

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
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
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

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
            res_dict, missing_position = num_bining(
                X=X[:, feature_idx].astype(float),
                y=y,
                min_pcnt_group=self.min_pcnt_group,
                max_bins=self.max_bins,
            )
            self.WOE_IV_dict.append(
                {
                    feature: res_dict,
                    "missing_bin": missing_position,
                    "type_feature": "num",
                }
            )

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transforms input arrays
        :param X: X data array
        :return: transformed data
        """
        for i, _ in enumerate(self.WOE_IV_dict):
            feature = list(self.WOE_IV_dict[i])[0]
            self._print(f"Transform {feature} feature")
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
            if (
                self.WOE_IV_dict[i][feature][0]["woe"]
                < self.WOE_IV_dict[i][feature][-1]["woe"]
            ):
                X[new_feature].fillna(
                    self.WOE_IV_dict[i][feature][0]["woe"], inplace=True
                )
            else:
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

    def save(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump(self.WOE_IV_dict, file, indent=4, cls=NpEncoder)

    def load(self, path: str) -> None:
        with open(path, "r") as file:
            self.WOE_IV_dict = json.load(file)

    def refit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:
        if isinstance(X, pd.DataFrame):
            if self.special_cols:
                X = X.drop(self.special_cols, axis=1)
            self.feature_names = X.columns
        elif isinstance(X, np.ndarray):
            self.feature_names = [f"X_{i}" for i in range(X.shape[-1])]
        else:
            raise TypeError("X vector is not np array neither data frame")

        X, y = self._check_inputs(X, y)

        self.temp_WOE_IV_dict = []

        for i in range(len(self.WOE_IV_dict)):
            feature_idx = list(self.feature_names).index(
                list(self.WOE_IV_dict[i].keys())[0]
            )
            self._print(f"Refiting {list(self.WOE_IV_dict[i].keys())[0]} feature")
            res_dict = refit_WOE_dict(
                X=X[:, feature_idx],
                y=y,
                bins=[
                    bin["bin"]
                    for bin in self.WOE_IV_dict[i][list(self.WOE_IV_dict[i].keys())[0]]
                ],
                type_feature=self.WOE_IV_dict[0]["type_feature"],
            )
            self.temp_WOE_IV_dict.append(
                {
                    list(self.WOE_IV_dict[i].keys())[0]: res_dict,
                    "missing_bin": self.WOE_IV_dict[i]["missing_bin"],
                    "type_feature": self.WOE_IV_dict[0]["type_feature"],
                }
            )
        self.WOE_IV_dict = self.temp_WOE_IV_dict
        del self.temp_WOE_IV_dict

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


class CreateModel(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_vars: int = 20,
        verbose: bool = False,
        special_cols: List = None,
        n_jobs: int = None,
        gini_threshold: float = 5.0,
        delta_train_test_threshold: float = 0.2,
        random_state: int = None,
        class_weight: str = None,
        direction: str = "forward",
        cv: int = 3,
        scoring: str = "roc_auc",
        save_reports: bool = True,
        path_to_save: str = None,
    ):

        self.max_vars = max_vars
        self.verbose = verbose
        self.special_cols = special_cols
        self.n_jobs = n_jobs
        self.gini_threshold = gini_threshold
        self.delta_train_test_threshold = delta_train_test_threshold
        self.random_state = random_state
        self.class_weight = class_weight
        self.direction = direction
        self.cv = cv
        self.scoring = scoring
        self.save_reports = save_reports
        self.path_to_save = path_to_save

        self.feature_names = []
        self.model = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            if self.special_cols:
                X = X.drop(self.special_cols, axis=1)
            self.feature_names = X.columns
        elif isinstance(X, np.ndarray):
            self.feature_names = [f"X_{i}" for i in range(X.shape[-1])]
        else:
            raise TypeError("X vector is not np array neither data frame")

        to_drop = []
        for i in range(len(self.feature_names)):
            if (
                self._calc_score(
                    X,
                    y,
                    self.feature_names[i],
                    self.random_state,
                    self.class_weight,
                    self.cv,
                    self.scoring,
                    self.n_jobs,
                )
                < self.gini_threshold
            ):
                to_drop.append(self.feature_names[i])

        self.feature_names = [var for var in self.feature_names if var not in to_drop]

        to_drop = []
        for i in range(len(self.feature_names)):
            ceeds = np.random.randint(0, 100, self.cv)
            list_pcnt = []
            for ceed in ceeds:
                temp_train_X, temp_test_X, temp_train_y, temp_test_y = train_test_split(
                    X[self.feature_names[i]].values.reshape(-1, 1),
                    y,
                    test_size=0.3,
                    stratify=y,
                    shuffle=True,
                    random_state=ceed,
                )

                LR = LogisticRegression(
                    random_state=ceed,
                    class_weight=self.class_weight,
                    n_jobs=self.n_jobs,
                )
                LR.fit(temp_train_X, temp_train_y)
                y_pred_train = LR.predict_proba(temp_train_X)[:, 1]
                y_pred_test = LR.predict_proba(temp_test_X)[:, 1]

                train_score = roc_auc_score(temp_train_y, y_pred_train)
                test_score = roc_auc_score(temp_test_y, y_pred_test)
                list_pcnt.append((train_score - test_score) / train_score)
            mean_pcnt = np.mean(list_pcnt)
            if mean_pcnt > self.delta_train_test_threshold:
                to_drop.append(self.feature_names[i])

        self.feature_names = [var for var in self.feature_names if var not in to_drop]

        sfs = SequentialFeatureSelector(
            LogisticRegression(
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
            ),
            n_features_to_select=self.max_vars,
            direction=self.direction,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=self.scoring,
        )
        sfs.fit(X[self.feature_names], y)

        self.feature_names = list(np.array(self.feature_names)[list(sfs.get_support())])

        for var_a in self.feature_names:
            for var_b in self.feature_names:
                if (
                    var_a != var_b
                    and abs(X[self.feature_names].corr()[var_a][var_b]) > 0.5
                ):
                    if self._calc_score(
                        X,
                        y,
                        var_a,
                        self.random_state,
                        self.class_weight,
                        self.cv,
                        self.scoring,
                        self.n_jobs,
                    ) > self._calc_score(
                        X,
                        y,
                        var_b,
                        self.random_state,
                        self.class_weight,
                        self.cv,
                        self.scoring,
                        self.n_jobs,
                    ):
                        self.feature_names.remove(var_b)
                    else:
                        self.feature_names.remove(var_a)
                    break

        temp_model = sm.Logit(y, sm.add_constant(X[self.feature_names])).fit()

        retrain = False
        for i, pvalue in enumerate(temp_model.wald_test_terms().table["pvalue"]):
            if pvalue > 0.05:
                self.feature_names.remove(temp_model.wald_test_terms().table.index[i])

        if retrain:
            temp_model = sm.Logit(y, sm.add_constant(X[self.feature_names])).fit()

        if self.save_reports:
            try:
                with open(
                    os.path.join(self.path_to_save, "model_summary.txt"), "w"
                ) as outfile:
                    outfile.write(temp_model.summary().as_text())

                with open(
                    os.path.join(self.path_to_save, "model_wald.txt"), "w"
                ) as outfile:
                    temp_model.wald_test_terms().summary_frame().to_string(outfile)
            except Exception as e:
                print(f"Problem with saving: {e}")

        self.model = LogisticRegression(
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X[self.feature_names], y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = check_array(X)
        prediction = self.model.predict(X)
        return prediction

    def predict_proba(self, X: pd.DataFrame) -> List[float]:
        X = check_array(X)
        predict_proba = self.model.predict_proba(X)
        return predict_proba

    def fit_predict(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        self.fit(X=X, y=y)
        prediction = self.predict(X=X)
        return prediction

    def fit_predict_proba(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> List[float]:
        self.fit(X=X, y=y)
        predict_proba = self.predict_proba(X)
        return predict_proba

    def _calc_score(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        var: str,
        random_state: int = None,
        class_weight: str = None,
        cv: int = 3,
        scoring: str = "roc_auc",
        n_jobs: int = None,
    ) -> float:
        model = LogisticRegression(
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )
        scores = cross_val_score(
            model,
            X[var].values.reshape(-1, 1),
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        return (np.mean(scores) * 2 - 1) * 100

    def _check_inputs(
        self, X: pd.DataFrame, y: Union[pd.DataFrame, np.ndarray]
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
