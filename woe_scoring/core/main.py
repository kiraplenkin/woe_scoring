import json
import os
from typing import List, Tuple, Union

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

from .functions import cat_binning, num_binning, refit_woe_dict


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
            max_bins: int = 10,
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
        self.max_bins = max_bins
        self.min_pct_group = min_pct_group
        self.cat_features = cat_features if cat_features else []
        self.special_cols = special_cols if special_cols else []
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
            else:
                if (
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
            max_vars: int = 20,
            special_cols: List = None,
            unused_cols: List = None,
            n_jobs: int = None,
            gini_threshold: float = 5.0,
            delta_train_test_threshold: float = 0.2,
            corr_threshold: float = 0.5,
            random_state: int = None,
            class_weight: str = None,
            direction: str = "forward",
            cv: int = 3,
            C: float = None,
            scoring: str = "roc_auc",
            save_report: bool = True,
            path_to_save: str = os.getcwd(),
    ):

        self.max_vars = max_vars
        self.special_cols = special_cols if special_cols else []
        self.unused_cols = unused_cols if unused_cols else []
        self.n_jobs = n_jobs
        self.gini_threshold = gini_threshold
        self.delta_train_test_threshold = delta_train_test_threshold
        self.corr_threshold = corr_threshold
        self.random_state = random_state
        self.class_weight = class_weight
        self.direction = direction
        self.cv = cv
        self.C = C
        self.scoring = scoring
        self.save_report = save_report
        self.path_to_save = path_to_save

        self.feature_names_: List[str] = []
        self.coef_: List[float] = []
        self.intercept_: float = 0.0
        self.model = None

    def _calc_score(
            self,
            x: pd.DataFrame,
            y: Union[pd.Series, np.ndarray],
            var: str,
            random_state: int = None,
            class_weight: str = None,
            cv: int = 3,
            c: float = None,
            scoring: str = "roc_auc",
            n_jobs: int = None,
    ) -> float:
        model = LogisticRegression(
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs,
            C=c,
        )
        scores = cross_val_score(
            model,
            x[var].values.reshape(-1, 1),
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        return (np.mean(scores) * 2 - 1) * 100

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

        #  check features for min gini threshold
        to_drop = []
        for i in range(len(self.feature_names_)):
            if (
                    self._calc_score(
                        x,
                        y,
                        self.feature_names_[i],
                        self.random_state,
                        self.class_weight,
                        self.cv,
                        self.C,
                        self.scoring,
                        self.n_jobs,
                    )
                    < self.gini_threshold
            ):
                to_drop.append(self.feature_names_[i])

        self.feature_names_ = [var for var in self.feature_names_ if var not in to_drop]

        #  check features for differance between train & test for min threshold
        to_drop = []
        for i in range(len(self.feature_names_)):
            ceeds = np.random.randint(0, 100, self.cv)
            list_pct = []
            for ceed in ceeds:
                temp_train_x, temp_test_x, temp_train_y, temp_test_y = train_test_split(
                    x[self.feature_names_[i]].values.reshape(-1, 1),
                    y,
                    test_size=0.3,
                    stratify=y,
                    shuffle=True,
                    random_state=ceed,
                )

                ls = LogisticRegression(
                    random_state=ceed,
                    class_weight=self.class_weight,
                    n_jobs=self.n_jobs,
                    C=self.C,
                )
                ls.fit(temp_train_x, temp_train_y)
                y_prediction_train = ls.predict_proba(temp_train_x)[:, 1]
                y_prediction_test = ls.predict_proba(temp_test_x)[:, 1]

                train_score = roc_auc_score(temp_train_y, y_prediction_train)
                test_score = roc_auc_score(temp_test_y, y_prediction_test)
                list_pct.append(np.abs((train_score - test_score) / train_score))
            mean_pct = np.mean(list_pct)
            if mean_pct > self.delta_train_test_threshold:
                to_drop.append(self.feature_names_[i])

        self.feature_names_ = [var for var in self.feature_names_ if var not in to_drop]

        #  find selected features
        sfs = SequentialFeatureSelector(
            LogisticRegression(
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_jobs=self.n_jobs,
                C=self.C,
            ),
            n_features_to_select=self.max_vars,
            direction=self.direction,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=self.scoring,
        )
        sfs.fit(x[self.feature_names_], y)

        self.feature_names_ = list(np.array(self.feature_names_)[list(sfs.get_support())])

        #  check correlation
        for var_a in self.feature_names_:
            for var_b in self.feature_names_:
                if (
                        var_a != var_b
                        and abs(x[self.feature_names_].corr()[var_a][var_b]) >= self.corr_threshold
                ):
                    if self._calc_score(
                            x,
                            y,
                            var_a,
                            self.random_state,
                            self.class_weight,
                            self.cv,
                            self.C,
                            self.scoring,
                            self.n_jobs,
                    ) > self._calc_score(
                        x,
                        y,
                        var_b,
                        self.random_state,
                        self.class_weight,
                        self.cv,
                        self.C,
                        self.scoring,
                        self.n_jobs,
                    ):
                        self.feature_names_.remove(var_b)
                    else:
                        self.feature_names_.remove(var_a)
                    break

        self.model = sm.Logit(y, sm.add_constant(x[self.feature_names_])).fit()

        #  check features for pvalue threshold
        retrain = False
        for i, pvalue in enumerate(self.model.wald_test_terms().table["pvalue"]):
            if pvalue > 0.05:
                self.feature_names_.remove(self.model.wald_test_terms().table.index[i])
                retrain = True
        if retrain:
            self.model = sm.Logit(y, sm.add_constant(x[self.feature_names_])).fit()

        self.results = pd.read_html(self.model.summary().tables[1].as_html(), header=0, index_col=0)[0].reset_index()
        self.intercept_ = self.results.iloc[0, 1]
        self.coef_ = list(self.results.iloc[1:, 1])
        self.feature_names_ = list(self.results.iloc[1:, 0])

        if self.save_report:
            try:
                with open(
                        os.path.join(self.path_to_save, "model_summary.txt"), "w"
                ) as outfile:
                    outfile.write(self.model.summary().as_text())

                with open(
                        os.path.join(self.path_to_save, "model_wald.txt"), "w"
                ) as outfile:
                    self.model.wald_test_terms().summary_frame().to_string(outfile)
            except Exception as e:
                print(f"Problem with saving: {e}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        x = check_array(x)
        prediction = self.model.predict(sm.add_constant(x))
        return prediction

    def generate_sql(self, encoder) -> str:
        sql = list()
        sql.append("with a as " + "(SELECT ")
        sql.append(",".join([var.replace("WOE_", "") for var in self.feature_names_]))
        sql.append("")

        for var in self.feature_names_:
            for i, _ in enumerate(encoder.woe_iv_dict):
                if list(encoder.woe_iv_dict[i])[0] == var.replace("WOE_", ""):
                    if encoder.woe_iv_dict[i]["type_feature"] == "cat":
                        sql.append(", CASE")
                        for bin in encoder.woe_iv_dict[i][var.replace("WOE_", "")]:
                            sql.append(
                                f" WHEN {var.replace('WOE_', '')} in {bin['bin']} THEN {bin['woe']}".replace(
                                    "[",
                                    "("
                                ).replace(
                                    "]", ")"
                                ).replace(", -1", "").replace(", Missing", "")
                            )
                        if encoder.woe_iv_dict[i]["missing_bin"] == "first":
                            sql.append(
                                f" WHEN {var.replace('WOE_', '')} IS NULL THEN {encoder.woe_iv_dict[i][var.replace('WOE_', '')][0]['woe']}"
                            )
                            sql.append(f" ELSE {encoder.woe_iv_dict[i][var.replace('WOE_', '')][0]['woe']}")
                        elif encoder.woe_iv_dict[i]["missing_bin"] == "last":
                            sql.append(
                                f" WHEN {var.replace('WOE_', '')} IS NULL THEN {encoder.woe_iv_dict[i][var.replace('WOE_', '')][len(encoder.woe_iv_dict[i][var.replace('WOE_', '')]) - 1]['woe']}"
                            )
                            sql.append(
                                f" ELSE {encoder.woe_iv_dict[i][var.replace('WOE_', '')][len(encoder.woe_iv_dict[i][var.replace('WOE_', '')]) - 1]['woe']}"
                            )
                        sql.append(f" END AS {var}")
                    else:
                        sql.append(", CASE")
                        if encoder.woe_iv_dict[i]["missing_bin"] == "first":
                            sql.append(
                                f" WHEN {var.replace('WOE_', '')} IS NULL THEN {encoder.woe_iv_dict[i][var.replace('WOE_', '')][0]['woe']}"
                            )
                        elif encoder.woe_iv_dict[i]["missing_bin"] == "last":
                            sql.append(
                                f" WHEN {var.replace('WOE_', '')} IS NULL THEN {encoder.woe_iv_dict[i][var.replace('WOE_', '')][len(encoder.woe_iv_dict[i][var.replace('WOE_', '')]) - 1]['woe']}"
                            )
                        for n, bin in enumerate(encoder.woe_iv_dict[i][var.replace("WOE_", "")]):
                            if n == 0:
                                sql.append(f" WHEN {var.replace('WOE_', '')} < {bin['bin'][1]} THEN {bin['woe']}")
                            elif n == len(encoder.woe_iv_dict[i][var.replace("WOE_", "")]) - 1:
                                sql.append(f" WHEN {var.replace('WOE_', '')} >= {bin['bin'][0]} THEN {bin['woe']}")
                            else:
                                sql.append(
                                    f" WHEN {var.replace('WOE_', '')} >= {bin['bin'][0]} AND {var.replace('WOE_', '')} < {bin['bin'][1]} THEN {bin['woe']}"
                                )
                        sql.append(f" END AS {var}")

        sql.append(f" FROM )")
        sql.append(", b as (")
        sql.append("SELECT a.*")
        sql.append(f", REPLACE(1 / (1 + EXP(-({self.results.iloc[0, 1]}")
        for idx in range(1, self.results.shape[0]):
            sql.append(f" + ({self.results.iloc[idx, 1]} * a.{self.results.iloc[idx, 0]})")
        sql.append("))), ',', '.') as PD")
        sql.append(" FROM a) ")
        sql.append("SELECT * FROM b")

        return "".join(sql)
