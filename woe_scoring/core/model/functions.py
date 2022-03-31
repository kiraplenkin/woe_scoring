import os
from operator import itemgetter
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def _calc_score(
        x: [pd.DataFrame, np.ndarray],
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


def _check_features_gini_threshold(
        x: [pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: List[str],
        gini_threshold: float,
        random_state: int = None,
        class_weight: str = None,
        cv: int = 3,
        c: float = None,
        scoring: str = "roc_auc",
        n_jobs: int = None
) -> List[str]:
    to_drop = [
        feature_name
        for feature_name in feature_names
        if _calc_score(
            x,
            y,
            feature_name,
            random_state,
            class_weight,
            cv,
            c,
            scoring,
            n_jobs,
        )
           < gini_threshold
    ]
    return [var for var in feature_names if var not in to_drop]


def _check_correlation_threshold(
        x: [pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: List[str],
        corr_threshold: float,
        random_state: int,
        class_weight: str,
        cv: int,
        c: float,
        scoring: str,
        n_jobs: int
) -> List[str]:
    for var_a in feature_names:
        for var_b in feature_names:
            if (
                    var_a != var_b
                    and abs(x[feature_names].corr()[var_a][var_b]) >= corr_threshold
            ):
                if _calc_score(
                        x,
                        y,
                        var_a,
                        random_state,
                        class_weight,
                        cv,
                        c,
                        scoring,
                        n_jobs,
                ) > _calc_score(
                    x,
                    y,
                    var_b,
                    random_state,
                    class_weight,
                    cv,
                    c,
                    scoring,
                    n_jobs,
                ):
                    feature_names.remove(var_b)
                else:
                    feature_names.remove(var_a)
                break
    return feature_names


def _feature_selector(
        x: [pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: List[str],
        random_state: int,
        class_weight: str,
        cv: int,
        c: float,
        n_jobs: int,
        max_vars: Union[int, float],
        direction: str,
        scoring: str,
) -> List[str]:
    sfs = SequentialFeatureSelector(
        LogisticRegression(
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs,
            C=c,
        ),
        n_features_to_select=max_vars,
        direction=direction,
        cv=cv,
        n_jobs=n_jobs,
        scoring=scoring,
    )
    sfs.fit(x[feature_names], y)
    return list(np.array(feature_names)[list(sfs.get_support())])


def sequential_feature_select(
        x: [pd.DataFrame],
        y: Union[pd.Series, np.ndarray],
        feature_names: List[str],
        gini_threshold: float,
        corr_threshold: float,
        random_state: int,
        class_weight: str,
        max_vars: Union[int, float],
        direction: str,
        cv: int,
        c: float,
        scoring: str,
        n_jobs: int,
) -> List[str]:
    feature_names = _check_features_gini_threshold(
        x, y,
        feature_names=feature_names,
        gini_threshold=gini_threshold,
        random_state=random_state,
        class_weight=class_weight,
        cv=cv,
        c=c,
        scoring=scoring,
        n_jobs=n_jobs
    )

    feature_names = _feature_selector(
        x, y,
        feature_names=feature_names,
        random_state=random_state,
        class_weight=class_weight,
        cv=cv,
        c=c,
        n_jobs=n_jobs,
        max_vars=max_vars,
        direction=direction,
        scoring=scoring,
    )

    feature_names = _check_correlation_threshold(
        x, y,
        feature_names=feature_names,
        corr_threshold=corr_threshold,
        random_state=random_state,
        class_weight=class_weight,
        cv=cv,
        c=c,
        scoring=scoring,
        n_jobs=n_jobs
    )
    return feature_names


def _calc_iv_dict(x: pd.DataFrame, y: np.ndarray, feature: str) -> Dict:
    _iv = 0
    for value in x[feature].sort_values().unique():
        bad = y[x[feature] == value].sum()
        good = len(y[x[feature] == value]) - bad
        all_bad = y.sum()
        all_good = len(y) - all_bad
        _iv += ((good / all_good) - (bad / all_bad)) * value
    return {feature: _iv}


def iv_feature_select(
        x: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        feature_names: List[str],
        iv_threshold: float,
        max_vars: int,
        n_jobs: int,
) -> List[str]:
    temp_res_dict = Parallel(n_jobs=n_jobs)(
        delayed(_calc_iv_dict)(x, y, feature) for feature in feature_names
    )
    res_dict = {}
    for d in temp_res_dict:
        res_dict.update(d)
    return [feature for feature in dict(sorted(res_dict.items(), key=itemgetter(1), reverse=True)) if
            res_dict[feature] >= iv_threshold][:max_vars]


def _check_pvalue(model: sm.Logit) -> List[int]:
    return [
        model.wald_test_terms().table.index[i]
        for i, pvalue in enumerate(model.wald_test_terms().table["pvalue"])
        if pvalue > 0.05
    ]


def create_model(
        x: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: List[str],
) -> sm.Logit:
    model = sm.Logit(y, sm.add_constant(x[feature_names])).fit()

    to_drop = _check_pvalue(model)
    while len(to_drop) > 0:
        feature_names = [feature for feature in feature_names if feature not in to_drop]
        model = sm.Logit(y, sm.add_constant(x[feature_names])).fit()
        to_drop = _check_pvalue(model)

    return model


def save_reports(
        model: sm.Logit,
        path: str = os.getcwd()
) -> None:
    try:
        with open(
                os.path.join(path, "model_summary.txt"), "w"
        ) as outfile:
            outfile.write(model.summary().as_text())

        with open(
                os.path.join(path, "model_wald.txt"), "w"
        ) as outfile:
            model.wald_test_terms().summary_frame().to_string(outfile)
    except Exception as e:
        print(f"Problem with saving: {e}")


def predict_proba(x: Union[pd.DataFrame, np.ndarray], model: sm.Logit):
    return model.predict(sm.add_constant(x))


def generate_sql(
        feature_names: List[str],
        encoder,
        results,
) -> str:
    sql = [
        "with a as " + "(SELECT ",
        ",".join([var.replace("WOE_", "") for var in feature_names]),
        "",
    ]
    for var in feature_names:
        for i, _ in enumerate(encoder.woe_iv_dict):
            if list(encoder.woe_iv_dict[i])[0] == var.replace("WOE_", ""):
                sql.append(", CASE")
                if encoder.woe_iv_dict[i]["type_feature"] == "cat":
                    sql.extend(
                        f" WHEN {var.replace('WOE_', '')} in {bin['bin']} THEN {bin['woe']}".replace(
                            "[", "("
                        )
                            .replace("]", ")")
                            .replace(", -1", "")
                            .replace(", Missing", "")
                        for bin in encoder.woe_iv_dict[i][
                            var.replace("WOE_", "")
                        ]
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
                else:
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
    sql.extend(
        (
            " FROM )",
            ", b as (",
            "SELECT a.*",
            f", REPLACE(1 / (1 + EXP(-({results.iloc[0, 1]}",
        )
    )

    sql.extend(
        f" + ({results.iloc[idx, 1]} * a.{results.iloc[idx, 0]})"
        for idx in range(1, results.shape[0])
    )
    sql.extend(("))), ',', '.') as PD", " FROM a) ", "SELECT * FROM b"))
    return "".join(sql)
