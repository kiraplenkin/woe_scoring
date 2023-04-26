import os
from typing import List, Union, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import product
from .model import Model
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def calculate_gini_score(
        data: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray],
        var: str,
        random_state: int,
        class_weight: str,
        cv: int,
        scoring: str,
        n_jobs: int
) -> float:
    """
    Calculate the Gini score for a given feature using Logistic Regression.

    Args:
        data: A pandas DataFrame or numpy array containing the feature and target data.
        target: A pandas Series or numpy array containing the target data.
        feature: A string representing the name of the feature to calculate the Gini score for.
        random_state: An integer representing the random state for the Logistic Regression estimator.
        class_weight: A string or dictionary representing the class weight for the Logistic Regression estimator.
        cv: An integer representing the number of cross-validation folds to use.
        scoring: A string representing the scoring metric to use for cross-validation.
        n_jobs: An integer representing the number of parallel jobs to run during cross-validation.

    Returns:
        A float representing the Gini score for the given feature.
    """

    scores = cross_val_score(
        estimator=LogisticRegression(
            random_state=random_state,
            class_weight=class_weight,
            max_iter=1000,
            n_jobs=n_jobs,
            warm_start=True,
        ),
        X=data[var].values.reshape(-1, 1),
        y=target,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    return (np.mean(scores) * 2 - 1) * 100


def check_features_gini_threshold(
        data: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray],
        feature_names: List[str],
        gini_threshold: float,
        random_state: int,
        class_weight: str,
        cv: int,
        scoring: str,
        n_jobs: int
) -> List[str]:
    """
    Returns the list of feature names whose gini score is above the given threshold.

    Args:
        data: The input data.
        target: The target variable.
        feature_names: The names of the features to check.
        gini_threshold: The threshold to use for the gini score.
        random_state: The random state to use for the cross-validation.
        class_weight: The class weight to use for the model.
        cv: The number of cross-validation folds.
        scoring: The scoring function to use for the cross-validation.
        n_jobs: The number of jobs to run in parallel for the cross-validation.

    Returns:
        A list of feature names whose gini score is above the given threshold.
    """

    features_to_drop = [
        feature_name
        for feature_name in feature_names
        if calculate_gini_score(
            data=data,
            target=target,
            var=feature_name,
            random_state=random_state,
            class_weight=class_weight,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
        ) < gini_threshold
    ]
    return [var for var in feature_names if var not in features_to_drop]


def check_correlation_threshold(
        data: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray],
        feature_names: List[str],
        corr_threshold: float,
        random_state: int,
        class_weight: str,
        cv: int,
        n_jobs: int,
        scoring: str,
) -> List[str]:
    """
    Returns a list of uncorrelated features based on a given correlation
    threshold and their respective scores.

    Args:
        data: The input data.
        target: The target variable.
        feature_names: The list of feature names to evaluate.
        corr_threshold: The correlation threshold to use for evaluation.
        random_state: The random state to use for evaluation.
        class_weight: The class weight to use for evaluation.
        cv: The number of cross-validation folds to use for evaluation.
        n_jobs: The number of jobs to run in parallel for evaluation.
        scoring: The scoring metric to use for evaluation.

    Returns:
        A list of uncorrelated feature names based on their respective scores.
    """

    correlation_matrix = data[feature_names].corr()
    for feature_a, feature_b in product(feature_names, feature_names):
        if (
            feature_a != feature_b
            and feature_a in feature_names
            and feature_b in feature_names
            and abs(correlation_matrix[feature_a][feature_b]) >= corr_threshold
        ):
            score_a = calculate_gini_score(
                data=data,
                target=target,
                var=feature_a,
                random_state=random_state,
                class_weight=class_weight,
                cv=cv,
                n_jobs=n_jobs,
                scoring=scoring
            )
            score_b = calculate_gini_score(
                data=data,
                target=target,
                var=feature_b,
                random_state=random_state,
                class_weight=class_weight,
                cv=cv,
                n_jobs=n_jobs,
                scoring=scoring
            )
            if score_a > score_b:
                feature_names.remove(feature_b)
            else:
                feature_names.remove(feature_a)
    return feature_names


def check_min_pct_group(
    data: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    min_pct_group: float,
) -> List[str]:
    """
    Check if a feature has a minimum percentage of values below a threshold.

    Args:
        data: DataFrame or numpy array.
        feature_names: List of features.
        min_pct_group: Minimum percentage of values below a threshold.

    Returns:
        List of features with a minimum percentage of values below a threshold.
    """

    return [
        feature_name for feature_name in feature_names 
        if feature_name not in [
            feature_name for feature_name in feature_names 
            if data[feature_name].value_counts(normalize=True).min() < min_pct_group
        ]
    ]


def find_bad_features(model: Model) -> List[int]:
    """Find features with high p-values and positive sign.
    Args:
        model: Model.
    Returns:
        List of features with high p-values and positive sign.
    """

    return [
        i
        for i in range(len(model.feature_names_))
        if model.pvalues_[i] > 0.05 or model.coef_[i] > 0
    ]


def calc_iv_dict(data: pd.DataFrame, target: np.ndarray, feature: str) -> Dict:
    """Calculate the information value (IV) of a categorical feature.

    Args:
        data: A pandas DataFrame containing the feature and target columns.
        target: A numpy array of binary labels (0 for good, 1 for bad).
        feature: A string with the name of the categorical feature.

    Returns:
        A dictionary with the feature name as key and the IV as value.
    """

    values = data[feature].values
    unique_values, value_counts = np.unique(values, return_counts=True)
    bad = np.zeros_like(unique_values)
    good = np.zeros_like(unique_values)
    for i, value in enumerate(unique_values):
        bad[i] = target[values == value].sum()
        good[i] = value_counts[i] - bad[i]
    all_bad = target.sum()
    all_good = len(target) - all_bad
    iv = ((good / all_good) - (bad / all_bad)) * unique_values
    return {feature: iv.sum()}


def save_reports(
        model: sm.Logit,
        path: str = os.getcwd()
) -> None:
    """Save model reports.
    Args:
        model: Model.
        path: Path to save reports."""

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


def generate_sql(
        encoder, feature_names: List[str], coef: List[float], intercept: float,
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
            f", REPLACE(1 / (1 + EXP(-({intercept}",
        )
    )

    sql.extend(
        f" + ({coef[idx]} * a.{feature_names[idx]})"
        for idx in range(len(feature_names))
    )
    sql.extend(("))), ',', '.') as PD", " FROM a) ", "SELECT * FROM b"))
    return "".join(sql)


def _calc_score_points(woe, coef, intercept, factor, offset: float, n_features: int) -> int:
    """Calculate score points.
    Args:
        woe: WOE.
        coef: Coefficient.
        intercept: Intercept.
        factor: Factor.
        offset: Offset.
        n_features: Number of features.
    Returns:
        Score points."""

    return -(woe * coef + intercept / n_features) * factor + offset / n_features


def _calc_stats_for_feature(
    idx,
    feature,
    feature_names: List[str],
    encoder,
    model_results,
    factor: float,
    offset: float,
) -> pd.DataFrame:
    """Calculate stats for feature.
    Args:
        idx: Index.
        feature: Feature.
        feature_names: Feature names.
        encoder: Encoder.
        model_results: Model results.
        factor: Factor.
        offset: Offset.
    Returns:
        Stats for feature.
    """
    result_dict = {
        "feature": [],
        "coef": [],
        "pvalue": [],
        "bin": [],
        "WOE": [],
        "IV": [],
        "percent_of_population": [],
        "total": [],
        "event_cnt": [],
        "non_event_cnt": [],
        "event_rate": [],
        "score_ball": [],
    }

    woe_iv_dict = encoder.woe_iv_dict
    intercept = model_results.iloc[0, 1]
    n_features = len(feature_names)

    if idx < 1:
        _update_result_dict(result_dict, feature, model_results, idx)
        for key, value in result_dict.items():
            if key not in ["feature", "coef", "pvalue"]:
                value.append("-")
    else:
        for woe_iv in woe_iv_dict:
            if list(woe_iv.keys())[0] == feature.replace("WOE_", ""):
                feature_woe_iv = woe_iv[feature.replace("WOE_", "")]
                for bin_info in feature_woe_iv:
                    _update_result_dict(result_dict, feature, model_results, idx)
                    bin_values = bin_info["bin"]
                    bin_values_str = [
                        str(val).replace("-1", "missing") if val == -1 else val
                        for val in bin_values
                    ]
                    result_dict["bin"].append(bin_values_str)
                    result_dict["WOE"].append(bin_info["woe"])
                    result_dict["IV"].append(bin_info["iv"])
                    result_dict["percent_of_population"].append(bin_info["pct"])
                    result_dict["total"].append(bin_info["total"])
                    result_dict["event_cnt"].append(bin_info["bad"])
                    result_dict["non_event_cnt"].append(bin_info["good"])
                    result_dict["event_rate"].append(bin_info["bad_rate"])
                    result_dict["score_ball"].append(
                        _calc_score_points(
                            woe=result_dict["WOE"][-1],
                            coef=result_dict["coef"][-1],
                            intercept=intercept,
                            factor=factor,
                            offset=offset,
                            n_features=n_features,
                        )
                    )

    return pd.DataFrame.from_dict(result_dict)


def _update_result_dict(result_dict, feature, model_results, idx) -> None:
    """Update result dict.
    Args:
        result_dict: Result dict.
        feature: Feature.
        model_results: Model results.
        idx: Index.
    Returns:
        None."""

    result_dict["feature"].append(feature.replace("WOE_", ""))
    result_dict["coef"].append(model_results.loc[idx, "coef"])
    result_dict["pvalue"].append(model_results.loc[idx, "P>|z|"])


def _calc_stats(
        feature_names: List[str],
        encoder,
        model_results,
        factor: float,
        offset: float,
) -> List[pd.DataFrame]:
    """Calculate stats.
    Args:
        feature_names: Feature names.
        encoder: Encoder.
        model_results: Model results.
        factor: Factor.
        offset: Offset.
    Returns:
        Stats."""

    return Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(_calc_stats_for_feature)(
            idx, feature, feature_names, encoder, model_results, factor, offset
        ).rename(feature.replace("WOE_", ""))
        for idx, feature in enumerate(model_results.iloc[:, 0])
    )


def _build_excel_sheet_with_charts(
        feature_stats: list[pd.DataFrame],
        writer: pd.ExcelWriter,
        width: int = 640,
        height: int = 480,
        first_plot_position: str = 'A',
        second_plot_position: str = "J",
) -> None:
    """Build excel sheet with charts.
    Args:
        feature_stats: Feature stats.
        writer: Writer.
        width: Width.
        height: Height.
        first_plot_position: First plot position.
        second_plot_position: Second plot position.
    Returns:
        None."""

    # Get workbook link
    workbook = writer.book
    # Create merge format
    merge_format = workbook.add_format(
        {
            'bold': 1,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        }
    )
    const = [result for result in feature_stats if result.name == 'const']
    iterator = [result for result in feature_stats if ((result is not None) and (result.name != 'const'))]
    scorecard_iterator = [*const, *iterator]
    indexes = np.cumsum([len(result) for result in scorecard_iterator])
    full_features = pd.concat(tuple(scorecard_iterator), ignore_index=True)
    full_features.to_excel(writer, sheet_name='Scorecard')
    worksheet = writer.sheets['Scorecard']
    area_start = 1
    for result, index in zip(scorecard_iterator, indexes):
        for column, column_width in zip([1, 2, 3], [20, 10, 10]):
            worksheet.merge_range(area_start, column, index, column, result.iloc[0, column - 1], merge_format)
            worksheet.set_column(column, column, column_width)
        area_start = index + 1

    for result in iterator:
        # Get dimensions of result Excel sheet and column indexes
        max_row = len(result)
        event_cnt = result.columns.get_loc('event_cnt') + 1
        non_event_cnt = result.columns.get_loc('non_event_cnt') + 1
        score_ball = result.columns.get_loc('score_ball') + 1
        woe = result.columns.get_loc('WOE') + 1
        event_rate = result.columns.get_loc('event_rate') + 1
        # Set sheet name, transfer data to sheet
        sheet_name = result.name
        result.to_excel(writer, sheet_name=sheet_name)
        # Get worksheet link
        worksheet = writer.sheets[sheet_name]
        # Create stacked column chart
        chart_events = workbook.add_chart({'type': 'column', 'subtype': 'stacked'})
        # Add event and non-event counts to chart
        chart_events.add_series(
            {
                'name': 'event_cnt ',
                'values': [sheet_name, 1, event_cnt, max_row, event_cnt]
            }
        )
        chart_events.add_series(
            {
                'name': 'non_event_cnt ',
                'values': [sheet_name, 1, non_event_cnt, max_row, non_event_cnt]
            }
        )
        # Create separate line chart for combination
        woe_line = workbook.add_chart({'type': 'line'})
        woe_line.add_series(
            {
                'name': 'WOE',
                'values': [sheet_name, 1, woe, max_row, woe],
                'smooth': False,
                'y2_axis': True,
            }
        )
        # Combine charts
        chart_events.combine(woe_line)
        # Create column chart for score_ball
        chart_score_ball = workbook.add_chart({'type': 'column'})
        chart_score_ball.add_series(
            {
                'name': 'score_ball ',
                'values': [sheet_name, 1, score_ball, max_row, score_ball]
            }
        )
        # Create separate line chart for combination
        event_rate_line = workbook.add_chart({'type': 'line'})
        event_rate_line.add_series(
            {
                'name': 'event_rate',
                'values': [sheet_name, 1, event_rate, max_row, event_rate],
                'smooth': False,
                'y2_axis': True,
            }
        )
        # Combine charts
        chart_score_ball.combine(event_rate_line)
        # Change size and legend of charts
        chart_events.set_size({'width': width, 'height': height})
        chart_events.set_legend({'position': 'bottom'})
        chart_score_ball.set_size({'width': width, 'height': height})
        chart_score_ball.set_legend({'position': 'bottom'})
        # Merge first 3 columns
        worksheet.merge_range(1, 1, max_row, 1, result.iloc[1, 0], merge_format)
        worksheet.set_column(1, 1, 20)
        worksheet.merge_range(1, 2, max_row, 2, result.iloc[1, 1], merge_format)
        worksheet.set_column(2, 2, 10)
        worksheet.merge_range(1, 3, max_row, 3, result.iloc[1, 2], merge_format)
        worksheet.set_column(3, 3, 10)
        # Insert charts
        worksheet.insert_chart(f'{first_plot_position}{max_row + 3}', chart_events)
        worksheet.insert_chart(f'{second_plot_position}{max_row + 3}', chart_score_ball)


def save_scorecard_fn(
        feature_names: List[str],
        encoder,
        model_results,
        base_scorecard_points: int,
        odds: int,
        points_to_double_odds: int,
        path: str,
) -> None:
    """Save scorecard.
    Args:
        feature_names: Feature names.
        encoder: Encoder.
        model_results: Model results.
        base_scorecard_points: Base scorecard points.
        odds: Odds.
        points_to_double_odds: Points to double odds.
        path: Path.
    Returns:
        None."""

    factor = points_to_double_odds / np.log(2)
    offset = base_scorecard_points - factor * np.log(odds)

    feature_stats = _calc_stats(
        feature_names=feature_names,
        encoder=encoder,
        model_results=model_results,
        factor=factor,
        offset=offset
    )

    try:
        writer = pd.ExcelWriter(os.path.join(path, "Scorecard.xlsx"), engine="xlsxwriter")
        _build_excel_sheet_with_charts(
            feature_stats=feature_stats,
            writer=writer
        )
        writer.save()
    except Exception as e:
        print(f"Problem with saving: {e}")
