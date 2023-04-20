import os
from typing import List, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def _calc_score(
        x: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        var: str,
        random_state: int = None,
        class_weight: str = None,
        cv: int = 3,
        c: float = 1.0,
        scoring: str = "roc_auc",
        n_jobs: int = 1,
) -> float:
    """Calculate the score of a feature.
    Args:
        x: DataFrame or numpy array.
        y: Series or numpy array.
        var: Name of the feature.
        random_state: Random state.
        class_weight: Class weight.
        cv: Number of folds.
        c: Regularization parameter.
        scoring: Scoring method.
        n_jobs: Number of jobs.
    Returns:
        Score of the feature."""

    model = LogisticRegression(
        random_state=random_state,
        class_weight=class_weight,
        solver='saga',
        max_iter=1000,
        warm_start=True,
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
        x: Union[pd.DataFrame, np.ndarray],
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
    """Check if a feature has a Gini score below a threshold.
    Args:
        x: DataFrame or numpy array.
        y: Series or numpy array.
        feature_names: List of features.
        gini_threshold: Gini threshold.
        random_state: Random state.
        class_weight: Class weight.
        cv: Number of folds.
        c: Regularization parameter.
        scoring: Scoring method.
        n_jobs: Number of jobs.
    Returns:
        List of features with a Gini score below a threshold."""

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
        x: Union[pd.DataFrame, np.ndarray],
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
    """Check if a feature has a correlation score below a threshold.

    Args:
        x: DataFrame or numpy array.
        y: Series or numpy array.
        feature_names: List of features.
        corr_threshold: Correlation threshold.
        random_state: Random state.
        class_weight: Class weight.
        cv: Number of folds.
        c: Regularization parameter.
        scoring: Scoring method.
        n_jobs: Number of jobs.

    Returns:
        List of features with a correlation score below a threshold.
    """

    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x, columns=feature_names)

    correlation = x[feature_names].corr()
    mask = np.tril(np.ones_like(correlation, dtype=bool), k=-1)
    correlation = correlation.where(mask)
    correlation = correlation.stack().reset_index()
    correlation.columns = ['feature_a', 'feature_b', 'correlation']
    correlation = correlation.query(f"correlation.abs() >= {corr_threshold}")
    features_to_remove = []

    for feature in feature_names:
        if feature in features_to_remove:
            continue
        correlated_features = correlation.query(f"feature_a == '{feature}' or feature_b == '{feature}'")
        if len(correlated_features) == 0:
            continue
        scores = []
        for _, row in correlated_features.iterrows():
            if row['feature_a'] == feature:
                other_feature = row['feature_b']
            else:
                other_feature = row['feature_a']
            score = _calc_score(x, y, other_feature, random_state, class_weight, cv, c, scoring, n_jobs)
            scores.append(score)
        if all(score <= scores[0] for score in scores):
            features_to_remove.append(feature)

    return [f for f in feature_names if f not in features_to_remove]


def _check_min_pct_group(
    x: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    min_pct_group: float,
) -> List[str]:
    """Check if a feature has a minimum percentage of values below a threshold.
    Args:
        x: DataFrame or numpy array.
        feature_names: List of features.
        min_pct_group: Minimum percentage of values below a threshold.
    Returns:
        List of features with a minimum percentage of values below a threshold."""
    
    to_drop = [
        feature_name for feature_name in feature_names if
        x[feature_name].value_counts(normalize=True).min() < min_pct_group
    ]
    return [var for var in feature_names if var not in to_drop]


def _get_high_pval_positive_vars(X: pd.DataFrame, y: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        """
        Returns variables with high p-values and positive sign.

        Args:
            X: DataFrame.
            y: Series or numpy array.
            feature_names: List of features.

        Returns:
            List of variables with high p-values and positive sign.
        """
        model = sm.Logit(y, sm.add_constant(X[feature_names])).fit()
        summary_table = model.summary().tables[1].data
        return [
            row[0]
            for row in summary_table[2:]
            if float(row[1]) > 0 or float(row[4]) > 0.05
        ]


def create_model(
        x: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: List[str],
) -> sm.Logit:
    """Create Logistic Regression model.
    Args:
        x: DataFrame or numpy array.
        y: Series or numpy array.
        feature_names: List of features.
    Returns:
        Model."""

    return sm.Logit(y, sm.add_constant(x[feature_names])).fit()


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


def predict_proba(x: Union[pd.DataFrame, np.ndarray], model: sm.Logit):
    """Predict probabilities.
    Args:
        x: DataFrame or numpy array.
        model: Model.
    Returns:
        Probabilities."""

    return model.predict(sm.add_constant(x))


def generate_sql(
        feature_names: List[str],
        encoder,
        model_results,
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
            f", REPLACE(1 / (1 + EXP(-({model_results.iloc[0, 1]}",
        )
    )

    sql.extend(
        f" + ({model_results.iloc[idx, 1]} * a.{model_results.iloc[idx, 0]})"
        for idx in range(1, model_results.shape[0])
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
