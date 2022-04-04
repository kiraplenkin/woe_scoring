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
    b = offset - factor * intercept
    s = -factor * coef * woe
    return int(round(s + b / n_features))


def _calc_stats_for_feature(
        idx,
        feature,
        feature_names: List[str],
        encoder,
        model_results,
        factor: float,
        offset: float,
) -> pd.DataFrame:
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
    if idx < 1:
        result_dict["feature"].append(feature.replace("WOE_", ""))
        result_dict["coef"].append(model_results.loc[idx, "coef"])
        result_dict["pvalue"].append(model_results.loc[idx, "P>|z|"])
        for key in result_dict:
            if key not in ["feature", "coef", "pvalue"]:
                result_dict[key].append("-")
    else:
        for i, _ in enumerate(encoder.woe_iv_dict):
            if list(encoder.woe_iv_dict[i])[0] == feature.replace("WOE_", ""):
                for _bin in encoder.woe_iv_dict[i][feature.replace("WOE_", "")]:
                    result_dict["feature"].append(feature.replace("WOE_", ""))
                    result_dict["coef"].append(model_results.loc[idx, "coef"])
                    result_dict["pvalue"].append(model_results.loc[idx, "P>|z|"])
                    result_dict["bin"].append(
                        [val if val != -1 else str(val).replace("-1", "missing") for val in _bin["bin"]]
                    )
                    result_dict["WOE"].append(_bin["woe"])
                    result_dict["IV"].append(_bin["iv"])
                    result_dict["percent_of_population"].append(_bin["pct"])
                    result_dict["total"].append(_bin["total"])
                    result_dict["event_cnt"].append(_bin["bad"])
                    result_dict["non_event_cnt"].append(result_dict["total"][-1] - result_dict["event_cnt"][-1])
                    result_dict["event_rate"].append(_bin["bad_rate"])
                    result_dict["score_ball"].append(
                        _calc_score_points(
                            woe=result_dict["WOE"][-1],
                            coef=result_dict["coef"][-1],
                            intercept=model_results.iloc[0, 1],
                            factor=factor,
                            offset=offset,
                            n_features=len(feature_names),
                        )
                    )
    return pd.DataFrame.from_dict(result_dict)


def _calc_stats(
        feature_names: List[str],
        encoder,
        model_results,
        factor: float,
        offset: float,
) -> List[pd.DataFrame]:
    feature_stats = []
    for idx, feature in enumerate(model_results.iloc[:, 0]):
        res_df = _calc_stats_for_feature(
            idx,
            feature,
            feature_names,
            encoder,
            model_results,
            factor,
            offset
        )
        res_df.name = feature.replace("WOE_", "")
        feature_stats.append(res_df)
    return feature_stats


def _build_excel_sheet_with_charts(
        feature_stats: list[pd.DataFrame],
        writer: pd.ExcelWriter,
        width: int = 640,
        height: int = 480,
        first_plot_position: str = 'A',
        second_plot_position: str = "J",
):
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
