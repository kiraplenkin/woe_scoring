import copy
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def _chi2(bad_rates: List[Dict], overall_rate: float) -> float:
    """Calculate the chi-squared statistic for the given bad rates and overall rate.
    Args:
        bad_rates (List[Dict]): List of bad rates.
        overall_rate (float): Overall rate.
    Returns:
        float: Chi-squared statistic."""

    f_obs = np.array([b["bad"] for b in bad_rates])
    f_exp = np.array([b["total"] * overall_rate for b in bad_rates])
    return np.sum((f_obs - f_exp) ** 2 / f_exp)


def _check_diff_woe(bad_rates: List[Dict], diff_woe_threshold: float) -> Union[None, int]:
    """Check if the difference in woe is greater than the threshold.
    Args:
        bad_rates (List[Dict]): List of bad rates.
        diff_woe_threshold (float): Difference in woe threshold.
    Returns:
        Union[None, int]: Index of the bad rate with the smallest difference in woe."""

    woe_values = [bad_rate["woe"] for bad_rate in bad_rates]
    woe_delta = np.abs(np.diff(woe_values))
    min_diff_woe_index = np.argmin(woe_delta)
    if woe_delta[min_diff_woe_index] < diff_woe_threshold:
        return min_diff_woe_index
    else:
        return None


def _mono_flags(bad_rates: List[Dict]) -> bool:
    """Check if the difference in bad rate is monotonic.
    Args:
        bad_rates (List[Dict]): List of bad rates.
    Returns:
        bool: True if the difference in bad rate is monotonic."""
    prev_bad_rate = bad_rates[0]["bad_rate"]
    
    increasing, decreasing = False, False
    for i in range(1, len(bad_rates)):
        curr_bad_rate = bad_rates[i]["bad_rate"]
        if curr_bad_rate < prev_bad_rate:
            increasing = False
        elif curr_bad_rate > prev_bad_rate:
            decreasing = False
        prev_bad_rate = curr_bad_rate
        if not (increasing or decreasing):
            return False
    return True


def _find_index_of_diff_flag(bad_rates: List[Dict]) -> int:
    """Find the index of the bad rate with the smallest difference in woe.
    Args:
        bad_rates (List[Dict]): List of bad rates.
    Returns:
        int: Index of the bad rate with the smallest difference in woe."""

    bad_rate_diffs = np.diff([bad_rate["bad_rate"] for bad_rate in bad_rates])
    return np.argmin(bad_rate_diffs)


def _merge_bins_chi(x, y: np.ndarray, bad_rates: List[Dict], bins: List) -> List[Dict]:
    """Merge the bins with the chi-squared statistic.
    Args:
        x (pd.DataFrame): Input data.
        y (np.ndarray): Output data.
        bad_rates (List[Dict]): List of bad rates.
        bins (List): List of bins.
    Returns:
        List[Dict]: List of bad rates."""

    idx = _find_index_of_diff_flag(bad_rates)
    if idx == 0:
        del bins[1]
    elif idx == len(bad_rates) - 2:
        del bins[len(bins) - 2]
    else:
        _extract_bin_by_chi2(bins, idx, x, y)
    bad_rates, _ = _bin_bad_rates(x, y, bins)
    return bad_rates, bins


def _extract_bin_by_chi2(bins, idx, x, y) -> None:
    """Extract the bins with the chi-squared statistic.
    Args:
        bins (List[Dict]): List of bins.
        idx (int): Index of the bad rate with the smallest difference in woe.
        x (pd.DataFrame): Input data.
        y (np.ndarray): Output data.
    Returns:
        None."""

    temp_bins = copy.deepcopy(bins)
    del temp_bins[idx + 1]
    temp_bad_rates, temp_overall_rate = _bin_bad_rates(x, y, temp_bins)
    chi_1 = _chi2(temp_bad_rates, temp_overall_rate)
    del temp_bins

    temp_bins = copy.deepcopy(bins)
    del temp_bins[idx + 2]
    temp_bad_rates, temp_overall_rate = _bin_bad_rates(x, y, temp_bins)
    chi_2 = _chi2(temp_bad_rates, temp_overall_rate)
    if chi_1 < chi_2:
        del bins[idx + 1]
    else:
        del bins[idx + 2]


def _merge_bins_iv(x, y: np.ndarray, bad_rates: List[Dict], bins: List) -> List[Dict]:
    """Merge the bins with the IV statistic.
    Args:
        x (pd.DataFrame): Input data.
        y (np.ndarray): Output data.
        bad_rates (List[Dict]): List of bad rates.
        bins (List): List of bins.
    Returns:
        List[Dict]: List of bad rates."""

    idx = _find_index_of_diff_flag(bad_rates)
    if idx == 0:
        del bins[1]
    elif idx == len(bad_rates) - 2:
        del bins[len(bins) - 2]
    else:
        _extract_bin_by_iv(bins, idx, x, y)
    bad_rates, _ = _bin_bad_rates(x, y, bins)
    return bad_rates, bins


def _extract_bin_by_iv(bins, idx, x, y) -> None:
    """Extract the bins with the IV statistic.
    Args:
        bins (List[Dict]): List of bins.
        idx (int): Index of the bad rate with the smallest difference in woe.
        x (pd.DataFrame): Input data.
        y (np.ndarray): Output data.
    Returns:
        None."""

    temp_bins = copy.deepcopy(bins)
    del temp_bins[idx + 1]
    temp_bad_rates, _ = _bin_bad_rates(x, y, temp_bins)
    iv_1 = sum(_bin["iv"] for _bin in temp_bad_rates)
    del temp_bins

    temp_bins = copy.deepcopy(bins)
    del temp_bins[idx + 2]
    temp_bad_rates, _ = _bin_bad_rates(x, y, temp_bins)
    iv_2 = sum(_bin["iv"] for _bin in temp_bad_rates)
    if iv_1 > iv_2:
        del bins[idx + 1]
    else:
        del bins[idx + 2]


def _merge_bins_min_pct(
    x, y: np.ndarray, bad_rates: Tuple[Dict], bins: List, cat: bool = False
) -> Tuple[List[Dict], List]:
    """Merge the bins with the minimum percentage.
    Args:
        x (pd.DataFrame): Input data.
        y (np.ndarray): Output data.
        bad_rates (Tuple[Dict]): Tuple of bad rates.
        bins (List): List of bins.
        cat (bool, optional): If True, the bins are merged into a categorical bin. Defaults to False.
    Returns:
        Tuple[List[Dict], List]: Tuple of bad rates and bins.
    """

    idx = [br["pct"] for br in bad_rates].index(min(br["pct"] for br in bad_rates))

    if cat:
        if idx == 0:
            bins[idx + 1] += bins[idx]
        elif idx == len(bad_rates) - 1 or bad_rates[idx - 1]["pct"] < bad_rates[idx + 1]["pct"]:
            bins[idx - 1] += bins[idx]
        else:
            bins[idx + 1] += bins[idx]
        bins = bins[:idx] + bins[idx+1:]
    elif idx == 0:
        bins = bins[1:]
    elif idx == len(bad_rates) - 1:
        bins = bins[:-1]
    else:
        bins = bins[:idx] + bins[idx+1:]

    bad_rates, _ = _bin_bad_rates(x, y, bins, cat=cat)
    return bad_rates, bins


def _calc_stats(
        x, y: np.ndarray, idx, all_bad, all_good: int, bins: List, cat: bool = False, refit_fl: bool = False
) -> Dict:
    """Calculate the statistics.
    Args:
        x (pd.DataFrame): Input data.
        y (np.ndarray): Output data.
        idx (int): Index of the bad rate with the smallest difference in woe.
        all_bad (int): Total number of bad rates.
        all_good (int): Total number of good rates.
        bins (List): List of bins.
        cat (bool, optional): If True, the bins are merged into a categorical bin. Defaults to False.
        refit_fl (bool, optional): If True, the bins are merged into a categorical bin. Defaults to False.
    Returns:
        Dict: Statistics."""

    if refit_fl:
        value = bins[idx]
    else:
        value = bins[idx] if cat else [bins[idx], bins[idx + 1]]
    x_not_na = x[~pd.isna(x)]
    y_not_na = y[~pd.isna(x)]
    if cat:
        x_in = x_not_na[np.isin(x_not_na, value)]
    else:
        x_in = x_not_na[[i for i in range(len(x_not_na)) if x_not_na[i] >= np.min(value) and x_not_na[i] < np.max(value)]]
    total = len(x_in)
    bad = y_not_na[np.isin(x_not_na, x_in)].sum()
    pct = np.sum(np.isin(x_not_na, x_in)) / len(x)
    bad_rate = bad / total if total != 0 else 0
    good = total - bad
    woe = np.log((good / all_good) / (bad / all_bad)) if good != 0 and bad != 0 else np.log(
        ((good + 0.5) / all_good) / ((bad + 0.5) / all_bad)
    )
    iv = ((good / all_good) - (bad / all_bad)) * woe
    return {
        "bin": value,
        "total": total,
        "bad": bad,
        "pct": pct,
        "bad_rate": bad_rate,
        "woe": woe,
        "iv": iv,
    }


def _bin_bad_rates(
        x: np.ndarray, y: np.ndarray, bins: List, cat: bool = False, refit_fl: bool = False
) -> Tuple[List[Dict], np.ndarray]:
    """Bin the bad rates.
    Args:
        x (pd.DataFrame): Input data.
        y (np.ndarray): Output data.
        bins (List): List of bins.
        cat (bool, optional): If True, the bins are merged into a categorical bin. Defaults to False.
        refit_fl (bool, optional): If True, the bins are merged into a categorical bin. Defaults to False.
    Returns:
        List[Dict]: List of bad rates."""

    all_bad = y.sum()
    all_good = len(y) - all_bad
    max_idx = len(bins) if cat or refit_fl else len(bins) - 1
    bad_rates = [_calc_stats(x, y, idx, all_bad, all_good, bins, cat, refit_fl) for idx in range(max_idx)]
    if cat:
        bad_rates.sort(key=lambda _x: _x["bad_rate"])
    overall_rate = None
    if not cat:
        bad = sum(bad_rate["bad"] for bad_rate in bad_rates)
        total = sum(bad_rate["total"] for bad_rate in bad_rates)
        overall_rate = bad / total
    return bad_rates, overall_rate


def _calc_max_bins(bins, max_bins: float) -> int:
    """
    Calculate the maximum number of bins.

    Args:
        bins (List): List of bins.
        max_bins (float): Maximum number of bins.

    Returns:
        int: Maximum number of bins.
    """
    return max(2, min(int(len(bins) * max_bins), int(max_bins)))


def prepare_data(data: pd.DataFrame, special_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare the data.

    Args:
        data (pd.DataFrame): Input data.
        special_cols (List[str], optional): List of special columns. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Prepared data.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data should be pandas data frame")
    if special_cols:
        data = data.drop(special_cols, axis=1)
    feature_names = data.columns
    return data, feature_names


def find_cat_features(x: pd.DataFrame, feature_names: List[str], cat_features_threshold: int) -> List[str]:
    """Find the categorical features.
    Args:
        x (pd.DataFrame): Input data.
        feature_names (List[str]): List of feature names.
        cat_features_threshold (int): Threshold of categorical features.
    Returns:
        List[str]: List of categorical features."""

    is_categorical = np.array([
        np.issubdtype(x.dtypes[i], np.object_) or np.issubdtype(x.dtypes[i], np.floating)
        or len(np.unique(x.iloc[:, i].values.astype(str))) < cat_features_threshold
        for i in range(len(feature_names))
    ])

    return list(np.array(feature_names)[is_categorical])


def _cat_binning(
        x, y: np.ndarray,
        min_pct_group: float,
        max_bins: Union[int, float],
        diff_woe_threshold: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Bin the categorical features.
    Args:
        x (pd.DataFrame): Input data.
        y (np.ndarray): Output data.
        min_pct_group (float): Minimum percent group.
        max_bins (Union[int, float]): Maximum number of bins.
        diff_woe_threshold (float): Difference of WOE threshold.
    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Prepared data."""

    missing_bin = None

    try:
        x = x.astype(float)
        data_type = "float"
    except ValueError:
        x = x.astype(str)
        data_type = "object"

    bins = [[_bin] for _bin in np.unique(x[~pd.isna(x)])]

    if max_bins < 1:
        max_bins = _calc_max_bins(bins, max_bins)

    if len(bins) > max_bins:
        bad_rates_dict = dict(
            sorted(
                {bins[i][0]: y[np.isin(x, bins[i])].sum() / len(y[np.isin(x, bins[i])]) for i in
                 range(len(bins))}.items(), key=lambda item: item[1]
            )
        )
        bad_rate_list = [bad_rates_dict[i] for i in bad_rates_dict]
        q_list = [0.0]
        q_list.extend(
            np.nanquantile(
                np.array(bad_rate_list), quantile / max_bins, axis=0
            )
            for quantile in range(1, max_bins)
        )

        q_list.append(1)
        q_list = list(sorted(set(q_list)))

        new_bins = [copy.deepcopy([list(bad_rates_dict.keys())[0]])]
        start = 1
        for i in range(len(q_list) - 1):
            for n in range(start, len(list(bad_rates_dict.keys()))):
                if bad_rate_list[n] >= q_list[i + 1]:
                    break
                elif (bad_rate_list[n] >= q_list[i]) & (
                        bad_rate_list[n] < q_list[i + 1]
                ):
                    try:
                        new_bins[i] += [list(bad_rates_dict.keys())[n]]
                        start += 1
                    except IndexError:
                        new_bins.append([])
                        new_bins[i] += [list(bad_rates_dict.keys())[n]]
                        start += 1

        bad_rates, _ = _bin_bad_rates(x, y, new_bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]
    else:
        bad_rates, _ = _bin_bad_rates(x, y, bins, cat=True)

    if len(y[pd.isna(x)]) > 0:
        if len(bins) < 2:
            bins.append([])
            if data_type == "object":
                bins[1] += ["Missing"]
                x[pd.isna(x)] = "Missing"
            else:
                bins[1] += [-1]
                x[pd.isna(x)] = -1
            bad_rates, _ = _bin_bad_rates(x, y, bins, cat=True)
            missing_bin = "first" if bad_rates[0]["bin"][0] in ["Missing", -1] else "last"
        else:
            na_bad_rate = y[pd.isna(x)].sum() / len(y[pd.isna(x)])
            if abs(na_bad_rate - bad_rates[0]["bad_rate"]) < abs(
                    na_bad_rate - bad_rates[len(bad_rates) - 1]["bad_rate"]
            ):
                missing_bin = "first"
                if data_type == "object":
                    bad_rates[0]["bin"] += ["Missing"]
                    x[pd.isna(x)] = "Missing"
                else:
                    bad_rates[0]["bin"] += [-1]
                    x[pd.isna(x)] = -1
            else:
                missing_bin = "last"
                if data_type == "object":
                    bad_rates[-1]["bin"] += ["Missing"]
                    x[pd.isna(x)] = "Missing"
                else:
                    bad_rates[-1]["bin"] += [-1]
                    x[pd.isna(x)] = -1
            bad_rates, _ = _bin_bad_rates(x, y, bins, cat=True)
            bins = [bad_rate["bin"] for bad_rate in bad_rates]

    if len(bins) <= 2:
        return bad_rates, missing_bin

    while (_check_diff_woe(bad_rates, diff_woe_threshold) is not None) and (
            len(bad_rates) > 2
    ):
        idx = _check_diff_woe(bad_rates, diff_woe_threshold)
        bins[idx + 1] += bins[idx]
        del bins[idx]
        bad_rates, _ = _bin_bad_rates(x, y, bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]

    if len(bins) <= 2:
        return bad_rates, missing_bin

    while (
            min(bad_rate["pct"] for bad_rate in bad_rates) <= min_pct_group
            and len(bins) > 2
    ):
        bad_rates, bins = _merge_bins_min_pct(x, y, bad_rates, bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]

    while len(bad_rates) > max_bins and len(bins) > 2:
        bad_rates, bins = _merge_bins_min_pct(x, y, bad_rates, bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]

    return bad_rates, missing_bin


def cat_processing(
        x: pd.Series,
        y: Union[np.ndarray, pd.Series],
        min_pct_group: float,
        max_bins: Union[int, float],
        diff_woe_threshold: float,
) -> Dict:
    """Cat binning function.
    Args:
        x: feature
        y: target
        min_pct_group: min pct group
        max_bins: max bins
        diff_woe_threshold: diff woe threshold
    Returns:
        Dict: binning result"""

    res_dict, missing_position = _cat_binning(
        x=x.values,
        y=y,
        min_pct_group=min_pct_group,
        max_bins=max_bins,
        diff_woe_threshold=diff_woe_threshold,
    )
    return {
        x.name: res_dict,
        "missing_bin": missing_position,
        "type_feature": "cat",
    }


def _num_binning(
        x, y: np.ndarray,
        min_pct_group: float,
        max_bins: Union[int, float],
        diff_woe_threshold: float,
        merge_type: str,
) -> Dict:
    """Num binning function.
    Args:
        x: feature
        y: target
        min_pct_group: min pct group
        max_bins: max bins
        diff_woe_threshold: diff woe threshold
    Returns:
        Dict: binning result"""

    missing_bin = None

    if max_bins < 1:
        max_bins = _calc_max_bins(list(np.unique(x[~pd.isna(x)])), max_bins)

    bins = [np.NINF]
    if len(np.unique(x[~pd.isna(x)])) > max_bins:
        bins.extend(
            np.nanquantile(x, quantile / max_bins, axis=0)
            for quantile in range(1, max_bins)
        )

        bins = list(np.unique(bins))
        if len(bins) == 2:
            bins.append(np.unique(x[~pd.isna(x)])[1])
    else:
        bins.extend(iter(sorted(np.unique(x[~pd.isna(x)]))))
    bins.append(np.inf)

    bad_rates, _ = _bin_bad_rates(x, y, bins)

    if (pd.isna(bad_rates[0]["bad_rate"])) and (len(bad_rates) > 2):
        del bins[1]
        bad_rates, _ = _bin_bad_rates(x, y, bins)

    if len(y[pd.isna(x)]) > 0:
        na_bad_rate = y[pd.isna(x)].sum() / len(y[pd.isna(x)])
        if len(bad_rates) == 2:
            if na_bad_rate < bad_rates[1]["bad_rate"]:
                x = np.nan_to_num(x, nan=np.amin(x[~pd.isna(x)]) - 1)
                bins = [np.NINF, np.amin(x[~pd.isna(x)])] + bins[1:]
                missing_bin = "first"
            else:
                x = np.nan_to_num(x, nan=np.amax(x[~pd.isna(x)]) + 1)
                bins = bins[:2] + [np.amax(x[~pd.isna(x)]), np.inf]
                missing_bin = "last"
        elif abs(
                na_bad_rate
                - np.mean(
                    [bad_rate["bad_rate"] for bad_rate in bad_rates[: len(bad_rates) // 2]]
                )
        ) < abs(
            na_bad_rate
            - np.mean(
                [bad_rate["bad_rate"] for bad_rate in bad_rates[len(bad_rates) // 2:]]
            )
        ):
            x = np.nan_to_num(x, nan=np.amin(x[~pd.isna(x)]))
            missing_bin = "first"
        else:
            x = np.nan_to_num(x, nan=np.amax(x[~pd.isna(x)]))
            missing_bin = "last"
        bad_rates, _ = _bin_bad_rates(x, y, bins)

    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    while (_mono_flags(bad_rates) is False) and (len(bad_rates) > 2):
        if merge_type == 'chi2':
            bad_rates, bins = _merge_bins_chi(x, y, bad_rates, bins)
        elif merge_type == "iv":
            bad_rates, bins = _merge_bins_iv(x, y, bad_rates, bins)
        else:
            raise NameError(f"Unexpected merge type: {merge_type}")

    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    while (
            min(bad_rate["pct"] for bad_rate in bad_rates) <= min_pct_group
            and len(bad_rates) > 2
    ):
        bad_rates, bins = _merge_bins_min_pct(x, y, bad_rates, bins)

    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    while (_check_diff_woe(bad_rates, diff_woe_threshold) is not None) and (
            len(bad_rates) > 2
    ):
        idx = _check_diff_woe(bad_rates, diff_woe_threshold) + 1
        del bins[idx]
        bad_rates, _ = _bin_bad_rates(x, y, bins)

    return bad_rates, missing_bin


def num_processing(
        x: pd.Series,
        y: Union[np.ndarray, pd.Series],
        min_pct_group: float,
        max_bins: Union[int, float],
        diff_woe_threshold: float,
        merge_type: str,
) -> Dict:
    """Num binning function.
    Args:
        x: feature
        y: target
        min_pct_group: min pct group
        max_bins: max bins
        diff_woe_threshold: diff woe threshold
    Returns:
        Dict: binning result"""

    res_dict, missing_position = _num_binning(
        x=x.values,
        y=y,
        min_pct_group=min_pct_group,
        max_bins=max_bins,
        diff_woe_threshold=diff_woe_threshold,
        merge_type=merge_type,
    )
    return {
        x.name: res_dict,
        "missing_bin": missing_position,
        "type_feature": "num",
    }


def _refit_woe_dict(x, y: np.ndarray, bins: List, type_feature: str, missing_bin: str) -> List[Dict]:
    """Refit woe dict.
    Args:
        x: feature
        y: target
        bins: bins
        type_feature: type of feature
        missing_bin: missing bin
    Returns:
        Dict: binning result"""

    cat = type_feature == "cat"
    if cat:
        na_value = -1.0 if np.issubdtype(x.dtype, np.floating) else "Missing"
        x[np.isnan(x)] = na_value
    elif missing_bin == "first":
        na_value = np.nanmin(x[~np.isnan(x)]) - 1
        x[np.isnan(x)] = na_value
    elif missing_bin == "last":
        na_value = np.nanmax(x[~np.isnan(x)]) + 1
        x[np.isnan(x)] = na_value
    bad_rates, _ = _bin_bad_rates(x, y, bins, cat=cat, refit_fl=True)
    return bad_rates


def refit(
        x, y: np.ndarray,
        bins: List,
        type_feature: str,
        missing_bin: str
) -> Dict:
    """Refit woe dict.

    Args:
        x: feature
        y: target
        bins: bins
        type_feature: type of feature
        missing_bin: missing bin

    Returns:
        Dict: binning result"""
        
    res_dict = _refit_woe_dict(
        x.values, y, bins, type_feature, missing_bin
    )
    return {
        x.name: res_dict,
        "missing_bin": missing_bin,
        "type_feature": type_feature,
    }
