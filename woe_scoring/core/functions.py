import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import chisquare


def _chi2(bad_rates: List[Dict], overall_rate: float) -> float:
    f_obs = [_bin["bad"] for _bin in bad_rates]
    f_exp = [_bin["total"] * overall_rate for _bin in bad_rates]

    chi2 = chisquare(f_obs=f_obs, f_exp=f_exp)[0]

    return chi2


def _check_diff_woe(bad_rates: List[Dict], diff_woe_threshold: float) -> Union[None, int]:
    woe_delta: np.ndarray = np.abs(np.diff([bad_rate["woe"] for bad_rate in bad_rates]))
    min_diff_woe = min(sorted(list(set(woe_delta))))
    if min_diff_woe < diff_woe_threshold:
        return list(woe_delta).index(min_diff_woe)
    else:
        return None


def _mono_flags(bad_rates: List[Dict]) -> bool:
    bad_rate_diffs = np.diff([bad_rate["bad_rate"] for bad_rate in bad_rates])
    positive_mono_diff = np.all(bad_rate_diffs > 0)
    negative_mono_diff = np.all(bad_rate_diffs < 0)
    return bool(positive_mono_diff | negative_mono_diff)


def _find_index_of_diff_flag(bad_rates: List[Dict]) -> int:
    bad_rate_diffs = np.diff([bad_rate["bad_rate"] for bad_rate in bad_rates])
    idx = list(bad_rate_diffs > 0).index(pd.Series(bad_rate_diffs > 0).value_counts().sort_values().index.tolist()[0])
    return idx


def _merge_bins_chi(x: np.ndarray, y: np.ndarray, bad_rates: List[Dict], bins: List):
    idx = _find_index_of_diff_flag(bad_rates)
    if idx == 0:
        del bins[1]
    elif idx == len(bad_rates) - 2:
        del bins[len(bins) - 2]
    else:
        temp_bins = copy.deepcopy(bins)
        del temp_bins[idx + 1]
        temp_bad_rates, temp_overall_rate = bin_bad_rate(x, y, temp_bins)
        chi_1 = _chi2(temp_bad_rates, temp_overall_rate)
        del temp_bins

        temp_bins = copy.deepcopy(bins)
        del temp_bins[idx + 2]
        temp_bad_rates, temp_overall_rate = bin_bad_rate(x, y, temp_bins)
        chi_2 = _chi2(temp_bad_rates, temp_overall_rate)
        if chi_1 < chi_2:
            del bins[idx + 1]
        else:
            del bins[idx + 2]
    bad_rates, _ = bin_bad_rate(x, y, bins)
    return bad_rates, bins


def _merge_bins_min_pct(
        x: np.ndarray, y: np.ndarray, bad_rates: List[Dict], bins: List, cat: bool = False
):
    idx = [
        pct for pct in [bad_rates[i]["pct"] for i in range(len(bad_rates))]
    ].index(min([bad_rate["pct"] for bad_rate in bad_rates]))

    if cat:
        if idx == 0:
            bins[idx + 1] += bins[idx]
        elif idx == len(bad_rates) - 1:
            bins[idx - 1] += bins[idx]
        else:
            if bad_rates[idx - 1]["pct"] < bad_rates[idx + 1]["pct"]:
                bins[idx - 1] += bins[idx]
            else:
                bins[idx + 1] += bins[idx]
        del bins[idx]
    else:
        if idx == 0:
            del bins[1]
        elif idx == len(bad_rates) - 1:
            del bins[len(bins) - 2]
        else:
            if bad_rates[idx - 1]["pct"] < bad_rates[idx + 1]["pct"]:
                del bins[idx]
            else:
                del bins[idx + 1]

    bad_rates, _ = bin_bad_rate(x, y, bins, cat=cat)
    if cat:
        bins = [bad_rate["bin"] for bad_rate in bad_rates]
    return bad_rates, bins


def bin_bad_rate(
        x: np.ndarray, y: np.ndarray, bins: List, cat: bool = False
) -> tuple[list[dict[str, Union[Union[list, int, float], Any]]], Union[Optional[float], Any]]:
    bad_rates = []
    if cat:
        max_idx = len(bins)
    else:
        max_idx = len(bins) - 1
    for i in range(max_idx):
        if cat:
            value = bins[i]
        else:
            value = [bins[i], bins[i + 1]]

        x_not_na = x[~pd.isna(x)]
        y_not_na = y[~pd.isna(x)]
        if cat:
            x_in = x_not_na[pd.Series(x_not_na).isin(value)]
        else:
            x_in = x_not_na[
                np.where((x_not_na >= np.min(value)) & (x_not_na < np.max(value)))
            ]
        total = len(x_in)
        all_bad = y[~pd.isna(x)].sum()
        all_good = len(y[~pd.isna(x)]) - all_bad
        bad = y_not_na[np.isin(x_not_na, x_in)].sum()
        pct = np.sum(np.isin(x_not_na, x_in)) * 1.0 / len(x)
        bad_rate = bad / total
        good = total - bad
        good_rate = good / total
        if good != 0 and bad != 0:
            woe = np.log((good / all_good) / (bad / all_bad))
        else:
            woe = np.log((good + 0.5 / all_good) / (bad + 0.5 / all_bad))
        iv = (good_rate - bad_rate) * woe
        stats = {
            "bin": value,
            "total": total,
            "bad": bad,
            "pct": pct,
            "bad_rate": bad_rate,
            "woe": woe,
            "iv": iv,
        }
        bad_rates.append(stats)

    if cat:
        bad_rates.sort(key=lambda _x: _x["bad_rate"])

    overall_rate = None
    if not cat:
        bad = sum([bad_rate["bad"] for bad_rate in bad_rates])
        total = sum([bad_rate["total"] for bad_rate in bad_rates])

        overall_rate = bad * 1.0 / total

    return bad_rates, overall_rate


def cat_binning(
        x: np.ndarray,
        y: np.ndarray,
        min_pct_group: float,
        max_bins: int,
        diff_woe_threshold: float,
) -> tuple[list[dict[str, Union[Union[list, int, float], Any]]], Optional[str]]:
    missing_bin = None

    try:
        x = x.astype(float)
        data_type = "float"
    except ValueError:
        x = x.astype(object)
        data_type = "object"

    bins = list([bin] for bin in np.unique(x[~pd.isna(x)]))

    if len(bins) > max_bins:
        bad_rates_dict = dict(
            sorted(
                {bins[i][0]: y[np.isin(x, bins[i])].sum() / len(y[np.isin(x, bins[i])]) for i in
                 range(len(bins))}.items(), key=lambda item: item[1]
            )
        )
        bad_rate_list = [bad_rates_dict[i] for i in bad_rates_dict]
        q_list = [0.0]
        for quantile in range(1, max_bins):
            q_list.append(
                np.nanquantile(np.array(bad_rate_list), quantile / max_bins, axis=0)
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

        bad_rates, _ = bin_bad_rate(x, y, new_bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]
    else:
        bad_rates, _ = bin_bad_rate(x, y, bins, cat=True)

    if len(y[pd.isna(x)]) > 0:
        if len(bins) < 2:
            bins.append([])
            if data_type == "object":
                bins[1] += ["Missing"]
                x[pd.isna(x)] = "Missing"
            else:
                bins[1] += [-1]
                x[pd.isna(x)] = -1
            bad_rates, _ = bin_bad_rate(x, y, bins, cat=True)
            if bad_rates[0]["bin"][0] == "Missing" or bad_rates[0]["bin"][0] == -1:
                missing_bin = "first"
            else:
                missing_bin = "last"
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
            bad_rates, _ = bin_bad_rate(x, y, bins, cat=True)
            bins = [bad_rate["bin"] for bad_rate in bad_rates]

    if len(bins) == 2:
        return bad_rates, missing_bin

    idx = _check_diff_woe(bad_rates, diff_woe_threshold)
    while idx is not None and len(bins) > 2:
        bins[idx + 1] += bins[idx]
        del bins[idx]
        bad_rates, _ = bin_bad_rate(x, y, bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]
        idx = _check_diff_woe(bad_rates, diff_woe_threshold)

    if len(bins) == 2:
        return bad_rates, missing_bin

    while (
            min([bad_rate["pct"] for bad_rate in bad_rates]) <= min_pct_group
            and len(bins) > 2
    ):
        bad_rates, bins = _merge_bins_min_pct(x, y, bad_rates, bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]

    while len(bad_rates) > max_bins and len(bins) > 2:
        bad_rates, bins = _merge_bins_min_pct(x, y, bad_rates, bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]

    return bad_rates, missing_bin


def num_binning(
        x: np.ndarray,
        y: np.ndarray,
        min_pct_group: float,
        max_bins: int,
        diff_woe_threshold: float,
) -> tuple[list[dict[str, Union[Union[list, int, float], Any]]], Optional[str]]:
    missing_bin = None
    bins = [np.NINF]
    if len(np.unique(x[~pd.isna(x)])) > max_bins:
        for quantile in range(1, max_bins):
            bins.append(np.nanquantile(x, quantile / max_bins, axis=0))
        bins = list(np.unique(bins))
        if len(bins) == 2:
            bins.append(np.unique(x[~pd.isna(x)])[1])
    else:
        for value in sorted(np.unique(x[~pd.isna(x)])):
            bins.append(value)

    bins.append(np.inf)

    bad_rates, _ = bin_bad_rate(x, y, bins)

    if (pd.isna(bad_rates[0]["bad_rate"])) and (len(bad_rates) > 2):
        del bins[1]
        bad_rates, _ = bin_bad_rate(x, y, bins)

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
        bad_rates, _ = bin_bad_rate(x, y, bins)

    if len(bad_rates) == 2:
        return bad_rates, missing_bin

    while (min([bad_rate["pct"] for bad_rate in bad_rates]) <= min_pct_group) and (
            len(bad_rates) > 2
    ):
        bad_rates, bins = _merge_bins_min_pct(x, y, bad_rates, bins)

    if len(bad_rates) == 2:
        return bad_rates, missing_bin

    while (_check_diff_woe(bad_rates, diff_woe_threshold) is not None) and (
            len(bad_rates) > 2
    ):
        idx = _check_diff_woe(bad_rates, diff_woe_threshold) + 1
        del bins[idx]
        bad_rates, _ = bin_bad_rate(x, y, bins)

    if len(bad_rates) == 2:
        return bad_rates, missing_bin

    while (~_mono_flags(bad_rates)) and (len(bad_rates) > 2):
        bad_rates, bins = _merge_bins_chi(x, y, bad_rates, bins)

    return bad_rates, missing_bin


def refit_woe_dict(x: np.ndarray, y: np.ndarray, bins: List, type_feature: str, missing_bin: str) -> List[Dict]:
    cat = True if type_feature == "cat" else False

    if cat:
        try:
            x = x.astype(float)
            x[pd.isna(x)] = -1
        except ValueError:
            x = x.astype(object)
            x[pd.isna(x)] = "Missing"
    else:
        if missing_bin == "first":
            x = np.nan_to_num(x, nan=np.amin(x[~pd.isna(x)]) - 1)
        elif missing_bin == "last":
            x = np.nan_to_num(x, nan=np.amax(x[~pd.isna(x)]) + 1)
        bins = list(set(item for sublist in bins for item in sublist))

    bad_rates, _ = bin_bad_rate(x, y, bins, cat=cat)
    return bad_rates
