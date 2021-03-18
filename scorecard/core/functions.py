import copy
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")


def _merge_bins_for_min_pcnt(
    X: np.ndarray, y: np.ndarray, bad_rates: Dict, bins: List, cat: bool = False
) -> Tuple[Dict, List, float]:

    min_idx = np.argmin([bad_rate["pcnt"] for bad_rate in bad_rates])

    if min_idx == 0:
        bins[min_idx + 1] += bins[min_idx]
    elif min_idx == len(bad_rates) - 1:
        bins[min_idx - 1] += bins[min_idx]
    else:
        if cat:
            if np.abs(
                bad_rates[min_idx]["bad_rate"] - bad_rates[min_idx - 1]["bad_rate"]
            ) < np.abs(
                bad_rates[min_idx]["bad_rate"] - bad_rates[min_idx + 1]["bad_rate"]
            ):
                bins[min_idx - 1] += bins[min_idx]
            else:
                bins[min_idx + 1] += bins[min_idx]
        else:
            temp_bins = copy.deepcopy(bins)
            temp_bins[min_idx] += temp_bins[min_idx - 1]
            del temp_bins[min_idx - 1]
            temp_bad_rates, temp_bins, overall_rate = bin_bad_rate(
                X=X, y=y, bins=temp_bins, cat=False
            )
            chi_1 = _chi2(bad_rates=temp_bad_rates, overall_rate=overall_rate)
            del temp_bins

            temp_bins = copy.deepcopy(bins)
            temp_bins[min_idx] += temp_bins[min_idx + 1]
            del temp_bins[min_idx + 1]
            temp_bad_rates, temp_bins, overall_rate = bin_bad_rate(
                X=X, y=y, bins=temp_bins, cat=False
            )
            chi_2 = _chi2(bad_rates=temp_bad_rates, overall_rate=overall_rate)
            del temp_bins

            if chi_1 < chi_2:
                bins[min_idx - 1] += bins[min_idx]
            else:
                bins[min_idx + 1] += bins[min_idx]

    del bins[min_idx]
    bad_rates, bins, overall_rate = bin_bad_rate(X=X, y=y, bins=bins, cat=True)
    return (bad_rates, bins, overall_rate)


def _chi2(bad_rates: Dict, overall_rate: float) -> float:

    f_obs = [bin["bad"] for bin in bad_rates]
    f_exp = [bin["total"] * overall_rate for bin in bad_rates]

    chi2 = chisquare(f_obs=f_obs, f_exp=f_exp)[0]

    return chi2


# TODO will be update
def _check_diff_woe(bad_rates: Dict) -> int:
    woe_delta = [
        abs(bad_rates[i]["woe"] - bad_rates[i - 1]["woe"])
        for i in range(1, len(bad_rates))
    ]
    for i, delta in enumerate(woe_delta):
        if delta < 0.05:
            return i


def _mono_flags(bad_rates: Dict) -> List:
    bad_rate_not_monotone_flags = [
        (
            bad_rates[i]["bad_rate"] < bad_rates[i + 1]["bad_rate"]
            and bad_rates[i]["bad_rate"] < bad_rates[i - 1]["bad_rate"]
        )
        or (
            bad_rates[i]["bad_rate"] > bad_rates[i + 1]["bad_rate"]
            and bad_rates[i]["bad_rate"] > bad_rates[i - 1]["bad_rate"]
        )
        for i in range(1, len(bad_rates) - 1)
    ]
    return bad_rate_not_monotone_flags


def _merge_bins_chi(
    X: np.ndarray, y: np.ndarray, bad_rates: Dict, bins: List
) -> Tuple[Dict, List]:
    idx = _mono_flags(bad_rates).index(True)
    if idx == 0:
        del bins[1]
    elif idx == len(bad_rates) - 2:
        del bins[len(bins) - 2]
    else:
        temp_bins = copy.deepcopy(bins)
        del temp_bins[idx + 1]
        temp_bad_rates, temp_overall_rate = bin_bad_rate(X, y, temp_bins)
        chi_1 = _chi2(temp_bad_rates, temp_overall_rate)
        del temp_bins

        temp_bins = copy.deepcopy(bins)
        del temp_bins[idx + 2]
        temp_bad_rates, temp_overall_rate = bin_bad_rate(X, y, temp_bins)
        chi_2 = _chi2(temp_bad_rates, temp_overall_rate)
        if chi_1 < chi_2:
            del bins[idx + 1]
        else:
            del bins[idx + 2]
    bad_rates, _ = bin_bad_rate(X, y, bins)
    return bad_rates, bins


def _merge_bins_min_pcnt(
    X: np.ndarray, y: np.ndarray, bad_rates: Dict, bins: List
) -> Tuple[Dict, List]:
    # TODO will be update
    idx = [
        pcnt for pcnt in [bad_rates[i]["pcnt"] for i in range(len(bad_rates))]
    ].index(min([bad_rate["pcnt"] for bad_rate in bad_rates]))
    if idx == 0:
        del bins[1]
    elif idx == len(bad_rates) - 1:
        del bins[len(bins) - 2]
    else:
        if bad_rates[idx - 1]["pcnt"] < bad_rates[idx + 1]["pcnt"]:
            del bins[idx]
        else:
            del bins[idx + 1]
    bad_rates, _ = bin_bad_rate(X, y, bins)
    return bad_rates, bins


def bin_bad_rate(
    X: np.ndarray, y: np.ndarray, bins: List, cat: bool = False
) -> Tuple[Dict, List, float]:

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

        X_not_na = X[~pd.isna(X)]
        y_not_na = y[~pd.isna(X)]
        if cat:
            X_isin = X_not_na[pd.Series(X_not_na).isin(value)]
        else:
            X_isin = X_not_na[
                np.where((X_not_na >= np.min(value)) & (X_not_na < np.max(value)))
            ]
        total = len(X_isin)
        bad = y_not_na[np.isin(X_not_na, X_isin)].sum()
        pcnt = np.sum(np.isin(X_not_na, X_isin)) * 1.0 / len(X)
        bad_rate = bad / total
        good = total - bad
        good_rate = good / total
        if bad_rate == 0:
            woe = np.log(good_rate / 0.000001)
        elif good_rate == 0:
            woe = np.log(0.000001 / bad_rate)
        else:
            woe = np.log(good_rate / bad_rate)
        iv = (good_rate - bad_rate) * woe
        stats = {
            "bin": value,
            "total": total,
            "bad": bad,
            "pcnt": pcnt,
            "bad_rate": bad_rate,
            "woe": woe,
            "iv": iv,
        }
        bad_rates.append(stats)

    if cat:
        bad_rates.sort(key=lambda x: x["bad_rate"])

    total, bad = 0, 0
    for bad_rate_bin in bad_rates:
        total += bad_rate_bin["total"]
        bad += bad_rate_bin["bad"]

    overall_rate = bad * 1.0 / total

    return (bad_rates, overall_rate)


def cat_bining(
    X: np.ndarray,
    y: np.ndarray,
    min_pcnt_group: float,
    max_bins: int,
) -> Dict:
    missing_bin = None
    X = X.astype(object)
    bins = list([bin] for bin in np.unique(X[~pd.isna(X)]))

    bad_rates, _ = bin_bad_rate(X, y, bins, cat=True)
    bins = [bad_rate["bin"] for bad_rate in bad_rates]

    if len(y[pd.isna(X)]) > 0:
        na_bad_rate = y[pd.isna(X)].sum() / len(y[pd.isna(X)])
        if abs(na_bad_rate - bad_rates[0]["bad_rate"]) < abs(
            na_bad_rate - bad_rates[len(bad_rates) - 1]["bad_rate"]
        ):
            missing_bin = "first"
            bad_rates[0]["bin"] += ["Missing"]
        else:
            missing_bin = "last"
            bad_rates[-1]["bin"] += ["Missing"]
        X[pd.isna(X)] = "Missing"
        bad_rates, _ = bin_bad_rate(X, y, bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]

    if len(bins) <= 2:
        return bad_rates, missing_bin

    if len(bins) > max_bins:
        bad_rate_list = [bad_rate["bad_rate"] for bad_rate in bad_rates]
        q_list = [0]
        for quantile in range(1, 10):
            q_list.append(
                np.nanquantile(np.array(bad_rate_list), quantile / 10, axis=0)
            )
        q_list.append(1)

        bins = [copy.deepcopy(bad_rates[0]["bin"])]
        for i in range(len(q_list) - 1):
            for n in range(1, len(bad_rates)):
                if (bad_rates[n]["bad_rate"] >= q_list[i]) & (
                    bad_rates[n]["bad_rate"] < q_list[i + 1]
                ):
                    try:
                        bins[i] += bad_rates[n]["bin"]
                    except IndexError:
                        bins.append([])
                        bins[i] += bad_rates[n]["bin"]

    idx = _check_diff_woe(bad_rates)
    while idx is not None and len(bins) > 2:
        bins[idx + 1] += bins[idx]
        del bins[idx]
        bad_rates, _ = bin_bad_rate(X, y, bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]
        idx = _check_diff_woe(bad_rates)

    while (
        min([bad_rate["pcnt"] for bad_rate in bad_rates]) <= min_pcnt_group
        and len(bins) > 2
    ):
        bad_rates, bins = _merge_bins_min_pcnt(X, y, bad_rates, bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]

    while len(bad_rates) > max_bins and len(bins) > 2:
        bad_rates, bins = _merge_bins_min_pcnt(X, y, bad_rates, bins, cat=True)
        bins = [bad_rate["bin"] for bad_rate in bad_rates]

    # TODO add proportion_to_dab_rate
    # proportion from best bin and small pcnt on values
    # prop = 0
    # while (prop < temperature and len(bad_rates) > 2):
    #     for i in range(len(bad_rates)):
    #         prop = bad_rates[i]['pcnt'] * bad_rates[i]['bad_rate']
    #         if prop < temperature:
    #             if bad_rates[len(bad_rates)-1]['pcnt'] \
    #                 > bad_rates[len(bad_rates)-2]['pcnt']:
    #                 bins[len(bad_rates)-2] += bins[i]
    #             else:
    #                 bins[len(bad_rates)-1] += bins[i]
    #             del bins[i]
    #             bad_rates, bins, _ = bin_bad_rate(X=X,
    #                                               y=y,
    #                                               bins=bins,
    #                                               cat=True)
    #             break

    return bad_rates, missing_bin


def num_bining(
    X: np.ndarray,
    y: np.ndarray,
    min_pcnt_group: float,
    max_bins: int,
) -> Dict:

    missing_bin = None
    bins = [np.NINF]
    if len(np.unique(X)) > max_bins:
        for quantile in range(1, 10):
            bins.append(np.nanquantile(X, quantile / 10, axis=0))
    else:
        for value in sorted(np.unique(X)):
            bins.append(value)
    bins.append(np.inf)

    bad_rates, _ = bin_bad_rate(X, y, bins)

    if len(y[pd.isna(X)]) > 0:
        na_bad_rate = y[pd.isna(X)].sum() / len(y[pd.isna(X)])
        if abs(na_bad_rate - bad_rates[0]["bad_rate"]) < abs(
            na_bad_rate - bad_rates[len(bad_rates) - 1]["bad_rate"]
        ):
            X = np.nan_to_num(X, nan=np.amin(X[~pd.isna(X)]))
            missing_bin = "first"
        else:
            X = np.nan_to_num(X, nan=np.amax(X[~pd.isna(X)]))
            missing_bin = "last"
        bad_rates, _ = bin_bad_rate(X, y, bins)
    if len(bins) <= 2:
        return bad_rates, missing_bin

    while True in _mono_flags(bad_rates) and len(bins) > 2:
        bad_rates, bins = _merge_bins_chi(X, y, bad_rates, bins)

    while _check_diff_woe(bad_rates) is not None and len(bins) > 2:
        idx = _check_diff_woe(bad_rates) + 1
        del bins[idx]
        bad_rates, _ = bin_bad_rate(X, y, bins)

    while (
        min([bad_rate["pcnt"] for bad_rate in bad_rates]) <= min_pcnt_group
        and len(bins) > 2
    ):
        bad_rates, bins = _merge_bins_min_pcnt(X, y, bad_rates, bins)

    return bad_rates, missing_bin
