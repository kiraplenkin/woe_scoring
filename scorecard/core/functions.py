import copy
import warnings
import numpy as np
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
) -> Tuple[Dict, float]:

    bad_rates = []
    for i in range(len(bins) - 1):
        value = [bins[i], bins[i + 1]]
        if cat:
            total = np.sum(np.isin(X, value))
            bad = y[np.isin(X, value)].sum()
            pcnt = np.sum(np.isin(X, value)) * 1.0 / len(X)
            bad_rate = y[np.isin(X, value)].sum() / len(y[np.isin(X, value)])
            good = total - bad
            good_pcnt = good / total
            bad_pcnt = bad / total
            if bad_pcnt != 0:
                woe = np.log(good_pcnt / bad_pcnt)
            else:
                woe = np.log(good_pcnt / 0.000001)
            iv = (good_pcnt - bad_pcnt) * woe
            stats = {
                "bin": value,
                "total": total,
                "bad": bad,
                "pcnt": pcnt,
                "bad_rate": bad_rate,
                "woe": woe,
                "iv": iv,
            }
        else:
            X_not_na = X[~np.isnan(X)]
            y_not_na = y[~np.isnan(X)]
            X_isin = X_not_na[
                np.where((X_not_na >= np.min(value)) & (X_not_na < np.max(value)))
            ]
            total = len(X_isin)
            bad = y_not_na[np.isin(X_not_na, X_isin)].sum()
            pcnt = np.sum(np.isin(X_not_na, X_isin)) * 1.0 / len(X)
            bad_rate = y_not_na[np.isin(X_not_na, X_isin)].sum() / len(
                y_not_na[np.isin(X_not_na, X_isin)]
            )
            good = total - bad
            good_pcnt = good / total
            bad_pcnt = bad / total
            if bad_pcnt != 0:
                woe = np.log(good_pcnt / bad_pcnt)
            else:
                woe = np.log(good_pcnt / 0.000001)
            iv = (good_pcnt - bad_pcnt) * woe
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
    n_finale: int,
    temperature: float,
    mask: str,
) -> Dict:

    bins = list([bin] for bin in np.unique(X))
    bad_rates, bins, _ = bin_bad_rate(X=X, y=y, bins=bins, cat=True)

    if len(bins) <= 2:
        return bad_rates
    else:
        # null to worst bin
        for i, bin in enumerate(bins):
            if bin[0] == mask:
                if i != len(bins) - 1 and len(bins) > 2:
                    bins[len(bins) - 1] += bins[i]
                    del bins[i]
                    bad_rates, bins, _ = bin_bad_rate(X=X, y=y, bins=bins, cat=True)

        # 0 bad_rate group to nearest bin
        while bad_rates[0]["bad_rate"] == 0 and len(bad_rates) > 2:
            bins[0] += bins[1]
            del bins[1]
            bad_rates, bins, _ = bin_bad_rate(X=X, y=y, bins=bins, cat=True)

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

        # min_pcnt_group threshold to nearest bad_rate group
        while min([bad_rate["pcnt"] for bad_rate in bad_rates]) < min_pcnt_group:
            bad_rates, bins, _ = _merge_bins_for_min_pcnt(
                X=X, y=y, bad_rates=bad_rates, bins=bins, cat=True
            )

        # count of bins > max_group
        while len(bad_rates) > n_finale:
            bad_rates, bins, _ = _merge_bins_for_min_pcnt(
                X=X, y=y, bad_rates=bad_rates, bins=bins, cat=True
            )

        return bad_rates


def num_bining(
    X: np.ndarray,
    y: np.ndarray,
    min_pcnt_group: float,
    max_bins: int,
) -> Tuple[Dict, str]:

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

    na_bad_rate = y[np.isnan(X)].sum() / len(y[np.isnan(X)])

    if abs(na_bad_rate - bad_rates[0]["bad_rate"]) < abs(
        na_bad_rate - bad_rates[len(bad_rates) - 1]["bad_rate"]
    ):
        X = np.nan_to_num(X, nan=np.amin(X[~np.isnan(X)]))
        missing_bin = "first"
    else:
        X = np.nan_to_num(X, nan=np.amax(X[~np.isnan(X)]))
        missing_bin = "last"

    bad_rates, _ = bin_bad_rate(X, y, bins)

    while True in _mono_flags(bad_rates):
        bad_rates, bins = _merge_bins_chi(X, y, bad_rates, bins)

    while _check_diff_woe(bad_rates) is not None:
        idx = _check_diff_woe(bad_rates) + 1
        del bins[idx]
        bad_rates, _ = bin_bad_rate(X, y, bins)

    while min([bad_rate["pcnt"] for bad_rate in bad_rates]) <= min_pcnt_group:
        bad_rates, bins = _merge_bins_min_pcnt(X, y, bad_rates, bins)

    return bad_rates, missing_bin
