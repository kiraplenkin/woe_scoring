import numpy as np
from scipy.stats import chisquare
from typing import Dict, List, Tuple


def _calc_bins(bad_rates: Dict) -> List:

    bins = []
    for bin in bad_rates:
        bins.append(bin['bin'])
    return bins


def _merge_bins_for_min_pcnt(X: np.ndarray,
                             y: np.ndarray,
                             bad_rates: Dict,
                             bins: List) -> Tuple[Dict, List, float]:

    min_idx = np.argmin([bad_rate['pcnt'] for bad_rate in bad_rates])

    if min_idx == 0:
        bins[min_idx+1] += bins[min_idx]
    elif min_idx == len(bad_rates)-1:
        bins[min_idx-1] += bins[min_idx]
    else:
        if np.abs(bad_rates[min_idx]['bad_rate'] - bad_rates[min_idx-1]['bad_rate']) \
            < np.abs(bad_rates[min_idx]['bad_rate'] - bad_rates[min_idx+1]['bad_rate']):
            bins[min_idx-1] += bins[min_idx]
        else:
            bins[min_idx+1] += bins[min_idx]
    del bins[min_idx]
    bad_rates, bins, overall_rate = bin_bad_rate(X=X,
                                                 y=y,
                                                 bins=bins,
                                                 cat=True)
    return (bad_rates, bins, overall_rate)


def _chi2(bad_rates: Dict,
          overall_rate: float) -> float:

    f_obs = [bin['bad'] for bin in bad_rates]
    f_exp = [bin['total'] * overall_rate for bin in bad_rates]

    chi2 = chisquare(f_obs=f_obs,
                     f_exp=f_exp)[0]
    
    return chi2


def bin_bad_rate(X: np.ndarray,
                 y: np.ndarray,
                 bins: List,
                 cat: bool = False) -> Tuple[Dict, List, float]:
    
    bad_rates = []
    for value in bins:
        stats = {
            'bin': value,
            'total': np.sum(np.isin(X, value)),
            'bad': y[np.isin(X, value)].sum(),
            'pcnt': np.sum(np.isin(X, value)) * 1.0 / len(X),
            'bad_rate': y[np.isin(X, value)].sum() / len(y[np.isin(X, value)])
        }
        bad_rates.append(stats)
        
    if cat:
        bad_rates.sort(key=lambda x: x['bad_rate'])
    
    N, B = 0, 0
    for bin in bad_rates:
        N += bin['total']
        B += bin['bad']
    
    overall_rate = B * 1.0 / N
    
    return (bad_rates, _calc_bins(bad_rates), overall_rate)


def cat_bining(X: np.ndarray,
               y: np.ndarray,
               min_pcnt_group: float,
               n_finale: int,
               temperature: float,
               mask: str):

    bins = list([bin] for bin in np.unique(X))
    bad_rates, bins, _ = bin_bad_rate(X=X,
                                      y=y,
                                      bins=bins,
                                      cat=True)

    # null to worst bin
    for i, bin in enumerate(bins):
        if bin[0] == mask:
            if (i != len(bins) and len(bins) > 2):
                bins[len(bins)-1] += bins[i]
                del bins[i]
                bad_rates, bins, _ = bin_bad_rate(X=X,
                                                  y=y,
                                                  bins=bins,
                                                  cat=True)
    
    # 0 bad_rate group to nearest bin
    while (bad_rates[0]['bad_rate'] == 0 and len(bad_rates) > 2):
        bins[0] += bins[1]
        del bins[1]
        bad_rates, bins, _ = bin_bad_rate(X=X,
                                          y=y,
                                          bins=bins,
                                          cat=True)

    # proportion from best bin and small pcnt on values
    prop = 0
    while (prop < temperature and len(bad_rates) > 2):
        for i in range(len(bad_rates)):
            prop = bad_rates[i]['pcnt'] * bad_rates[i]['bad_rate']
            if prop < temperature:
                if bad_rates[len(bad_rates)-1]['pcnt'] > bad_rates[len(bad_rates)-2]['pcnt']:
                    bins[len(bad_rates)-2] += bins[i]
                else:
                    bins[len(bad_rates)-1] += bins[i]
                del bins[i]
                bad_rates, bins, _ = bin_bad_rate(X=X,
                                                  y=y,
                                                  bins=bins,
                                                  cat=True)
                break

    # min_pcnt_group threshold to nearest bad_rate group
    while min([bad_rate['pcnt'] for bad_rate in bad_rates]) < min_pcnt_group:
        bad_rates, bins, _ = _merge_bins_for_min_pcnt(X=X,
                                                      y=y,
                                                      bad_rates=bad_rates,
                                                      bins=bins)

    # count of bins > max_group
    while len(bad_rates) > n_finale:
        bad_rates, bins, _ = _merge_bins_for_min_pcnt(X=X,
                                                      y=y,
                                                      bad_rates=bad_rates,
                                                      bins=bins)
    
    return bad_rates


def bad_rate_monotone(bad_rates: Dict) -> bool:
    
    if len(bad_rates) <= 2:
        return True
    
    bad_rate_not_monotone = [(bad_rates[i]['bad_rate'] < bad_rates[i+1]['bad_rate']
                             and bad_rates[i]['bad_rate'] < bad_rates[i-1]['bad_rate'])
                             or (bad_rates[i]['bad_rate'] > bad_rates[i+1]['bad_rate']
                             and bad_rates[i]['bad_rate'] > bad_rates[i-1]['bad_rate'])
                             for i in range(1, len(bad_rates)-1)]

    if bad_rate_not_monotone:
        return False
    else:
        return True
