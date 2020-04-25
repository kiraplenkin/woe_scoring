import numpy as np
from typing import Dict, List, Tuple


def _calc_bins(bad_rates: Dict) -> List:

    bins = []
    for bin in bad_rates:
        bins.append(bin['bin'])
    return bins


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
               y: np.ndarray):

    bins = list([bin] for bin in np.unique(X))
    bad_rates, bins, _ = bin_bad_rate(X=X,
                                      y=y,
                                      bins=bins,
                                      cat=True)
    
    while (bad_rates[0]['bad_rate'] == 0 and len(bad_rates) > 2):
        bins[0] += bins[1]
        del bins[1]
        bad_rates, bins, _ = bin_bad_rate(X=X,
                                          y=y,
                                          bins=bins,
                                          cat=True)
    
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
