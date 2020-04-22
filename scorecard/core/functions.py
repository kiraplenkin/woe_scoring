import numpy as np
from typing import Dict, List, Tuple


def _calc_bins(bad_rates: Dict) -> List:
    bins = []
    for bin in bad_rates:
        bins.append(bin['bin'])
    return bins


def calc_bad_rate(X: np.ndarray,
                  y: np.ndarray,
                  feature_idx: int,
                  bins: List,
                  cat: bool = False) -> Tuple[Dict, List]:
    
    bad_rates = []
    for value in bins:
        stats = {
            'bin': value,
            'bad_rate': y[np.isin(X[:, feature_idx], value)].sum() / len(y[np.isin(X[:, feature_idx], value)])
        }
        bad_rates.append(stats)
        
    if cat:
        bad_rates.sort(key=lambda x: x['bad_rate'])
    
    return bad_rates, _calc_bins(bad_rates)
