import pandas as pd
import numpy as np
from typing import Dict, Union, List


def bin_bad_rate(df: pd.DataFrame,
                 col: str,
                 target: str,
                 grant_rate_indicator: int = 0,
                 cat: bool = False):
    
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})

    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})

    pcnt = df.groupby([col])[col].count() * 1.0 / df.shape[0]
    pcnt = pd.DataFrame({'pcnt': pcnt})

    regroup = total.merge(
        bad, left_index=True, right_index=True, how='left'
    ).merge(
        pcnt, left_index=True, right_index=True, how='left'
    )  # try .reset_index(level=0)
    regroup.reset_index(level=0, inplace=True)

    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)

    if cat:
        regroup.sort_values(by=['bad_rate'], inplace=True)
        regroup.reset_index(level=0, inplace=True)

    dicts = dict(zip(regroup[col], regroup['bad_rate']))

    if grant_rate_indicator == 0:
        return (dicts, regroup)

    N = sum(regroup['total'])
    B = sum(regroup['bad'])

    overall_rate = B * 1.0 / N
    return (dicts, regroup, overall_rate)


def bad_rate_monotone(df: pd.DataFrame,
                      sort_by_var: str,
                      target: str,
                      special_attribute: List = None,
                      cat: bool = False) -> bool:
    
    df2 = df.loc[~df[sort_by_var].isin(special_attribute)]

    if len(set(df2[sort_by_var])) <= 2:
        return True

    bad_rate = [x for x in bin_bad_rate(df=df2, col=sort_by_var, target=target)[1]['bad_rate']]
    bad_rate_not_monotone = [bad_rate[i] < bad_rate[i+1] and bad_rate[i] < bad_rate[i-1]
                             or bad_rate[i] > bad_rate[i+1] and bad_rate[i] > bad_rate[i-1]
                             for i in range(1, len(bad_rate) - 1)]

    if True in bad_rate_not_monotone:  # try if bad_rate_not_monotone
        return False
    else:
        return True


def calc_WOE(df: pd.DataFrame,
             col: str,
             target: str,
             cat: bool = False) -> Dict:
    
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})

    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})

    regroup = total.merge(
        bad, left_index=True, right_index=True, how='left'
    )  # try .reset_index(level=0) 
    regroup.reset_index(level=0, inplace=True)

    N = sum(regroup['total'])
    B = sum(regroup['bad'])

    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)

    try:
        regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    except:
        regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt * 1.0 / 0.000001), axis=1)

    WOE_dict = regroup[[col, 'WOE']].set_index(col).to_dict(orient='index')

    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']

    if cat:
        WOE_dict = {k: v for k, v in sorted(WOE_dict.items(), key=lambda x: x[1], reverse=True)}

    try:
        IV = regroup.apply(lambda x: (x.good_pcnt - x.bad_pcnt) * np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    except:
        IV = pd.Series(data=10e-9)
    IV = sum(IV)

    return {'WOE': WOE_dict, 'IV': IV}


def assign_bin(x: Union[int, float],
               cut_of_points, 
               group_intervals: Union[int, float, str]) -> str:

    num_bin = len(cut_of_points) + 1

    if x <= cut_of_points[0]:
        return str(group_intervals[0])
    elif x > cut_of_points[-1]:
        return str(group_intervals[num_bin - 1])
    else:
        for i in range(0, num_bin-1):
            if cut_of_points[i] < x <= cut_of_points[i+1]:
                return str(group_intervals[i+1])


def assingn_group(x,
                  bin):
    
    N = len(bin)

    if x <= min(bin):
        return min(bin)
    elif x > max(bin):
        return 10e10
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1]


def split_data(df: pd.DataFrame,
               col: str,
               num_of_split: int,
               special_attribute: List = None):
    
    df2 = df.copy()  # try df2 = df[col].copy()

    if special_attribute != []:
        df2 = df.loc[~df[col].isin(special_attribute)]

    N = df2.shape[0]
    n = N // num_of_split
    split_point_index = [i * n for i in range(1, num_of_split)]

    raw_values = sorted(list(df2[col]))
    split_point = [raw_values[i] for i in split_point_index]
    split_point = sorted(list(set(split_point)))

    return split_point


def chi2(df: pd.DataFrame,
         total_col: int,
         bad_col: int,
         overall_rate: float) -> float:
    
    df2 = df.copy()
    df2['expected'] = df[total_col].apply(lambda x: x * overall_rate)
    combined = zip(df2['expected'], df[bad_col])
    chi = [(i[0] - i[1])**2 / i[0] for i in combined]
    shi2 = sum(chi)

    return chi2
