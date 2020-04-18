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
               cut_off_points, 
               group_intervals: Union[int, float, str]) -> str:

    num_bin = len(cut_off_points) + 1

    if x <= cut_off_points[0]:
        return str(group_intervals[0])
    elif x > cut_off_points[-1]:
        return str(group_intervals[num_bin - 1])
    else:
        for i in range(0, num_bin-1):
            if cut_off_points[i] < x <= cut_off_points[i+1]:
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
         total_col: str,
         bad_col: str,
         overall_rate: float) -> float:
    
    df2 = df.copy()
    df2['expected'] = df[total_col].apply(lambda x: x * overall_rate)
    combined = zip(df2['expected'], df[bad_col])
    chi = [(i[0] - i[1])**2 / i[0] for i in combined]
    chi2 = sum(chi)
    return chi2


def chi_merge(df: pd.DataFrame,
              col: str,
              target: str,
              max_interval: int,
              initial_max_group: int = 150,
              special_attribute: List = None,
              min_bin_pcnt: float = 0.0):
    
    col_levels = sorted(list(set(df[col])))
    n_distinct = len(col_levels)

    if n_distinct <= max_interval:
        raise ValueError(f'The value of the original attribute {col} is lower then max_interval = {max_interval}')
    else:
        if len(special_attribute) > 0:
            print(f'{col} case with missing values')  # ??
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df[[col, target]].copy()
        n_distinct = len(list(set(df2[col])))

        if n_distinct > initial_max_group:
            split_x = split_data(df=df2,
                                 col=col,
                                 num_of_split=initial_max_group)
            df2['temp'] = df2[col].map(lambda x: assingn_group(x=x,
                                                               bin=split_x))
        else:
            df['temp'] = df2[col]

        (binbadrate, regroup, overall_rate) = bin_bad_rate(df=df2,
                                                           col='temp',
                                                           target=target,
                                                           grant_rate_indicator=1)
        col_levels = sorted(list(set(df2['temp'])))
        group_intervals = [[i] for i in col_levels]

        split_intervals = max_interval - len(special_attribute)
        while (len(group_intervals) > split_intervals):
            chiq_list = []
            for k in range(len(group_intervals) - 1):
                temp_group = group_intervals[k] + group_intervals[k+1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                chisq = chi2(df=df2b,
                             total_col='total',
                             bad_col='bad',
                             overall_rate=overall_rate)
                chiq_list.append(chisq)
            best_combined = chiq_list.index(min(chiq_list))
            group_intervals[best_combined] = group_intervals[best_combined] + group_intervals[best_combined+1]
            group_intervals.remove(group_intervals[best_combined+1])
        group_intervals = [sorted(i) for i in group_intervals]
        cut_off_points = [max(i) for i in group_intervals[:-1]]

        grouped_values = df2['temp'].apply(lambda x: assign_bin(x=x,
                                                                cut_off_points=cut_off_points,
                                                                group_intervals=group_intervals))
        df2['temp_Bin'] = grouped_values
        (binbadrate, regroup) = bin_bad_rate(df=df2,
                                             col='temp_Bin',
                                             target=target)
        [min_bad_rate, max_bad_rate] = [min(binbadrate.values()), max(binbadrate.values())]

        while min_bad_rate == 0 or max_bad_rate == 1:
            index_for_bad_01 = regroup[regroup['bad_rate'].isin([0, 1])].temp_Bin.tolist()
            bin = index_for_bad_01[0]

            if bin == max(regroup.temp_Bin):
                cut_off_points = cut_off_points[:-1]
                group_intervals = group_intervals[:-1]
            elif bin == min(regroup.temp_Bin):
                cut_off_points = cut_off_points[1:]
                group_intervals = group_intervals[1:]
            else:
                current_index = list(regroup.temp_Bin).index(bin)
                prev_index = list(regroup.temp_Bin)[current_index-1]
                df3 = df2.loc[df2['temp_Bin'].isin([prev_index, bin])]
                (binbadrate, df2b) = bin_bad_rate(df=df3,
                                                  col='temp_Bin',
                                                  target=target)
                chisq1 = chi2(df=df2b,
                              total_col='total',
                              bad_col='bad',
                              overall_rate=overall_rate)
                
                later_index = list(regroup.temp_Bin)[current_index+1]
                df3b = df2.loc[df2['temp_Bin'].isin([later_index, bin])]
                (binbadrate, df2b) = bin_bad_rate(df=df3,
                                                  col='temp_Bin',
                                                  target=target)
                chisq2 = chi2(df=df2b,
                              total_col='total',
                              bad_col='bad',
                              overall_rate=overall_rate)
                
                if chisq1 < chisq2:
                    cut_off_points.remove(cut_off_points[current_index-1])
                    group_intervals[current_index-1] = group_intervals[current_index-1] + group_intervals[current_index]
                    del group_intervals[current_index]
                else:
                    cut_off_points.remove(cut_off_points[current_index])
                    group_intervals[current_index] = group_intervals[current_index] + group_intervals[current_index+1]
                    del group_intervals[current_index+1]
                
            grouped_values = df2['temp'].apply(lambda x: assign_bin(x=x,
                                                                    cut_off_points=cut_off_points,
                                                                    group_intervals=group_intervals))
            df2['temp_Bin'] = grouped_values
            (binbadrate, regroup) = bin_bad_rate(df=df2,
                                                 col='temp_Bin',
                                                 target=target)
            [min_bad_rate, max_bad_rate] = [min(binbadrate.values()), max(binbadrate.values())]

        if min_bin_pcnt > 0:
            grouped_values = df2['temp'].apply(lambda x: assign_bin(x=x,
                                                                    cut_off_points=cut_off_points,
                                                                    group_intervals=group_intervals))
            df2['temp_Bin'] = grouped_values

            value_counts = grouped_values.value_counts().to_frame()
            N = sum(value_counts['temp'])
            value_counts['pcnt'] = value_counts['temp'].apply(lambda x: x * 1.0 / N)
            value_counts = value_counts.sort_index()
            min_pcnt = min(value_counts['pcnt'])

            while min_pcnt < min_bin_pcnt and len(cut_off_points) > 2:
                index_for_min_pcnt = value_counts[value_counts['pcnt'] == min_pcnt].index.tolist()[0]
                if index_for_min_pcnt == max(value_counts.index):
                    cut_off_points = cut_off_points[:-1]
                    group_intervals = group_intervals[:-1]
                elif index_for_min_pcnt == min(value_counts.index):
                    cut_off_points = cut_off_points[1:]
                    group_intervals = group_intervals[1:]
                else:
                    current_index = list(value_counts.index).index(index_for_min_pcnt)
                    prev_index = list(value_counts.index)[current_index-1]
                    df3 = df2.loc[df2['temp_Bin'].isin([prev_index, index_for_min_pcnt])]
                    (binbadrate, df2b) = bin_bad_rate(df=df3,
                                                      col='temp_Bin',
                                                      target=target)
                    chisq1 = chi2(df=df2b,
                                  total_col='total',
                                  bad_col='bad',
                                  overall_rate=overall_rate)
                    
                    later_index = list(value_counts.index)[current_index+1]
                    df3b = df2.loc[df2['temp_Bin'].isin([later_index, index_for_min_pcnt])]
                    (binbadrate, df2b) = bin_bad_rate(df=df3b,
                                                      col='temp_Bin',
                                                      target=target)
                    chisq2 = chi2(df=df2b,
                                  total_col='total',
                                  bad_col='bad',
                                  overall_rate=overall_rate)

                    if chisq1 < chisq2:
                        cut_off_points.remove(cut_off_points[current_index-1])
                        group_intervals[current_index-1] = group_intervals[current_index-1] + group_intervals[current_index]
                        del group_intervals[current_index]
                    else:
                        cut_off_points.remove(cut_off_points[current_index])
                        group_intervals[current_index] = group_intervals[current_index] + group_intervals[current_index+1]
                        del group_intervals[current_index+1]

                grouped_values = df2['temp'].apply(lambda x: assign_bin(x=x,
                                                                        cut_off_points=cut_off_points,
                                                                        group_intervals=group_intervals))
                df2['temp_Bin'] = grouped_values
                value_counts = grouped_values.value_counts().to_frame()
                N = sum(value_counts['temp'])
                value_counts['pcnt'] = value_counts['temp'].apply(lambda x: x * 1.0 / N)
                value_counts = value_counts.sort_index()
                min_pcnt = min(value_counts['pcnt'])
        
        cut_off_points = special_attribute + cut_off_points
        return cut_off_points, group_intervals


def cat_feature_bining(df: pd.DataFrame,
                       var: str,
                       target: str,
                       max_bin: int,
                       verbose: bool) -> None:
    print(f'Preparing {var}')
    regroup = bin_bad_rate(df=df,
                           col=var,
                           target=target,
                           cat=True)[1]
    bad_rate = regroup['bad_rate']
    for i in range(len(bad_rate)):
        if bad_rate[i] == 0:
            if regroup.iloc[i, 0] == 'Missing':
                new_value = str(str(regroup.iloc[i, 0]) + ', ' + str(regroup.iloc[regroup.shape[0]-1, 0]))
                df[var] = df[var].apply(lambda x: x if x != regroup.loc[i, var]
                                                    else regroup.loc[regroup.shape[0]-1, var])
                df[var] = df[var].apply(lambda x: x if x != regroup.loc[regroup.shape[0]-1, var]
                                                    else new_value)
            elif i < len(bad_rate)-1:
                new_value = str(str(regroup.iloc[i, 0]) + ', ' + str(regroup.iloc[i+1, 0]))
                df[var] = df[var].apply(lambda x: x if x != regroup.loc[i, var]
                                                    else regroup.loc[i+1, var])
                df[var] = df[var].apply(lambda x: x if x != regroup.loc[i+1, var]
                                                    else new_value)
            else:
                pass

    while min(bin_bad_rate(df=df,
                           col=var,
                           target=target,
                           cat=True)[1]['pcnt']) < 0.05:
        regroup = bin_bad_rate(df=df,
                               col=var,
                               target=target,
                               cat=True)[1]
        bad_rate = regroup['bad_rate']
        i = 0
        while i < len(bad_rate):
            if regroup['pcnt'][i] < 0.05:
                if i < len(bad_rate)-1:
                    new_value = str(str(regroup.iloc[i, 0]) + ', ' + str(regroup.iloc[i+1, 0]))
                    df[var] = df[var].apply(lambda x: x if x != regroup.loc[i, var]
                                                        else regroup.loc[i+1, var])
                    df[var] = df[var].apply(lambda x: x if x != regroup.loc[i+1, var]
                                                        else new_value)
                    i += 1
                else:
                    new_value = str(str(regroup.iloc[i-1, 0]) + ', ' + str(regroup.iloc[i, 0]))
                    df[var] = df[var].apply(lambda x: x if x != regroup.loc[i, var]
                                                        else regroup.loc[i-1, var])
                    df[var] = df[var].apply(lambda x: x if x != regroup.loc[i-1, var]
                                                        else new_value)
            i += 1

    while bin_bad_rate(df=df,
                       col=var,
                       target=target,
                       cat=True)[1].shape[0] > max_bin:
        regroup = bin_bad_rate(df=df,
                               col=var,
                               target=target,
                               cat=True)[1]
        min_pcnt_index = regroup[regroup['pcnt'] == regroup['pnct'].min()].index.values[0]
        if min_pcnt_index == 0:
            new_value = str(str(regroup.iloc[min_pcnt_index, 0]) + ', ' + str(regroup.iloc[min_pcnt_index+1, 0]))
            df[var] = df[var].apply(lambda x: x if x != regroup.loc[min_pcnt_index, var]
                                                else regroup.loc[min_pcnt_index-1, var])
            df[var] = df[var].apply(lambda x: x if x != regroup.loc[min_pcnt_index-1, var]
                                                else new_value)
        else:
            if regroup.loc[min_pcnt_index-1, 'pcnt'] < regroup.loc[min_pcnt_index+1, 'pcnt']:
                new_value = str(str(regroup.iloc[min_pcnt_index-1, 0]) + ', ' + str(regroup.iloc[min_pcnt_index, 0]))
                df[var] = df[var].apply(lambda x: x if x != regroup.loc[min_pcnt_index, var]
                                                    else regroup.loc[min_pcnt_index-1, var])
                df[var] = df[var].apply(lambda x: x if x != regroup.loc[min_pcnt_index-1, var]
                                                    else new_value)
            else:
                new_value = str(str(regroup.iloc[min_pcnt_index, 0]) + ', ' + str(regroup.iloc[min_pcnt_index+1, 0]))
                df[var] = df[var].apply(lambda x: x if x != regroup.loc[min_pcnt_index, var]
                                                    else regroup.loc[min_pcnt_index+1, var])
                df[var] = df[var].apply(lambda x: x if x != regroup.loc[min_pcnt_index+1, var]
                                                    else new_value)