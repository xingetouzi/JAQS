# encoding = utf-8

import numpy as np
import pandas as pd
from . import performance as pfm
from jaqs.trade import common
import scipy.stats as scst


def compute_downside_returns(price,
                             low,
                             can_exit=None,
                             period=5,
                             compound=True):
    """
    Finds the N period downside_returns for each asset provided.

    Parameters
    ----------
    price : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    low : pd.DataFrame
        Low pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    can_exit:bool
        shape like price&low
    period : int
        periods to compute returns on.
    compound : bool


    Returns
    -------
    downside_returns : pd.DataFrame
        downside_returns in indexed by date
    """
    if compound:
        downside_ret = (low.rolling(period).min() - price.shift(period)) / price.shift(period)
    else:
        downside_ret = (low.rolling(period).min() - price.shift(period)) / price.iloc[0]
    if can_exit is not None:
        low_can_exit = low.copy()
        low_can_exit[~can_exit] = np.NaN
        low_can_exit = low_can_exit.fillna(method="bfill")
        if compound:
            downside_ret_can_exit = (low_can_exit.rolling(period).min() - price.shift(period)) / price.shift(period)
        else:
            downside_ret_can_exit = (low_can_exit.rolling(period).min() - price.shift(period)) / price.iloc[0]
        downside_ret[~can_exit] = (downside_ret[downside_ret <= downside_ret_can_exit].fillna(0) + \
                                   downside_ret_can_exit[downside_ret_can_exit < downside_ret].fillna(0))[~can_exit]

    return downside_ret


def compute_upside_returns(price,
                           high,
                           can_exit=None,
                           period=5,
                           compound=True):
    """
    Finds the N period upside_returns for each asset provided.

    Parameters
    ----------
    price : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    high : pd.DataFrame
        High pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    can_exit:bool
        shape like price&low
    period : int
        periods to compute returns on.
    compound : bool


    Returns
    -------
    upside_returns : pd.DataFrame
        upside_returns in indexed by date
    """
    if compound:
        upside_ret = (high.rolling(period).max() - price.shift(period)) / price.shift(period)
    else:
        upside_ret = (high.rolling(period).max() - price.shift(period)) / price.iloc[0]
    if can_exit is not None:
        high_can_exit = high.copy()
        high_can_exit[~can_exit] = np.NaN
        high_can_exit = high_can_exit.fillna(method="bfill")
        if compound:
            upside_ret_can_exit = (high_can_exit.rolling(period).max() - price.shift(period)) / price.shift(period)
        else:
            upside_ret_can_exit = (high_can_exit.rolling(period).max() - price.shift(period)) / price.iloc[0]
        upside_ret[~can_exit] = (upside_ret[upside_ret >= upside_ret_can_exit].fillna(0) + \
                                 upside_ret_can_exit[upside_ret_can_exit > upside_ret].fillna(0))[~can_exit]

    return upside_ret


def cal_rets_stats(rets, period):
    ret_summary_table = pd.DataFrame()
    ratio = (1.0 * common.CALENDAR_CONST.TRADE_DAYS_PER_YEAR / period)
    mean = rets.mean()
    std = rets.std()
    annual_ret, annual_vol = mean * ratio, std * np.sqrt(ratio)
    t_stats, p_values = scst.ttest_1samp(rets, np.zeros(rets.shape[1]), axis=0)
    ret_summary_table['t-stat'] = t_stats
    ret_summary_table['p-value'] = np.round(p_values, 5)
    ret_summary_table["skewness"] = scst.skew(rets, axis=0)
    ret_summary_table["kurtosis"] = scst.kurtosis(rets, axis=0)
    ret_summary_table['Ann. Ret'] = annual_ret
    ret_summary_table['Ann. Vol'] = annual_vol
    ret_summary_table['Ann. IR'] = annual_ret / annual_vol
    ret_summary_table['occurance'] = len(rets)
    return ret_summary_table.T


def ic_stats(signal_data):
    ic = pfm.calc_signal_ic(signal_data)
    ic = ic.dropna()
    ic.index = pd.to_datetime(ic.index, format="%Y%m%d")
    ic_summary_table = pfm.calc_ic_stats_table(ic).T
    return ic_summary_table


def return_stats(signal_data, is_event, period):
    rets = get_rets(signal_data, is_event)
    stats = []
    for ret_type in rets.keys():
        if len(rets[ret_type]) > 0:
            ret_stats = cal_rets_stats(rets[ret_type].values.reshape((-1, 1)), period)
            ret_stats.columns = [ret_type]
            stats.append(ret_stats)
    if len(stats) > 0:
        stats = pd.concat(stats, axis=1)
    return stats


def get_rets(signal_data, is_event):
    rets = dict()
    signal_data = signal_data.copy()
    n_quantiles = signal_data['quantile'].max()

    if is_event:
        rets["long_ret"] = signal_data[signal_data['signal'] == 1]["return"]
        rets['short_ret'] = signal_data[signal_data['signal'] == -1]["return"] * -1
    else:
        rets['long_ret'] = \
            pfm.calc_period_wise_weighted_signal_return(signal_data, weight_method='long_only')
        rets['short_ret'] = \
            pfm.calc_period_wise_weighted_signal_return(signal_data, weight_method='short_only')
    rets['long_short_ret'] = \
        pfm.calc_period_wise_weighted_signal_return(signal_data, weight_method='long_short')
    # quantile return
    if not is_event:
        rets['top_quantile_ret'] = signal_data[signal_data['quantile'] == n_quantiles]["return"]
        rets['bottom_quantile_ret'] = signal_data[signal_data['quantile'] == 1]["return"]
        period_wise_quantile_ret_stats = pfm.calc_quantile_return_mean_std(signal_data, time_series=True)
        rets['tmb_ret'] = pfm.calc_return_diff_mean_std(period_wise_quantile_ret_stats[n_quantiles],
                                                        period_wise_quantile_ret_stats[1])['mean_diff']

    return rets


def analysis(signal_data, is_event, period):
    if is_event:
        return {"ret": return_stats(signal_data, True, period)}
    else:
        return {
            "ic": ic_stats(signal_data),
            "ret": return_stats(signal_data, False, period)
        }
