# encoding=utf-8
from functools import reduce

from . import process
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
import jaqs.util as jutil
from . import performance as pfm
from . import SignalDigger


# 因子间存在较强同质性时，使用施密特正交化方法对因子做正交化处理，用得到的正交化残差作为因子,默认对Admin里加载的所有因子做调整
def orthogonalize(factors_dict=None,
                  standardize_type="rank"):
    """
    # 因子间存在较强同质性时，使用施密特正交化方法对因子做正交化处理，用得到的正交化残差作为因子,默认对Admin里加载的所有因子做调整
    :param factors_dict: 若干因子组成的字典(dict),形式为:
                         {"factor_name_1":factor_1,"factor_name_2":factor_2}
                       　每个因子值格式为一个pd.DataFrame，索引(index)为date,column为asset
    :param standardize_type: 标准化方法，有"rank"（排序标准化）,"z_score"(z-score标准化)两种（"rank"/"z_score"）
    :return: factors_dict（new) 正交化处理后所得的一系列新因子。
    """

    from scipy import linalg
    from functools import partial

    def Schmidt(data):
        return linalg.orth(data)

    def get_vector(date, factor):
        return factor.loc[date]

    if not factors_dict or len(list(factors_dict.keys())) < 2:
        raise ValueError("你需要给定至少２个因子")

    new_factors_dict = {}  # 用于记录正交化后的因子值
    for factor_name in factors_dict.keys():
        new_factors_dict[factor_name] = []
        # 处理非法值
        factors_dict[factor_name] = jutil.fillinf(factors_dict[factor_name])

    factor_name_list = list(factors_dict.keys())
    factor_value_list = list(factors_dict.values())
    # 施密特正交
    for date in factor_value_list[0].index:
        data = list(map(partial(get_vector, date), factor_value_list))
        data = pd.concat(data, axis=1, join="inner")
        if len(data) == 0:
            continue
        data = data.dropna()
        data = pd.DataFrame(Schmidt(data), index=data.index)
        data.columns = factor_name_list
        for factor_name in factor_name_list:
            row = pd.DataFrame(data[factor_name]).T
            row.index = [date, ]
            new_factors_dict[factor_name].append(row)

    # 因子标准化
    for factor_name in factor_name_list:
        factor_value = pd.concat(new_factors_dict[factor_name])
        if standardize_type == "z_score":
            new_factors_dict[factor_name] = process.standardize(factor_value)
        else:
            new_factors_dict[factor_name] = process.rank_standardize(factor_value)

    return new_factors_dict


# 获取因子的ic序列
def get_factors_ic_df(factors_dict,
                      price,
                      benchmark_price=None,
                      period=5,
                      quantiles=5,
                      mask=None,
                      can_enter=None,
                      can_exit=None,
                      commisson=0.0008,
                      forward=True,
                      **kwargs):
    """
    获取多个因子ic值序列矩阵
    :param factors_dict: 若干因子组成的字典(dict),形式为:
                         {"factor_name_1":factor_1,"factor_name_2":factor_2}
    :param pool: 股票池范围（list),如：["000001.SH","600300.SH",......]
    :param start: 起始时间 (int)
    :param end: 结束时间 (int)
    :param period: 指定持有周期(int)
    :param quantiles: 根据因子大小将股票池划分的分位数量(int)
    :param price : 包含了pool中所有股票的价格时间序列(pd.Dataframe)，索引（index)为datetime,columns为各股票代码，与pool对应。
    :param benchmark_price:基准收益，不为空计算相对收益，否则计算绝对收益
    :return: ic_df 多个因子ｉc值序列矩阵
             类型pd.Dataframe,索引（index）为datetime,columns为各因子名称，与factors_dict中的对应。
             如：

            　　　　　　　　　　　BP	　　　CFP	　　　EP	　　ILLIQUIDITY	REVS20	　　　SRMI	　　　VOL20
            date
            2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832	0.214377	0.068445
            2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890	0.202724	0.081748
            2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691	0.122554	0.042489
            2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805	0.053339	0.079592
            2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902	0.077293	-0.050667
    """

    ic_table = []
    sd = SignalDigger(output_format=None)
    # 获取factor_value的时间（index）,将用来生成 factors_ic_df 的对应时间（index）
    times = sorted(
        pd.concat([pd.Series(factors_dict[factor_name].index) for factor_name in factors_dict.keys()]).unique())
    for factor_name in factors_dict.keys():
        factors_dict[factor_name] = jutil.fillinf(factors_dict[factor_name])
        factor_value = factors_dict[factor_name]
        sd.process_signal_before_analysis(factor_value, price,
                                          benchmark_price=benchmark_price,
                                          period=period,
                                          n_quantiles=quantiles,
                                          mask=mask,
                                          can_enter=can_enter,
                                          can_exit=can_exit,
                                          forward=forward,
                                          commission=commisson)
        signal_data = sd.signal_data
        ic = pd.DataFrame(pfm.calc_signal_ic(signal_data))
        ic.columns = [factor_name, ]
        ic_table.append(ic)

    ic_df = pd.concat(ic_table, axis=1).dropna().reindex(times)

    return ic_df


# 根据样本协方差矩阵估算结果求最大化IC_IR下的多因子组合权重
def max_IR_weight(ic_df,
                  holding_period,
                  rollback_period=120,
                  covariance_type="shrink"):
    """
    输入ic_df(ic值序列矩阵),指定持有期和滚动窗口，给出相应的多因子组合权重
    :param ic_df: ic值序列矩阵 （pd.Dataframe），索引（index）为datetime,columns为各因子名称。
             如：

            　　　　　　　　　　　BP	　　　CFP	　　　EP	　　ILLIQUIDITY	REVS20	　　　SRMI	　　　VOL20
            date
            2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832	0.214377	0.068445
            2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890	0.202724	0.081748
            2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691	0.122554	0.042489
            2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805	0.053339	0.079592
            2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902	0.077293	-0.050667

    :param holding_period: 持有周期(int)
    :param rollback_period: 滚动窗口，即计算每一天的因子权重时，使用了之前rollback_period下的IC时间序列来计算IC均值向量和IC协方差矩阵（int)。
    :param covariance_type:"shrink"/"simple" 协防差矩阵估算方式　Ledoit-Wolf压缩估计或简单估计
    :return: ic_weight_df:使用Sample协方差矩阵估算方法得到的因子权重(pd.Dataframe),
             索引（index)为datetime,columns为待合成的因子名称。
    """
    n = rollback_period
    ic_weight_df = pd.DataFrame(index=ic_df.index, columns=ic_df.columns)
    lw = LedoitWolf()
    for dt in ic_df.index:
        ic_dt = ic_df[ic_df.index < dt].tail(n)
        if len(ic_dt) < n:
            continue
        if covariance_type == "shrink":
            try:
                ic_cov_mat = lw.fit(ic_dt.as_matrix()).covariance_
            except:
                ic_cov_mat = np.mat(np.cov(ic_dt.T.as_matrix()).astype(float))
        else:
            ic_cov_mat = np.mat(np.cov(ic_dt.T.as_matrix()).astype(float))
        inv_ic_cov_mat = np.linalg.inv(ic_cov_mat)
        weight = inv_ic_cov_mat * np.mat(ic_dt.mean()).reshape(len(inv_ic_cov_mat), 1)
        weight = np.array(weight.reshape(len(weight), ))[0]
        ic_weight_df.ix[dt] = weight / np.sum(weight)

    return ic_weight_df.shift(holding_period)


def combine_factors(factors_dict=None,
                    standardize_type="rank",
                    weighted_method="equal_weight",
                    max_IR_props=None):
    """
    # 因子间存在较强同质性时，使用施密特正交化方法对因子做正交化处理，用得到的正交化残差作为因子,默认对Admin里加载的所有因子做调整
    :param max_IR_props:
    :param factors_dict: 若干因子组成的字典(dict),形式为:
                         {"factor_name_1":factor_1,"factor_name_2":factor_2}
                       　每个因子值格式为一个pd.DataFrame，索引(index)为date,column为asset
    :param standardize_type: 标准化方法，有"rank"（排序标准化）,"z_score"(z-score标准化),为空则不进行标准化操作
    :param weighted_method 组合方法，有"equal_weight","max_IR".若选择max_IR，则还需配置max_IR_props参数．
    :return: new_factor 合成后所得的新因子。
    """

    def generate_max_IR_props():
        max_IR_props = {
            'dataview': None,
            "data_api": None,
            'price': None,
            'benchmark_price': None,
            'period': 5,
            'mask': None,
            'can_enter': None,
            'can_exit': None,
            'forward': True,
            'commission': 0.0008,
            "covariance_type": "simple",  # 还可以为"shrink"
            "rollback_period": 120
        }
        return max_IR_props

    def standarize_factors(factors,
                           standardize_type=None):
        if isinstance(factors,pd.DataFrame):
            factors_dict = {"factor":factors}
        else:
            factors_dict = factors
        factor_name_list = factors_dict.keys()
        for factor_name in factor_name_list:
            factors_dict[factor_name] = jutil.fillinf(factors_dict[factor_name])
            if standardize_type == "z_score":
                factors_dict[factor_name] = process.standardize(factors_dict[factor_name])
            elif standardize_type == "rank":
                factors_dict[factor_name] = process.rank_standardize(factors_dict[factor_name])
            elif standardize_type is not None:
                raise ValueError("standardize_type 只能为'z_score'/'rank'/None")
        return factors_dict

    def _cal_max_IR_weight():
        props = generate_max_IR_props()
        if not (max_IR_props is None):
            props.update(max_IR_props)
        if props["price"] is None:
            factors_name = factors_dict.keys()
            factor_0 = factors_dict[factors_name[0]]
            pools = list(factor_0.columns)
            start = factor_0.index[0]
            end = factor_0.index[-1]
            dv = process._prepare_data(pools, start, end,
                                       dv=props["dataview"],
                                       ds=props["data_api"])
            props["price"] = dv.get_ts("close_adj")
        ic_df = get_factors_ic_df(factors_dict=factors_dict,
                                  **props)
        return max_IR_weight(ic_df,
                             props['period'],
                             props["rollback_period"],
                             props["covariance_type"])

    def sum_weighted_factors(x, y):
        return x + y

    if not factors_dict or len(list(factors_dict.keys())) < 2:
        raise ValueError("你需要给定至少２个因子")
    factors_dict = standarize_factors(factors_dict, standardize_type)

    if weighted_method == "max_IR":
        weight = _cal_max_IR_weight()
        weighted_factors = {}
        factor_name_list = factors_dict.keys()
        for factor_name in factor_name_list:
            w = pd.DataFrame(data=weight[factor_name], index=factors_dict[factor_name].index)
            w = pd.concat([w for i in factors_dict[factor_name].columns],axis=1)
            w.columns = factors_dict[factor_name].columns
            weighted_factors[factor_name] = factors_dict[factor_name] * w
    elif weighted_method == "equal_weight":
        weighted_factors = factors_dict
    else:
        raise ValueError('weighted_method 只能为equal_weight/max_IR')
    new_factor = reduce(sum_weighted_factors, weighted_factors.values())
    new_factor = standarize_factors(new_factor,standardize_type)["factor"]
    return new_factor

