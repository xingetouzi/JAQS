# encoding=utf-8
# 数据处理

import jaqs.util as jutil
import pandas as pd
import numpy as np
from sklearn import linear_model
from jaqs.data import DataView
from jaqs.data import RemoteDataService

data_config = {
    "remote.data.address": "tcp://data.tushare.org:8910",
    "remote.data.username": "18566262672",
    "remote.data.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTI3MDI3NTAyMTIiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg1NjYyNjI2NzIifQ.O_-yR0zYagrLRvPbggnru1Rapk4kiyAzcwYt2a3vlpM"
}


# 横截面标准化 - 对Dataframe数据
def standardize(factor_df):
    """
    对因子值做z-score标准化
    :param factor_df: 因子值 (pandas.Dataframe类型),index为datetime, colunms为股票代码。
                      形如:
                                  　AAPL	　　　     BA	　　　CMG	　　   DAL	      LULU	　　
                        date
                        2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832
                        2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890
                        2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691
                        2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805
                        2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902
    :return:z-score标准化后的因子值(pandas.Dataframe类型),index为datetime, colunms为股票代码。
    """

    factor_df = jutil.fillinf(factor_df)
    return factor_df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)


# 横截面去极值 - 对Dataframe数据
def winsorize(factor_df):
    """
    对因子值做去极值操作
    :param factor_df: 因子值 (pandas.Dataframe类型),index为datetime, colunms为股票代码。
                      形如:
                                  　AAPL	　　　     BA	　　　CMG	　　   DAL	      LULU	　　
                        date
                        2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832
                        2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890
                        2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691
                        2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805
                        2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902
    :return:去极值后的因子值(pandas.Dataframe类型),index为datetime, colunms为股票代码。
    """

    def winsorize_series(se):
        q = se.quantile([0.025, 0.975])
        se[se < q.iloc[0]] = q.iloc[0]
        se[se > q.iloc[1]] = q.iloc[1]
        return se

    factor_df = jutil.fillinf(factor_df)
    return factor_df.apply(lambda x: winsorize_series(x), axis=1)


# 横截面排序并归一化
def rank_standardize(factor_df, ascending=True):
    """
    输入因子值, 将因子用排序分值重构，并处理到0-1之间(默认为升序——因子越大 排序分值越大(越好)
        :param factor_df: 因子值 (pandas.Dataframe类型),index为datetime, colunms为股票代码。
                      形如:
                                  　AAPL	　　　     BA	　　　CMG	　　   DAL	      LULU	　　
                        date
                        2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832
                        2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890
                        2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691
                        2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805
                        2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902

    :param ascending: 因子值按升序法排序对应还是降序法排序对应。具体根据因子对收益的相关关系而定，为正则应用升序,为负用降序。(bool)
    :return: 排序重构后的因子值。 取值范围在0-1之间
    """
    factor_df = jutil.fillinf(factor_df)
    num = len(factor_df.columns)
    return factor_df.apply(lambda x: x.rank(method="min", ascending=ascending) / num, axis=1)


def _prepare_neutralize_data(pools,
                             start,
                             end,
                             group_field="sw1",
                             dv=None,
                             ds=None):
    if dv is not None:
        if (group_field in dv.fields) and ("float_mv" in dv.fields) \
                and (set(pools) - set(dv.symbol) == 0) \
                and (start >= dv.start_date and end <= dv.end_date):
            if not('LFLO' in dv.fields):
                dv.add_formula('LFLO', "Log(float_mv)", is_quarterly=False)
            return dv
    dv = DataView()
    if ds is None:
        ds = RemoteDataService()
        ds.init_from_config(data_config)
        dv.data_api = ds
    else:
        dv.data_api = ds
    dv.symbol = sorted(pools)
    dv.start_date = start
    dv.end_date = end
    dv.extended_start_date_d = jutil.shift(dv.start_date, n_weeks=-8)  # query more data
    dv.extended_start_date_q = jutil.shift(dv.start_date, n_weeks=-80)
    dv.freq = 1
    dv.fields = [group_field, "float_mv"]
    dv.prepare_data()
    dv.add_formula('LFLO', "Log(float_mv)",is_quarterly=False)
    return dv


# 行业、市值中性化 - 对Dataframe数据
def neutralize(factor_df,
               factorIsMV=False,
               group_field="sw1",
               dv=None,
               ds=None):
    """
    对因子做行业、市值中性化
    :param ds: data_api
    :param dv: dataview
    :param group_field:
    :param factor_df: 因子值 (pandas.Dataframe类型),index为datetime, colunms为股票代码。
                      形如:
                                  　AAPL	　　　     BA	　　　CMG	　　   DAL	      LULU	　　
                        date
                        2016-06-24	0.165260	0.002198	0.085632	-0.078074	0.173832
                        2016-06-27	0.165537	0.003583	0.063299	-0.048674	0.180890
                        2016-06-28	0.135215	0.010403	0.059038	-0.034879	0.111691
                        2016-06-29	0.068774	0.019848	0.058476	-0.049971	0.042805
                        2016-06-30	0.039431	0.012271	0.037432	-0.027272	0.010902
    :param factorIsMV: 待中性化的因子是否是市值类因子(bool)。是则为True,默认为False
    :return: 中性化后的因子值(pandas.Dataframe类型),index为datetime, colunms为股票代码。
    """

    def _deal_industry_class(industry_class):
        industry_class = industry_class.apply(lambda x: x.rank(method="dense", ascending=True), axis=1)
        symbols = industry_class.columns
        X = {}
        for _, se in industry_class.iterrows():
            class_num = se.max()
            frame = pd.DataFrame(0, index=np.arange(class_num) + 1, columns=symbols)
            for symbol in symbols:
                this_class = se[symbol]
                frame.loc[this_class, symbol] = 1
            X[_] = frame
        return X

    # 剔除有过多无效数据的个股
    empty_data = pd.isnull(factor_df).sum()
    pools = empty_data[empty_data < len(factor_df) * 0.1].index  # 保留空值比例低于0.1的股票
    factor_df = factor_df.loc[:, pools]

    # 剔除过多值为空的截面
    factor_df = factor_df.dropna(thresh=len(factor_df.columns) * 0.9)  # 保留空值比例低于0.9的截面
    start = factor_df.index[0]
    end = factor_df.index[-1]
    dv = _prepare_neutralize_data(pools,start,end,group_field,dv=dv,ds=ds)

    # 获取对数流动市值，并去极值、标准化。市值类因子不需进行这一步
    if not factorIsMV:
        x1 = standardize(winsorize(dv.get_ts("LFLO")))

    # 获取行业分类信息
    industry_class = dv.get_ts(group_field)
    X_dict = _deal_industry_class(industry_class)
    result = []
    # 逐个截面进行回归，留残差作为中性化后的因子值
    for i in factor_df.index:
        # 获取行业分类信息
        X = X_dict[i]
        if not factorIsMV:
            nfactors = int(X.index[-1]+1)
            DataAll = pd.concat([X.T, x1.loc[i], factor_df.loc[i]], axis=1)
        else:
            nfactors = int(X.index[-1])
            DataAll = pd.concat([X.T, factor_df.loc[i]], axis=1)
        # 剔除截面中值含空的股票
        DataAll = DataAll.dropna()
        DataAll.columns = list(range(0, nfactors + 1))
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(np.matrix(DataAll.iloc[:, 0:nfactors]), np.transpose(np.matrix(DataAll.iloc[:, nfactors])))
        residuals = np.transpose(np.matrix(DataAll.iloc[:, nfactors])) - regr.predict(
            np.matrix(DataAll.iloc[:, 0:nfactors]))
        residuals = pd.DataFrame(data=residuals, index=np.transpose(np.matrix(DataAll.index.values)))
        residuals.index = DataAll.index.values
        residuals.columns = [i]
        result.append(residuals)

    result = pd.concat(result, axis=1).T

    return result
