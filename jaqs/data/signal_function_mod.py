# encoding=utf-8

import pandas as pd
import numpy as np
from talib import abstract


# talib函数库,自动剔除为空的日期,用于计算signal
def ta(ta_method='MA',
       ta_column=0,
       Open=None,
       High=None,
       Low=None,
       Close=None,
       *args,
       **kwargs):
    if not isinstance(ta_method, str):
        raise ValueError("格式错误!Ta方法需指定调用的talib函数名(str),检测到传入的为%s,需要传入str" % (type(ta_method)))
    else:
        if not (ta_method in abstract.__dict__):
            raise ValueError("指定的talib函数名有误,检测到传入的为%s,调用的talib库仅支持%s" % (ta_method, str(abstract.__dict__.keys())))

    if not (isinstance(Open, pd.DataFrame)) \
            or not (isinstance(High, pd.DataFrame)) \
            or not (isinstance(Low, pd.DataFrame)) \
            or not (isinstance(Close, pd.DataFrame)):
        raise ValueError("Open,High,Low,Close均需要按顺序被传入,不能为空,需要为pd.Dataframe")

    results = []
    candle_pannel = pd.Panel.from_dict({"open": Open, "high": High, "low": Low, "close": Close})
    for sec in candle_pannel.minor_axis:
        df = candle_pannel.minor_xs(sec).dropna()
        if len(df) == 0:
            continue
        result = pd.DataFrame(getattr(abstract, ta_method)(df, *args, **kwargs))

        if isinstance(ta_column, int):
            if ta_column >= len(result.columns) or ta_column < 0:
                raise ValueError("非法的ta_column,列号不能为负且不得超过%s,输入为%s" % (len(result.columns) - 1, ta_column))
            result = pd.DataFrame(result.iloc[:, ta_column])
        elif isinstance(ta_column, str):
            if not (ta_column in result.columns):
                raise ValueError("非法的ta_column,可选的列名有%s,输入为%s" % (str(result.columns), ta_column))
            result = pd.DataFrame(result.loc[:, ta_column])
        else:
            raise ValueError("ta_column格式有误,错误的类型为%s,请指定合法的列号(int),或列名(str)" % (type(ta_column)))

        result.columns = [sec, ]
        results.append(result)

    return pd.concat(results, axis=1)


# 最大值的坐标
def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax) + 1


# 最小值的坐标
def ts_argmin(df, window=10):
    return df.rolling(window).apply(np.argmin) + 1
