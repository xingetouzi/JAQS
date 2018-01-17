# encoding=utf-8

import pandas as pd
from talib import abstract


# talib函数库,自动剔除为空的日期,用于计算signal
def ta(ta_method='MA',
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
        result.columns = [sec, ]
        results.append(result)

    return pd.concat(results, axis=1)
