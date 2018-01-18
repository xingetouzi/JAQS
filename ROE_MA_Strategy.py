from jaqs.data import DataView
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
dataview_folder = 'JAQS_Data/Factor'
dv = DataView()

dv.load_dataview(dataview_folder)
from jaqs.data import DataView
from datetime import datetime
import pandas as pd
import warnings
import alphalens
warnings.filterwarnings("ignore")
dataview_folder = 'JAQS_Data/factor'
dv = DataView()
dv.load_dataview(dataview_folder)
def change_columns_index(signal):
    new_names = {}
    for c in signal.columns:
        if c.endswith('SZ'):
            new_names[c] = c.replace('SZ', 'XSHE')
        elif c.endswith('SH'):
            new_names[c] = c.replace('SH', 'XSHG')
    signal = signal.rename_axis(new_names, axis=1)
    signal.index = pd.Index(map(lambda x: datetime.strptime(str(x),"%Y%m%d") , signal.index))

    return signal
all_factor = ['pb', 'roe', 'price_div_dps', 'ps_ttm','hsigma']
origin_factors = {f: change_columns_index(dv.get_ts(f).loc[20140101:]) for f in all_factor}

# dic={'adad6':origin_factors['ad']/origin_factors['acd6']}
prices = change_columns_index(dv.get_ts('close_adj'))
from jaqs.research.signaldigger import process
#去极值,z_score标准化,加干扰值
PN_disturbed = pd.Panel({name: process.get_disturbed_factor(process.standardize(process.winsorize(frame)))
                         for name, frame in origin_factors.items()})

def cal_monthly_ic(factor_df):
    factor_data = alphalens.utils.get_clean_factor_and_forward_returns(factor_df.stack(), prices, quantiles=5)
    # print(factor_data)
    # factor_data = alphalens.utils.get_clean_factor_and_forward_returns(factor_df.stack(), prices, quantiles=5,
    #                                                                   )
    # print('ppppp')
    # print(factor_data)
    return alphalens.performance.mean_information_coefficient(factor_data, by_time='D')
monthly_ic = {key: cal_monthly_ic(value) for key, value in PN_disturbed.shift(1).iteritems()}

monthly_ic_mean = pd.DataFrame(
    list(map(lambda frame: frame.mean(), monthly_ic.values())),
    monthly_ic.keys()
)
monthly_ic_std = pd.DataFrame(
    list(map(lambda frame: frame.std(), monthly_ic.values())),
    monthly_ic.keys()
)

origin_factors = {f: change_columns_index(dv.get_ts(f).loc[20140101:20161231]) for f in all_factor}

# dic={'adad6':origin_factors['ad']/origin_factors['acd6']}


def change_columns_index(signal):
    new_names = {}
    for c in signal.columns:
        if c.endswith('SZ'):
            new_names[c] = c.replace('SZ', 'XSHE')
        elif c.endswith('SH'):
            new_names[c] = c.replace('SH', 'XSHG')
    signal = signal.rename_axis(new_names, axis=1)
    signal.index = pd.Index(map(lambda x: datetime.strptime(str(x),"%Y%m%d") , signal.index))
    signal.index = pd.Index(map(lambda x: x+timedelta(hours=15), signal.index))
    return signal

print(monthly_ic)

ROE_Data = change_columns_index(dv.get_ts('acd6').shift(1, axis=0))
# print(ROE_Data)
prices = change_columns_index(dv.get_ts('close_adj'))
def get_largest(se, n=20):
    largest_list = []
    largest_list = se.nlargest(n)

    return largest_list
def get_sma(se, n=5):
    largest_list = []
    # print(se.values)
    # print(type(se.values))
    largest_list = se.nsmallest(n)

    return largest_list
# stock_df = get_largest(ROE_Data).dropna(how='all', axis=1)

import numpy as np
import talib as ta
import pandas as pd
import rqalpha
from rqalpha.api import *
#读取文件位置

dict={}
def init(context):
    context.codes={}
    context.stocks = []
    context.PERIOD = 50
    context.P = 90
    context.changeday=0
    context.change=1
    # scheduler.run_weekly(find_pool, tradingday=1)
    scheduler.run_daily(find_pool)
def find_pool(context, bar_dict):

    if context.changeday==0:
        context.changeday=context.now
    elif (context.now <=context.changeday+timedelta(days=context.change)):
        pass
    else:

        # print(context.now >context.changeday+timedelta(days=context.change))
        # print(context.now)
        context.changeday = context.now
        lar = 0
        i = 'roe'

        for a in all_factor:

            if (abs(monthly_ic[a][1][context.now + timedelta(hours=-15)]) > lar):
                lar = abs(monthly_ic[a][1][context.now + timedelta(hours=-15)])
                i = a
                # print(a)
                # print(lar)

        if ((monthly_ic[a][1][context.now - timedelta(hours=15)]) > 0):
            context.codes = get_largest(change_columns_index(dv.get_ts(i).shift(1, axis=0)).loc[context.now]).dropna(how='all')
        else:
            context.codes = get_sma(change_columns_index(dv.get_ts(i).shift(1, axis=0)).loc[context.now]).dropna(how='all')


        # context.codes = get_sma(change_columns_index(dv.get_ts(i).shift(1, axis=0)).loc[context.now],30).dropna(how='all')
        # context.codes=context.codes.loc[context.now]
        # print(context.codes)
        # print(lar)
        # print(i)
        # print('xxxxxxxxx')
        dict[context.now]=str(lar)+i;
        # print(dict)



        stocks = context.codes
        context.stocks = stocks
def handle_bar(context, bar_dict):
    buy(context, bar_dict)
def buy(context, bar_dict):
    pool = context.codes
    lens=0
    code=[]
    low=10000
    # print (pool)
    if pool is not None:
        stocks_len = len(pool)
        for stocks in context.portfolio.positions:
            price = history_bars(stocks, context.PERIOD + 10, '1d', 'close')
            if  (price[-5]>price[-4]>price[-3] > price[-2] > price[-1]):
                order_target_percent(stocks, 0)
            if stocks not in pool:
                order_target_percent(stocks, 0)
        for codes in pool.keys():
            # print(codes)
            try:
                price = history_bars(codes, context.PERIOD+10, '1d', 'close')

                short_avg = ta.SMA(price, 10)
                long_avg = ta.MA(price, timeperiod=20)
                # print('short_avg:',short_avg)
                # print('long_avg:', long_avg)
                # if (short_avg[-1] < long_avg[-1]) and not(short_avg[-2] < long_avg[-2]):
                #     order_target_percent(codes, 0)
                # if (short_avg[-1]> long_avg[-1]) and not(short_avg[-2]> long_avg[-2]):
                #     order_target_percent(codes, 1.0/stocks_len)
                #
                # if price[-1]> buyp:
                #     buyp=price[-1]
                # cur_position = context.portfolio.positions[codes].quantity
                # if (short_avg[-1]> long_avg[-1]) and not(short_avg[-2]> long_avg[-2]):
                #     order_target_percent(codes,  1.0/stocks_len)
                #     buyp=price[-1]
                #
                if price[-2]*0.95>price[-1] and price[-1]<short_avg[-1] and price[-2]>short_avg[-2] :
                    # if not (price[-3] > price[-2] > price[-1]):
                        code.append(codes)
                        lens+=1

                prices = history_bars(codes, context.PERIOD + 2, '1d', 'close')
                upperband, middleband, lowerband = ta.BBANDS(prices, context.PERIOD)
                sigma = (upperband[-1] - prices[-1]) / (2 * prices[-1])

                if prices[-2] <= upperband[-2] and prices[-1] >= upperband[-1] and sigma < 0.005:
                    code.append(codes)
                    lens += 1
                    # print('asdsad')

                # if buyp*98/100>price[-1]:
                #     order_target_percent(codes, 1.0 / stocks_len)
                #     buyp=price[-1]

            except Exception:
                pass
        try:
            for codes in code:
                order_target_percent(codes, 1.0 / lens)
        except Exception:
            pass
config = {
  "base": {
    "start_date": "2014-01-04",
    "end_date": "2016-12-04",
    "accounts": {'stock':1000000},
    "benchmark": "000300.XSHG"
  },
  "extra": {
    "log_level": "error",
  },
  "mod": {
    "sys_analyser": {
      "report_save_path": '.',
      "enabled": True,
      "plot": True
    }
  }
}
config2 = {
  "base": {
    "start_date": "2017-01-04",
    "end_date": "2017-12-4",
    "accounts": {'stock':1000000},
    "benchmark": "000300.XSHG"
  },
  "extra": {
    "log_level": "error",
  },
  "mod": {
    "sys_analyser": {
      "report_save_path": '.',
      "enabled": True,
      "plot": True
    }
  }
}

# if __name__=='main':
#
#     rqalpha.run_func(init=init, handle_bar=handle_bar, config=config)
    # rqalpha.run_func(init=init, handle_bar=handle_bar, config=config2)
rqalpha.run_func(init=init, handle_bar=handle_bar, config=config)
rqalpha.run_func(init=init, handle_bar=handle_bar, config=config2)
# print(dict)
# dict1={}
# dict1['y']= dict
#
# dict={}
# rqalpha.run_func(init=init, handle_bar=handle_bar, config=config2)
# dict1['17']= dict
# frame_data = pd.DataFrame(dict1)
# frame_data.to_excel('17.xlsx')