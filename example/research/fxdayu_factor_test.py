# encoding: utf-8

import numpy as np

from jaqs.data import DataView
from jaqs.data import RemoteDataService
from jaqs.research import SignalDigger
import JAQS.jaqs.util as jutil
import os

data_config = {
    "remote.data.address": "tcp://data.tushare.org:8910",
    "remote.data.username": "18566262672",
    "remote.data.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTI3MDI3NTAyMTIiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg1NjYyNjI2NzIifQ.O_-yR0zYagrLRvPbggnru1Rapk4kiyAzcwYt2a3vlpM"
}

dataview_folder = 'output/prepared/data'
if not (os.path.isdir(dataview_folder)):
    os.makedirs(dataview_folder)

def save_dataview():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()

    props = {'start_date': 20170101, 'end_date': 20171001, 'universe': '000300.SH',
             'fields': 'volume,turnover,float_mv,pb,total_mv',
             'freq': 1}

    dv.init_from_config(props, ds)
    dv.prepare_data()

    # for convenience to check limit reachers
    dv.add_formula('random', 'StdDev(volume, 20)', is_quarterly=False)
    dv.add_formula('momentum', 'Return(close_adj, 20)', is_quarterly=False)
    dv.save_dataview(dataview_folder)


def analyze_signal():
    # --------------------------------------------------------------------------------
    # Step.1 load dataview
    dv = DataView()
    dv.load_dataview(dataview_folder)

    # --------------------------------------------------------------------------------
    # Step.2 calculate mask (to mask those ill data points)
    df_index_member = dv.get_ts('index_member')
    mask_index_member = ~(df_index_member > 0) #定义信号过滤条件-非指数成分

    # 定义可买卖条件——未停牌、未涨跌停
    trade_status = dv.get_ts('trade_status')
    mask_sus = trade_status == u'停牌'

    # 涨停
    dv.add_formula('up_limit', '(open - Delay(close, 1)) / Delay(close, 1) > 0.095', is_quarterly=False)
    # 跌停
    dv.add_formula('down_limit', '(open - Delay(close, 1)) / Delay(close, 1) < -0.095', is_quarterly=False)
    can_enter = np.logical_and(dv.get_ts('up_limit') < 1, ~mask_sus) # 未涨停未停牌
    can_exit = np.logical_and(dv.get_ts('down_limit') < 1, ~mask_sus) # 未跌停未停牌

    # --------------------------------------------------------------------------------
    # Step.3 get signal, benchmark and price data
    dv.add_formula('divert', '- Correlation(vwap_adj, volume, 10)', is_quarterly=False)

    signal = dv.get_ts('divert')
    price = dv.get_ts('close_adj')
    price_bench = dv.data_benchmark

    # Step.4 analyze!
    my_period = 5
    obj = SignalDigger(output_folder='output/test_signal',
                       output_format='pdf')
    obj.process_signal_before_analysis(signal,
                                       price=price,
                                       n_quantiles=5,
                                       mask=mask_index_member,
                                       can_enter=can_enter,
                                       can_exit=can_exit,
                                       period=my_period,
                                       benchmark_price=price_bench,
                                       forward=True,
                                       )
    res = obj.create_full_report()
    signal_masked =  obj.signal_data
    print("signal_masked is")
    print(signal_masked)

    # import cPickle as pickle
    # pickle.dump(res, open('_res.pic', 'w'))


def analyze_event():
    # --------------------------------------------------------------------------------
    # Step.1 load dataview
    dv = DataView()
    dv.load_dataview(dataview_folder)

    # --------------------------------------------------------------------------------
    # Step.2 calculate mask (to mask those ill data points)
    df_index_member = dv.get_ts('index_member')
    mask_index_member = ~(df_index_member > 0)  # 定义信号过滤条件-非指数成分

    # 定义可买卖条件——未停牌、未涨跌停
    trade_status = dv.get_ts('trade_status')
    mask_sus = trade_status == u'停牌'

    # 涨停
    dv.add_formula('up_limit', '(open - Delay(close, 1)) / Delay(close, 1) > 0.095', is_quarterly=False)
    # 跌停
    dv.add_formula('down_limit', '(open - Delay(close, 1)) / Delay(close, 1) < -0.095', is_quarterly=False)
    can_enter = np.logical_and(dv.get_ts('up_limit') < 1, ~mask_sus)  # 未涨停未停牌
    can_exit = np.logical_and(dv.get_ts('down_limit') < 1, ~mask_sus)  # 未跌停未停牌

    # --------------------------------------------------------------------------------
    # Step.3 get signal, benchmark and price data
    dv.add_formula('sig', 'close_adj >= Ts_Max(close_adj, 300)', is_quarterly=False)
    #dv.add_formula('new_high_delay', 'Delay(Ts_Max(new_high, 300), 1)', is_quarterly=False)
    #dv.add_formula('sig', 'new_high && (! new_high_delay)', is_quarterly=False)

    signal = dv.get_ts('sig')
    print(signal)
    # 选股信号 可以直接保存
    price = dv.get_ts('close_adj')
    price_bench = dv.data_benchmark

    # Step.4 analyze!
    obj = SignalDigger(output_folder="output/test_event",
                       output_format='pdf')

    df_all, df_events, df_stats = obj.create_binary_event_report(signal,
                                                                 price,
                                                                 mask_index_member,
                                                                 can_enter,
                                                                 can_exit,
                                                                 price_bench,
                                                                 periods=[5, 20, 40])

    # dic_res 里是当前的信号和持有period天后的持有期收益
    print(df_all, df_events, df_stats)



if __name__ == "__main__":
    save_dataview()
    analyze_signal()
    analyze_event()