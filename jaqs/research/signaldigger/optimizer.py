# encoding=utf-8
# 参数优化器

from .digger import SignalDigger
from .analysis import analysis

opt_tars = {

}

class Optimizer(object):
    '''
    :param dataview: 包含了计算公式所需要的所有数据的jaqs.data.DataView对象
    :param formula: str 需要优化的公式：如'(open - Delay(close, l1)) / Delay(close, l2)'
    :param params: dict 需要优化的参数范围：如{"l1"：range(1,10,1),"l2":range(1,10,1)}
    :param name: str (N) 信号的名称
    :param in_sample_range: float (0.75) 定义样本内优化范围(0-1),为１则在全样本上做优化.
    :param price: dataFrame (N) 价格与ret不能同时存在
    :param ret: dataFrame (N) 收益
    :param benchmark_price: dataFrame (N) 基准价格　若不为空收益计算模式为相对benchmark的收益
    :param period: int (5) 选股持有期
    :param n_quantiles: int (5)
    :param mask: 过滤条件 dataFrame (N)
    :param can_enter: dataFrame (N) 是否能进场
    :param can_exit: dataFrame (N) 是否能出场
    :param forward: bool(True) 是否forward return
    :param commission:　float(0.0008) 手续费率
    :param is_event: bool(False) 是否是事件(0/1因子)
    :param is_quarterly: bool(False) 是否是季度因子
    '''
    def __init__(self,
                 dataview,
                 formula,
                 params,
                 name=None,
                 in_sample_range=0.75,
                 price=None,
                 ret=None,
                 benchmark_price=None,
                 period=5,
                 n_quantiles=5,
                 mask=None,
                 can_enter=None,
                 can_exit=None,
                 forward=True,
                 commission=0.0008,
                 is_event=False,
                 is_quarterly=False,
                 ):
        self.dataview = dataview
        self.formula = formula
        self.params = params
        self.name = name if name else formula
        self.in_sanmple_range = in_sample_range
        self.price = price
        self.ret = ret
        if self.price is None and self.ret is None:
            try:
                self.price = dataview.get_ts('close_adj')
            except:
                pass
        self.benchmark_price = benchmark_price
        self.period = period
        self.n_quantiles = n_quantiles
        if is_event:
            self.n_quantiles = 1
        else:
            self.n_quantiles = n_quantiles
        self.mask = mask
        self.can_enter = can_enter
        self.can_exit = can_exit
        self.forward = forward
        self.commission = commission
        self.is_event = is_event
        self.is_quarterly = is_quarterly
        self.signal_digger = SignalDigger(output_format=None)
        self.all_signals_perf = None

    def enumerate_optimizer(self,optimization_target):
        if self.all_signals_perf is None:
            self.enumerate_all_signals_perf()

    def enumerate_all_signals_perf(self):
        from itertools import product

        keys = list(self.params.keys())
        for value in product(*self.params.values()):
            para_dict = dict(zip(keys, value))
            formula = self.formula
            for vars in para_dict.keys():
                formula = formula.replace(vars,para_dict[vars])
            self.dataview.add_formula(self.name+str(para_dict),
                                      formula,
                                      is_quarterly=self.is_quarterly)
            signal = self.dataview.get_ts(self.name+str(para_dict))
            data_end = int(len(signal)*self.in_sanmple_range)-1
            if data_end<0:
                data_end = 0
            signal = signal.iloc[:data_end]
            self.all_signals_perf[self.name+str(para_dict)] = self.cal_perf(signal)

    @staticmethod
    def cal_perf(self,signal):
        signal_data = self.signal_digger.process_signal_before_analysis(
                                           signal,
                                           price=self.price,
                                           ret=self.ret,
                                           n_quantiles=self.n_quantiles,
                                           mask=self.mask,
                                           can_enter=self.can_enter,
                                           can_exit=self.can_exit,
                                           period=self.period,
                                           benchmark_price=self.benchmark_price,
                                           forward=self.forward,
                                           commission=self.commission)

        perf = {}
        if len(signal_data)>0:
            perf = analysis(signal_data,self.is_event,self.period)
        return perf