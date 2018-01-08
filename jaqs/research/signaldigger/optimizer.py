# encoding=utf-8
# 参数优化器

from .digger import SignalDigger
from .analysis import analysis
from itertools import product

target_types = {
    'factor': {
        "ic": ["ic"],
        "ret": [
            "long_ret",
            "short_ret",
            "long_short_ret",
            'top_quantile_ret',
            'bottom_quantile_ret',
            "tmb_ret"]
    },
    "event": {
        "ret": [
            "long_ret",
            "short_ret",
            "long_short_ret",
        ]}
}

targets = {
    "ic": ["IC Mean", "IC Std.", "t-stat(IC)", "p-value(IC)", "IC Skew", "IC Kurtosis", "Ann. IR"],
    "ret": ['t-stat', "p-value", "skewness", "kurtosis", "Ann. Ret", "Ann. Vol", "Ann. IR", "occurance"],
}


class Optimizer(object):
    '''
    :param dataview: 包含了计算公式所需要的所有数据的jaqs.data.DataView对象
    :param formula: str 需要优化的公式：如'(open - Delay(close, l1)) / Delay(close, l2)'
    :param params: dict 需要优化的参数范围：如{"l1"：range(1,10,1),"l2":range(1,10,1)}
    :param name: str (N) 信号的名称
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
                 dataview=None,
                 formula="",
                 params=None,
                 name=None,
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
        self._judge_params()
        self.name = name if name else formula
        self.price = price
        self.ret = ret
        if self.price is None and self.ret is None:
            try:
                self.price = dataview.get_ts('close_adj')
            except:
                raise ValueError("One of price / ret must be provided.")
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
        self.all_signals = None
        self.all_signals_perf = None
        self.in_sample_range = None

    # 判断参数命名的规范性
    def _judge_params(self):
        for para in self.params.keys():
            if len(para) < 2 or not para.isupper():
                raise ValueError("参数变量的命名不符合要求!参数名称需全部由大写英文字母组成,且字母数不少于2")

    # 判断target合法性
    def _judge_target(self, target_type, target):
        legal = True
        if self.is_event:
            if target_type in target_types["event"]["ret"]:
                if not (target in targets["ret"]):
                    legal = False
                    print("可选的优化目标仅能从%s选取" % (str(targets["ret"])))
            else:
                legal = False
                print("可选的优化类型仅能从%s选取" % (str(target_types["event"]["ret"])))
        else:
            if target_type in target_types["factor"]["ret"]:
                if not (target in targets["ret"]):
                    legal = False
                    print("可选的优化目标仅能从%s选取" % (str(targets["ret"])))
            elif target_type in target_types["factor"]["ic"]:
                if not (target in targets["ic"]):
                    legal = False
                    print("可选的优化目标仅能从%s选取" % (str(targets["ic"])))
            else:
                print("可选的优化类型仅能从%s选取" % (str(target_types["factor"]["ret"] + target_types["factor"]["ic"])))
        return legal

    def enumerate_optimizer(self,
                            target_type="long_ret",
                            target="Ann. IR",
                            ascending=False,
                            in_sample_range=None):
        '''
        :param target_type: 目标种类
        :param target: 优化目标
        :param ascending: bool(False)升序or降序排列
        :param in_sample_range: [date_start(int),date_end(int)] (N) 定义样本内优化范围.
        :return:
        '''

        if self._judge_target(target_type, target):  # 判断target合法性
            self.get_all_signals_perf(in_sample_range)
            if len(self.all_signals_perf) == 0:
                return []
            if target_type in (target_types["factor"]["ic"]):
                order_index = "ic"
            else:
                order_index = "ret"
            ordered_perf = self.all_signals_perf.values()
            return sorted(ordered_perf,
                          key=lambda x: x[order_index].loc[target, target_type],
                          reverse=(ascending == False))

    def get_all_signals(self):
        if self.all_signals is None:
            self.all_signals = dict()
            keys = list(self.params.keys())
            for value in product(*self.params.values()):
                para_dict = dict(zip(keys, value))
                formula = self.formula
                for vars in para_dict.keys():
                    formula = formula.replace(vars, str(para_dict[vars]))
                self.dataview.add_formula(self.name,
                                          formula,
                                          is_quarterly=self.is_quarterly)
                signal = self.dataview.get_ts(self.name)
                self.dataview.remove_field(self.name)
                self.all_signals[self.name + str(para_dict)] = self.cal_signal(signal)

    def get_all_signals_perf(self, in_sample_range=None):
        self.get_all_signals()
        if self.all_signals_perf is None or \
                (self.in_sample_range != in_sample_range) or \
                (len(set(self.all_signals_perf.keys())-set(self.all_signals.keys())) != 0):
            self.all_signals_perf = dict()
            for sig_name in self.all_signals.keys():
                perf = self.cal_perf(self.all_signals[sig_name], in_sample_range)
                if perf is not None:
                    self.all_signals_perf[sig_name] = perf
                    self.all_signals_perf[sig_name]["signal_name"] = sig_name
            if len(self.all_signals_perf) == 0:
                print("没有计算出可用的信号绩效，请确保至少有一个信号可用.(可尝试增加样本内数据的时间范围以确保有信号发生)")
            self.in_sample_range = in_sample_range

    def cal_signal(self, signal):
        self.signal_digger.process_signal_before_analysis(
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
        return self.signal_digger.signal_data

    def cal_perf(self, signal_data, in_sample_range=None):
        perf = None
        if signal_data is not None:
            if in_sample_range is not None:
                signal_data = signal_data.loc[in_sample_range[0]:in_sample_range[1]]
            if len(signal_data) > 0:
                perf = analysis(signal_data, self.is_event, self.period)
        return perf
