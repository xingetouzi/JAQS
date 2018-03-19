from jaqs.data.dataview import DataView
from jaqs.data import RemoteDataService, DataApi
import pandas as pd


PF = "prepare_fields"


def get_api(data_api):
    if isinstance(data_api, RemoteDataService):
        return data_api.data_api
    elif isinstance(data_api, DataApi):
        return data_api
    else:
        raise TypeError("Type of data_api should be jaqs.data.RemoteDataService or jaqs.data.DataApi")


class FxdayuDataView(DataView):

    def __init__(self):
        super(FxdayuDataView, self).__init__()
        self.factor_fields = set()

    def init_from_config(self, props, data_api):
        _props = props.copy()
        if _props.pop(PF, False):
            self.prepare_fields(data_api)
        super(FxdayuDataView, self).init_from_config(_props, data_api)

    def prepare_fields(self, data_api):
        api = get_api(data_api)

        table, msg = api.query("help.apiParam", "api=factor&ptype=OUT", "param")
        if msg == "0,":
            self.factor_fields = set(table["param"])
            self.custom_daily_fields.extend(self.factor_fields)

    def _get_fields(self, field_type, fields, complement=False, append=False):
        """
        Get list of fields that are in ref_quarterly_fields.

        Parameters
        ----------
        field_type : {'market_daily', 'ref_daily', 'income', 'balance_sheet', 'cash_flow', 'daily', 'quarterly'
        fields : list of str
        complement : bool, optional
            If True, get fields that are NOT in ref_quarterly_fields.

        Returns
        -------
        list

        """
        pool_map = {'market_daily': self.market_daily_fields,
                    'ref_daily': self.reference_daily_fields,
                    'income': self.fin_stat_income,
                    'balance_sheet': self.fin_stat_balance_sheet,
                    'cash_flow': self.fin_stat_cash_flow,
                    'fin_indicator': self.fin_indicator,
                    'group': self.group_fields,
                    'factor': self.factor_fields}
        pool_map['daily'] = set.union(pool_map['market_daily'],
                                      pool_map['ref_daily'],
                                      pool_map['group'],
                                      self.custom_daily_fields)
        pool_map['quarterly'] = set.union(pool_map['income'],
                                          pool_map['balance_sheet'],
                                          pool_map['cash_flow'],
                                          pool_map['fin_indicator'],
                                          self.custom_quarterly_fields)

        pool = pool_map.get(field_type, None)
        if pool is None:
            raise NotImplementedError("field_type = {:s}".format(field_type))

        s = set.intersection(set(pool), set(fields))
        if not s:
            return []

        if complement:
            s = set(fields) - s

        if field_type == 'market_daily' and self.all_price:
            # turnover will not be adjusted
            s.update({'open', 'high', 'close', 'low', 'vwap'})

        if append:
            s.add('symbol')
            if field_type == 'market_daily' or field_type == 'ref_daily':
                s.add('trade_date')
                if field_type == 'market_daily':
                    s.add(self.TRADE_STATUS_FIELD_NAME)
            elif (field_type == 'income'
                  or field_type == 'balance_sheet'
                  or field_type == 'cash_flow'
                  or field_type == 'fin_indicator'):
                s.add(self.ANN_DATE_FIELD_NAME)
                s.add(self.REPORT_DATE_FIELD_NAME)

        l = list(s)
        return l

    def get_factor(self, symbol, start, end, fields):
        if isinstance(symbol, list):
            symbol = ",".join(symbol)
        if isinstance(fields, list):
            fields = ",".join(fields)

        api = get_api(self.data_api)
        data, msg = api.query(
            "factor",
            "symbol={}&start={}&end={}".format(symbol, start, end),
            fields
        )
        if msg == "0,":
            data["symbol"] = data["symbol"].apply(lambda s: s[:6]+".SH" if s.startswith("6") else s[:6]+".SZ")
            data.rename_axis({"datetime": "trade_date"}, 1, inplace=True)
            return data
        else:
            raise Exception(msg)

    def _query_data(self, symbol, fields):
        """
        Query data using different APIs, then store them in dict.
        period, start_date and end_date are fixed.
        Keys of dict are securitites.

        Parameters
        ----------
        symbol : list of str
        fields : list of str

        Returns
        -------
        daily_list : list
        quarterly_list : list

        """
        sep = ','
        symbol_str = sep.join(symbol)

        if self.freq == 1:
            daily_list = []
            quarterly_list = []

            # TODO : use fields = {field: kwargs} to enable params
            fields_market_daily = self._get_fields('market_daily', fields, append=True)
            if fields_market_daily:
                print("NOTE: price adjust method is [{:s} adjust]".format(self.adjust_mode))
                # no adjust prices and other market daily fields
                df_daily, msg1 = self.distributed_query('daily', symbol_str,
                                                        start_date=self.extended_start_date_d, end_date=self.end_date,
                                                        adjust_mode=None, fields=sep.join(fields_market_daily),
                                                        limit=100000)
                # df_daily, msg1 = self.data_api.daily(symbol_str, start_date=self.extended_start_date_d, end_date=self.end_date,
                #                                     adjust_mode=None, fields=sep.join(fields_market_daily))

                if self.all_price:
                    adj_cols = ['open', 'high', 'low', 'close', 'vwap']
                    # adjusted prices
                    # df_daily_adjust, msg11 = self.data_api.daily(symbol_str, start_date=self.extended_start_date_d, end_date=self.end_date,
                    #                                             adjust_mode=self.adjust_mode, fields=','.join(adj_cols))
                    df_daily_adjust, msg1 = self.distributed_query('daily', symbol_str,
                                                                   start_date=self.extended_start_date_d,
                                                                   end_date=self.end_date,
                                                                   adjust_mode=self.adjust_mode,
                                                                   fields=sep.join(fields_market_daily), limit=100000)

                    df_daily = pd.merge(df_daily, df_daily_adjust, how='outer',
                                        on=['symbol', 'trade_date'], suffixes=('', '_adj'))
                daily_list.append(df_daily.loc[:, fields_market_daily])

            fields_ref_daily = self._get_fields('ref_daily', fields, append=True)
            if fields_ref_daily:
                df_ref_daily, msg2 = self.distributed_query('query_lb_dailyindicator', symbol_str,
                                                            start_date=self.extended_start_date_d,
                                                            end_date=self.end_date,
                                                            fields=sep.join(fields_ref_daily), limit=20000)
                daily_list.append(df_ref_daily.loc[:, fields_ref_daily])

            # ----------------------------- query factor -----------------------------
            factor_fields = self._get_fields("factor", fields)
            if factor_fields:
                df_factor = self.get_factor(symbol, self.extended_start_date_d, self.end_date, factor_fields)
                daily_list.append(df_factor)

            # ----------------------------- query factor -----------------------------

            fields_income = self._get_fields('income', fields, append=True)
            if fields_income:
                df_income, msg3 = self.data_api.query_lb_fin_stat('income', symbol_str, self.extended_start_date_q,
                                                                  self.end_date,
                                                                  sep.join(fields_income),
                                                                  drop_dup_cols=['symbol', self.REPORT_DATE_FIELD_NAME])
                quarterly_list.append(df_income.loc[:, fields_income])

            fields_balance = self._get_fields('balance_sheet', fields, append=True)
            if fields_balance:
                df_balance, msg3 = self.data_api.query_lb_fin_stat('balance_sheet', symbol_str,
                                                                   self.extended_start_date_q, self.end_date,
                                                                   sep.join(fields_balance), drop_dup_cols=['symbol',
                                                                                                            self.REPORT_DATE_FIELD_NAME])
                quarterly_list.append(df_balance.loc[:, fields_balance])

            fields_cf = self._get_fields('cash_flow', fields, append=True)
            if fields_cf:
                df_cf, msg3 = self.data_api.query_lb_fin_stat('cash_flow', symbol_str, self.extended_start_date_q,
                                                              self.end_date,
                                                              sep.join(fields_cf),
                                                              drop_dup_cols=['symbol', self.REPORT_DATE_FIELD_NAME])
                quarterly_list.append(df_cf.loc[:, fields_cf])

            fields_fin_ind = self._get_fields('fin_indicator', fields, append=True)
            if fields_fin_ind:
                df_fin_ind, msg4 = self.data_api.query_lb_fin_stat('fin_indicator', symbol_str,
                                                                   self.extended_start_date_q, self.end_date,
                                                                   sep.join(fields_fin_ind), drop_dup_cols=['symbol',
                                                                                                            self.REPORT_DATE_FIELD_NAME])
                quarterly_list.append(df_fin_ind.loc[:, fields_fin_ind])

        else:
            raise NotImplementedError("freq = {}".format(self.freq))
        return daily_list, quarterly_list

