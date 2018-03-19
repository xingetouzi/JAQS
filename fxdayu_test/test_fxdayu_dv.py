# encoding: utf-8
from jaqs.data.fxdayu_dv import FxdayuDataView
from jaqs.data import RemoteDataService
from .data_config import data_config


def test_save_dataview(props):
    ds = RemoteDataService()

    ds.init_from_config(data_config)
    dv = FxdayuDataView()

    dv.init_from_config(props, ds)
    dv.prepare_data()
    print(dv.get_ts("DDNBT").head())
    print(dv.fields)
    dv.add_field("CMRA")
    print(dv.get_ts("CMRA").head())
    print(dv.fields)

if __name__ == "__main__":
    start = 20180104
    end = 20180320
    hs300_props = {'start_date': start, 'end_date': end, 'universe': '000300.SH',
                   "prepare_fields":True,
                   'fields': 'DDNBT,close,pe_ttm,ps_ttm,pb,pcf_ocfttm,ebit,roe,roa,price_div_dps',
                   'freq': 1}
    gem_props = {'start_date': start, 'end_date': end, 'universe': '399606.SZ',
                 'fields': 'pe_ttm,ps_ttm,pb,pcf_ocfttm,ebit,roe,roa,price_div_dps',
                 'freq': 1}
    test_save_dataview(hs300_props)
    test_save_dataview(gem_props)