from __future__ import print_function
import gate_api
from gate_api.exceptions import ApiException, GateApiException
from gate_datas.configs.configs import gate_api_client as api_client
from gate_datas.gate_info_getter.api_caller import time_getter_now
from gate_datas.gate_info_getter.entity import FuturePriceInfo
"""
        api_response = api_instance.list_futures_candlesticks(settle, contract, _from=_from, to=to, limit=limit,
                                                              interval=interval)
                                                              """


def get_future_data(currency_pair, _from, limit=100, interval='10s'):
    # Create an instance of the API class
    api_instance = gate_api.FuturesApi(api_client)
    limit = limit  # int | Maximum recent data points to return. `limit` is conflicted with `from` and `to`. If either `from` or `to` is specified, request will be rejected. (optional) (default to 100)
    _from = _from  # int | End time of candlesticks, formatted in Unix timestamp in seconds. Default to current time (optional)
    settle = 'usdt'
    ret = []
    try:
        # Get futures candlesticks

        api_response = api_instance.list_futures_candlesticks(settle, currency_pair, _from=_from, limit=limit,
                                                              interval=interval)
        return api_response
    except GateApiException as ex:
        print("Gate api exception, label: %s, message: %s\n" % (ex.label, ex.message))
        return
    except ApiException as e:
        print("Exception when calling FuturesApi->list_futures_candlesticks: %s\n" % e)
        return


def future_funding_rate():
    api_instance = gate_api.FuturesApi(api_client)
    settle = 'usdt'  # str | Settle currency
    contract = 'BTC_USDT'  # str | Futures contract
    limit = 100  # int | Maximum number of records to be returned in a single list (optional) (default to 100)

    try:
        # Funding rate history
        api_response = api_instance.list_futures_funding_rate_history(settle, contract, limit=limit)
        print(api_response)
    except GateApiException as ex:
        print("Gate api exception, label: %s, message: %s\n" % (ex.label, ex.message))
    except ApiException as e:
        print("Exception when calling FuturesApi->list_futures_funding_rate_history: %s\n" % e)


if __name__ == '__main__':
    # pair = 'ZK_USDT'
    # # ts = int(time_getter_now() - 10000 * 9.899)
    # now = time_getter_now()
    # # int(time_getter_now() - 10 * 1000000)
    # ret = get_future_data(pair,_from=time_getter_now()-10*500000)
    # res = []
    # for each in ret:
    #     if now - each.t < 10:
    #         break
    #     res.append(FuturePriceInfo(pair,int(each.t),each.sum,each.h,each.l,each.o,each.c,True))

    future_funding_rate()

