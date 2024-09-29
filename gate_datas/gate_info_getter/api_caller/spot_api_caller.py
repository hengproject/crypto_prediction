from __future__ import print_function
import gate_api
from gate_api.exceptions import ApiException, GateApiException
from gate_datas.configs.configs import gate_api_client as api_client
from gate_datas.gate_info_getter.api_caller import time_getter_now

"""
    api_response = api_instance.list_candlesticks(currency_pair, limit=limit, _from=_from, to=to, interval=interval)
"""
def get_spot_data(currency_pair, _from, limit=100,interval='10s'):

    # Create an instance of the API class
    api_instance = gate_api.SpotApi(api_client)
    currency_pair = currency_pair  # str | Currency pair
    limit = limit  # int | Maximum recent data points to return. `limit` is conflicted with `from` and `to`. If either `from` or `to` is specified, request will be rejected. (optional) (default to 100)
    _from = _from  # int | End time of candlesticks, formatted in Unix timestamp in seconds. Default to current time (optional)

    try:
        # Market candlesticks
        api_response = api_instance.list_candlesticks(currency_pair, limit=limit, to=_from, interval=interval)
        return api_response
    except GateApiException as ex:
        print("Gate api exception, label: %s, message: %s\n" % (ex.label, ex.message))
    except ApiException as e:
        print("Exception when calling SpotApi->list_candlesticks: %s\n" % e)

if __name__ == '__main__':
    ret = get_spot_data('BTC_USDT',_from=time_getter_now())
    print(ret[0])

