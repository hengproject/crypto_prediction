import json
import time

from gate_datas.gate_info_getter.api_caller.api_caller import time_getter_now, get_latest_secure_time
from gate_datas.gate_info_getter.api_caller.future_api_caller import get_future_data
from gate_datas.gate_info_getter.api_caller.spot_api_caller import get_spot_data
from gate_datas.gate_info_getter.entity import SpotPriceInfo, FuturePriceInfo


def frequency_test():
    for i in range(202):
        get_future_data('btc_usdt', 100, _from=time_getter_now())
        if i > 199:
            print(i)



def spot_test():
    res = []

    data = get_spot_data('btc_usdt', 100, _from=get_latest_secure_time())
    print(data[0])
    _s = time.time()
    for each in data:
        res.append(SpotPriceInfo('btc_usdt',*each))
    print(f"res time = {time.time() - _s}")
    return res

def future_test():
    res = []

    data = get_spot_data('btc_usdt', 100, _from=get_latest_secure_time())
    print(data[0])
    _s = time.time()
    for each in data:
        res.append(FuturePriceInfo('btc_usdt',*each))
    print(f"res time = {time.time() - _s}")
    return res


if __name__ == '__main__':
    s = time.time()
    res = future_test()
    d =time.time() - s
    print(d)
    print(res[0])
    print(res[1])
    json_s = "\""+str(res[1])+ "\""
    # res2 = FuturePriceInfo.from_dict()
    jl = json.loads(json_s)

