from __future__ import print_function

import time

import gate_api
from gate_api.exceptions import ApiException, GateApiException
from gate_datas.configs.configs import crypto_future_type_list,db
from gate_datas.gate_info_getter.api_caller.future_api_caller import get_future_data


def get_creat_time(contract):
    configuration = gate_api.Configuration(
        host="https://api.gateio.ws/api/v4"
    )

    api_client = gate_api.ApiClient(configuration)
    # Create an instance of the API class
    api_instance = gate_api.FuturesApi(api_client)

    # Get a single contract
    api_response = api_instance.get_futures_contract('usdt', contract)
    return int(api_response.create_time)


def create_time_getter(start):
    not_working = []
    for i,pair in enumerate(crypto_future_type_list):
        if i < start:
            continue
        print(f'{pair} starts')
        create_time = 0
        try:
            create_time = get_creat_time(pair)
        except Exception as e:
            not_working.append(pair)
            print(f'not working {pair}')
        db['create_time'].insert_one({'ts':int(create_time),'type':'future','pair':pair})
        print(f'{pair} done {i}')
    print(not_working)

def get_earliest_from_time():
    for i, pair in enumerate(crypto_future_type_list):
        ret = []
        k = 0
        dbret = db['create_time'].find_one({'type': 'future', 'pair': pair})
        ts = int(dbret['ts'])
        while len(ret) == 0:
            try:
                ret = get_future_data(pair, _from=ts+k*3600)
                k+=1
                print(f'{pair} try {k}')
            except Exception as e:
                print(f'{i} not working {pair}')
                time.sleep(20)
                continue

        from_time = ret[0].t
        db['from_time'].insert_one({'ts': int(from_time), 'type': 'future', 'pair': pair})
        print(f'{i}::::{pair} {k}  from_time: {from_time}')

if __name__ == '__main__':
    get_earliest_from_time()

