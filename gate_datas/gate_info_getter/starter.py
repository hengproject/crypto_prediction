from gate_datas.configs.configs import crypto_spot_type_list, crypto_future_type_list,db
from gate_datas.gate_info_getter.api_caller import time_getter_now, get_latest_secure_time, get_spot_data, \
    get_future_data
from gate_datas.gate_info_getter.entity import SpotPriceInfo, FuturePriceInfo


def get_starter():
    data_collection = db['spot_future_data']
    from_collection = db['from_time']
    while True:
        for spot_crypto_type in crypto_spot_type_list:
            # 10s间隔
            from_time = get_latest_secure_time(spot_crypto_type, "spot")
            if from_time - time_getter_now() > 10:
                spot_ret = get_spot_data(spot_crypto_type, from_time)
                for each in spot_ret:
                    if not each[7]:
                        continue
                    info = SpotPriceInfo(spot_crypto_type, int(each[0]),each[1],each[3],each[4],each[5],each[2],each[7])
                    info.save(data_collection)
                    from_time = max(from_time,int(each[0]))
            from_collection.update({"pair":spot_crypto_type,"type":"spot"},{"$set": {"ts": from_time}},upsert=True)
        for future_crypto_type in crypto_future_type_list:
            from_time = get_latest_secure_time(future_crypto_type, "future")
            if from_time - time_getter_now() > 10:
                future_ret = get_future_data(future_crypto_type,from_time)
                for each in future_ret:
                    if from_time - each.t < 10:
                        continue
                    info = FuturePriceInfo(future_crypto_type,int(each.t),each.sum,each.h,each.l,each.o,each.c,True)
                    info.save(data_collection)
                    from_time = max(from_time, int(each.t))
            from_collection.update({"pair": future_crypto_type, "type": "future"}, {"$set": {"ts": from_time}}, upsert=True)

