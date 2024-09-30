import logging

from gate_datas.configs.configs import crypto_spot_type_list, crypto_future_type_list, db
from gate_datas.gate_info_getter.api_caller.api_caller import get_latest_secure_time, time_getter_now

from gate_datas.gate_info_getter.api_caller.future_api_caller import future_funding_rate, get_future_data
from gate_datas.gate_info_getter.api_caller.spot_api_caller import get_spot_data
from gate_datas.gate_info_getter.entity import SpotPriceInfo, FuturePriceInfo, RateInfo


def start():
    logger = logging.getLogger(__name__)
    data_collection = db['spot_future_data']
    from_collection = db['from_time']
    rate_collection = db['rates']
    while True:
        for spot_crypto_type in crypto_spot_type_list:
            # 10s间隔
            from_time = get_latest_secure_time(spot_crypto_type, "spot")
            if time_getter_now() - from_time > 10:
                spot_ret = get_spot_data(spot_crypto_type, from_time)
                for each in spot_ret:
                    if not each[7]:
                        continue
                    info = SpotPriceInfo(spot_crypto_type, int(each[0]), each[1], each[3], each[4], each[5], each[2],
                                         each[7])
                    info.save(data_collection)
                    from_time = max(from_time, int(each[0]))
                logger.info(f"Spot Price for {spot_crypto_type} saved to {from_time}")
            from_collection.update_one({"pair": spot_crypto_type, "type": "spot"}, {"$set": {"ts": from_time}}, upsert=True)
        for future_crypto_type in crypto_future_type_list:
            from_time = get_latest_secure_time(future_crypto_type, "future")
            if time_getter_now() - from_time > 10:
                future_ret = get_future_data(future_crypto_type, from_time)
                for each in future_ret:
                    if from_time - each.t < 10:
                        continue
                    info = FuturePriceInfo(future_crypto_type, int(each.t), each.sum, each.h, each.l, each.o, each.c,
                                           True)
                    info.save(data_collection)
                    from_time = max(from_time, int(each.t))
            from_collection.update_one({"pair": future_crypto_type, "type": "future"}, {"$set": {"ts": from_time}},
                                   upsert=True)
            logger.info(f"Future Price for {future_crypto_type} saved to {from_time}")

            rate_from_time = get_latest_secure_time(future_crypto_type, "future_rate")
            rate_now = time_getter_now()
            if rate_now - rate_from_time > 3600 * 4:
                limit = min(1000, int((rate_now - rate_from_time) // 3600 * 4))
                api_response = future_funding_rate(future_crypto_type, limit)
                for each in api_response:
                    if each.t < rate_from_time:
                        continue
                    info = RateInfo(int(each.t), each.r,future_crypto_type,"future_rate")
                    info.save(rate_collection)
                    from_time = max(from_time, int(each.t))
                from_collection.update_one({"pair": future_crypto_type, "type": "future_rate"}, {"$set": {"ts": from_time}},
                                       upsert=True)
                logger.info(f"Future Rate for {future_crypto_type} saved to {from_time}")


if __name__ == "__main__":
    start()
