import time
from gate_datas.configs.configs import db, default_from_time


def time_getter_now():
    return int(time.time())


# time_shift_sec = 1000 * 10
# to_time = time_getter_now() - time_shift_sec * 9.899
# ret = get_spot_data('SOL_USDT', limit=100, _from=int(to_time))
# print(ret)
# print(len(ret))


# 定义为最新的确保可以作为from使用的timestamp,在(-inf, ts]是可以不用考虑的，只需要考虑ts+min_step就可以了
# 从mongoDB中查询 else 给定 else  [time_getter_now()- time_shift_sec* 9.899]
def get_latest_secure_time(name,type):
    from_time = 0
    # for spot sol
    query = {"name": name, "type": type}
    from_collection = db['from']
    from_query = from_collection.find_one(query)
    if from_query == 1:
        from_time = int(from_query)
    elif default_from_time != 0:
        from_time = default_from_time
    else:
        from_time = int(time_getter_now() - 10000 * 9.899)
    return from_time
