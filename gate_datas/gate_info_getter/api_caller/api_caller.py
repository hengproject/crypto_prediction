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
def get_latest_secure_time(pair,type):
    from_time = 0
    # for spot sol
    query = {"pair": pair, "type": type}
    from_collection = db['from_time']
    from_query = from_collection.find_one(query)
    farest_time = 0
    if type == "spot":
        farest_time = int(time_getter_now() - 10*9899)
    if from_query:
        from_time = int(from_query['ts'])
        from_time = max(from_time,farest_time)
    elif default_from_time != 0:
        from_time = default_from_time
    else:
        if type == 'future_rate':
            from_time = 0
        else:
            from_time = farest_time
    return from_time

if __name__ == '__main__':
    ret = get_latest_secure_time('SOL_USDT',"future")
    print(ret)