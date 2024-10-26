import pymongo
from datetime import datetime
from processors.configs.passwds import db_uri, db_name

def aggregate_data(interval_sec, source_collection, target_collection, interval_label,count_num):

    with pymongo.MongoClient(db_uri) as client:
        print(f"连接到数据库... 处理时间间隔: {interval_label}")
        db = client[db_name]

        # 确保索引存在
        print(f"确保索引存在... {target_collection}")
        db[target_collection].create_index([("pair", pymongo.ASCENDING), ("timestamp", pymongo.ASCENDING)], unique=True)

        collection = db[source_collection]

        print("获取上次处理的时间戳...")
        last_processed_document = db[target_collection].find_one(
            sort=[('timestamp', pymongo.DESCENDING)]
        )

        if last_processed_document:
            last_processed_timestamp = last_processed_document['timestamp']
            print(f"上次处理时间戳：{last_processed_timestamp}")
        else:
            last_processed_timestamp = 0  # 设置初始值
            print("没有上次处理时间戳，使用初始值")

        # 创建聚合管道
        print("创建聚合管道...")
        pipeline = [
            {
                "$match": {
                    "timestamp": {
                        "$gte": max(0, last_processed_timestamp)  # 筛选新的数据
                    }
                }
            },
            {
                "$addFields": {
                    "custom_interval": {
                        "$subtract": [
                            {"$toLong": "$timestamp"},
                            {"$mod": [
                                {"$toLong": "$timestamp"},
                                interval_sec  # 时间间隔的timestamp量，单位为秒
                            ]}
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "custom_interval": "$custom_interval",
                        "pair": "$pair",
                        "type": "$type"
                    },
                    "count": {"$sum": 1},
                    "open": {"$first": "$open"},
                    "high": {"$max": "$high"},
                    "low": {"$min": "$low"},
                    "close": {"$last": "$close"},
                    "vol_as_u": {"$sum": "$vol_as_u"},
                    "is_closed": {"$first": "$is_closed"},
                    "min_timestamp": {"$min": "$timestamp"}  # 找到每组的最小时间戳
                }
            },
            {
                "$match": {
                    "count": {"$eq": count_num}  # 确保每组有相应数量的数据
                }
            },
            {
                "$project": {
                    "pair": "$_id.pair",
                    "timestamp": "$min_timestamp",  # 使用最小的时间戳
                    "interval": {"$literal": interval_label},
                    "type": "$_id.type",
                    "open": 1,
                    "high": 1,
                    "low": 1,
                    "close": 1,
                    "vol_as_u": 1,
                    "is_closed": 1
                }
            },
            {
                "$merge": {
                    "into": target_collection,
                    "on": ["pair", "timestamp"],
                    "whenMatched": "merge",
                    "whenNotMatched": "insert"
                }
            }
        ]

        # 执行聚合管道
        print("执行聚合管道...")
        collection.aggregate(pipeline)
        print("聚合管道执行完成")

# 调用函数处理不同时间间隔的数据聚合
aggregate_data(60, 'spot_future_data', 'spot_future_data_1m', '1m',6)
aggregate_data(300, 'spot_future_data_1m', 'spot_future_data_5m', '5m',5)
aggregate_data(900, 'spot_future_data_5m', 'spot_future_data_15m', '15m',3)
aggregate_data(1800, 'spot_future_data_15m', 'spot_future_data_30m', '30m',2)
aggregate_data(3600, 'spot_future_data_30m', 'spot_future_data_1h', '1h',2)
aggregate_data(14400, 'spot_future_data_1h', 'spot_future_data_4h', '4h',4)
aggregate_data(43200, 'spot_future_data_4h', 'spot_future_data_12h', '12h',3)
aggregate_data(86400, 'spot_future_data_12h', 'spot_future_data_1d', '1d',2)
aggregate_data(604800, 'spot_future_data_1d', 'spot_future_data_1w', '1w',7)
