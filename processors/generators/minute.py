import pymongo
from datetime import datetime

from processors.configs.passwds import db_uri, db_name


def aggregate_to_1min_data():
    with pymongo.MongoClient(db_uri) as client:
        print("连接到数据库...")
        db = client[db_name]

        # 确保索引存在
        print("确保索引存在...")
        db.spot_future_data_1min.create_index([("pair", pymongo.ASCENDING), ("timestamp", pymongo.ASCENDING)],
                                              unique=True)

        collection = db['spot_future_data']

        print("获取上次处理的时间戳...")
        last_processed_document = db['spot_future_data_1min'].find_one(
            sort=[('timestamp', pymongo.DESCENDING)]
        )

        if last_processed_document:
            last_processed_timestamp = last_processed_document['timestamp']
            print(f"上次处理时间戳：{last_processed_timestamp}")
        else:
            last_processed_timestamp = 0 # 设置初始值
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
                                60  # 每分钟的timestamp量，单位为秒
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
                    "count": {"$eq": 6}  # 确保每组有6条数据
                }
            },
            {
                "$project": {
                    "pair": "$_id.pair",
                    "timestamp": "$min_timestamp",  # 使用最小的时间戳
                    "interval": {"$literal": "1m"},
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
                    "into": "spot_future_data_1min",
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


# 调用函数
aggregate_to_1min_data()
