-- 创建集合（MongoDB 的集合在插入数据时自动创建，这里只是模拟创建过程）
db.createCollection("spot_future_data_1min")

-- 创建复合索引
db.spot_future_data_1min.createIndex({"pair": 1, "timestamp": 1}, {unique: true})
