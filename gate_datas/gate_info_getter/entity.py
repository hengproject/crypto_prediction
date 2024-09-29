import json


class MongoDBAbstract:
    @classmethod
    def from_dict(cls, data):
        data = json.loads(data)
        return cls.from_dict(data)

    def to_dict(self):
        return self.__dict__

    def save(self, db_collection):
        return db_collection.insert_one(self.to_dict())

    def __str__(self):
        return str(self.__dict__)


"""
每个时间粒度的 K 线数据，从左到右依次为:

- 秒(s)精度的 Unix 时间戳
- 计价货币交易额
- 收盘价
- 最高价
- 最低价
- 开盘价
- 基础货币交易量
- 窗口是否关闭，true 代表此段K线蜡烛图数据结束，false 代表此段K线蜡烛图数据尚未结束
"""


class PriceInfo(MongoDBAbstract):
    pair=""
    type = ""
    timestamp = 0
    interval = ""
    open = 0
    close = 0
    high = 0
    low = 0
    is_closed = False
    # 以U计算的交易额
    vol_as_u = 0
    # 以币本位计算的交易额
    vol_as_c = 0
    fee = 0

    def __init__(self, pair,timestamp, vol_as_u, high, low, open,close, is_closed,interval='10s',fee=0):
        self.pair = pair
        self.timestamp = timestamp
        self.vol_as_u = vol_as_u
        self.close = close
        self.high = high
        self.low = low
        self.is_closed = is_closed
        self.open = open
        self.interval = interval
        self.fee = fee


class SpotPriceInfo(PriceInfo):
    def __init__(self, *args):
        super().__init__(*args)
        self.type = "spot"


class FuturePriceInfo(PriceInfo):
    def __init__(self, *args):
        super().__init__(*args)
        self.type = "future"
