import pymongo
from gate_datas.configs.configs import db_uri, db_name

# 创建MongoDB客户端
client = pymongo.MongoClient(db_uri)


def get_db(name: str = db_name):
    return client[name]


if __name__ == '__main__':
    db = get_db(db_name)
    print(db)
