from gate_datas.configs.configs import db

collection = db['test']
def test_find():
    finds = collection.find()
    for doc in finds:
        print(doc)

def test_insert():
    inserts = [
        {"name":"btc","_type":"spot","from":114514},
        {"name":"eth","_type":"spot","from":114515},
        {"name":"btc","_type":"future","from":114516},
    ]
    collection.insert_many(inserts)

def test_delete():
    delete_all_result = collection.delete_many({"name":"btc","_type":"spot"})

def test_upsert():
    # upsert_result = collection.update_many(
    #     {
    #         "$or":[
    #             {"name": "sol"},
    #             {"name":"btc","_type":"future"}
    #         ]
    #     },  # 查询条件
    #     {"$set": {"from": 99999}},  # 更新操作
    #     upsert=True  # 启用 upsert
    # )
    upsert_result = collection.update_many(
        {"name": "sol"},  # 查询条件
        {"$set": {"from": 10083}},  # 更新操作
        upsert=True  # 启用 upsert
    )

def test_find_not():
    finds = collection.find({'name':'not_exist'})
    print(len(list(finds)))

def test_find_out():
    finds = collection.find_one({'name': 'eth', '_type': 'spot', 'from': 114515})
    if finds:
        print("found")


if __name__ == '__main__':
    test_find_out()