import gate_api
import pymongo
from gate_datas.configs.config_helper import crypto_type_spot_list_helper,crypto_type_future_list_helper
from gate_datas.configs.passwds import db_uri,db_name

db = pymongo.MongoClient(db_uri)[db_name]

_configuration = gate_api.Configuration(
    host = "https://api.gateio.ws/api/v4"
)
gate_api_client = gate_api.ApiClient(_configuration)

# 何时开始存储数据，若为0，则忽略该选项
default_from_time = 0

# 虚拟货币类型 top市值
crypto_spot_type_list = crypto_type_spot_list_helper
crypto_future_type_list = crypto_type_future_list_helper
