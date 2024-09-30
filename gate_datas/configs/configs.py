import os

import gate_api
import pymongo
from gate_datas.configs.config_helper import crypto_type_spot_list_helper,crypto_type_future_list_helper
from gate_datas.configs.passwds import db_uri,db_name

import logging

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

# 定义日志文件的路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_log_file_path = os.path.join(_current_dir, '..','..', '..','logs', 'starter.log')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('_log_file_path', 'a', 'utf-8')])

# logger = logging.getLogger(__name__)

