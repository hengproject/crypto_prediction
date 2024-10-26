import pymongo
import pandas as pd
import plotly.graph_objects as go
from processors.configs.configs import db_uri,db_name

# Using with statement to ensure the MongoDB client is properly closed
with pymongo.MongoClient(db_uri) as client:
    db = client[db_name]
    collection = db['spot_future_data']
    print(000)
    # 查询数据
    data = collection.find({'pair': 'BTC_USDT','type':'future'}).sort('timestamp', pymongo.DESCENDING).limit(100)
    data_list = list(data)
    print(111)
    # 将数据转换为DataFrame
    df = pd.DataFrame(data_list)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# 创建一个图表
fig = go.Figure()

# 添加数据
fig.add_trace(go.Candlestick(
    x=df['timestamp'],
    open=df['open'].astype(float),
    high=df['high'].astype(float),
    low=df['low'].astype(float),
    close=df['close'].astype(float),
    name='BTC_USDT'
))

# 设置布局
fig.update_layout(
    title='BTC_USDT Price Over Time',
    xaxis_title='Timestamp',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False
)

# 显示图表
fig.show()
