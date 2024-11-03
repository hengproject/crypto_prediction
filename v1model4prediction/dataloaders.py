import pandas as pd
import numpy as np
import datetime

import torch
from torch.utils.data import Dataset, DataLoader

# ts smaller than gte will be neglected
class TimeSeriesDataset(Dataset):
    def __init__(self, df, input_features, output_features,gte=0):
        self.df = df.copy()
        self.input_features = input_features
        self.output_features = output_features
        # for aggregation and cache
        self.min_timestamp_list = []
        self.hour_timestamp_list = []
        # 预计算并缓存聚合结果
        self.future_min_agg = {}
        self.future_h_agg = {}
        self.available_timestamp_list = []
        self.prepare_aggregation(gte)

    def _aggregate_time_range(self,start_timestamp, unit_duration):
        high, low, open_price, close_price = -np.inf, np.inf, None, None
        ts = start_timestamp
        data_slice = self.df.loc[ts:ts + unit_duration - 1]
        if not data_slice.empty:
            open_price = data_slice.iloc[0]['open']
            high = max(high, data_slice['high'].max())
            low = min(low, data_slice['low'].min())
            close_price = data_slice.iloc[-1]['close']
            # vol = data_slice['vol_as_u'].sum()
        return [open_price, high, low, close_price]

    def prepare_aggregation(self,gte):
        timestamps = self.df['timestamp']
        self.df.set_index('timestamp',inplace=True)
        # 获取最早和最晚的时间戳
        start_time = timestamps.min()
        end_time = timestamps.max()
        # 构建分钟级别的时间戳列表
        current_time = (start_time // 60) * 60  # 向下取整到整分钟
        while current_time <= end_time:
            self.min_timestamp_list.append(current_time)
            current_time += 60  # 每次增加 60 秒（1 分钟）

        # 构建小时级别的时间戳列表
        current_time = (start_time // 3600) * 3600  # 向下取整到整小时
        while current_time <= end_time:
            self.hour_timestamp_list.append(current_time)
            current_time += 3600  # 每次增加 3600 秒（1 小时）
        e = current_time - 13 * 3600
        self.available_timestamp_list = self.df.index[(self.df.index >= gte) & (self.df.index <= e)].tolist()
        print('init_done')

    def __len__(self):
        return len(self.available_timestamp_list)

    def __getitem__(self, idx):
        current_timestamp = self.available_timestamp_list[idx]

        # 获取输入特征
        X = self.df.loc[current_timestamp, self.input_features].values.astype(np.float32)

        current_minute = current_timestamp - current_timestamp%60
        current_hour = current_timestamp - current_timestamp%3600

        y_min = []
        y_h = []
        for i in range(61):
            aim = current_minute + 60*i
            if aim not in self.future_min_agg:
                self.future_min_agg[aim] = self._aggregate_time_range(aim,60)
            y_min.append(self.future_min_agg[aim])
        for i in range(13):
            aim = current_hour + 3600 * i
            if aim not in self.future_h_agg:
                self.future_h_agg[aim] = self._aggregate_time_range(aim, 3600)
            y_h.append(self.future_h_agg[aim])
        return [torch.tensor(X, dtype=torch.float32), torch.tensor(y_min, dtype=torch.float32), torch.tensor(y_h, dtype=torch.float32)]

# 测试流程
if __name__ == "__main__":
    # 读取 CSV 文件
    df = pd.read_csv('btc_future_only_10s_a1.csv', header=0)

    # 定义输入和输出特征
    input_features = ['open', 'high', 'low', 'close', 'vol_as_u']
    output_features = ['open', 'high', 'low', 'close']

    # 创建数据集实例
    dataset = TimeSeriesDataset(df, input_features, output_features)

    # 按照时间顺序划分数据集
    train_size = int(len(dataset) * 0.9)
    train_dataset = TimeSeriesDataset(df.iloc[:train_size], input_features, output_features)
    test_dataset = TimeSeriesDataset(df.iloc[train_size:], input_features, output_features)

    # 创建 DataLoader 实例，确保 batch size 不足时进行填充
    batch_size = 600
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    X, y_60min, y_12h = train_dataset[0]
    X, y_60min, y_12h = train_dataset[1]
    X, y_60min, y_12h = train_dataset[2]
    X, y_60min, y_12h = train_dataset[3]
    # 迭代数据加载器
    for data in train_loader:
        print("Batch input features:", data.shape)
        break
