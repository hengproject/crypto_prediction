import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=600):
        self.data = data
        self.seq_length = seq_length
        self.features = self.data[['open', 'high', 'low', 'close', 'vol_as_u']].values
        self.targets = self.data[['open', 'high', 'low', 'close']].values
    def __len__(self):
        return len(self.data) - self.seq_length
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        print(x.shape)
        # Prepare future predictions for minute and hour levels
        y_minute = self.targets[idx + self.seq_length:idx + self.seq_length + 10]
        print(y_minute.shape)
        y_hour = self.targets[idx + self.seq_length * 6:idx + self.seq_length * 6 + 10]

        return torch.tensor(x, dtype=torch.float32), (
        torch.tensor(y_minute, dtype=torch.float32), torch.tensor(y_hour, dtype=torch.float32))

if __name__ == '__main__':
    batch_size = 600

    csv_file = 'btc_future_only_10s_s1.csv'
    data = pd.read_csv(csv_file)
    train_dataset = TimeSeriesDataset(data, 10)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    for data, labels in train_loader:
        print(data)
        print(labels)
        break
