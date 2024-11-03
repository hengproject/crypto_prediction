import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# 假设您的数据加载器保存在 dataloader.py 文件中
from v1model4prediction.dataloaders import TimeSeriesDataset


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size_60min, output_size_24h):
        super(LSTMModel, self).__init__()

        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 共享的特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 分支 1：预测未来 60 分钟
        self.branch_60min = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size_60min * 2)  # 输出均值和标准差
        )

        # 分支 2：预测未来 24 小时
        self.branch_24h = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size_24h * 2)  # 输出均值和标准差
        )

    def forward(self, x):
        # x 的形状：batch_size x seq_len x input_size
        lstm_out, _ = self.lstm(x)
        lstm_last_output = lstm_out[:, -1, :]
        features = self.feature_extractor(lstm_last_output)

        output_60min = self.branch_60min(features)
        mean_60min, std_60min = torch.chunk(output_60min, 2, dim=1)
        mean_60min = mean_60min.view(-1, 61, 4)  # 重塑为 [batch_size, 61, 4]
        std_60min = F.softplus(std_60min) + 1e-6
        std_60min = std_60min.view(-1, 61, 4)

        output_24h = self.branch_24h(features)
        mean_24h, std_24h = torch.chunk(output_24h, 2, dim=1)
        mean_24h = mean_24h.view(-1, 13, 4)  # 重塑为 [batch_size, 13, 4]
        std_24h = F.softplus(std_24h) + 1e-6
        std_24h = std_24h.view(-1, 13, 4)

        return (mean_60min, std_60min), (mean_24h, std_24h)


# 定义高斯负对数似然损失函数
def gaussian_nll_loss(mean, std, target):
    var = std ** 2
    loss = 0.5 * torch.log(2 * torch.pi * var) + ((target - mean) ** 2) / (2 * var)
    return loss.mean()


def train_model():


    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 超参数设置
    input_size = 5  # 输入特征数量
    hidden_size = 128
    num_layers = 2
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # 读取数据集
    df = pd.read_csv('btc_future_only_10s_a1.csv', header=0)

    input_features = ['open', 'high', 'low', 'close', 'vol_as_u']
    output_features = ['open', 'high', 'low', 'close']

    # 创建数据集实例
    dataset = TimeSeriesDataset(df, input_features, output_features)

    # 按照时间顺序划分数据集
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    # 创建 DataLoader 实例
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 模型参数
    output_size_60min = 61 * 4  # 61 个时间步，每个时间步 4 个特征
    output_size_24h = 13 * 4  # 13 个时间步，每个时间步 4 个特征

    # 初始化模型、优化器和损失函数
    model = LSTMModel(input_size, hidden_size, num_layers, output_size_60min, output_size_24h).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 检查是否存在检查点以继续训练
    import os
    checkpoint_path = 'model_checkpoint.pth'
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("正在加载检查点...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从第 {start_epoch} 个 epoch 继续训练")

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (X, y_60min, y_24h) in enumerate(train_loader):
            # 将数据移动到设备上
            X = X.to(device)
            y_60min = y_60min.to(device)
            y_24h = y_24h.to(device)

            # 前向传播
            (mean_60min, std_60min), (mean_24h, std_24h) = model(X)

            # 计算损失
            loss_60min = gaussian_nll_loss(mean_60min, std_60min, y_60min)
            loss_24h = gaussian_nll_loss(mean_24h, std_24h, y_24h)
            loss = loss_60min + loss_24h

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] 平均损失: {avg_epoch_loss:.4f}')

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    # 保存最终模型
    torch.save(model.state_dict(), 'final_model.pth')
    print("训练完成。模型已保存为 'final_model.pth'")


if __name__ == "__main__":
    train_model()
