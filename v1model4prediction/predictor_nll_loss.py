import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# 假设您的数据加载器保存在 dataloader.py 文件中
from v1model4prediction.dataloaders import TimeSeriesDataset
import pandas as pd
import numpy as np
import os
import glob
import re
import time
from sklearn.preprocessing import StandardScaler


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
        x = x.unsqueeze(1)
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
    batch_size = 600
    learning_rate = 0.001

    # 读取数据集
    df = pd.read_csv('btc_future_only_10s_a1.csv', header=0)
    #手动选数据
    df = df[2581205:]
    
    input_features = ['open', 'high', 'low', 'close', 'vol_as_u']
    output_features = ['open', 'high', 'low', 'close']

    # 创建 scaler 实例
    scaler = StandardScaler()
    output_scaler = StandardScaler()
    # 对输入特征进行标准化
    output_scaler.fit_transform(df[output_features].copy())
    df[input_features] = scaler.fit_transform(df[input_features])

    # 保存 scaler 以便在预测时使用
    joblib.dump(scaler, 'scaler.save')
    joblib.dump(output_scaler, 'output_scaler.save')

    # 创建数据集实例
    all_dataset = TimeSeriesDataset(df, input_features, output_features)
    train_size = int(len(all_dataset) * 0.9)
    train_dataset = TimeSeriesDataset(df.iloc[:train_size], input_features, output_features)
    # test_dataset = TimeSeriesDataset(df.iloc[train_size:], input_features, output_features)
    # 计算总的批次数
    total_samples = len(train_dataset)
    total_batches = total_samples // batch_size

    # 初始化模型和优化器
    output_size_60min = 61 * 4  # 61 个时间步，每个时间步 4 个特征
    output_size_24h = 13 * 4  # 13 个时间步，每个时间步 4 个特征

    model = LSTMModel(input_size, hidden_size, num_layers, output_size_60min, output_size_24h).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7)

    # 检查是否存在检查点以继续训练
    start_batch = 0

    # 寻找最新的检查点文件
    checkpoint_files = glob.glob('model_checkpoint_batch*.pth')
    if checkpoint_files:
        # 找到最大的batch编号
        max_batch_num = -1
        for file in checkpoint_files:
            match = re.search(r'model_checkpoint_batch(\d+).pth', file)
            if match:
                batch_num = int(match.group(1))
                if batch_num > max_batch_num:
                    max_batch_num = batch_num
                    latest_checkpoint_path = file
        if max_batch_num >= 0:
            start_batch = max_batch_num
            print(f"正在加载检查点 {latest_checkpoint_path}，从第 {start_batch} 个 batch 继续训练")
            checkpoint = torch.load(latest_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("未找到有效的检查点文件，重新开始训练")
    else:
        print("未找到检查点文件，重新开始训练")

    # 创建自定义的 DataLoader，从指定的批次开始
    # 使用 Subset，将数据集从 start_index 开始
    start_index = start_batch * batch_size
    remaining_indices = list(range(start_index, total_samples))
    subset_dataset = Subset(train_dataset, remaining_indices)
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    current_batch = start_batch

    # 初始化 previous_checkpoint_path 变量
    previous_checkpoint_path = None
    if start_batch > 0:
        previous_checkpoint_path = f'model_checkpoint_batch{start_batch}.pth'

    # 在开始训练前，记录起始时间
    start_time = time.time()

    # 训练循环
    model.train()
    for batch_idx, (X, y_60min, y_24h) in enumerate(data_loader):
        current_batch = start_batch + batch_idx

        # 将数据移动到设备上
        X = X.to(device)
        y_60min = y_60min.to(device)
        y_24h = y_24h.to(device)

        # 前向传播
        (mean_60min, std_60min), (mean_24h, std_24h) = model(X)

        # **确保标准差为正值，避免数值问题**
        std_60min = F.softplus(std_60min) + 1e-6
        std_24h = F.softplus(std_24h) + 1e-6

        # 计算损失
        loss_60min = gaussian_nll_loss(mean_60min, std_60min, y_60min)
        loss_24h = gaussian_nll_loss(mean_24h, std_24h, y_24h)
        loss = loss_60min + loss_24h

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()

        # **添加梯度裁剪，防止梯度爆炸**
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        # peek
        # if (current_batch + 1) % 100 == 0:
        #     # 获取第一个样本的预测均值
        #     first_mean_60min = mean_60min[0].cpu().detach().numpy()
        #     first_mean_60min_flat = first_mean_60min.reshape(-1, 4)
        #     first_mean_60min_original = output_scaler.inverse_transform(first_mean_60min_flat)
        #     first_mean_60min_original = first_mean_60min_original.reshape(61, 4)
        #
        #     print(f'Batch {current_batch + 1} First Sample Predictions (60min):')
        #     print(first_mean_60min_original[0])
        #
        #     # 获取第一个样本的真实目标值
        #     first_y_60min = y_60min[0].cpu().detach().numpy()
        #     first_y_60min_flat = first_y_60min.reshape(-1, 4)
        #     first_y_60min_original = output_scaler.inverse_transform(first_y_60min_flat)
        #     first_y_60min_original = first_y_60min_original.reshape(61, 4)
        #
        #     print(f'Batch {current_batch + 1} First Sample True Values (60min):')
        #     print(first_y_60min_original[0])
        #
        #     # 可选：计算绝对误差
        #     absolute_error = np.abs(first_mean_60min_original - first_y_60min_original)
        #     print(f'Batch {current_batch + 1} First Sample Absolute Error (60min):')
        #     print(absolute_error[0])


        # 计算当前进度和剩余时间
        elapsed_time = time.time() - start_time
        batches_done = current_batch + 1
        batches_left = total_batches - batches_done
        time_per_batch = elapsed_time / (batches_done - start_batch)
        estimated_remaining_time = batches_left * time_per_batch

        # 将剩余时间格式化为易读的格式
        est_hours = int(estimated_remaining_time // 3600)
        est_minutes = int((estimated_remaining_time % 3600) // 60)
        est_seconds = int(estimated_remaining_time % 60)
        if (current_batch + 1) % 100 == 0:
            # 显示进度和预估剩余时间
            print(f'Batch [{batches_done}/{total_batches}], Loss: {loss.item():.4f}, '
                  f'Estimated Remaining Time: {est_hours}h {est_minutes}m {est_seconds}s')

        # 保存检查点
        if (current_batch + 1) % 1000 == 0:
            checkpoint_path = f'model_checkpoint_batch{current_batch + 1}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"已保存检查点：{checkpoint_path}")

            # 删除上一个检查点文件
            if previous_checkpoint_path and os.path.exists(previous_checkpoint_path):
                os.remove(previous_checkpoint_path)
                print(f"已删除上一个检查点：{previous_checkpoint_path}")

            # 更新 previous_checkpoint_path
            previous_checkpoint_path = checkpoint_path

    print("训练完成。")

    # 保存最终模型
    torch.save(model.state_dict(), 'final_model.pth')
    print("模型已保存为 'final_model.pth'")

    # 删除最后一个检查点文件
    if previous_checkpoint_path and os.path.exists(previous_checkpoint_path):
        os.remove(previous_checkpoint_path)
        print(f"已删除最后的检查点文件：{previous_checkpoint_path}")


if __name__ == "__main__":
    train_model()
