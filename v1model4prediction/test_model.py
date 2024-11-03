import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# 从您的模型文件中导入 LSTMModel 和 TimeSeriesDataset 类
from v1model4prediction.dataloaders import TimeSeriesDataset

# 定义 LSTM 模型（与训练时相同）
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

def test_model():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 超参数设置（与训练时相同）
    input_size = 5  # 输入特征数量
    hidden_size = 128
    num_layers = 2
    batch_size = 600

    # 定义模型
    output_size_60min = 61 * 4  # 61 个时间步，每个时间步 4 个特征
    output_size_24h = 13 * 4    # 13 个时间步，每个时间步 4 个特征

    model = LSTMModel(input_size, hidden_size, num_layers, output_size_60min, output_size_24h).to(device)

    # 加载训练好的模型参数
    model.load_state_dict(torch.load('final_model.pth', map_location=device))
    model.eval()

    # 加载 scaler 对象
    scaler = joblib.load('scaler.save')
    output_scaler = joblib.load('output_scaler.save')

    # 读取测试数据集
    df = pd.read_csv('btc_future_only_10s_a1.csv', header=0)
    df = df[2581205:]  # 根据您的数据集选择

    input_features = ['open', 'high', 'low', 'close', 'vol_as_u']
    output_features = ['open', 'high', 'low', 'close']

    # **选择测试数据集的一个子集**
    test_subset_size = 500000  # 您可以根据需要调整子集大小
    df_test = df[:test_subset_size]

    # 对输入特征进行标准化
    X_test = df_test[input_features].copy()
    X_test_scaled = scaler.transform(X_test)

    # 创建测试数据集实例
    test_dataset = TimeSeriesDataset(df_test, input_features, output_features)

    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # **限制处理的批次数**
    max_batches = 1  # 您可以根据需要调整处理的批次数

    # 定义用于存储预测值和真实值的列表
    all_predictions_60min = []
    all_true_values_60min = []
    all_std_60min = []

    all_predictions_24h = []
    all_true_values_24h = []
    all_std_24h = []

    # 禁用梯度计算
    with torch.no_grad():
        for batch_idx, (X_batch, y_60min_batch, y_24h_batch) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break  # 超过最大批次数，退出循环

            X_batch = X_batch.to(device)

            # 前向传播
            (mean_60min, std_60min), (mean_24h, std_24h) = model(X_batch)

            # **处理 60 分钟预测**

            # 获取预测的均值和标准差，形状：[batch_size, 61, 4]
            predictions_60min = mean_60min.cpu().numpy()
            std_60min_np = std_60min.cpu().numpy()

            # **逆变换预测均值**
            # 将预测值展平成二维数组，以便进行 inverse_transform
            predictions_60min_flat = predictions_60min.reshape(-1, 4)
            # 对预测值进行逆变换
            predictions_60min_original_flat = output_scaler.inverse_transform(predictions_60min_flat)
            # 将逆变换后的预测值恢复原始形状
            predictions_60min_original = predictions_60min_original_flat.reshape(batch_size, 61, 4)

            # **逆变换标准差**
            # 标准差在标准化时被缩放，需要乘以 scaler 的 scale_
            std_60min_flat = std_60min_np.reshape(-1, 4)
            std_60min_original_flat = std_60min_flat * output_scaler.scale_
            std_60min_original = std_60min_original_flat.reshape(batch_size, 61, 4)

            # **获取真实值**
            y_60min_batch = y_60min_batch.cpu().numpy()  # 形状：[batch_size, 61, 4]

            # **将预测值、真实值和标准差添加到列表中**
            all_predictions_60min.append(predictions_60min_original)
            all_true_values_60min.append(y_60min_batch)
            all_std_60min.append(std_60min_original)

            # **处理 24 小时预测**

            # 获取预测的均值和标准差，形状：[batch_size, 13, 4]
            predictions_24h = mean_24h.cpu().numpy()
            std_24h_np = std_24h.cpu().numpy()

            # **逆变换预测均值**
            predictions_24h_flat = predictions_24h.reshape(-1, 4)
            predictions_24h_original_flat = output_scaler.inverse_transform(predictions_24h_flat)
            predictions_24h_original = predictions_24h_original_flat.reshape(batch_size, 13, 4)

            # **逆变换标准差**
            std_24h_flat = std_24h_np.reshape(-1, 4)
            std_24h_original_flat = std_24h_flat * output_scaler.scale_
            std_24h_original = std_24h_original_flat.reshape(batch_size, 13, 4)

            # **获取真实值**
            y_24h_batch = y_24h_batch.cpu().numpy()  # 形状：[batch_size, 13, 4]

            # **将预测值、真实值和标准差添加到列表中**
            all_predictions_24h.append(predictions_24h_original)
            all_true_values_24h.append(y_24h_batch)
            all_std_24h.append(std_24h_original)

        # **将列表转换为数组**
        all_predictions_60min = np.concatenate(all_predictions_60min, axis=0)  # 形状：[样本数量, 61, 4]
        all_true_values_60min = np.concatenate(all_true_values_60min, axis=0)
        all_std_60min = np.concatenate(all_std_60min, axis=0)

        all_predictions_24h = np.concatenate(all_predictions_24h, axis=0)      # 形状：[样本数量, 13, 4]
        all_true_values_24h = np.concatenate(all_true_values_24h, axis=0)
        all_std_24h = np.concatenate(all_std_24h, axis=0)

        # **计算评估指标**

        from sklearn.metrics import mean_squared_error

        # 计算 60 分钟预测的 MSE 和 MAPE
        mse_60min = mean_squared_error(
            all_true_values_60min.reshape(-1, 4),
            all_predictions_60min.reshape(-1, 4)
        )
        mape_60min = np.mean(
            np.abs((all_true_values_60min - all_predictions_60min) / all_true_values_60min)
        ) * 100

        print(f'60 分钟预测的均方误差（MSE）：{mse_60min:.4f}')
        print(f'60 分钟预测的平均绝对百分比误差（MAPE）：{mape_60min:.2f}%')

        # 计算 24 小时预测的 MSE 和 MAPE
        mse_24h = mean_squared_error(
            all_true_values_24h.reshape(-1, 4),
            all_predictions_24h.reshape(-1, 4)
        )
        mape_24h = np.mean(
            np.abs((all_true_values_24h - all_predictions_24h) / all_true_values_24h)
        ) * 100

        print(f'24 小时预测的均方误差（MSE）：{mse_24h:.4f}')
        print(f'24 小时预测的平均绝对百分比误差（MAPE）：{mape_24h:.2f}%')

        # **可视化 Close 价格的预测结果**

        import matplotlib.pyplot as plt

        # 随机选择一个样本进行可视化（60 分钟）
        sample_index_60min = np.random.randint(0, all_predictions_60min.shape[0])
        time_steps_60min = range(61)

        # 获取预测的均值、标准差和真实值（Close 价格，索引 3）
        mean_60min_sample = all_predictions_60min[sample_index_60min, :, 3]
        std_60min_sample = all_std_60min[sample_index_60min, :, 3]
        true_values_60min_sample = all_true_values_60min[sample_index_60min, :, 3]

        # 绘制 60 分钟预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps_60min, true_values_60min_sample, label='真实值', color='blue')
        plt.plot(time_steps_60min, mean_60min_sample, label='预测均值', color='red')
        plt.fill_between(
            time_steps_60min,
            mean_60min_sample - std_60min_sample,
            mean_60min_sample + std_60min_sample,
            color='red', alpha=0.2, label='预测标准差'
        )
        plt.xlabel('时间步')
        plt.ylabel('Close 价格')
        plt.title('未来 60 分钟 Close 价格预测与真实值')
        plt.legend()
        plt.show()

        # 随机选择一个样本进行可视化（24 小时）
        sample_index_24h = np.random.randint(0, all_predictions_24h.shape[0])
        time_steps_24h = range(13)

        # 获取预测的均值、标准差和真实值（Close 价格，索引 3）
        mean_24h_sample = all_predictions_24h[sample_index_24h, :, 3]
        std_24h_sample = all_std_24h[sample_index_24h, :, 3]
        true_values_24h_sample = all_true_values_24h[sample_index_24h, :, 3]

        # 绘制 24 小时预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps_24h, true_values_24h_sample, label='真实值', color='blue')
        plt.plot(time_steps_24h, mean_24h_sample, label='预测均值', color='red')
        plt.fill_between(
            time_steps_24h,
            mean_24h_sample - std_24h_sample,
            mean_24h_sample + std_24h_sample,
            color='red', alpha=0.2, label='预测标准差'
        )
        plt.xlabel('时间步')
        plt.ylabel('Close 价格')
        plt.title('未来 24 小时 Close 价格预测与真实值')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    test_model()



