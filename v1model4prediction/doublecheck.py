import pandas as pd
import numpy as np
from collections import Counter


def check_timestamps(file_path):
    try:
        # 读取CSV文件，跳过表头
        df = pd.read_csv(file_path, names=['timestamp', 'open', 'high', 'low', 'close', 'vol_as_u'],
                         skiprows=1)

        # 确保timestamp是整数类型
        df['timestamp'] = df['timestamp'].astype(int)

        # 获取时间戳序列
        timestamps = df['timestamp'].values

        # 检查重复的时间戳
        timestamp_counts = Counter(timestamps)
        duplicates = {ts: count for ts, count in timestamp_counts.items() if count > 1}

        if duplicates:
            print("\n发现重复的时间戳:")
            print("时间戳         出现次数    出现的行号")
            print("-" * 50)
            for ts in duplicates:
                # 找出该时间戳出现的所有行号
                rows = df.index[df['timestamp'] == ts].tolist()
                print(f"{ts:<12d}    {duplicates[ts]:<8d}    {rows}")
        else:
            print("\n没有发现重复的时间戳！")

        # 计算相邻时间戳之差
        time_diffs = np.diff(timestamps)

        # 找出不连续的点（差值不等于10的位置）
        discontinuities = np.where(time_diffs != 10)[0]

        if len(discontinuities) == 0:
            print("\n所有时间戳都是连续的！")
        else:
            print(f"\n发现 {len(discontinuities)} 处时间戳不连续的地方:")
            print("\n缺失的时间戳段:")
            print("索引位置    当前时间戳    下一个时间戳    缺失数量    缺失的时间戳")
            print("-" * 70)

            for idx in discontinuities:
                current_ts = timestamps[idx]
                next_ts = timestamps[idx + 1]
                missing_count = (next_ts - current_ts) // 10 - 1

                # 生成缺失的时间戳
                missing_timestamps = [current_ts + (i + 1) * 10 for i in range(int(missing_count))]

                print(
                    f"{idx:<8d}    {current_ts:<12d}    {next_ts:<12d}    {missing_count:<8d}    {missing_timestamps}")

        # 检查是否所有时间戳都能被10整除
        invalid_timestamps = timestamps[timestamps % 10 != 0]
        if len(invalid_timestamps) > 0:
            print("\n警告：发现不能被10整除的时间戳：")
            for ts in invalid_timestamps:
                print(f"时间戳: {ts}")

        # 打印统计信息
        print("\n统计信息:")
        print(f"总行数: {len(timestamps)}")
        print(f"重复时间戳数: {len(duplicates)}")
        print(f"不连续点数: {len(discontinuities)}")
        print(f"非10整除时间戳数: {len(invalid_timestamps)}")

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        print("\n文件前几行内容：")
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"行 {i + 1}: {line.strip()}")
                else:
                    break


if __name__ == "__main__":
    file_path = 'btc_future_only_10s_a1.csv'  # 替换为你的CSV文件路径
    check_timestamps(file_path)