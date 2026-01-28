import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 统一时间轴：0 到 15 秒，步长 0.1 秒
time_grid = np.arange(0, 15.01, 0.1)  # 包含 15.0
total_queue = np.zeros_like(time_grid)
total_throughput = np.zeros_like(time_grid)

# 遍历每个流，插值到统一时间轴并累加
for fid in range(5):
    try:
        df = pd.read_csv(f'./log/ts_flow{fid}_buffer.csv')
        # 只保留 <=15s 的数据
        df = df[df['time'] <= 15]
        if df.empty:
            continue

        # 使用线性插值将流的数据对齐到 time_grid
        queue_interp = np.interp(
            time_grid,
            df['time'],
            df['queue_length'],
            left=0, right=0  # 时间范围外视为 0
        )
        tp_interp = np.interp(
            time_grid,
            df['time'],
            df['instant_throughput_mbps'],
            left=0, right=0
        )

        total_queue += queue_interp
        total_throughput += tp_interp

    except FileNotFoundError:
        print(f"Warning: ts_flow{fid}_buffer.csv not found. Skipping.")
        continue

# 开始绘图
plt.figure(figsize=(14, 5))

# 子图1：总 buffer occupancy（包数）
plt.subplot(1, 2, 1)
plt.plot(time_grid, total_queue, linewidth=1.5)
plt.title('Overall Buffer Occupancy (All Flows)')
plt.xlabel('Time (s)')
plt.ylabel('Total Packets in Buffer')
plt.xlim(0, 15)
plt.grid(True, linestyle='--', alpha=0.5)

# 子图2：总瞬时吞吐量
plt.subplot(1, 2, 2)
plt.plot(time_grid, total_throughput, color='orange', linewidth=1.5)
plt.title('Overall Instant Throughput (All Flows)')
plt.xlabel('Time (s)')
plt.ylabel('Total Throughput (Mbps)')
plt.xlim(0, 15)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('overall_system_metrics.png', dpi=200)
plt.close()

print("✅ Overall system metrics plot saved as 'overall_system_metrics.png'")