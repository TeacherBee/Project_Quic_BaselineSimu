import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 统一时间轴：0 到 15 秒，步长 0.1 秒
time_grid = np.arange(0, 15.01, 0.1)  # 包含 15.0
total_queue_packets = np.zeros_like(time_grid)
total_throughput_mbps = np.zeros_like(time_grid)

# 假设每个包大小为 1250 字节（1500 字节 MTU - 头部）
PKT_SIZE_BYTES = 1250

path = 'mine'

# 遍历每个流，插值到统一时间轴并累加
for fid in range(5):
    try:
        df = pd.read_csv(f'./log/{path}/ts_flow{fid}_buffer.csv')
        df = df[df['time'] <= 15]
        if df.empty:
            continue

        queue_interp = np.interp(
            time_grid,
            df['time'],
            df['queue_length'],
            left=0, right=0
        )
        tp_interp = np.interp(
            time_grid,
            df['time'],
            df['instant_throughput_mbps'],
            left=0, right=0
        )

        total_queue_packets += queue_interp
        total_throughput_mbps += tp_interp

    except FileNotFoundError:
        print(f"Warning: ts_flow{fid}_buffer.csv not found. Skipping.")
        continue

# 将队列长度转为字节数
total_queue_bytes = total_queue_packets * PKT_SIZE_BYTES

# 计算瞬时效率：Mbps → bps, then / bytes
efficiency_bps_per_byte = np.zeros_like(time_grid)
nonzero = total_queue_bytes > 1  # 避免除以极小值
efficiency_bps_per_byte[nonzero] = (
    total_throughput_mbps[nonzero] * 1e6 / total_queue_bytes[nonzero]
)

# 🔍 调试：打印几个点
print("Sample efficiencies (bps/byte):")
for i in [0, 50, 100, 150]:
    t = time_grid[i]
    q = total_queue_packets[i]
    tp = total_throughput_mbps[i]
    eff = efficiency_bps_per_byte[i]
    print(f"  t={t:.1f}s: queue={q:.0f}, tp={tp:.1f} Mbps, eff={eff:.2f} bps/byte")

# 开始绘图
plt.figure(figsize=(21, 7))

# 子图1：总 buffer occupancy（包数）
plt.subplot(1, 3, 1)
plt.plot(time_grid, total_queue_packets, linewidth=1.5)
plt.title('Overall Buffer Occupancy (All Flows)')
plt.xlabel('Time (s)')
plt.ylabel('Total Packets in Buffer')
plt.xlim(0, 15)
plt.grid(True, linestyle='--', alpha=0.5)

# 子图2：总瞬时吞吐量
plt.subplot(1, 3, 2)
plt.plot(time_grid, total_throughput_mbps, color='orange', linewidth=1.5)
plt.title('Overall Instant Throughput (All Flows)')
plt.xlabel('Time (s)')
plt.ylabel('Total Throughput (Mbps)')
plt.xlim(0, 15)
plt.grid(True, linestyle='--', alpha=0.5)

# 子图3：单位存储占用下的吞吐率（正确版本）
plt.subplot(1, 3, 3)
plt.plot(time_grid, efficiency_bps_per_byte, color='green', linewidth=1.5)
plt.title('Throughput per Unit Storage (bps/byte)')
plt.xlabel('Time (s)')
plt.ylabel('Efficiency (bps/byte)')
plt.xlim(0, 15)
plt.ylim(0, 30)  # 根据你的数据，最大约 20
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'./result_png/{path}_overall_system_metrics_with_correct_efficiency.png', dpi=200)
plt.close()

print("✅ Plot saved. Max efficiency:", np.max(efficiency_bps_per_byte))