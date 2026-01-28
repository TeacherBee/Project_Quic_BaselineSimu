import os
import numpy as np
import pandas as pd

os.makedirs('./log', exist_ok=True)

# 定义每个流的参数（贴合你的输出）

# ------------- mine -------------
# flows = [
#     {"fid": 0, "avg_queue": 3294.7, "throughput": 58.57},
#     {"fid": 1, "avg_queue": 3760.3, "throughput": 66.83},
#     {"fid": 2, "avg_queue": 5763.3, "throughput": 56.85},
#     {"fid": 3, "avg_queue": 9012.8, "throughput": 57.12},
#     {"fid": 4, "avg_queue": 2528.1, "throughput": 36.61},
# ]
# 重传：11698 18040 62750 72586 6041
# 效率bps/bit：113.8 113.8 63.1 40.6 94.9

# ------------- SR -------------

# flows = [
#     {"fid": 0, "avg_queue": 3294.7, "throughput": 58.57},
#     {"fid": 1, "avg_queue": 3760.3, "throughput": 66.83},
#     {"fid": 2, "avg_queue": 5969.4, "throughput": 57.23},
#     {"fid": 3, "avg_queue": 9411.7, "throughput": 59.30},
#     {"fid": 4, "avg_queue": 2596.3, "throughput": 38.99},
# ]
# 重传：11698 18040 61344 72586 6041
# 效率：113.8 113.8 61.4 40.3 96.1

# ------------- RSR -------------
# flows = [
#     {"fid": 0, "avg_queue": 3299.6, "throughput": 54.45},
#     {"fid": 1, "avg_queue": 3730.9, "throughput": 64.63},
#     {"fid": 2, "avg_queue": 6736.4, "throughput": 57.65},
#     {"fid": 3, "avg_queue": 9031.9, "throughput": 58.72},
#     {"fid": 4, "avg_queue": 2579.6, "throughput": 33.01},
# ]
# 重传：11854 18129 62414 70939 5995
# 效率：105.6 110.9 54.8 41.6 81.9

# ------------- FPS -------------
# flows = [
#     {"fid": 0, "avg_queue": 3027.5, "throughput": 50.00},
#     {"fid": 1, "avg_queue": 3637.8, "throughput": 56.84},
#     {"fid": 2, "avg_queue": 5559.7, "throughput": 57.36},
#     {"fid": 3, "avg_queue": 8006.3, "throughput": 53.49},
#     {"fid": 4, "avg_queue": 2757.9, "throughput": 33.72},
# ]
# 重传：13130 19750 60451 70219 6364
# 效率：105.7 100.0 66.0 42.8 78.2

# ------------- none -------------
# flows = [
#     {"fid": 0, "avg_queue": 3294.7, "throughput": 58.57},
#     {"fid": 1, "avg_queue": 3760.3, "throughput": 66.83},
#     {"fid": 2, "avg_queue": 5763.3, "throughput": 56.85},
#     {"fid": 3, "avg_queue": 9707.1, "throughput": 60.20},
#     {"fid": 4, "avg_queue": 2552.5, "throughput": 35.52},
# ]
# 重传：11698 18040 62750 76596 5875
# 效率：113.8 113.8 63.1 39.7 89.0

# ------------- replicate -------------
# flows = [
#     {"fid": 0, "avg_queue": 4220.2, "throughput": 54.89},
#     {"fid": 1, "avg_queue": 4575.4, "throughput": 56.83},
#     {"fid": 2, "avg_queue": 5184.9, "throughput": 51.67},
#     {"fid": 3, "avg_queue": 9365.5, "throughput": 60.51},
#     {"fid": 4, "avg_queue": 2574.5, "throughput": 39.99},
# ]
# 重传：10544 17095 59630 73148 5899
# 效率：83.2 79.5 63.8 41.3 99.4

# ------------- xor -------------
flows = [
    {"fid": 0, "avg_queue": 200.0, "throughput": 14.77},
    {"fid": 1, "avg_queue": 1418.6, "throughput": 18.32},
    {"fid": 2, "avg_queue": 1766.8, "throughput": 25.21},
    {"fid": 3, "avg_queue": 6607.7, "throughput": 60.51},
    {"fid": 4, "avg_queue": 256.4, "throughput": 14.71},
]
# 重传：3466 29649 57020 72544 1942
# 效率：59.05 82.6 91.3 58.6 367.2



time_grid = np.arange(0, 15.1, 0.1)
PKT_SIZE_BYTES = 1250

for flow in flows:
    fid = flow["fid"]
    avg_q = flow["avg_queue"]
    tp_mbps = flow["throughput"]

    # 模拟 queue_length：围绕 avg_q 波动 ±20%
    noise = np.random.normal(0, 0.2 * avg_q, len(time_grid))
    queue = np.clip(avg_q + noise, 0, None).astype(int)

    # cumulative delivered = throughput × time / (8 * pkt_size) × 1e6
    cum_delivered = (tp_mbps * 1e6 * time_grid) / (8 * PKT_SIZE_BYTES)
    cum_delivered = cum_delivered.astype(int)

    # instant throughput：基本恒定，加小噪声
    inst_tp = np.full_like(time_grid, tp_mbps)
    inst_tp += np.random.normal(0, 2, len(time_grid))
    inst_tp = np.clip(inst_tp, 0, None)

    df = pd.DataFrame({
        'time': time_grid,
        'queue_length': queue,
        'cumulative_delivered': cum_delivered,
        'instant_throughput_mbps': inst_tp
    })

    df.to_csv(f'./ts_flow{fid}_buffer.csv', index=False)
    print(f"✅ Generated ts_flow{fid}_buffer.csv (avg_queue={avg_q:.1f})")