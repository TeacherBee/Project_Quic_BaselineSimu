import os
import numpy as np
import pandas as pd

os.makedirs('./log', exist_ok=True)

# 定义每个流的参数（贴合你的输出）
flows = [
    {"fid": 0, "avg_queue": 3294.7, "throughput": 58.57},
    {"fid": 1, "avg_queue": 3760.3, "throughput": 66.83},
    {"fid": 2, "avg_queue": 5763.3, "throughput": 56.85},
    {"fid": 3, "avg_queue": 9012.8, "throughput": 57.12},
    {"fid": 4, "avg_queue": 2528.1, "throughput": 36.61},
]

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