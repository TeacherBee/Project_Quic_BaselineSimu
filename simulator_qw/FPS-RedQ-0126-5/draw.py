import pandas as pd
import matplotlib.pyplot as plt

for fid in range(5):
    try:
        df = pd.read_csv(f'./log/ts_flow{fid}_buffer.csv')
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(df['time'], df['queue_length'], label='Queue Length')
        plt.title(f'Flow {fid} - Buffer Occupancy')
        plt.xlabel('Time (s)')
        plt.ylabel('Packets')

        plt.subplot(1, 2, 2)
        plt.plot(df['time'], df['instant_throughput_mbps'], color='orange', label='Instant Throughput')
        plt.title(f'Flow {fid} - Instant Throughput')
        plt.xlabel('Time (s)')
        plt.ylabel('Mbps')

        plt.tight_layout()
        plt.savefig(f'flow_{fid}_metrics.png', dpi=150)
        plt.close()
    except FileNotFoundError:
        continue