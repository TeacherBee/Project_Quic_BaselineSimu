# sweep_avg.py —— 支持自定义策略列表
import sys
import os
import random
from statistics import mean, stdev

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FindRedThreshold_EventDrivenSimu import simulate

def main():
    # Define your candidate strategies
    STRATEGIES = [
        'none',
        'replicate',
        'erasure_2_1',   # XOR
        'erasure_4_2',   # RS(6,4)
        'erasure_5_3',   # RS(8,5)
        'erasure_10_5',  # Strong protection
    ]
    LOSS_RATES = [i / 100 for i in range(0, 15, 2)]  # 0% to 14%
    TRIALS = 10
    BASE_SEED = 42

    print(f"Running sweep: {len(LOSS_RATES)} loss rates × {len(STRATEGIES)} strategies × {TRIALS} trials")
    results = {}

    for loss in LOSS_RATES:
        for red in STRATEGIES:
            print(f"Testing: loss={loss:.1%}, redundancy={red} ...")
            throughputs = []
            queues = []
            for trial in range(TRIALS):
                seed = BASE_SEED + hash((loss, red, trial)) % 100000
                random.seed(seed)
                try:
                    res = simulate(loss, red)
                    throughputs.append(res['throughput_mbps'])
                    queues.append(res['avg_queue_length'])
                except Exception as e:
                    print(f"  Trial {trial+1} failed: {e}")
                    throughputs.append(0.0)
                    queues.append(0.0)

            results[(loss, red)] = {
                'throughput': mean(throughputs),
                'tp_std': stdev(throughputs) if len(throughputs) > 1 else 0.0,
                'queue': mean(queues),
                'q_std': mean(queues) if len(queues) > 1 else 0.0
            }

    # Print table
    print("\n=== Throughput (Mbps) ===")
    header = "Loss% | " + " | ".join(f"{s:>10}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))
    for loss in LOSS_RATES:
        row = f"{loss:5.1%} |"
        for red in STRATEGIES:
            tp = results.get((loss, red), {'throughput': 0.0})['throughput']
            row += f" {tp:10.2f} |"
        print(row)

    # Save CSV
    with open("sweep_results.csv", "w") as f:
        f.write("LossRate,Strategy,AvgThroughput_Mbps,AvgQueue\n")
        for (loss, red), r in results.items():
            f.write(f"{loss:.3f},{red},{r['throughput']:.3f},{r['queue']:.3f}\n")
    print("\n✅ Results saved to sweep_results.csv")

if __name__ == '__main__':
    main()