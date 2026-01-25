# FindRedThreshold__EventDrivenScript.py —— 支持自定义策略列表
import sys
import os
import random
from statistics import mean, stdev

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FindRedThreshold_EventDrivenSimu import simulate

def main():
    # Define the candidate strategies
    STRATEGIES = [
        'none',
        'replicate',
        'replicate_2_1',
        'replicate_3_1',
        'replicate_5_1',
        'replicate_10_1',
        'xor_2_1',
        'xor_3_1',
        'xor_5_1',
        'xor_10_1',
    ]
    LOSS_RATES = [i / 100 for i in range(0, 21, 2)]  # 0% to 20%
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
    header = "Loss% | " + " | ".join(f"{s:>12}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))
    for loss in LOSS_RATES:
        row = f"{loss:5.1%} |"
        for red in STRATEGIES:
            tp = results.get((loss, red), {'throughput': 0.0})['throughput']
            row += f" {tp:12.2f} |"
        print(row)

    print("\n=== Average Queue Length ===")
    header = "Loss% | " + " | ".join(f"{s:>12}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))
    for loss in LOSS_RATES:
        row = f"{loss:5.1%} |"
        for red in STRATEGIES:
            q_len = results.get((loss, red), {'queue': 0.0})['queue']
            row += f" {q_len:12.2f} |"
        print(row)

    # Save CSV
    csv_filename = "threshold_test_results.csv"
    with open(csv_filename, "w") as f:
        f.write("LossRate,Strategy,AvgThroughput_Mbps,AvgQueueLength,ThroughputStd,QueueStd\n")
        for (loss, red), r in results.items():
            f.write(f"{loss:.3f},{red},{r['throughput']:.3f},{r['queue']:.3f},{r['tp_std']:.3f},{r['q_std']:.3f}\n")
    print(f"\n✅ Results saved to {csv_filename}")

if __name__ == '__main__':
    main()