# sweep_avg.py
import sys
import os
import random
from statistics import mean, stdev

# Add current directory to path to import fast_sim
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from FindRedThreshold_EventDrivenSimu import simulate

def main():
    # Configuration
    LOSS_RATES = [i / 100 for i in range(0, 15, 2)]  # 0%, 2%, ..., 14%
    STRATEGIES = ['none', 'replicate']
    TRIALS = 10
    BASE_SEED = 42

    print(f"Running sweep: {len(LOSS_RATES)} loss rates × {len(STRATEGIES)} strategies × {TRIALS} trials")
    print("=" * 70)
    
    results = {}

    for loss in LOSS_RATES:
        for red in STRATEGIES:
            print(f"Testing: loss={loss:.1%}, redundancy={red} ...")
            throughputs = []
            queues = []

            for trial in range(TRIALS):
                # Use different seed per trial for independence
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

            avg_tp = mean(throughputs)
            std_tp = stdev(throughputs) if len(throughputs) > 1 else 0.0
            avg_q = mean(queues)
            std_q = stdev(queues) if len(queues) > 1 else 0.0

            results[(loss, red)] = {
                'throughput': avg_tp,
                'tp_std': std_tp,
                'queue': avg_q,
                'q_std': std_q
            }

    # Print throughput table
    print("\n=== Average Throughput (Mbps) ===")
    print("Loss% |   None   | Replicate")
    print("-" * 30)
    for loss in LOSS_RATES:
        none = results.get((loss, 'none'), {'throughput': 0.0})
        rep = results.get((loss, 'replicate'), {'throughput': 0.0})
        print(f"{loss:5.1%} | {none['throughput']:8.2f} | {rep['throughput']:9.2f}")

    # Print queue length table
    print("\n=== Avg Receiver Queue Length (packets) ===")
    print("Loss% |   None   | Replicate")
    print("-" * 30)
    for loss in LOSS_RATES:
        none = results.get((loss, 'none'), {'queue': 0.0})
        rep = results.get((loss, 'replicate'), {'queue': 0.0})
        print(f"{loss:5.1%} | {none['queue']:8.1f} | {rep['queue']:9.1f}")

    # Save to CSV
    save_to_csv(results, LOSS_RATES, STRATEGIES)

def save_to_csv(results, loss_rates, strategies, filename="sweep_results.csv"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("LossRate,Strategy,AvgThroughput_Mbps,StdThroughput,AvgQueue,StdQueue\n")
        for loss in loss_rates:
            for red in strategies:
                r = results.get((loss, red), {})
                tp = r.get('throughput', 0.0)
                tp_std = r.get('tp_std', 0.0)
                q = r.get('queue', 0.0)
                q_std = r.get('q_std', 0.0)
                f.write(f"{loss:.3f},{red},{tp:.3f},{tp_std:.3f},{q:.3f},{q_std:.3f}\n")
    print(f"\n✅ Results saved to {filename}")

if __name__ == '__main__':
    main()