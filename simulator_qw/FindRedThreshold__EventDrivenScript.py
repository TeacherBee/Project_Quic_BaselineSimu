# FindRedThreshold__EventDrivenScript.py —— 支持自定义策略列表
import sys
import os
import random
from statistics import mean, stdev
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FindRedThreshold_EventDrivenSimu import simulate

def plot_results(results, loss_rates, strategies):
    """
    Plots three graphs based on the simulation results.

    Args:
        results (dict): The results dictionary from the sweep.
        loss_rates (list): List of loss rates tested.
        strategies (list): List of strategies tested.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot 1: Average Throughput vs Loss Rate
    ax1 = axes[0]
    for strategy in strategies:
        x_values = []
        y_values = []
        for loss in loss_rates:
            key = (loss, strategy)
            if key in results:
                x_values.append(loss * 100)  # Convert to percentage for x-axis
                y_values.append(results[key]['throughput'])
        
        if y_values: # Only plot if there are data points for this strategy
            ax1.plot(x_values, y_values, marker='o', label=strategy, linewidth=2)
    
    ax1.set_title('Average Throughput vs Loss Rate')
    ax1.set_xlabel('Loss Rate (%)')
    ax1.set_ylabel('Average Throughput (Mbps)')
    ax1.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Average Queue Length vs Loss Rate
    ax2 = axes[1]
    for strategy in strategies:
        x_values = []
        y_values = []
        for loss in loss_rates:
            key = (loss, strategy)
            if key in results:
                x_values.append(loss * 100)  # Convert to percentage for x-axis
                y_values.append(results[key]['queue'])
        
        if y_values: # Only plot if there are data points for this strategy
            ax2.plot(x_values, y_values, marker='s', label=strategy, linewidth=2)
    
    ax2.set_title('Average Queue Length vs Loss Rate')
    ax2.set_xlabel('Loss Rate (%)')
    ax2.set_ylabel('Average Queue Length')
    ax2.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Average Throughput Rate vs Loss Rate
    ax3 = axes[2]
    for strategy in strategies:
        x_values = []
        y_values = []
        for loss in loss_rates:
            key = (loss, strategy)
            if key in results:
                x_values.append(loss * 100)  # Convert to percentage for x-axis
                avg_tp = results[key]['throughput']
                avg_q = results[key]['queue']
                if avg_q != 0:
                    thr_rate = avg_tp / avg_q
                else:
                    thr_rate = float('inf') # Handle division by zero if needed, though unlikely for queue length
                y_values.append(thr_rate)
        
        if y_values: # Only plot if there are data points for this strategy
            ax3.plot(x_values, y_values, marker='^', label=strategy, linewidth=2)
    
    ax3.set_title('Average Throughput Rate vs Loss Rate')
    ax3.set_xlabel('Loss Rate (%)')
    ax3.set_ylabel('Average Throughput Rate (Mbps/packet)')
    ax3.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


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
    TRIALS = 2
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

    # New Table: Average Throughput Rate
    print("\n=== Average Throughput Rate (Mbps/packet) ===")
    header = "Loss% | " + " | ".join(f"{s:>12}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))
    for loss in LOSS_RATES:
        row = f"{loss:5.1%} |"
        for red in STRATEGIES:
            avg_tp = results.get((loss, red), {'throughput': 0.0})['throughput']
            avg_q = results.get((loss, red), {'queue': 0.0})['queue']
            if avg_q != 0:
                thr_rate = avg_tp / avg_q
            else:
                thr_rate = float('inf') # Handle potential division by zero gracefully
            row += f" {thr_rate:12.2f} |"
        print(row)

    # Save CSV (now includes Throughput Rate)
    csv_filename = "threshold_test_results.csv"
    with open(csv_filename, "w") as f:
        f.write("LossRate,Strategy,AvgThroughput_Mbps,AvgQueueLength,ThroughputStd,QueueStd,AvgThroughputRate\n")
        for (loss, red), r in results.items():
            avg_tp = r['throughput']
            avg_q = r['queue']
            if avg_q != 0:
                thr_rate = avg_tp / avg_q
            else:
                thr_rate = float('inf')
            f.write(f"{loss:.3f},{red},{avg_tp:.3f},{avg_q:.3f},{r['tp_std']:.3f},{r['q_std']:.3f},{thr_rate:.3f}\n")
    print(f"\n✅ Results saved to {csv_filename}")

    # Call the plotting function
    print("\nGenerating plots...")
    plot_results(results, LOSS_RATES, STRATEGIES)
    print("✅ Plots displayed.")


if __name__ == '__main__':
    main()