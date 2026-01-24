### 运行基本仿真：
# 直接运行
python main.py

# 带参数运行
python main.py --duration 10 --rate 100 --loss 0.3 --strategy fixed --redundancy 0.4 --visualize

# 比较不同策略
python main.py --strategy compare

# 安静模式运行并导出结果
python main.py --duration 5 --loss 0.4 --strategy adaptive --quiet --export ./results




### 作为模块使用：
from quic_simulator import QUICSimulator, SimulationConfig

# 创建配置
config = SimulationConfig(
    duration=10.0,
    packet_rate=100.0,
    loss_rate=0.3,
    redundancy_strategy="fixed",
    redundancy_factor=0.3
)

# 创建并运行仿真
simulator = QUICSimulator(config)
simulator.run()

# 获取结果
results = simulator.get_results()
print(f"交付包数: {results['receiver_stats']['delivered']}")
print(f"缓冲区峰值: {results['receiver_stats']['buffer_max']}")

# 可视化
simulator.visualize()