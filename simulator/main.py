"""主程序入口"""
import sys
import os
import argparse
import random
from typing import List, Dict, Any

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SimulationConfig
from redundancy import RedundancyController, RedundancyConfig
from sender import QUICSender
from receiver import QUICReceiver
from network import NetworkTopology
from stats import StatisticsCollector
from visualizer import QUICVisualizer

class QUICSimulator:
    """QUIC协议仿真器（主控制器）"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # 仿真状态
        self.time = 0.0
        self.next_send_time = 0.0
        self.simulation_running = False
        self.timeline_data = []
        
        # 初始化统计收集器
        self.stats_collector = StatisticsCollector()
        
        # 初始化组件
        self._init_components()
        
    def _init_components(self):
        """初始化所有组件"""
        # 冗余控制器
        redundancy_config = RedundancyConfig(
            strategy=self.config.redundancy_strategy,
            factor=self.config.redundancy_factor,
            adaptive_threshold=self.config.adaptive_threshold,
            fec_enabled=self.config.fec_enabled,
            fec_group_size=self.config.fec_group_size
        )
        self.redundancy_controller = RedundancyController(redundancy_config)
        
        # 发送端
        self.sender = QUICSender(self.config, self.redundancy_controller)
        
        # 接收端
        self.receiver = QUICReceiver(self.config)
        
        # 网络拓扑
        self.network = NetworkTopology(self.config)
        
        # 记录初始化事件
        self.stats_collector.record_event(
            'INIT',
            f'仿真初始化完成 - 策略: {self.config.redundancy_strategy}, '
            f'冗余因子: {self.config.redundancy_factor}, '
            f'丢包率: {self.config.loss_rate}',
            self.time
        )
    
    def run(self, verbose: bool = True):
        """运行仿真"""
        self.simulation_running = True
        
        if verbose:
            print(f"\n开始QUIC协议仿真...")
            print(f"仿真时长: {self.config.duration}秒")
            print(f"发包速率: {self.config.packet_rate}包/秒")
            print(f"丢包率: {self.config.loss_rate:.1%}")
            print(f"冗余策略: {self.config.redundancy_strategy}")
            print(f"冗余因子: {self.config.redundancy_factor}")
            print("-" * 50)
        
        packet_interval = 1.0 / self.config.packet_rate
        
        while self.time < self.config.duration and self.simulation_running:
            # 1. 发送数据包
            if self.time >= self.next_send_time:
                packets = self.sender.generate_packet(self.time)
                if packets:
                    for packet in packets:
                        arrival_time = self.network.send_packet(
                            packet, self.time, link_idx=0
                        )
                        if arrival_time:
                            # 记录发送事件
                            self.stats_collector.record_event(
                                'SEND',
                                f'发送包 seq={packet.seq_num}',
                                self.time,
                                {'seq_num': packet.seq_num, 
                                 'is_redundant': packet.is_redundant,
                                 'size': packet.payload_size}
                            )
                    
                    self.next_send_time += packet_interval
            
            # 2. 从网络接收数据包
            network_result = self.network.get_next_packet(self.time)
            while network_result is not None:
                arrival_time, packet = network_result
                
                # 接收包
                delivered = self.receiver.receive_packet(packet, arrival_time)
                
                if delivered:
                    # 记录交付事件
                    self.stats_collector.record_event(
                        'DELIVER',
                        f'交付包 seq={packet.seq_num}',
                        arrival_time,
                        {'seq_num': packet.seq_num, 
                         'delay': arrival_time - packet.send_time}
                    )
                
                # 获取ACK并处理
                ack_info = self.receiver.get_next_ack(arrival_time)
                if ack_info:
                    rtt_sample = arrival_time - packet.send_time
                    self.sender.process_ack(
                        ack_info['ack_num'], 
                        rtt_sample, 
                        arrival_time
                    )
                    
                    # 记录ACK事件
                    self.stats_collector.record_event(
                        'ACK',
                        f'ACK seq={ack_info["ack_num"]}',
                        arrival_time,
                        {'ack_num': ack_info['ack_num'],
                         'rtt': rtt_sample}
                    )
                
                network_result = self.network.get_next_packet(self.time)
            
            # 3. 检查超时和丢包
            lost_packets = self.sender.check_timeouts(self.time)
            for seq_num in lost_packets:
                self.stats_collector.record_event(
                    'LOSS',
                    f'包丢失 seq={seq_num}',
                    self.time,
                    {'seq_num': seq_num}
                )
            
            # 4. 收集统计数据
            if self.time % self.config.stats_interval < 0.001:
                self._collect_statistics()
            
            # 5. 打印进度（可选）
            if verbose and self.time % self.config.log_interval < 0.001:
                self._print_progress()
            
            # 6. 时间前进
            self.time += 0.001  # 1ms步进
        
        # 仿真结束
        self.simulation_running = False
        self._finalize_simulation()
        
        if verbose:
            print("\n仿真完成!")
            print("=" * 50)
    
    def _collect_statistics(self):
        """收集统计数据"""
        # 获取组件统计
        sender_stats = self.sender.get_stats()
        receiver_stats = self.receiver.get_stats()
        network_stats = self.network.get_stats()
        
        # 记录时间序列数据
        self.stats_collector.record_time_series(
            'buffer_occupancy',
            self.time,
            receiver_stats.get('buffer_current', 0)
        )
        
        missing_count, gap_size = self.receiver.get_delivery_gap()
        self.stats_collector.record_time_series(
            'delivery_gap',
            self.time,
            gap_size,
            {'missing_count': missing_count}
        )
        
        # 计算瞬时吞吐量
        if len(receiver_stats['buffer_occupancy']) > 1:
            last_time, last_bytes = receiver_stats['buffer_occupancy'][-1]
            if len(receiver_stats['buffer_occupancy']) > 2:
                prev_time, prev_bytes = receiver_stats['buffer_occupancy'][-2]
                throughput = (last_bytes - prev_bytes) * 8 / (last_time - prev_time) / 1_000_000
                self.stats_collector.record_time_series('throughput', self.time, throughput)
        
        # 记录其他指标
        if sender_stats.get('rto'):
            self.stats_collector.record_time_series('rtt', self.time, sender_stats['rto'])
        
        if sender_stats.get('cwnd'):
            self.stats_collector.record_time_series('cwnd', self.time, sender_stats['cwnd'])
        
        if sender_stats.get('redundancy_effectiveness'):
            self.stats_collector.record_time_series(
                'redundancy_ratio',
                self.time,
                sender_stats.get('redundancy_effectiveness', 0)
            )
        
        # 更新累计统计
        cumulative_stats = {
            'total_packets_sent': sender_stats.get('total_sent', 0),
            'total_packets_delivered': receiver_stats.get('delivered', 0),
            'total_bytes_sent': sender_stats.get('bytes_sent', 0),
            'total_bytes_delivered': receiver_stats.get('bytes_received', 0),
            'total_retransmits': sender_stats.get('retransmits', 0),
            'total_duplicates': receiver_stats.get('duplicates', 0),
            'simulation_duration': self.time
        }
        self.stats_collector.update_cumulative_stats(cumulative_stats)
    
    def _print_progress(self):
        """打印仿真进度"""
        progress = min(100, (self.time / self.config.duration) * 100)
        
        sender_stats = self.sender.get_stats()
        receiver_stats = self.receiver.get_stats()
        
        print(f"时间: {self.time:.1f}s [{progress:.0f}%] | "
              f"发送: {sender_stats.get('total_sent', 0)} | "
              f"接收: {receiver_stats.get('delivered', 0)} | "
              f"缓冲: {receiver_stats.get('buffer_current', 0)} | "
              f"CWND: {sender_stats.get('cwnd', 0):.1f}")
    
    def _finalize_simulation(self):
        """仿真结束处理"""
        # 记录结束事件
        self.stats_collector.record_event(
            'END',
            '仿真结束',
            self.time,
            {'final_time': self.time}
        )
        
        # 更新最终统计
        self.stats_collector.update_cumulative_stats({
            'simulation_duration': self.time
        })
    
    def get_results(self) -> Dict[str, Any]:
        """获取仿真结果"""
        # 收集所有统计
        results = {
            'config': self.config.to_dict(),
            'sender_stats': self.sender.get_stats(),
            'receiver_stats': self.receiver.get_stats(),
            'network_stats': self.network.get_stats(),
            'summary_stats': self.stats_collector.get_summary_stats(),
            'timeline': self.timeline_data
        }
        
        return results
    
    def visualize(self, save_dir: str = None):
        """可视化结果"""
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 创建可视化器
        visualizer = QUICVisualizer(self.stats_collector)
        
        # 绘制综合图表
        save_path = os.path.join(save_dir, 'quic_simulation_summary.png') if save_dir else None
        visualizer.plot_comprehensive(save_path)
        
        # 导出数据
        if save_dir:
            self.stats_collector.export_to_csv(
                os.path.join(save_dir, 'time_series.csv')
            )
            self.stats_collector.export_events(
                os.path.join(save_dir, 'events.csv')
            )
            
            # 保存配置
            import json
            with open(os.path.join(save_dir, 'config.json'), 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            print(f"\n数据已导出到: {save_dir}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='QUIC协议仿真器')
    
    # 仿真参数
    parser.add_argument('--duration', type=float, default=10.0,
                       help='仿真时长（秒）')
    parser.add_argument('--rate', type=float, default=100.0,
                       help='发包速率（包/秒）')
    parser.add_argument('--loss', type=float, default=0.3,
                       help='网络丢包率')
    parser.add_argument('--packet-size', type=int, default=1200,
                       help='包大小（字节）')
    
    # 冗余策略
    parser.add_argument('--strategy', type=str, default='fixed',
                       choices=['none', 'fixed', 'adaptive', 'fec', 'hybrid'],
                       help='冗余策略')
    parser.add_argument('--redundancy', type=float, default=0.3,
                       help='冗余因子')
    parser.add_argument('--adaptive-threshold', type=float, default=0.1,
                       help='自适应阈值')
    
    # 输出选项
    parser.add_argument('--visualize', action='store_true',
                       help='启用可视化')
    parser.add_argument('--export', type=str, default=None,
                       help='导出数据目录')
    parser.add_argument('--quiet', action='store_true',
                       help='安静模式，减少输出')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    
    return parser.parse_args()

def compare_strategies():
    """比较不同冗余策略"""
    strategies = [
        {'name': '无冗余', 'strategy': 'none', 'factor': 0.0},
        {'name': '固定20%', 'strategy': 'fixed', 'factor': 0.2},
        {'name': '固定40%', 'strategy': 'fixed', 'factor': 0.4},
        {'name': '自适应', 'strategy': 'adaptive', 'factor': 0.3},
        {'name': '混合策略', 'strategy': 'hybrid', 'factor': 0.3}
    ]
    
    results = []
    
    for strategy_info in strategies:
        print(f"\n{'='*60}")
        print(f"测试策略: {strategy_info['name']}")
        print('='*60)
        
        # 创建配置
        config = SimulationConfig(
            duration=5.0,
            packet_rate=50.0,
            loss_rate=0.4,
            redundancy_strategy=strategy_info['strategy'],
            redundancy_factor=strategy_info['factor'],
            adaptive_threshold=0.1
        )
        
        # 运行仿真
        simulator = QUICSimulator(config)
        simulator.run(verbose=not strategy_info.get('quiet', False))
        
        # 收集结果
        result = simulator.get_results()
        results.append({
            'name': strategy_info['name'],
            'strategy': strategy_info['strategy'],
            'factor': strategy_info['factor'],
            'delivered': result['receiver_stats']['delivered'],
            'buffer_max': result['receiver_stats']['buffer_max'],
            'duplicates': result['receiver_stats']['duplicates'],
            'retransmits': result['sender_stats']['retransmits'],
            'throughput': result['summary_stats'].get('avg_throughput_mbps', 0)
        })
    
    # 打印比较结果
    print(f"\n{'='*70}")
    print("策略比较结果:")
    print(f"{'策略':<12} {'交付包数':<10} {'缓冲峰值':<10} {'重复包':<8} {'重传':<8} {'吞吐量':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<12} "
              f"{result['delivered']:<10} "
              f"{result['buffer_max']:<10} "
              f"{result['duplicates']:<8} "
              f"{result['retransmits']:<8} "
              f"{result['throughput']:<10.2f}")
    
    return results

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        print(f"随机种子设置为: {args.seed}")
    
    if args.strategy == 'compare':
        # 比较模式
        results = compare_strategies()
        return
    
    # 创建配置
    config = SimulationConfig(
        duration=args.duration,
        packet_rate=args.rate,
        loss_rate=args.loss,
        packet_size=args.packet_size,
        redundancy_strategy=args.strategy,
        redundancy_factor=args.redundancy,
        adaptive_threshold=args.adaptive_threshold
    )
    
    # 创建并运行仿真器
    simulator = QUICSimulator(config)
    simulator.run(verbose=not args.quiet)
    
    # 打印统计摘要
    simulator.stats_collector.print_summary()
    
    # 可视化和导出
    if args.visualize or args.export:
        export_dir = args.export or 'results'
        simulator.visualize(export_dir)

if __name__ == "__main__":
    main()