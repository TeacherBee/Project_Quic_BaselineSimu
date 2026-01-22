import heapq
import random
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

class PacketStatus(Enum):
    """数据包状态"""
    IN_FLIGHT = 1      # 传输中
    DELIVERED = 2      # 已交付
    LOST = 3           # 丢失
    DUPLICATE = 4      # 重复包

@dataclass
class Packet:
    """数据包类"""
    seq_num: int           # 序列号
    send_time: float       # 发送时间
    is_original: bool      # 是否是原始包
    is_redundant: bool = False  # 是否是冗余包
    redundancy_id: int = 0      # 冗余批次ID
    status: PacketStatus = PacketStatus.IN_FLIGHT
    
    def __lt__(self, other):
        return self.seq_num < other.seq_num

class QUICSender:
    """QUIC发送端模拟"""
    
    def __init__(self, redundancy_strategy: str = "none", 
                 redundancy_factor: float = 0.0,
                 window_size: int = 10):
        """
        初始化发送端
        
        Args:
            redundancy_strategy: 冗余策略 
                "none": 无冗余
                "fixed": 固定比例冗余
                "adaptive": 自适应冗余
            redundancy_factor: 冗余因子（0-1之间）
            window_size: 发送窗口大小
        """
        self.redundancy_strategy = redundancy_strategy
        self.redundancy_factor = redundancy_factor
        self.window_size = window_size

        self.seq_num = 0
        self.inflight_packets = {}  # 传输中的数据包 {seq_num: Packet}
        self.sent_packets = []      # 已发送的数据包列表
        self.ack_received = set()   # 已确认的序列号
        self.loss_count = 0         # 丢失包计数
        self.redundancy_batch_id = 0
        self.stats = {
            'total_sent': 0,
            'original_sent': 0,
            'redundant_sent': 0,
            'redundancy_ratio': 0.0
        }
    
    def generate_packets(self, time: float, num_packets: int = 1) -> List[Packet]:
        """生成数据包（包含可能的冗余包）"""
        packets = []
        
        for _ in range(num_packets):
            # 生成原始包
            packet = Packet(
                seq_num=self.seq_num,
                send_time=time,
                is_original=True,
                status=PacketStatus.IN_FLIGHT
            )
            # 将原始包加入待发送列表
            packets.append(packet)
            self.inflight_packets[self.seq_num] = packet
            self.stats['original_sent'] += 1
            
            # 根据策略生成冗余包
            if self.redundancy_strategy != "none" and self.redundancy_factor > 0:
                self._add_redundant_packets(packet, packets, time)
            
            self.seq_num += 1
        
        # 将生成的 num_packets 个数据包（肯呢个包括冗余包）加入已发送列表
        self.sent_packets.extend(packets)
        self.stats['total_sent'] = len(self.sent_packets)

        # 更新冗余比例统计
        self.stats['redundancy_ratio'] = (
            self.stats['redundant_sent'] / 
            max(1, self.stats['original_sent'])
        )
        
        return packets
    
    def _add_redundant_packets(self, original_packet: Packet, 
                              packets: List[Packet], time: float):
        """添加冗余包"""
        if self.redundancy_strategy == "fixed":
            # 固定比例冗余：为每个包发送固定数量的冗余副本
            redundancy_count = int(1 / (1 - self.redundancy_factor) - 1)
            for i in range(redundancy_count):
                redundant_packet = Packet(
                    seq_num=original_packet.seq_num,
                    send_time=time + (i+1)*0.001,  # 稍微延迟发送
                    is_original=False,
                    is_redundant=True,
                    redundancy_id=self.redundancy_batch_id,
                    status=PacketStatus.IN_FLIGHT
                )
                packets.append(redundant_packet)
                self.stats['redundant_sent'] += 1
            
            self.redundancy_batch_id += 1
            
        elif self.redundancy_strategy == "adaptive":
            # 简单自适应：基于历史丢包率调整
            loss_rate = self.loss_count / max(1, len(self.sent_packets))
            if loss_rate > 0.1:  # 丢包率超过10%，增加冗余
                redundant_packet = Packet(
                    seq_num=original_packet.seq_num,
                    send_time=time + 0.001,
                    is_original=False,
                    is_redundant=True,
                    redundancy_id=self.redundancy_batch_id,
                    status=PacketStatus.IN_FLIGHT
                )
                packets.append(redundant_packet)
                self.stats['redundant_sent'] += 1
    
    def process_ack(self, seq_num: int):
        """处理ACK确认"""
        if seq_num in self.inflight_packets:
            self.ack_received.add(seq_num)
            del self.inflight_packets[seq_num]
    
    def detect_losses(self, timeout: float = 0.1):
        """检测丢包（超时重传）"""
        lost_packets = []
        for seq_num, packet in list(self.inflight_packets.items()):
            # 简化的丢包检测
            if random.random() < 0.1:  # 10%概率检测为丢包
                packet.status = PacketStatus.LOST
                lost_packets.append(seq_num)
                self.loss_count += 1
                del self.inflight_packets[seq_num]
        
        return lost_packets

class QUICReceiver:
    """QUIC接收端模拟"""
    
    def __init__(self):
        self.next_expected_seq = 0
        self.received_buffer = {}      # 接收缓冲区 {seq_num: Packet}
        self.delivered_packets = []    # 已交付的数据包
        self.duplicate_packets = []    # 重复接收的数据包
        self.buffer_occupancy = []     # 缓冲区占用记录 [(time, occupancy)]
        self.stats = {
            'total_received': 0,
            'delivered': 0,
            'duplicates': 0,
            'buffer_max': 0,
            'out_of_order': 0
        }
    
    def receive_packet(self, packet: Packet, time: float):
        """接收一个数据包"""
        self.stats['total_received'] += 1
        
        # 检查是否是重复包
        if packet.seq_num in self.received_buffer:
            self.duplicate_packets.append(packet)
            self.stats['duplicates'] += 1
            packet.status = PacketStatus.DUPLICATE
            return False
        
        # 存储到接收缓冲区
        self.received_buffer[packet.seq_num] = packet
        
        # 记录缓冲区占用
        buffer_size = len(self.received_buffer)
        self.buffer_occupancy.append((time, buffer_size))
        self.stats['buffer_max'] = max(self.stats['buffer_max'], buffer_size)
        
        # 检查是否可以按序交付
        return self._try_deliver(time)
    
    def _try_deliver(self, time: float) -> bool:
        """尝试按序交付数据包"""
        delivered = False
        
        while self.next_expected_seq in self.received_buffer:
            packet = self.received_buffer.pop(self.next_expected_seq)
            packet.status = PacketStatus.DELIVERED
            self.delivered_packets.append(packet)
            self.next_expected_seq += 1
            self.stats['delivered'] += 1
            delivered = True
        
        return delivered
    
    def get_delivery_gap(self) -> int:
        """获取交付间隙（当前期望但未收到的包数）"""
        if not self.received_buffer:
            return 0
        
        max_received = max(self.received_buffer.keys())
        gap = max_received - self.next_expected_seq + 1 - len(self.received_buffer)
        
        if gap > 0:
            self.stats['out_of_order'] += gap
            
        return max(0, gap)

class NetworkPath:
    """网络路径模拟（包含丢包和延迟）"""
    
    def __init__(self, loss_rate: float = 0.3, 
                 delay_mean: float = 0.01,
                 delay_std: float = 0.002):
        """
        初始化网络路径
        
        Args:
            loss_rate: 丢包率 (0-1)
            delay_mean: 平均延迟（秒）
            delay_std: 延迟标准差
        """
        self.loss_rate = loss_rate
        self.delay_mean = delay_mean
        self.delay_std = delay_std
        self.packet_queue = []  # 使用堆队列实现按到达时间排序
        self.dropped_packets = []
        self.stats = {
            'packets_sent': 0,
            'packets_dropped': 0,
            'avg_delay': 0.0
        }
    
    def send_packet(self, packet: Packet, send_time: float):
        """发送数据包到网络路径"""
        self.stats['packets_sent'] += 1
        
        # 模拟丢包
        if random.random() < self.loss_rate:
            packet.status = PacketStatus.LOST
            self.dropped_packets.append(packet)
            self.stats['packets_dropped'] += 1
            return None
        
        # 添加随机延迟
        delay = max(0, random.normalvariate(self.delay_mean, self.delay_std))
        arrival_time = send_time + delay
        
        # 使用堆队列按到达时间排序
        heapq.heappush(self.packet_queue, (arrival_time, packet))
        
        return arrival_time
    
    def get_next_packet(self, current_time: float) -> Optional[Tuple[float, Packet]]:
        """获取下一个到达的数据包"""
        if not self.packet_queue:
            return None
        
        # 检查是否有包已经到达
        if self.packet_queue[0][0] <= current_time:
            return heapq.heappop(self.packet_queue)
        
        return None

class QUICSimulator:
    """QUIC协议仿真器"""
    
    def __init__(self, duration: float = 10.0, 
                 packet_rate: float = 100.0,
                 loss_rate: float = 0.3,
                 redundancy_strategy: str = "fixed",
                 redundancy_factor: float = 0.3):
        """
        初始化仿真器
        
        Args:
            duration: 仿真时长（秒）
            packet_rate: 发包速率（包/秒）
            loss_rate: 网络丢包率
            redundancy_strategy: 冗余策略
            redundancy_factor: 冗余因子
        """
        self.duration = duration
        self.packet_rate = packet_rate
        self.loss_rate = loss_rate
        
        # 初始化组件
        self.sender = QUICSender(
            redundancy_strategy=redundancy_strategy,
            redundancy_factor=redundancy_factor
        )
        self.receiver = QUICReceiver()
        self.network = NetworkPath(
            loss_rate=loss_rate,
            delay_mean=0.01,
            delay_std=0.003
        )
        
        # 统计
        self.time = 0.0
        self.packet_interval = 1.0 / packet_rate
        self.next_send_time = 0.0
        self.timeline_data = []
        
    def run(self):
        """运行仿真"""
        print(f"开始QUIC仿真...")
        print(f"策略: {self.sender.redundancy_strategy}, "
              f"冗余因子: {self.sender.redundancy_factor}, "
              f"丢包率: {self.loss_rate}")
        
        while self.time < self.duration:
            # 1. 发送数据包
            if self.time >= self.next_send_time:
                packets = self.sender.generate_packets(self.time)
                for packet in packets:
                    self.network.send_packet(packet, self.time)
                self.next_send_time += self.packet_interval
            
            # 2. 从网络接收数据包
            network_result = self.network.get_next_packet(self.time)
            while network_result is not None:
                arrival_time, packet = network_result
                self.receiver.receive_packet(packet, arrival_time)
                network_result = self.network.get_next_packet(self.time)
            
            # 3. 检测丢包并可能触发重传（简化）
            if self.time % 0.1 < 0.001:  # 每100ms检测一次
                lost_packets = self.sender.detect_losses()
                # 简化的重传：重新发送丢失的包
                for seq_num in lost_packets:
                    retransmit_packet = Packet(
                        seq_num=seq_num,
                        send_time=self.time,
                        is_original=False,
                        is_redundant=False,
                        status=PacketStatus.IN_FLIGHT
                    )
                    self.network.send_packet(retransmit_packet, self.time)
            
            # 4. 收集统计数据
            if self.time % 0.1 < 0.001:  # 每100ms记录一次
                self._record_stats()
            
            # 5. 时间前进
            self.time += 0.001  # 1ms步进
        
        print("仿真完成!")
        self._print_statistics()
    
    def _record_stats(self):
        """记录时间点统计数据"""
        buffer_occupancy = len(self.receiver.received_buffer)
        delivery_gap = self.receiver.get_delivery_gap()
        
        self.timeline_data.append({
            'time': self.time,
            'buffer_occupancy': buffer_occupancy,
            'delivery_gap': delivery_gap,
            'delivered': self.receiver.stats['delivered'],
            'duplicates': self.receiver.stats['duplicates']
        })
    
    def _print_statistics(self):
        """打印统计结果"""
        print("\n=== 仿真统计 ===")
        print(f"发送端:")
        print(f"  总发送包数: {self.sender.stats['total_sent']}")
        print(f"  原始包数: {self.sender.stats['original_sent']}")
        print(f"  冗余包数: {self.sender.stats['redundant_sent']}")
        print(f"  冗余比例: {self.sender.stats['redundancy_ratio']:.2%}")
        print(f"  丢包数: {self.sender.loss_count}")
        
        print(f"\n接收端:")
        print(f"  总接收包数: {self.receiver.stats['total_received']}")
        print(f"  已交付包数: {self.receiver.stats['delivered']}")
        print(f"  重复包数: {self.receiver.stats['duplicates']}")
        print(f"  缓冲区峰值: {self.receiver.stats['buffer_max']}")
        print(f"  乱序间隙总计: {self.receiver.stats['out_of_order']}")
        
        print(f"\n网络:")
        print(f"  总发包数: {self.network.stats['packets_sent']}")
        print(f"  丢弃包数: {self.network.stats['packets_dropped']}")
        print(f"  实际丢包率: {self.network.stats['packets_dropped']/max(1, self.network.stats['packets_sent']):.2%}")
        
        # 计算有效吞吐量
        effective_throughput = self.receiver.stats['delivered'] / self.duration
        print(f"\n有效吞吐量: {effective_throughput:.1f} packets/sec")
        
        # 计算冗余效率
        if self.sender.stats['redundant_sent'] > 0:
            redundancy_efficiency = (
                (self.receiver.stats['delivered'] - self.sender.stats['original_sent']) /
                self.sender.stats['redundant_sent']
            )
            print(f"冗余效率: {redundancy_efficiency:.2f} (每冗余包带来的额外交付)")
    
    def plot_results(self):
        """绘制结果图表"""
        if not self.timeline_data:
            print("无数据可绘图")
            return
        
        times = [d['time'] for d in self.timeline_data]
        buffer_occ = [d['buffer_occupancy'] for d in self.timeline_data]
        delivery_gaps = [d['delivery_gap'] for d in self.timeline_data]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 缓冲区占用情况
        axes[0, 0].plot(times, buffer_occ, 'b-', linewidth=1)
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel('缓冲区占用 (包数)')
        axes[0, 0].set_title('接收端缓冲区占用情况')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 交付间隙
        axes[0, 1].plot(times, delivery_gaps, 'r-', linewidth=1)
        axes[0, 1].set_xlabel('时间 (秒)')
        axes[0, 1].set_ylabel('交付间隙 (包数)')
        axes[0, 1].set_title('因丢包导致的交付间隙')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 累计交付包数
        delivered = [d['delivered'] for d in self.timeline_data]
        axes[1, 0].plot(times, delivered, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('时间 (秒)')
        axes[1, 0].set_ylabel('累计交付包数')
        axes[1, 0].set_title('累计交付进度')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 重复包统计
        duplicates = [d['duplicates'] for d in self.timeline_data]
        axes[1, 1].plot(times, duplicates, 'm-', linewidth=1)
        axes[1, 1].set_xlabel('时间 (秒)')
        axes[1, 1].set_ylabel('累计重复包数')
        axes[1, 1].set_title('重复包累积情况')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'QUIC仿真结果 - 策略:{self.sender.redundancy_strategy}, '
                    f'冗余因子:{self.sender.redundancy_factor}, '
                    f'丢包率:{self.loss_rate}')
        plt.tight_layout()
        plt.show()

def compare_strategies():
    """比较不同策略的性能"""
    strategies = [
        ("none", 0.0, "无冗余"),
        ("fixed", 0.2, "20%冗余"),
        ("fixed", 0.4, "40%冗余"),
        ("fixed", 0.6, "60%冗余"),
        ("adaptive", 0.3, "自适应冗余")
    ]
    
    results = []
    
    for strategy, factor, name in strategies:
        print(f"\n{'='*50}")
        print(f"测试策略: {name}")
        
        simulator = QUICSimulator(
            duration=5.0,
            packet_rate=50.0,
            loss_rate=0.4,  # 高丢包率环境
            redundancy_strategy=strategy,
            redundancy_factor=factor
        )
        
        simulator.run()
        
        # 收集关键指标
        results.append({
            'name': name,
            'delivered': simulator.receiver.stats['delivered'],
            'buffer_max': simulator.receiver.stats['buffer_max'],
            'duplicates': simulator.receiver.stats['duplicates'],
            'effective_tput': simulator.receiver.stats['delivered'] / simulator.duration,
            'redundancy_ratio': simulator.sender.stats['redundancy_ratio']
        })
    
    # 打印比较结果
    print(f"\n{'='*60}")
    print("策略比较结果:")
    print(f"{'策略':<15} {'交付包数':<10} {'缓冲区峰值':<12} {'重复包数':<10} {'吞吐量':<10} {'冗余比例':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<15} "
              f"{result['delivered']:<10} "
              f"{result['buffer_max']:<12} "
              f"{result['duplicates']:<10} "
              f"{result['effective_tput']:<10.1f} "
              f"{result['redundancy_ratio']:<10.2%}")
    
    # 绘制比较图表
    names = [r['name'] for r in results]
    delivered = [r['delivered'] for r in results]
    buffer_max = [r['buffer_max'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 交付包数比较
    bars1 = axes[0].bar(names, delivered, color=['blue', 'green', 'orange', 'red', 'purple'])
    axes[0].set_xlabel('策略')
    axes[0].set_ylabel('交付包数')
    axes[0].set_title('不同策略下的交付包数')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 缓冲区峰值比较
    bars2 = axes[1].bar(names, buffer_max, color=['blue', 'green', 'orange', 'red', 'purple'])
    axes[1].set_xlabel('策略')
    axes[1].set_ylabel('缓冲区峰值 (包数)')
    axes[1].set_title('不同策略下的缓冲区堆积情况')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
    
    plt.suptitle('QUIC不同冗余策略性能比较 (丢包率40%)')
    plt.tight_layout()
    plt.show()
    
    return results

# 运行示例
if __name__ == "__main__":
    # 设置随机种子以便复现
    random.seed(42)
    
    print("=== QUIC协议冗余发包策略仿真 ===")
    
    # 单次仿真示例
    print("\n1. 运行单次仿真...")
    simulator = QUICSimulator(
        duration=5.0,           # 仿真5秒
        packet_rate=50.0,       # 每秒50个包
        loss_rate=0.4,          # 40%丢包率
        redundancy_strategy="fixed",  # 固定比例冗余
        redundancy_factor=0.4   # 40%冗余（每2.5个原始包发送1个冗余包）
    )
    
    simulator.run()
    simulator.plot_results()
    
    # 策略比较
    print("\n2. 运行策略比较...")
    compare_strategies()