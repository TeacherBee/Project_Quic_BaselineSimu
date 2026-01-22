"""网络链路模块"""
import random
import heapq
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from packet import Packet
from config import SimulationConfig

@dataclass
class LinkStats:
    """链路统计"""
    packets_sent: int = 0
    packets_delivered: int = 0
    packets_dropped: int = 0
    bytes_sent: int = 0
    bytes_delivered: int = 0
    avg_delay: float = 0.0
    max_delay: float = 0.0
    min_delay: float = float('inf')
    delay_samples: List[float] = None
    
    def __post_init__(self):
        if self.delay_samples is None:
            self.delay_samples = []
    
    def record_delivery(self, delay: float, packet_size: int):
        """记录包交付"""
        self.packets_delivered += 1
        self.bytes_delivered += packet_size
        self.delay_samples.append(delay)
        
        self.avg_delay = sum(self.delay_samples) / len(self.delay_samples)
        self.max_delay = max(self.max_delay, delay)
        self.min_delay = min(self.min_delay, delay)

class NetworkLink:
    """网络链路模拟"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # 链路状态
        self.packet_queue = []  # (arrival_time, packet)
        self.dropped_packets: List[Packet] = []
        
        # 统计
        self.stats = LinkStats()
        
        # 延迟模型参数
        self.delay_history = []
        self.jitter = 0.0
        self.loss_burst_state = False
        self.loss_burst_length = 0
    
    def send_packet(self, packet: Packet, send_time: float) -> Optional[float]:
        """发送数据包到链路"""
        self.stats.packets_sent += 1
        self.stats.bytes_sent += packet.payload_size
        
        # 模拟丢包
        if self._should_drop_packet():
            packet.status = PacketStatus.LOST
            self.dropped_packets.append(packet)
            self.stats.packets_dropped += 1
            return None
        
        # 计算延迟（考虑拥塞和抖动）
        delay = self._calculate_delay(send_time)
        
        # 计算到达时间
        arrival_time = send_time + delay
        
        # 添加到队列（按到达时间排序）
        heapq.heappush(self.packet_queue, (arrival_time, packet))
        
        return arrival_time
    
    def _should_drop_packet(self) -> bool:
        """决定是否丢弃包（模拟丢包）"""
        # 简单的丢包模型
        if self.loss_burst_state:
            # 丢包突发状态
            self.loss_burst_length -= 1
            if self.loss_burst_length <= 0:
                self.loss_burst_state = False
            return True
        
        # 随机丢包
        if random.random() < self.config.loss_rate:
            # 开始丢包突发
            if random.random() < 0.3:  # 30%的概率开始突发
                self.loss_burst_state = True
                self.loss_burst_length = random.randint(2, 5)
            return True
        
        return False
    
    def _calculate_delay(self, send_time: float) -> float:
        """计算包延迟"""
        # 基础延迟
        base_delay = random.normalvariate(self.config.delay_mean, self.config.delay_std)
        
        # 队列延迟（模拟拥塞）
        queue_delay = len(self.packet_queue) * 0.001  # 每个排队包增加1ms
        
        # 抖动
        jitter = random.uniform(-self.jitter, self.jitter)
        
        # 总延迟
        total_delay = base_delay + queue_delay + jitter
        
        # 限制延迟范围
        total_delay = max(self.config.delay_min, 
                         min(self.config.delay_max, total_delay))
        
        # 更新抖动估计
        if self.stats.delay_samples:
            recent_delays = self.stats.delay_samples[-10:]
            if recent_delays:
                self.jitter = max(0.001, np.std(recent_delays))
        
        return total_delay
    
    def get_next_packet(self, current_time: float) -> Optional[Tuple[float, Packet]]:
        """获取下一个到达的数据包"""
        if not self.packet_queue:
            return None
        
        # 检查是否有包已经到达
        if self.packet_queue[0][0] <= current_time:
            arrival_time, packet = heapq.heappop(self.packet_queue)
            
            # 记录统计
            delay = arrival_time - packet.send_time
            self.stats.record_delivery(delay, packet.payload_size)
            
            return arrival_time, packet
        
        return None
    
    def get_queue_size(self) -> int:
        """获取当前队列大小"""
        return len(self.packet_queue)
    
    def get_stats(self) -> dict:
        """获取链路统计信息"""
        stats_dict = {
            'packets_sent': self.stats.packets_sent,
            'packets_delivered': self.stats.packets_delivered,
            'packets_dropped': self.stats.packets_dropped,
            'loss_rate': self.stats.packets_dropped / max(1, self.stats.packets_sent),
            'bytes_sent': self.stats.bytes_sent,
            'bytes_delivered': self.stats.bytes_delivered,
            'avg_delay': self.stats.avg_delay,
            'max_delay': self.stats.max_delay,
            'min_delay': self.stats.min_delay if self.stats.min_delay != float('inf') else 0.0,
            'queue_size': self.get_queue_size(),
            'jitter': self.jitter
        }
        
        return stats_dict
    
    def reset(self):
        """重置链路状态"""
        self.packet_queue = []
        self.dropped_packets = []
        self.stats = LinkStats()
        self.loss_burst_state = False
        self.loss_burst_length = 0

class NetworkTopology:
    """网络拓扑（支持多链路）"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.links = []
        
        # 创建默认的单链路
        self.default_link = NetworkLink(config)
        self.links.append(self.default_link)
        
        # 统计
        self.stats = {
            'total_packets': 0,
            'total_delay': 0.0,
            'link_utilization': []
        }
    
    def send_packet(self, packet: Packet, send_time: float, link_idx: int = 0) -> Optional[float]:
        """通过指定链路发送包"""
        if 0 <= link_idx < len(self.links):
            link = self.links[link_idx]
            return link.send_packet(packet, send_time)
        return None
    
    def get_next_packet(self, current_time: float) -> Optional[Tuple[float, Packet]]:
        """从所有链路中获取下一个到达的包"""
        next_packets = []
        
        for link in self.links:
            result = link.get_next_packet(current_time)
            if result:
                next_packets.append(result)
        
        if not next_packets:
            return None
        
        # 返回最早到达的包
        return min(next_packets, key=lambda x: x[0])
    
    def get_stats(self) -> dict:
        """获取总体网络统计"""
        total_stats = {
            'total_packets_sent': 0,
            'total_packets_delivered': 0,
            'total_packets_dropped': 0,
            'total_bytes_sent': 0,
            'total_bytes_delivered': 0,
            'avg_delay_across_links': 0.0,
            'links': []
        }
        
        total_delay = 0
        link_count = 0
        
        for i, link in enumerate(self.links):
            link_stats = link.get_stats()
            total_stats['links'].append({
                'link_id': i,
                **link_stats
            })
            
            total_stats['total_packets_sent'] += link_stats['packets_sent']
            total_stats['total_packets_delivered'] += link_stats['packets_delivered']
            total_stats['total_packets_dropped'] += link_stats['packets_dropped']
            total_stats['total_bytes_sent'] += link_stats['bytes_sent']
            total_stats['total_bytes_delivered'] += link_stats['bytes_delivered']
            
            if link_stats['packets_delivered'] > 0:
                total_delay += link_stats['avg_delay'] * link_stats['packets_delivered']
                link_count += 1
        
        if total_stats['total_packets_delivered'] > 0:
            total_stats['avg_delay_across_links'] = total_delay / total_stats['total_packets_delivered']
        
        return total_stats