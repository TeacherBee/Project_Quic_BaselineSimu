"""接收端模块"""
from typing import List, Dict, Set, Optional, Tuple
from collections import deque, defaultdict
import heapq

from packet import Packet, PacketStatus
from config import SimulationConfig

class QUICReceiver:
    """QUIC接收端"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # 接收状态
        self.next_expected_seq = 0
        self.highest_received_seq = -1
        self.last_delivered_seq = -1
        
        # 数据结构
        self.receive_buffer: Dict[int, Packet] = {}  # 接收缓冲区
        self.delivered_packets: List[Packet] = []     # 已交付的数据包
        self.duplicate_packets: List[Packet] = []     # 重复包
        self.ack_queue = []                           # 待发送的ACK队列
        
        # 乱序处理
        self.out_of_order_gaps = []                   # 乱序间隙
        
        # 统计
        self.stats = {
            'total_received': 0,
            'delivered': 0,
            'duplicates': 0,
            'out_of_order': 0,
            'buffer_max': 0,
            'buffer_occupancy': [],    # (time, occupancy)
            'delivery_gaps': [],       # (time, gap_size)
            'ack_sent': 0,
            'bytes_received': 0
        }
        
        # FEC恢复（如果启用）
        self.fec_buffer = {}
    
    def receive_packet(self, packet: Packet, current_time: float) -> bool:
        """接收一个数据包"""
        self.stats['total_received'] += 1
        self.stats['bytes_received'] += packet.payload_size
        
        # 更新最高接收序列号
        if packet.seq_num > self.highest_received_seq:
            self.highest_received_seq = packet.seq_num
        
        # 检查是否是重复包
        if packet.seq_num in self.receive_buffer:
            self.duplicate_packets.append(packet)
            self.stats['duplicates'] += 1
            packet.status = PacketStatus.DUPLICATE
            return False
        
        # 检查是否已经交付过（快速路径）
        if packet.seq_num < self.next_expected_seq:
            self.duplicate_packets.append(packet)
            self.stats['duplicates'] += 1
            packet.status = PacketStatus.DUPLICATE
            return False
        
        # 存储到接收缓冲区
        self.receive_buffer[packet.seq_num] = packet
        
        # 更新缓冲区占用统计
        buffer_size = len(self.receive_buffer)
        self.stats['buffer_occupancy'].append((current_time, buffer_size))
        self.stats['buffer_max'] = max(self.stats['buffer_max'], buffer_size)
        
        # 检查乱序
        if packet.seq_num != self.next_expected_seq:
            self.stats['out_of_order'] += 1
            gap_start = self.next_expected_seq
            gap_end = packet.seq_num
            missing_count = 0
            
            # 计算缺失的包数
            for seq in range(gap_start, gap_end):
                if seq not in self.receive_buffer:
                    missing_count += 1
            
            if missing_count > 0:
                self.out_of_order_gaps.append((current_time, missing_count))
                self.stats['delivery_gaps'].append((current_time, missing_count))
        
        # 生成ACK
        self._generate_ack(packet.seq_num, current_time)
        
        # 尝试交付数据包
        delivered = self._try_deliver(current_time)
        
        return delivered
    
    def _try_deliver(self, current_time: float) -> bool:
        """尝试按序交付数据包"""
        delivered = False
        
        while self.next_expected_seq in self.receive_buffer:
            packet = self.receive_buffer.pop(self.next_expected_seq)
            packet.status = PacketStatus.DELIVERED
            self.delivered_packets.append(packet)
            self.last_delivered_seq = self.next_expected_seq
            self.next_expected_seq += 1
            self.stats['delivered'] += 1
            delivered = True
        
        return delivered
    
    def _generate_ack(self, seq_num: int, current_time: float):
        """生成ACK"""
        # 检查是否需要发送ACK
        # 简单的ACK策略：每收到2个包或收到乱序包时发送ACK
        ack_needed = False
        
        if len(self.ack_queue) == 0:
            ack_needed = True
        elif seq_num != self.ack_queue[-1] + 1:  # 乱序
            ack_needed = True
        elif len(self.ack_queue) >= 2:  # 收到连续2个包
            ack_needed = True
        
        if ack_needed:
            # 记录ACK信息
            ack_info = {
                'ack_num': seq_num,
                'time': current_time,
                'delivered': self.stats['delivered'],
                'buffer_size': len(self.receive_buffer)
            }
            heapq.heappush(self.ack_queue, (current_time, ack_info))
            self.stats['ack_sent'] += 1
    
    def get_next_ack(self, current_time: float) -> Optional[dict]:
        """获取下一个要发送的ACK"""
        if not self.ack_queue:
            return None
        
        # 检查是否有ACK需要发送
        if self.ack_queue[0][0] <= current_time:
            _, ack_info = heapq.heappop(self.ack_queue)
            return ack_info
        
        return None
    
    def get_delivery_gap(self) -> Tuple[int, int]:
        """获取当前交付间隙信息"""
        if not self.receive_buffer:
            return 0, 0
        
        # 计算当前缺失的包数
        missing_count = 0
        current_seq = self.next_expected_seq
        
        # 找出缓冲区中的最大序列号
        if self.receive_buffer:
            max_buffered = max(self.receive_buffer.keys())
            for seq in range(current_seq, max_buffered + 1):
                if seq not in self.receive_buffer:
                    missing_count += 1
        
        # 计算间隙大小
        gap_size = 0
        if self.receive_buffer:
            min_buffered = min(self.receive_buffer.keys())
            if min_buffered > current_seq:
                gap_size = min_buffered - current_seq
        
        return missing_count, gap_size
    
    def get_stats(self) -> dict:
        """获取接收端统计信息"""
        stats = self.stats.copy()
        stats['buffer_current'] = len(self.receive_buffer)
        stats['next_expected_seq'] = self.next_expected_seq
        stats['last_delivered_seq'] = self.last_delivered_seq
        
        # 计算交付延迟（如果数据足够）
        if self.delivered_packets:
            total_delay = 0
            for packet in self.delivered_packets:
                if hasattr(packet, 'ack_time') and packet.ack_time:
                    total_delay += packet.ack_time - packet.send_time
            stats['avg_delivery_delay'] = total_delay / len(self.delivered_packets)
        
        return stats