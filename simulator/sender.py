"""发送端模块"""
import random
import heapq
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict

from packet import Packet, PacketStatus
from redundancy import RedundancyController
from config import SimulationConfig

class QUICSender:
    """QUIC发送端"""
    
    def __init__(self, config: SimulationConfig, redundancy_controller: Optional[RedundancyController] = None):
        self.config = config
        self.redundancy_controller = redundancy_controller
        
        # 状态变量
        self.seq_num = 0
        self.next_seq_to_send = 0
        self.last_acked_seq = -1
        
        # 数据结构
        self.inflight_packets: Dict[int, Packet] = {}      # 传输中的数据包
        self.retransmit_queue = []                         # 重传队列
        self.sent_packets: List[Packet] = []               # 已发送的数据包列表
        self.ack_received: Set[int] = set()                # 已确认的序列号
        
        # 拥塞控制
        self.cwnd = config.initial_cwnd
        self.ssthresh = config.max_cwnd
        self.rtt_samples = []
        self.rto = config.rto_initial
        self.timeout_queue = []  # (timeout_time, seq_num)
        
        # 统计
        self.stats = {
            'total_sent': 0,
            'original_sent': 0,
            'redundant_sent': 0,
            'retransmits': 0,
            'loss_detected': 0,
            'bytes_sent': 0,
            'cwnd_changes': [],
            'rtt_samples': []
        }
    
    def can_send(self) -> bool:
        """检查是否可以发送新包"""
        # 检查拥塞窗口和发送窗口
        inflight_count = len(self.inflight_packets)
        window_available = (self.next_seq_to_send - self.last_acked_seq) < self.config.window_size
        
        return (inflight_count < self.cwnd) and window_available
    
    def generate_packet(self, current_time: float, is_retransmit: bool = False) -> Optional[Packet]:
        """生成数据包"""
        if not self.can_send() and not is_retransmit:
            return None
        
        if is_retransmit:
            # 重传逻辑
            if not self.retransmit_queue:
                return None
            
            seq_num = heapq.heappop(self.retransmit_queue)
            if seq_num in self.inflight_packets:
                packet = self.inflight_packets[seq_num]
                packet.retransmit_count += 1
                packet.send_time = current_time
                packet.status = PacketStatus.RETRANSMITTED
                self.stats['retransmits'] += 1
                return packet
            return None
        
        # 生成新包
        packet = Packet(
            seq_num=self.seq_num,
            send_time=current_time,
            is_original=True,
            payload_size=self.config.packet_size,
            status=PacketStatus.IN_FLIGHT
        )
        
        # 添加到传输中
        self.inflight_packets[self.seq_num] = packet
        self.sent_packets.append(packet)
        
        # 设置超时
        timeout_time = current_time + self.rto
        heapq.heappush(self.timeout_queue, (timeout_time, self.seq_num))
        
        # 更新统计
        self.stats['original_sent'] += 1
        self.stats['total_sent'] += 1
        self.stats['bytes_sent'] += packet.payload_size
        
        # 生成冗余包（如果启用）
        redundant_packets = []
        if self.redundancy_controller:
            loss_rate = self.stats['loss_detected'] / max(1, self.stats['total_sent'])
            redundant_packets = self.redundancy_controller.generate_redundancy(
                packet, loss_rate, current_time
            )
            
            for red_packet in redundant_packets:
                red_packet.seq_num = self.seq_num  # 保持相同序列号
                self.sent_packets.append(red_packet)
                self.stats['redundant_sent'] += 1
                self.stats['bytes_sent'] += red_packet.payload_size
        
        self.seq_num += 1
        self.next_seq_to_send = self.seq_num
        
        return [packet] + redundant_packets if redundant_packets else [packet]
    
    def process_ack(self, seq_num: int, rtt_sample: float = None, current_time: float = None):
        """处理ACK确认"""
        if seq_num in self.ack_received:
            # 重复ACK，可能是快速重传的信号
            if seq_num in self.inflight_packets:
                # 触发快速重传
                self._schedule_retransmit(seq_num)
            return
        
        # 记录ACK
        self.ack_received.add(seq_num)
        
        # 更新RTT估计
        if rtt_sample and current_time:
            self._update_rtt(rtt_sample)
            
            # 从统计中移除RTT样本
            if seq_num in self.inflight_packets:
                packet = self.inflight_packets[seq_num]
                packet.ack_time = current_time
                
                # 拥塞控制：收到ACK，增加cwnd
                self._congestion_control_ack()
        
        # 从传输中移除
        if seq_num in self.inflight_packets:
            del self.inflight_packets[seq_num]
        
        # 更新最后确认的序列号
        if seq_num > self.last_acked_seq:
            self.last_acked_seq = seq_num
            
            # 检查是否有新的包可以确认
            while (self.last_acked_seq + 1) in self.ack_received:
                self.last_acked_seq += 1
    
    def _update_rtt(self, rtt_sample: float):
        """更新RTT估计"""
        self.rtt_samples.append(rtt_sample)
        
        # 简单平滑：使用指数加权移动平均
        alpha = 0.125
        beta = 0.25
        
        if len(self.rtt_samples) == 1:
            self.rtt_estimate = rtt_sample
            self.rtt_variance = rtt_sample / 2
        else:
            sample_error = abs(rtt_sample - self.rtt_estimate)
            self.rtt_estimate = (1 - alpha) * self.rtt_estimate + alpha * rtt_sample
            self.rtt_variance = (1 - beta) * self.rtt_variance + beta * sample_error
        
        # 计算RTO
        self.rto = max(
            self.config.rto_min,
            min(self.config.rto_max, self.rtt_estimate + 4 * self.rtt_variance)
        )
        
        self.stats['rtt_samples'].append((len(self.stats['rtt_samples']), self.rtt_estimate))
    
    def _congestion_control_ack(self):
        """拥塞控制：收到ACK时的处理"""
        if self.cwnd < self.ssthresh:
            # 慢启动阶段
            self.cwnd += 1
        else:
            # 拥塞避免阶段
            self.cwnd += 1.0 / self.cwnd
        
        self.cwnd = min(self.cwnd, self.config.max_cwnd)
        self.stats['cwnd_changes'].append((len(self.stats['cwnd_changes']), self.cwnd))
    
    def _congestion_control_loss(self):
        """拥塞控制：检测到丢包时的处理"""
        self.ssthresh = max(self.config.min_cwnd, self.cwnd / 2)
        self.cwnd = self.config.min_cwnd
        self.stats['cwnd_changes'].append((len(self.stats['cwnd_changes']), self.cwnd))
    
    def _schedule_retransmit(self, seq_num: int):
        """安排重传"""
        if seq_num not in self.retransmit_queue:
            heapq.heappush(self.retransmit_queue, seq_num)
    
    def check_timeouts(self, current_time: float) -> List[int]:
        """检查超时的包"""
        lost_packets = []
        
        while self.timeout_queue and self.timeout_queue[0][0] <= current_time:
            timeout_time, seq_num = heapq.heappop(self.timeout_queue)
            
            if seq_num in self.inflight_packets and seq_num not in self.ack_received:
                # 包超时，标记为丢失
                packet = self.inflight_packets[seq_num]
                if packet.retransmit_count < self.config.max_retransmits:
                    packet.status = PacketStatus.LOST
                    lost_packets.append(seq_num)
                    self.stats['loss_detected'] += 1
                    
                    # 安排重传
                    self._schedule_retransmit(seq_num)
                    
                    # 拥塞控制：检测到丢包
                    self._congestion_control_loss()
                else:
                    # 超过最大重传次数，放弃
                    del self.inflight_packets[seq_num]
        
        return lost_packets
    
    def get_stats(self) -> dict:
        """获取发送端统计信息"""
        stats = self.stats.copy()
        stats['inflight_count'] = len(self.inflight_packets)
        stats['retransmit_queue_size'] = len(self.retransmit_queue)
        stats['cwnd'] = self.cwnd
        stats['rto'] = self.rto
        
        if self.redundancy_controller:
            stats.update(self.redundancy_controller.get_stats())
        
        return stats