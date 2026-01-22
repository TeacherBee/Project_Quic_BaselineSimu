"""冗余控制模块"""
import random
from typing import List, Optional
from dataclasses import dataclass

from packet import Packet
from config import SimulationConfig

@dataclass
class RedundancyConfig:
    """冗余配置"""
    strategy: str = "none"        # 冗余策略
    factor: float = 0.0           # 冗余因子
    adaptive_threshold: float = 0.1  # 自适应阈值
    fec_enabled: bool = False     # 是否启用FEC
    fec_group_size: int = 4       # FEC组大小

class RedundancyController:
    """冗余控制器"""
    
    def __init__(self, config: RedundancyConfig):
        self.config = config
        self.stats = {
            'redundant_packets_generated': 0,
            'redundancy_bytes': 0,
            'redundancy_effectiveness': 0.0,
            'fec_groups_created': 0,
            'fec_recovery_attempts': 0,
            'fec_recovery_success': 0
        }
        self.redundancy_batch_id = 0
        self.packet_history = []  # 用于FEC的历史包记录
        
    def generate_redundancy(self, original_packet: Packet, 
                           loss_rate: float = 0.0,
                           current_time: float = 0.0) -> List[Packet]:
        """根据策略生成冗余包"""
        packets = []
        
        if self.config.strategy == "none" or self.config.factor <= 0:
            return packets
        
        elif self.config.strategy == "fixed":
            # 固定比例冗余
            return self._fixed_redundancy(original_packet, current_time)
        
        elif self.config.strategy == "adaptive":
            # 自适应冗余
            return self._adaptive_redundancy(original_packet, loss_rate, current_time)
        
        elif self.config.strategy == "fec" and self.config.fec_enabled:
            # FEC冗余
            return self._fec_redundancy(original_packet, current_time)
        
        elif self.config.strategy == "hybrid":
            # 混合冗余策略
            return self._hybrid_redundancy(original_packet, loss_rate, current_time)
        
        return packets
    
    def _fixed_redundancy(self, original_packet: Packet, 
                         current_time: float) -> List[Packet]:
        """固定比例冗余策略"""
        packets = []
        # 计算冗余包数量
        redundancy_count = int(original_packet.payload_size * self.config.factor / 100)
        
        for i in range(redundancy_count):
            redundant_packet = Packet(
                seq_num=original_packet.seq_num,
                send_time=current_time + (i+1)*0.001,
                is_original=False,
                is_redundant=True,
                redundancy_id=self.redundancy_batch_id,
                payload_size=original_packet.payload_size,
                status=original_packet.status
            )
            packets.append(redundant_packet)
            self.stats['redundant_packets_generated'] += 1
            self.stats['redundancy_bytes'] += original_packet.payload_size
        
        self.redundancy_batch_id += 1
        return packets
    
    def _adaptive_redundancy(self, original_packet: Packet, 
                           loss_rate: float, current_time: float) -> List[Packet]:
        """自适应冗余策略"""
        packets = []
        
        # 根据丢包率动态调整冗余量
        if loss_rate > self.config.adaptive_threshold:
            # 高丢包率：增加冗余
            redundancy_level = min(3, int(loss_rate / self.config.adaptive_threshold))
            for i in range(redundancy_level):
                redundant_packet = Packet(
                    seq_num=original_packet.seq_num,
                    send_time=current_time + (i+1)*0.002,
                    is_original=False,
                    is_redundant=True,
                    redundancy_id=self.redundancy_batch_id,
                    payload_size=original_packet.payload_size,
                    status=original_packet.status
                )
                packets.append(redundant_packet)
                self.stats['redundant_packets_generated'] += 1
                self.stats['redundancy_bytes'] += original_packet.payload_size
            
            self.redundancy_batch_id += 1
        
        return packets
    
    def _fec_redundancy(self, original_packet: Packet, 
                       current_time: float) -> List[Packet]:
        """FEC冗余策略（简化版）"""
        packets = []
        
        # 将包添加到历史中
        self.packet_history.append(original_packet)
        
        # 当收集到足够包时生成FEC冗余包
        if len(self.packet_history) >= self.config.fec_group_size:
            self.stats['fec_groups_created'] += 1
            
            # 生成简单的异或FEC包（简化实现）
            fec_packet = Packet(
                seq_num=-self.redundancy_batch_id,  # 负序列号表示FEC包
                send_time=current_time,
                is_original=False,
                is_redundant=True,
                redundancy_id=self.redundancy_batch_id,
                payload_size=original_packet.payload_size,
                status=original_packet.status
            )
            packets.append(fec_packet)
            self.stats['redundant_packets_generated'] += 1
            
            # 清空历史
            self.packet_history = []
            self.redundancy_batch_id += 1
        
        return packets
    
    def _hybrid_redundancy(self, original_packet: Packet,
                          loss_rate: float, current_time: float) -> List[Packet]:
        """混合冗余策略"""
        packets = []
        
        # 基础固定冗余
        packets.extend(self._fixed_redundancy(original_packet, current_time))
        
        # 根据丢包率添加自适应冗余
        if loss_rate > self.config.adaptive_threshold * 2:
            adaptive_packets = self._adaptive_redundancy(original_packet, loss_rate, current_time)
            packets.extend(adaptive_packets)
        
        return packets
    
    def update_statistics(self, recovered_packets: int = 0):
        """更新统计信息"""
        if self.stats['redundant_packets_generated'] > 0:
            self.stats['redundancy_effectiveness'] = (
                recovered_packets / self.stats['redundant_packets_generated']
            )
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return self.stats.copy()