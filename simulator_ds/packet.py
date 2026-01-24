"""数据包定义模块"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class PacketStatus(Enum):
    """数据包状态枚举"""
    IN_FLIGHT = 1      # 传输中
    DELIVERED = 2      # 已交付
    LOST = 3           # 丢失
    DUPLICATE = 4      # 重复包
    ACKED = 5          # 已确认
    RETRANSMITTED = 6  # 重传包

@dataclass
class Packet:
    """数据包类"""
    seq_num: int           # 序列号
    send_time: float       # 发送时间
    is_original: bool      # 是否是原始包
    payload_size: int = 1200  # 有效载荷大小（字节）
    is_redundant: bool = False  # 是否是冗余包
    redundancy_id: int = 0      # 冗余批次ID
    retransmit_count: int = 0   # 重传次数
    status: PacketStatus = PacketStatus.IN_FLIGHT
    ack_time: Optional[float] = None  # 确认时间
    
    def __lt__(self, other):
        return self.seq_num < other.seq_num
    
    def __repr__(self):
        return (f"Packet(seq={self.seq_num}, "
                f"time={self.send_time:.3f}, "
                f"original={self.is_original}, "
                f"redundant={self.is_redundant}, "
                f"status={self.status.name})")