import re
from collections import defaultdict

# --- 新增的冗余管理类 ---
class RedundancyManager:
    """
    负责管理所有冗余相关逻辑。
    它接收原始包ID (sn)，并返回需要发送的物理包列表。
    """
    def __init__(self, redundancy_mode, k_rep=None):
        self.mode = redundancy_mode
        self.k_rep = k_rep
        self.xor_groups = {} # For xor_k_1: maps group_id -> {pkts_sent, pkts_arrived}

        # 验证并设置模式参数
        if redundancy_mode == 'none':
            self.mode = 'none'
        elif redundancy_mode == 'replicate':
            self.mode = 'replicate'
            self.k_rep = 1
        elif redundancy_mode.startswith('replicate_'):
            match = re.match(r'replicate_(\d+)_1', redundancy_mode)
            if not match:
                raise ValueError("Use 'replicate_k_1', e.g., 'replicate_4_1'")
            self.k_rep = int(match.group(1))
            if self.k_rep <= 0:
                raise ValueError("k must be positive")
            self.mode = 'replicate_k_1'
        elif redundancy_mode.startswith('xor_'):
            match = re.match(r'xor_(\d+)_1', redundancy_mode)
            if not match:
                raise ValueError("Use 'xor_k_1', e.g., 'xor_4_1'")
            self.k_rep = int(match.group(1))
            if self.k_rep <= 0:
                raise ValueError("k must be positive")
            self.mode = 'xor_k_1'
        else:
            raise ValueError("Redundancy must be: 'none', 'replicate', 'replicate_k_1' (e.g., 'replicate_4_1'), or 'xor_k_1' (e.g., 'xor_4_1')")

    def get_packets_to_send(self, original_sn):
        """
        根据冗余模式，为给定的原始包ID (original_sn) 返回需要发送的物理包列表。
        列表中的每个元素是一个字典，包含 'sn' 和 'is_redundant' 标识。
        """
        if self.mode == 'none':
            return [{'sn': original_sn, 'is_redundant': False}]
        elif self.mode == 'replicate':
            return [
                {'sn': original_sn, 'is_redundant': False},
                {'sn': original_sn, 'is_redundant': True} # Replicated packet
            ]
        elif self.mode == 'replicate_k_1':
            packets = [{'sn': original_sn, 'is_redundant': False}]
            # 检查是否是组的最后一个包或最后一个包
            if (original_sn + 1) % self.k_rep == 0 or original_sn == self.total_original_packets - 1:
                 packets.append({'sn': original_sn, 'is_redundant': True}) # Replicated packet for group
            return packets
        elif self.mode == 'xor_k_1':
            packets = [{'sn': original_sn, 'is_redundant': False}]
            # 检查是否是组的最后一个包或最后一个包
            if (original_sn + 1) % self.k_rep == 0 or original_sn == self.total_original_packets - 1:
                 # XOR包的标识符，这里使用负数sn作为约定
                 xor_sn = -(original_sn)
                 packets.append({'sn': xor_sn, 'is_redundant': True, 'type': 'xor'})
            return packets
        else:
            # Should not happen due to validation in __init__
            return [{'sn': original_sn, 'is_redundant': False}]

    def set_total_packets(self, num_pkts):
        """告知管理器总共有多少个原始包需要发送，以便进行边界检查。"""
        self.total_original_packets = num_pkts

    def get_effective_send_interval_factor(self, base_interval, num_pkts):
        """
        计算基于冗余开销的有效发送间隔因子。
        返回一个乘数，最终间隔为 base_interval * factor。
        """
        if self.mode == 'none':
            return 1.0
        elif self.mode == 'replicate':
            return 2.0 # 1 orig + 1 red per packet
        elif self.mode == 'replicate_k_1' or self.mode == 'xor_k_1':
            # 每 k 个原始包 -> k + 1 个物理包
            return (self.k_rep + 1) / self.k_rep
        else:
            return 1.0 # Default case
