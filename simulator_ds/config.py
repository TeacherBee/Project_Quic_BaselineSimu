"""仿真配置模块"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SimulationConfig:
    """仿真配置"""
    # 仿真参数
    duration: float = 10.0           # 仿真时长（秒）
    packet_rate: float = 100.0       # 发包速率（包/秒）
    packet_size: int = 1200          # 包大小（字节）
    
    # 网络参数
    loss_rate: float = 0.1           # 丢包率
    delay_mean: float = 0.1         # 平均延迟（秒）
    delay_std: float = 0.002         # 延迟标准差
    delay_min: float = 0.05         # 最小延迟
    delay_max: float = 0.3          # 最大延迟
    
    # 发送端参数
    window_size: int = 10            # 发送窗口大小
    initial_cwnd: int = 10           # 初始拥塞窗口
    max_cwnd: int = 64               # 最大拥塞窗口
    min_cwnd: int = 1                # 最小拥塞窗口
    
    # 冗余策略参数
    redundancy_strategy: str = "fixed"  # 冗余策略
    redundancy_factor: float = 0.3      # 冗余因子
    adaptive_threshold: float = 0.1     # 自适应阈值
    fec_enabled: bool = False          # 是否启用FEC
    fec_group_size: int = 4            # FEC组大小
    
    # 重传参数
    rto_initial: float = 0.1          # 初始RTO
    rto_min: float = 0.01             # 最小RTO
    rto_max: float = 1.0              # 最大RTO
    max_retransmits: int = 3          # 最大重传次数
    
    # 接收端参数
    recv_buffer_size: int = 100       # 接收缓冲区大小
    max_out_of_order: int = 50        # 最大乱序容忍
    
    # 统计参数
    stats_interval: float = 0.1       # 统计间隔
    log_interval: float = 1.0         # 日志间隔
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
    
    def update(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)