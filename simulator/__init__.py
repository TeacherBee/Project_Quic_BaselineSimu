"""QUIC协议仿真器包"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "QUIC协议冗余发包策略仿真器"

# 导出主要类
from .config import SimulationConfig
from .packet import Packet, PacketStatus
from .redundancy import RedundancyController, RedundancyConfig
from .sender import QUICSender
from .receiver import QUICReceiver
from .network import NetworkLink, NetworkTopology
from .stats import StatisticsCollector
from .visualizer import QUICVisualizer
from .main import QUICSimulator, main, compare_strategies

__all__ = [
    'SimulationConfig',
    'Packet',
    'PacketStatus',
    'RedundancyController',
    'RedundancyConfig',
    'QUICSender',
    'QUICReceiver',
    'NetworkLink',
    'NetworkTopology',
    'StatisticsCollector',
    'QUICVisualizer',
    'QUICSimulator',
    'main',
    'compare_strategies'
]