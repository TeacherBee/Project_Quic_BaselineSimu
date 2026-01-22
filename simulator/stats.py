"""统计和监控模块"""
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import csv
from datetime import datetime

@dataclass
class TimeSeriesPoint:
    """时间序列数据点"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class StatisticsCollector:
    """统计收集器"""
    
    def __init__(self, simulation_id: str = None):
        self.simulation_id = simulation_id or f"sim_{int(time.time())}"
        self.start_time = time.time()
        
        # 时间序列数据
        self.time_series = {
            'buffer_occupancy': [],
            'delivery_gap': [],
            'throughput': [],
            'loss_rate': [],
            'rtt': [],
            'cwnd': [],
            'redundancy_ratio': [],
            'queue_length': []
        }
        
        # 累计统计
        self.cumulative_stats = {
            'total_packets_sent': 0,
            'total_packets_delivered': 0,
            'total_bytes_sent': 0,
            'total_bytes_delivered': 0,
            'total_retransmits': 0,
            'total_duplicates': 0,
            'simulation_duration': 0.0
        }
        
        # 事件日志
        self.event_log = []
    
    def record_time_series(self, metric_name: str, timestamp: float, 
                          value: float, metadata: Dict = None):
        """记录时间序列数据"""
        if metric_name in self.time_series:
            point = TimeSeriesPoint(timestamp, value, metadata or {})
            self.time_series[metric_name].append(point)
    
    def record_event(self, event_type: str, description: str, 
                    timestamp: float = None, details: Dict = None):
        """记录事件"""
        if timestamp is None:
            timestamp = time.time() - self.start_time
        
        event = {
            'timestamp': timestamp,
            'type': event_type,
            'description': description,
            'details': details or {}
        }
        self.event_log.append(event)
    
    def update_cumulative_stats(self, stats: Dict[str, Any]):
        """更新累计统计"""
        for key, value in stats.items():
            if key in self.cumulative_stats:
                if isinstance(value, (int, float)):
                    self.cumulative_stats[key] += value
                else:
                    self.cumulative_stats[key] = value
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """获取汇总统计"""
        summary = self.cumulative_stats.copy()
        
        # 计算衍生指标
        if summary['total_packets_sent'] > 0:
            summary['overall_loss_rate'] = (
                1 - (summary['total_packets_delivered'] / summary['total_packets_sent'])
            )
        
        if summary['simulation_duration'] > 0:
            summary['avg_throughput_mbps'] = (
                summary['total_bytes_delivered'] * 8 / 
                summary['simulation_duration'] / 1_000_000
            )
        
        # 添加时间序列摘要
        for metric, points in self.time_series.items():
            if points:
                values = [p.value for p in points]
                summary[f'{metric}_avg'] = sum(values) / len(values)
                summary[f'{metric}_max'] = max(values)
                summary[f'{metric}_min'] = min(values)
                summary[f'{metric}_std'] = self._calculate_std(values)
        
        return summary
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def export_to_csv(self, filename: str):
        """导出时间序列数据到CSV"""
        # 合并所有时间序列
        all_points = {}
        
        for metric, points in self.time_series.items():
            for point in points:
                timestamp = point.timestamp
                if timestamp not in all_points:
                    all_points[timestamp] = {'timestamp': timestamp}
                all_points[timestamp][metric] = point.value
        
        # 写入CSV
        if all_points:
            timestamps = sorted(all_points.keys())
            fieldnames = ['timestamp'] + list(self.time_series.keys())
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for ts in timestamps:
                    writer.writerow(all_points[ts])
    
    def export_events(self, filename: str):
        """导出事件日志"""
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'type', 'description', 'details']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in self.event_log:
                row = event.copy()
                row['details'] = json.dumps(row['details'])
                writer.writerow(row)
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary_stats()
        
        print("\n" + "="*60)
        print("仿真统计摘要")
        print("="*60)
        
        print(f"\n基本信息:")
        print(f"  仿真ID: {self.simulation_id}")
        print(f"  仿真时长: {summary['simulation_duration']:.2f}秒")
        
        print(f"\n流量统计:")
        print(f"  发送包数: {summary['total_packets_sent']}")
        print(f"  交付包数: {summary['total_packets_delivered']}")
        print(f"  总丢包率: {summary.get('overall_loss_rate', 0):.2%}")
        print(f"  发送字节: {summary['total_bytes_sent']:,} bytes")
        print(f"  交付字节: {summary['total_bytes_delivered']:,} bytes")
        print(f"  平均吞吐: {summary.get('avg_throughput_mbps', 0):.2f} Mbps")
        
        print(f"\n性能指标:")
        print(f"  缓冲区占用峰值: {summary.get('buffer_occupancy_max', 0):.1f}")
        print(f"  平均交付间隙: {summary.get('delivery_gap_avg', 0):.1f}")
        print(f"  平均RTT: {summary.get('rtt_avg', 0)*1000:.1f} ms")
        print(f"  重传次数: {summary['total_retransmits']}")
        print(f"  重复包数: {summary['total_duplicates']}")
        
        print(f"\n拥塞控制:")
        print(f"  平均CWND: {summary.get('cwnd_avg', 0):.1f}")
        print(f"  CWND峰值: {summary.get('cwnd_max', 0):.1f}")
        
        print("="*60)