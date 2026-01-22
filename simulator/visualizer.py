"""数据可视化模块"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Dict, List, Any, Optional
from stats import StatisticsCollector

class QUICVisualizer:
    """QUIC仿真可视化器"""
    
    def __init__(self, stats_collector: StatisticsCollector):
        self.stats = stats_collector
        self.figure = None
    
    def plot_comprehensive(self, save_path: Optional[str] = None):
        """绘制综合可视化图表"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # 1. 缓冲区占用和交付间隙
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_buffer_and_gap(ax1)
        
        # 2. 吞吐量和丢包率
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_throughput_and_loss(ax2)
        
        # 3. RTT和CWND变化
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_rtt_and_cwnd(ax3)
        
        # 4. 冗余效率
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_redundancy_efficiency(ax4)
        
        # 5. 包交付时序图
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_packet_delivery_timeline(ax5)
        
        # 6. 队列长度和网络延迟
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_queue_and_delay(ax6)
        
        # 7. 统计摘要表格
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_summary_table(ax7)
        
        plt.suptitle('QUIC协议仿真综合分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def _plot_buffer_and_gap(self, ax):
        """绘制缓冲区占用和交付间隙"""
        buffer_points = self.stats.time_series.get('buffer_occupancy', [])
        gap_points = self.stats.time_series.get('delivery_gap', [])
        
        if buffer_points:
            times = [p.timestamp for p in buffer_points]
            values = [p.value for p in buffer_points]
            ax.plot(times, values, 'b-', linewidth=1.5, label='缓冲区占用')
            ax.fill_between(times, 0, values, alpha=0.3, color='blue')
        
        if gap_points:
            times = [p.timestamp for p in gap_points]
            values = [p.value for p in gap_points]
            ax.plot(times, values, 'r--', linewidth=1.5, label='交付间隙')
        
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('包数')
        ax.set_title('接收端缓冲区占用 vs 交付间隙')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_throughput_and_loss(self, ax):
        """绘制吞吐量和丢包率"""
        throughput_points = self.stats.time_series.get('throughput', [])
        loss_points = self.stats.time_series.get('loss_rate', [])
        
        if throughput_points and len(throughput_points) > 10:
            # 计算移动平均
            window = min(10, len(throughput_points) // 4)
            times = [p.timestamp for p in throughput_points]
            values = [p.value for p in throughput_points]
            
            # 简单移动平均
            ma_values = []
            for i in range(len(values)):
                start = max(0, i - window)
                end = i + 1
                ma_values.append(np.mean(values[start:end]))
            
            ax.plot(times, ma_values, 'g-', linewidth=2, label='吞吐量')
            ax.set_ylabel('吞吐量 (Mbps)', color='green')
            ax.tick_params(axis='y', labelcolor='green')
        
        if loss_points:
            ax2 = ax.twinx()
            times = [p.timestamp for p in loss_points]
            values = [p.value for p in loss_points]
            ax2.plot(times, values, 'r-', linewidth=1, alpha=0.7, label='丢包率')
            ax2.set_ylabel('丢包率', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, 1)
        
        ax.set_xlabel('时间 (秒)')
        ax.set_title('吞吐量和丢包率')
        ax.grid(True, alpha=0.3)
    
    def _plot_rtt_and_cwnd(self, ax):
        """绘制RTT和CWND变化"""
        rtt_points = self.stats.time_series.get('rtt', [])
        cwnd_points = self.stats.time_series.get('cwnd', [])
        
        if rtt_points:
            ax2 = ax.twinx()
            times = [p.timestamp for p in rtt_points]
            values = [p.value * 1000 for p in rtt_points]  # 转换为ms
            ax2.plot(times, values, 'orange', linewidth=1.5, label='RTT')
            ax2.set_ylabel('RTT (ms)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        
        if cwnd_points:
            times = [p.timestamp for p in cwnd_points]
            values = [p.value for p in cwnd_points]
            ax.plot(times, values, 'purple', linewidth=2, label='CWND')
            ax.set_ylabel('CWND (包数)', color='purple')
            ax.tick_params(axis='y', labelcolor='purple')
        
        ax.set_xlabel('时间 (秒)')
        ax.set_title('RTT和拥塞窗口变化')
        ax.grid(True, alpha=0.3)
    
    def _plot_redundancy_efficiency(self, ax):
        """绘制冗余效率"""
        redundancy_points = self.stats.time_series.get('redundancy_ratio', [])
        
        if redundancy_points:
            times = [p.timestamp for p in redundancy_points]
            values = [p.value * 100 for p in redundancy_points]  # 转换为百分比
            
            ax.bar(times, values, width=0.1, alpha=0.7, color='teal')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('冗余比例 (%)')
            ax.set_title('冗余包比例变化')
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_packet_delivery_timeline(self, ax):
        """绘制包交付时序图"""
        # 这是一个简化版本，实际需要从事件日志中提取
        events = self.stats.event_log
        
        if events:
            delivery_events = [e for e in events if 'deliver' in e['type'].lower()]
            loss_events = [e for e in events if 'loss' in e['type'].lower()]
            
            if delivery_events:
                times = [e['timestamp'] for e in delivery_events[:100]]  # 限制数量
                seq_nums = [e['details'].get('seq_num', i) for i, e in enumerate(delivery_events[:100])]
                
                ax.scatter(times, seq_nums, s=10, alpha=0.6, color='green', label='交付')
            
            if loss_events:
                times = [e['timestamp'] for e in loss_events[:50]]
                seq_nums = [e['details'].get('seq_num', i) for i, e in enumerate(loss_events[:50])]
                
                ax.scatter(times, seq_nums, s=30, alpha=0.8, color='red', 
                          marker='x', label='丢失')
            
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('序列号')
            ax.set_title('包交付时序图')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_queue_and_delay(self, ax):
        """绘制队列长度和网络延迟"""
        queue_points = self.stats.time_series.get('queue_length', [])
        
        if queue_points:
            times = [p.timestamp for p in queue_points]
            values = [p.value for p in queue_points]
            
            ax.plot(times, values, 'brown', linewidth=1.5, label='队列长度')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('队列长度 (包数)')
            ax.set_title('网络队列长度')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
    
    def _plot_summary_table(self, ax):
        """绘制统计摘要表格"""
        ax.axis('tight')
        ax.axis('off')
        
        summary = self.stats.get_summary_stats()
        
        # 选择关键指标
        key_metrics = {
            '仿真时长': f"{summary.get('simulation_duration', 0):.1f}秒",
            '发送包数': f"{summary.get('total_packets_sent', 0):,}",
            '交付包数': f"{summary.get('total_packets_delivered', 0):,}",
            '总丢包率': f"{summary.get('overall_loss_rate', 0):.2%}",
            '平均吞吐': f"{summary.get('avg_throughput_mbps', 0):.2f} Mbps",
            '缓冲区峰值': f"{summary.get('buffer_occupancy_max', 0):.1f}",
            '平均RTT': f"{summary.get('rtt_avg', 0)*1000:.1f} ms",
            '重传次数': f"{summary.get('total_retransmits', 0):,}"
        }
        
        # 创建表格
        table_data = [[k, v] for k, v in key_metrics.items()]
        table = ax.table(cellText=table_data, 
                        colLabels=['指标', '数值'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.4, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # 设置样式
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[i, j]
                if i == 0:
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                elif i % 2 == 1:
                    cell.set_facecolor('#f0f0f0')