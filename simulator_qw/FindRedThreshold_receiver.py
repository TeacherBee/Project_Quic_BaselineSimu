# receiver.py
import heapq
import threading
import time
from collections import defaultdict

# 全局通信队列（由 sender 创建并传入）
net_to_sender = None  # ACK 队列: put(ack_msg)
net_to_receiver = None  # 数据包队列: get() -> (arrival_time, sn)

class Receiver:
    def __init__(self):
        self.next_expected = 0
        self.received = set()      # 所有收到的 SN
        self.delivered = 0         # 已交付数量
        self.buffer_size_history = []  # (time, buffer_size)
        self.last_event_time = 0.0

    def send_ack(self, cumulative, selective):
        ack_msg = {
            'cumulative': cumulative,
            'selective': list(selective),
            'time': self.last_event_time
        }
        net_to_sender.put(ack_msg)

    def run(self):
        receiver_buffer = set()
        last_ack_time = -1

        while True:
            try:
                arrival_time, sn = net_to_receiver.get(timeout=0.1)
                if arrival_time is None:  # 停止信号
                    break
            except:
                continue

            self.last_event_time = arrival_time

            if sn < self.next_expected:
                continue  # duplicate

            self.received.add(sn)

            # 尝试交付
            if sn == self.next_expected:
                self.delivered += 1
                self.next_expected += 1
                while self.next_expected in receiver_buffer:
                    receiver_buffer.remove(self.next_expected)
                    self.delivered += 1
                    self.next_expected += 1

            elif sn > self.next_expected:
                receiver_buffer.add(sn)

            # 记录 buffer 大小（用于平均排队）
            current_buffer = len(receiver_buffer)
            self.buffer_size_history.append((arrival_time, current_buffer))

            # 每 10ms 或关键事件发 ACK（避免太频繁）
            if arrival_time - last_ack_time > 0.01 or sn == self.next_expected - 1:
                # Cumulative ACK = next_expected - 1
                # Selective ACK = 所有 > cumulative 的 received 包
                cumulative = self.next_expected - 1
                selective = {s for s in self.received if s > cumulative}
                self.send_ack(cumulative, selective)
                last_ack_time = arrival_time

        # 返回指标
        if not self.buffer_size_history:
            avg_queue = 0.0
        else:
            # 时间加权平均
            total = 0.0
            prev_t = self.buffer_size_history[0][0]
            for t, size in self.buffer_size_history:
                total += size * (t - prev_t)
                prev_t = t
            avg_queue = total / (prev_t - self.buffer_size_history[0][0]) if prev_t > self.buffer_size_history[0][0] else 0.0

        return {
            'delivered': self.delivered,
            'avg_queue_length': avg_queue,
            'last_time': self.last_event_time
        }

def start_receiver(_net_to_sender, _net_to_receiver):
    global net_to_sender, net_to_receiver
    net_to_sender = _net_to_sender
    net_to_receiver = _net_to_receiver
    receiver = Receiver()
    return receiver.run()