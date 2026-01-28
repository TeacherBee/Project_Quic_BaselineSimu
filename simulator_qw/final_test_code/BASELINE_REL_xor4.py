import heapq
import random
import sys
import datetime
from redundancy import RedundancyManager

useLog = False

class Config:
    PKT_SIZE = 1250         # bytes
    FLOW_SIZE = 1 * 1024 * 1024  # fallback for old mode
    RTO = 0.5               # Timeout

    # 链路 A (快路径)
    B_A = 100e6             # 100 Mbps
    RTT_A = 0.2            # 50 ms one-way → 100ms RTT? But you use /2 later, so keep as is
    LOSS_RATE_A = 0.05

    # 链路 B (慢路径)
    B_B = 20e6              # 20 Mbps
    RTT_B = 0.4            # 150 ms one-way
    LOSS_RATE_B = 0.2

    FLOW_SIZE_THRESHOLD = B_A / 2 / 8  # unused in new mode, but kept


class Flow:
    def __init__(self, flow_id, arrival_time, rate_bps=None, duration_sec=None, total_bytes=None):
        self.flow_id = flow_id
        self.arrival_time = arrival_time
        
        if total_bytes is not None:
            # Backward-compatible burst flow (e.g., Web object)
            self.total_bytes = total_bytes
            # Assume it wants to finish quickly (e.g., within 0.1s)
            self.duration_sec = 0.1
            self.rate_bps = self.total_bytes * 8 / self.duration_sec
        else:
            # CBR-like flow
            self.rate_bps = rate_bps
            self.duration_sec = duration_sec
            self.total_bytes = int(rate_bps * duration_sec // 8)


def determine_redundancy_mode(loss_rate_a):
    # if loss_rate_a < 0.04:
    #     return 'none'
    # elif loss_rate_a <= 0.12:
    #     return 'xor_4_1'
    # else:
    #     return 'replicate_4_1'
    return 'xor_4_1'


# --- FastSim: unchanged except for is_large_flow_override ---
class FastSim:
    def __init__(self, loss_rate_a_func, loss_rate_b_func, redundancy_mode, flow, is_large_flow_override=None):      
        self.redundancy_manager = RedundancyManager(redundancy_mode)
        
        cfg = Config()
        self.B_A = cfg.B_A
        self.RTT_A = cfg.RTT_A
        # self.loss_rate_a = loss_rate_a
        self.loss_rate_a_func = loss_rate_a_func
        self.loss_rate_b_func = loss_rate_b_func
        self.B_B = cfg.B_B
        self.RTT_B = cfg.RTT_B
        # self.loss_rate_b = loss_rate_b
        self.PKT_SIZE = cfg.PKT_SIZE
        self.flow_total_bytes = flow.total_bytes
        self.RTO = cfg.RTO
        self.threshold = cfg.FLOW_SIZE_THRESHOLD
        self.arrival_time = flow.arrival_time
        self.flow = flow

        # ✅ Use external large-flow decision
        if is_large_flow_override is not None:
            if is_large_flow_override:
                self.sendMode = 'FPS'
            else:
                self.sendMode = self._decide_small_flow_mode()
        else:
            # Fallback for single-flow mode
            self.sendMode = self.mode_determine(flow)

        print(f"Determined send mode: {self.sendMode} for flow {flow.flow_id}")

        self.delay_time_diff = self.RTT_B / 2 - self.RTT_A / 2
        self.delay_packet_diff = self.delay_time_diff * (self.B_A / 8) / self.PKT_SIZE
        self.num_pkts = (self.flow_total_bytes + self.PKT_SIZE - 1) // self.PKT_SIZE
        self.redundancy_manager.set_total_packets(self.num_pkts)

        if self.redundancy_manager.mode == 'xor_k_1':
            self.current_group_start = None
            self.current_group_status = set()
            self.group_size = self.redundancy_manager.k_rep + 1

        self.fast_path = 'A'
        self.slow_path = 'B'
        self.last_fast_sn = -1
        self.last_slow_sn = -1
        self.current_batch_start_sn = 0
        self.batch_size = max(1, int(self.B_A // self.B_B))

        self.next_sn = 0
        self.unacked = {}
        self.next_expected = 0
        self.delivered = 0
        self.receiver_buffer = {}
        self.buffer_history = []
        self.events = []
        self.event_id = 0
        self.max_window = 100

        self.last_send_time_A = - (self.PKT_SIZE * 8 / self.B_A)
        self.last_send_time_B = - (self.PKT_SIZE * 8 / self.B_B)

        # ===== 新增统计变量 =====
        self.retransmit_count = 0
        self.packet_deliveries = []  # 记录每个包的交付时间 (t, sn)
        self.buffer_samples = []     # 按固定间隔采样队列长度
        self.sample_interval = 0.1   # 每 0.1 秒采样一次（可调）
        self.next_sample_time = self.arrival_time
        self.total_simulation_time = 0.0

        if useLog:
            initial_loss_a = self.loss_rate_a_func(self.arrival_time)
            initial_loss_b = self.loss_rate_b_func(self.arrival_time)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_tx_filename = f"./log/log_tx_flow{flow.flow_id}_{redundancy_mode}_loss_{initial_loss_a:.2f}_{initial_loss_b:.2f}_{timestamp}.txt"
            self.log_rx_filename = f"./log/log_rx_flow{flow.flow_id}_{redundancy_mode}_loss_{initial_loss_a:.2f}_{initial_loss_b:.2f}_{timestamp}.txt"
            try:
                self.log_tx_handle = open(self.log_tx_filename, 'w')
                self.log_rx_handle = open(self.log_rx_filename, 'w')
                header_info = f"--- Simulation Log Started at {datetime.datetime.now()} ---\n"
                config_info = f"Mode: {self.redundancy_manager.mode}, k: {self.redundancy_manager.k_rep}, Loss Rates: A={initial_loss_a:.2f}, B={initial_loss_b:.2f}\nTotal Packets: {self.num_pkts}\n\n"
                self.log_tx_handle.write(header_info + "LOG TYPE: TRANSMISSION EVENTS\n" + config_info)
                self.log_rx_handle.write(header_info + "LOG TYPE: RECEPTION EVENTS\n" + config_info)
                self.log_tx_handle.flush()
                self.log_rx_handle.flush()
            except IOError as e:
                print(f"Error opening log files: {e}", file=sys.stderr)
                raise

    def _decide_small_flow_mode(self):
        if self._should_send_redundancy_on_bad_path():
            return 'RSR'
        else:
            return 'SP'

    def mode_determine(self, flow):
        # Legacy fallback
        if flow.total_bytes >= self.threshold:
            return 'FPS'
        else:
            return self._decide_small_flow_mode()

    def log_tx_event(self, message):
        if hasattr(self, 'log_tx_handle') and self.log_tx_handle and useLog:
            self.log_tx_handle.write(message + "\n")
            self.log_tx_handle.flush()

    def log_rx_event(self, message):
        if hasattr(self, 'log_rx_handle') and self.log_rx_handle and useLog:
            self.log_rx_handle.write(message + "\n")
            self.log_rx_handle.flush()

    def close_log_tx(self):
        if hasattr(self, 'log_tx_handle') and self.log_tx_handle and not self.log_tx_handle.closed and useLog:
            self.log_tx_handle.close()
            self.log_tx_handle = None

    def close_log_rx(self):
        if hasattr(self, 'log_rx_handle') and self.log_rx_handle and not self.log_rx_handle.closed and useLog:
            self.log_rx_handle.close()
            self.log_rx_handle = None

    def close_logs(self):
        self.close_log_tx()
        self.close_log_rx()

    def _should_send_redundancy_on_bad_path(self):
        loss_rate_a = self.loss_rate_a_func(self.arrival_time)
        loss_rate_b = self.loss_rate_b_func(self.arrival_time)
        L_A = loss_rate_a
        L_B = loss_rate_b
        T_A = self.RTT_A
        T_B = self.RTT_B
        RTO = self.RTO

        if L_A <= 0:
            return False
        if L_B >= 1.0:
            return False

        E1 = T_A + (L_A ** 2) * RTO
        term1 = (1 - L_A) * T_A
        term2 = L_A * (1 - L_B) * (T_A + T_B) / 2.0
        term3 = L_A * L_B * (RTO + T_A)
        E2 = term1 + term2 + term3
        return E2 < E1

    def schedule(self, t, ev_type, data=None):
        heapq.heappush(self.events, (t, self.event_id, ev_type, data))
        self.event_id += 1

    def run(self):
        self.log_rx_event("--- Simulation Run Started ---")
        base_initial_time = self.arrival_time
        for sn in range(min(self.max_window, self.num_pkts)):
            self._send_packet(sn, base_initial_time)

        try:
            while self.events and (self.next_sn < self.num_pkts or self.unacked):
                t, _, ev_type, data = heapq.heappop(self.events)
                if ev_type == 'arrive':
                    self._on_arrival(t, data['sn'], data.get('path', 'N/A'))
                elif ev_type == 'ack':
                    self._on_ack(t, data['cumulative'], data['selective'])
                elif ev_type == 'timeout':
                    if data['sn'] in self.unacked:
                        self._retransmit(t, data['sn'], data.get('path', 'N/A'))
            self.log_rx_event("\n--- Simulation Run Ended ---")
        finally:
            self.close_logs()

        # 确保采样覆盖到仿真结束
        last_time = max((t for t, _ in self.buffer_history), default=self.arrival_time + 0.1)
        delivered_bytes = self.delivered * self.PKT_SIZE
        throughput_bps = (delivered_bytes * 8) / (last_time - self.arrival_time) if (last_time > self.arrival_time) else 0.0
        throughput_mbps = throughput_bps / 1e6

        # ✅ 使用原始 buffer_history 计算 avg_queue（这才是对的！）
        if len(self.buffer_history) < 2:
            avg_queue = 0.0
        else:
            total_area = 0
            prev_t, prev_q = self.buffer_history[0]
            for t, q in self.buffer_history[1:]:
                total_area += prev_q * (t - prev_t)
                prev_t, prev_q = t, q
            time_span = prev_t - self.buffer_history[0][0]
            avg_queue = total_area / time_span

        avg_buffer_bytes = avg_queue * self.PKT_SIZE
        efficiency = throughput_bps / avg_buffer_bytes if avg_buffer_bytes > 0 else 0.0

        # 写入时间序列数据到 CSV（用于绘图）
        self._save_time_series_to_csv()

        return {
            'throughput_mbps': throughput_mbps,
            'avg_queue_length': avg_queue,
            'avg_buffer_bytes': avg_buffer_bytes,
            'efficiency_bps_per_byte': efficiency,          # 新增核心指标
            'retransmit_count': self.retransmit_count,      # 新增
            'delivered_packets': self.delivered,
            'total_simulation_time': self.total_simulation_time
        }

    # --- SEND LOGIC (unchanged from your original) ---
    def _send_packet(self, original_sn, base_time):
        if self.sendMode == 'FPS':
            self.fps_send_packet(original_sn, base_time)
        elif self.sendMode == 'SP':
            self.sp_send_packet(original_sn, base_time)
        elif self.sendMode == 'RSR':
            self.rsr_send_packet(original_sn, base_time)
        else:
            raise ValueError(f"Unsupported sendMode: {self.sendMode}")

    def _send_physical_packet_on_path(self, sn, send_time, path, is_redundant=False, pkt_type='data'):
        # ✅ 动态丢包率：根据发送时间查询
        if path == 'A':
            loss_rate = self.loss_rate_a_func(send_time)
        else:
            loss_rate = self.loss_rate_b_func(send_time)

        rtt = self.RTT_A if path == 'A' else self.RTT_B

        if random.random() >= loss_rate:
            arrive_time = send_time + rtt / 2
            self.schedule(arrive_time, 'arrive', {'sn': sn, 'path': path})
            log_type = "REPLICATE" if is_redundant and pkt_type != 'xor' else \
                       "XOR" if pkt_type == 'xor' else \
                       "REPLICATE_K_1" if is_redundant else "ORIGINAL"
            self.log_tx_event(f"[SEND] Scheduling {log_type} packet {sn} on Path {path} arrival at {arrive_time:.4f}")
        else:
            log_type = "REPLICATE" if is_redundant and pkt_type != 'xor' else \
                       "XOR" if pkt_type == 'xor' else \
                       "REPLICATE_K_1" if is_redundant else "ORIGINAL"
            self.log_tx_event(f"[SEND] {log_type} packet {sn} on Path {path} LOST at {send_time:.4f}")

    def rsr_send_packet(self, original_sn, base_time):
        if original_sn >= self.num_pkts:
            return
        packets_to_send = self.redundancy_manager.get_packets_to_send(original_sn)
        use_bad_path = self._should_send_redundancy_on_bad_path()
        for idx, pkt_info in enumerate(packets_to_send):
            sn = pkt_info['sn']
            is_redundant = pkt_info.get('is_redundant', False)
            pkt_type = pkt_info.get('type', 'data')
            if idx == 0:
                path = 'A'
                interval = self.PKT_SIZE * 8 / self.B_A
                send_time = max(base_time, self.last_send_time_A + interval)
                self.last_send_time_A = send_time
            else:
                if use_bad_path and self.B_B > 0:
                    path = 'B'
                    interval = self.PKT_SIZE * 8 / self.B_B
                    send_time = max(base_time, self.last_send_time_B + interval)
                    self.last_send_time_B = send_time
                else:
                    path = 'A'
                    interval = self.PKT_SIZE * 8 / self.B_A
                    send_time = max(base_time, self.last_send_time_A + interval)
                    self.last_send_time_A = send_time
            self._send_physical_packet_on_path(sn, send_time, path, is_redundant, pkt_type)
        self.unacked[original_sn] = (base_time, 'A')
        self.schedule(base_time + self.RTO, 'timeout', {'sn': original_sn, 'path': 'A'})

    def sp_send_packet(self, original_sn, base_time):
        if original_sn >= self.num_pkts:
            return
        path = 'A'
        path_interval_A = self.PKT_SIZE * 8 / self.B_A
        send_time = max(base_time, self.last_send_time_A + path_interval_A)
        self.last_send_time_A = send_time
        self._send_packet_on_path(original_sn, send_time, path)

    def fps_send_packet(self, original_sn, base_time):
        if original_sn >= self.num_pkts:
            return
        if original_sn < self.delay_packet_diff:
            path = self.fast_path
            self.last_fast_sn = original_sn
        else:
            if self.last_fast_sn < self.current_batch_start_sn + self.batch_size - 1:
                path = self.fast_path
                self.last_fast_sn = original_sn
                if self.last_fast_sn == self.current_batch_start_sn + self.batch_size - 1:
                    self.log_tx_event(f"[FPS] Fast batch [{self.last_fast_sn - self.batch_size + 1}, {self.last_fast_sn}] completed.")
            else:
                expected_slow_sn = self.last_fast_sn + 1
                if original_sn == expected_slow_sn:
                    path = self.slow_path
                    self.last_slow_sn = original_sn
                    self.current_batch_start_sn = self.last_slow_sn + 1
                    self.log_tx_event(f"[FPS] Sent slow packet {original_sn}. Next fast batch starts at {self.current_batch_start_sn}.")
                elif original_sn == self.last_slow_sn + 1:
                    path = self.fast_path
                    self.last_fast_sn = original_sn
                    self.log_tx_event(f"[FPS] Packet {original_sn} continues fast path after slow packet.")
                else:
                    path = self.fast_path
                    self.last_fast_sn = original_sn
                    self.log_tx_event(f"[FPS] Packet {original_sn} routed to fast path (default).")

        path_intervals = {'A': self.PKT_SIZE * 8 / self.B_A, 'B': self.PKT_SIZE * 8 / self.B_B}
        if path == 'A':
            send_time = max(base_time, self.last_send_time_A + path_intervals['A'])
            self.last_send_time_A = send_time
        elif path == 'B':
            send_time = max(base_time, self.last_send_time_B + path_intervals['B'])
            self.last_send_time_B = send_time
        else:
            send_time = base_time
        self._send_packet_on_path(original_sn, send_time, path)

    def _send_packet_on_path(self, original_sn, send_time, path):
        if original_sn >= self.num_pkts:
            return
        packets_to_send = self.redundancy_manager.get_packets_to_send(original_sn)
        current_offset = 0.0
        base_interval = self.PKT_SIZE * 8 / (self.B_A if path == 'A' else self.B_B)
        for pkt_info in packets_to_send:
            sn = pkt_info['sn']
            is_redundant = pkt_info.get('is_redundant', False)
            pkt_type = pkt_info.get('type', 'data')
            physical_time = send_time + current_offset
            self._send_physical_packet_on_path(sn, physical_time, path, is_redundant, pkt_type)
            current_offset += base_interval
        self.unacked[original_sn] = (send_time, path)
        self.schedule(send_time + self.RTO, 'timeout', {'sn': original_sn, 'path': path})

    # --- RECEIVE LOGIC (unchanged) ---
    def _on_arrival(self, t, pkt_identifier, path_received='N/A'):
        sn = pkt_identifier
        self.log_rx_event(f"[{t:.4f}] RX: Packet {sn} arrived on Path {path_received}.")
        if self.redundancy_manager.mode == 'xor_k_1':
            if sn == 0:
                self.current_group_start = 0
                self.current_group_status = set()
                self.log_rx_event(f"[{t:.4f}]  -> Packet {sn} starts the first XOR group.")
            if self.current_group_start is not None:
                group_end = self.current_group_start + self.redundancy_manager.k_rep - 1
                expected_xor_sn = -group_end
                relevant_group_elements = set(range(self.current_group_start, group_end + 1))
                relevant_group_elements.add(expected_xor_sn)
                if sn in relevant_group_elements:
                    if sn not in self.current_group_status:
                        self.current_group_status.add(sn)
                        self.log_rx_event(f"[{t:.4f}]     Added {sn} to group. Status: {sorted(list(self.current_group_status))}")
                    if len(self.current_group_status) >= self.group_size - 1:
                        self._attempt_recovery(t, t)
        if sn < 0:
            pass
        else:
            if sn < self.next_expected:
                self.buffer_history.append((t, len(self.receiver_buffer)))
                return
            if sn == self.next_expected:
                self.delivered += 1
                self.next_expected += 1
                while self.next_expected in self.receiver_buffer:
                    self.delivered += 1
                    del self.receiver_buffer[self.next_expected]
                    self.next_expected += 1
                cumulative = self.next_expected - 1
                selective = [s for s in self.receiver_buffer if s > cumulative]
                self.schedule(t + self.RTT_A / 2, 'ack', {'cumulative': cumulative, 'selective': selective})
                while (self.next_sn < self.num_pkts and (self.next_sn - (self.next_expected - 1)) < self.max_window):
                    send_time = self.arrival_time + self.next_sn * (self.PKT_SIZE * 8 / self.B_A)
                    self._send_packet(self.next_sn, send_time)
                    self.next_sn += 1
            else:
                if sn not in self.receiver_buffer:
                    self.receiver_buffer[sn] = t
                cumulative = self.next_expected - 1
                selective = [s for s in self.receiver_buffer if s > cumulative]
                self.schedule(t + 1e-9, 'ack', {'cumulative': cumulative, 'selective': selective})
        if self.redundancy_manager.mode == 'xor_k_1':
            if (-sn + 1) % self.redundancy_manager.k_rep == 0 and sn != 0 and sn < 0:
                new_group_start = -sn + 1
                if new_group_start != self.current_group_start:
                    self.current_group_start = new_group_start
                    self.current_group_status = set()
        self.buffer_history.append((t, len(self.receiver_buffer)))

        # 记录成功交付（仅原始数据包）
        if sn >= 0 and sn < self.num_pkts:
            self.packet_deliveries.append(t)

        # 按固定时间间隔采样 buffer 长度（用于时间序列分析）
        while t >= self.next_sample_time:
            current_queue_len = len(self.receiver_buffer)
            self.buffer_samples.append((self.next_sample_time, current_queue_len))
            self.next_sample_time += self.sample_interval

    def _attempt_recovery(self, t, event_time):
        if self.current_group_start is None:
            return
        group_start = self.current_group_start
        group_end = group_start + self.redundancy_manager.k_rep - 1
        expected_xor_sn = -group_end
        all_group_elements = set(range(group_start, group_end + 1))
        all_group_elements.add(expected_xor_sn)
        missing_elements = all_group_elements - self.current_group_status
        missing_data_packets = [elem for elem in missing_elements if elem >= 0]
        if len(missing_data_packets) == 1:
            recovered_sn = missing_data_packets[0]
            if recovered_sn == self.next_expected:
                self.delivered += 1
                self.next_expected += 1
                while self.next_expected in self.receiver_buffer:
                    self.delivered += 1
                    del self.receiver_buffer[self.next_expected]
                    self.next_expected += 1
                cumulative_ack = self.next_expected - 1
                selective_ack = [s for s in self.receiver_buffer if s > cumulative_ack]
                self.schedule(event_time + 1e-9, 'ack', {'cumulative': cumulative_ack, 'selective': selective_ack})
                while (self.next_sn < self.num_pkts and (self.next_sn - (self.next_expected - 1)) < self.max_window):
                    self._send_packet(self.next_sn, event_time)
                    self.next_sn += 1
            else:
                if recovered_sn not in self.receiver_buffer:
                    self.receiver_buffer[recovered_sn] = event_time
                    cumulative_ack = self.next_expected - 1
                    selective_ack = [s for s in self.receiver_buffer if s > cumulative_ack]
                    self.schedule(event_time + 1e-9, 'ack', {'cumulative': cumulative_ack, 'selective': selective_ack})
        self.current_group_start = None
        self.current_group_status = set()

    def _on_ack(self, t, cumulative, selective):
        to_remove = [sn for sn in self.unacked if sn <= cumulative]
        for sn in to_remove:
            del self.unacked[sn]

    def _retransmit(self, t, sn, path_for_retransmit):
        if sn not in self.unacked:
            return
        self.retransmit_count += 1
        # ✅ Use current time 't' for loss rate
        if path_for_retransmit == 'A':
            loss_rate = self.loss_rate_a_func(t)
        else:
            loss_rate = self.loss_rate_b_func(t)
        rtt = self.RTT_A if path_for_retransmit == 'A' else self.RTT_B
        if random.random() >= loss_rate:
            arrive_time = t + rtt / 2
            self.schedule(arrive_time, 'arrive', {'sn': sn, 'path': path_for_retransmit})
            self.log_tx_event(f"[RETRANS] Scheduling RETRANSMISSION packet {sn} on {path_for_retransmit} arrival at {arrive_time:.4f}")
        else:
            self.log_tx_event(f"[RETRANS] RETRANSMISSION packet {sn} on {path_for_retransmit} LOST at {t:.4f}")
        self.unacked[sn] = (t, path_for_retransmit)
        self.schedule(t + self.RTO, 'timeout', {'sn': sn, 'path': path_for_retransmit})

    def _save_time_series_to_csv(self):
        """保存时间序列数据：时间, 队列长度, 累计交付包数"""
        import csv
        filename = f"./log/ts_flow{self.flow.flow_id}_buffer.csv"
        cumulative_delivered = 0
        delivery_iter = iter(sorted(self.packet_deliveries))
        next_delivery = next(delivery_iter, None)

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'queue_length', 'cumulative_delivered', 'instant_throughput_mbps'])
                prev_time = self.arrival_time
                for t, q_len in self.buffer_samples:
                    # 累计交付包数到当前时间 t
                    while next_delivery is not None and next_delivery <= t:
                        cumulative_delivered += 1
                        next_delivery = next(delivery_iter, None)
                    # 瞬时吞吐（过去 sample_interval 内的交付速率）
                    if t > prev_time:
                        interval = t - prev_time
                        # 找出 [prev_time, t] 内的交付包数（简化：用累计差）
                        # 更精确的做法是记录每个交付时间，但这里用近似
                        instant_delivered = sum(1 for d in self.packet_deliveries if prev_time < d <= t)
                        instant_thr_mbps = (instant_delivered * self.PKT_SIZE * 8) / interval / 1e6 if interval > 0 else 0.0
                    else:
                        instant_thr_mbps = 0.0
                    writer.writerow([f"{t:.3f}", q_len, cumulative_delivered, f"{instant_thr_mbps:.3f}"])
                    prev_time = t
        except Exception as e:
            print(f"Warning: Could not write time-series CSV for flow {self.flow.flow_id}: {e}")


# ======================
# Multi-Flow Runner
# ======================
class MultiFlowRunner:
    def __init__(self, loss_rate_a_func, loss_rate_b_func, flows):
        self.loss_rate_a_func = loss_rate_a_func
        self.loss_rate_b_func = loss_rate_b_func
        self.flows = sorted(flows, key=lambda f: f.arrival_time)
        self.allocated_bw = 0.0
        self.B_A = Config().B_A
        self.results = []

    def run(self):
        for flow in self.flows:
            remaining_bw = self.B_A - self.allocated_bw
            is_large = flow.rate_bps > remaining_bw
            self.allocated_bw += flow.rate_bps

            current_loss_a = self.loss_rate_a_func(flow.arrival_time)
            redundancy_mode = determine_redundancy_mode(current_loss_a)
            sim = FastSim(
                loss_rate_a_func=self.loss_rate_a_func,
                loss_rate_b_func=self.loss_rate_b_func,
                redundancy_mode=redundancy_mode,
                flow=flow,
                is_large_flow_override=is_large
            )
            result = sim.run()
            self.results.append((flow.flow_id, flow.arrival_time, flow.duration_sec, result))

            self.allocated_bw -= flow.rate_bps
        return self.results


# ======================
# Flow Generators
# ======================
def generate_bursty_flows():
    flows = []
    rates = [40e6, 60e6, 125e6, 80e6, 20e6]
    duration = 3.0
    for i, rate in enumerate(rates):
        arrival = i * duration
        flows.append(Flow(flow_id=i, arrival_time=arrival, rate_bps=rate, duration_sec=duration))
    return flows

def generate_mixed_flows():
    flows = []
    flows.append(Flow(flow_id=0, arrival_time=0.1, total_bytes=100*1024))
    flows.append(Flow(flow_id=1, arrival_time=0.5, total_bytes=500*1024))
    flows.append(Flow(flow_id=2, arrival_time=0.2, rate_bps=60e6, duration_sec=5))
    flows.append(Flow(flow_id=3, arrival_time=0.8, rate_bps=50e6, duration_sec=4))
    return flows

# 示例 1：固定丢包率（兼容旧用法）
def constant_loss(rate):
    return lambda t: rate

# 示例 2：周期性高丢包（每 5 秒，第 4～5 秒丢包率 0.3）
def periodic_bursty_loss(baseline=0.05, burst_rate=0.3, period=150.0, burst_duration=3.0):
    def func(t):
        phase = t % period
        if phase >= (period - burst_duration):
            return burst_rate
        else:
            return baseline
    return func

# 示例 3：随机突发（简单版：每整数秒有 20% 概率进入 2 秒高丢包）
def random_burst_loss(baseline=0.05, burst_rate=0.4, burst_duration=2.0, burst_prob=0.2):
    bursts = {}  # cache to keep burst consistent during duration
    def func(t):
        base_sec = int(t)
        if base_sec not in bursts:
            bursts[base_sec] = (random.random() < burst_prob)
        if bursts[base_sec]:
            # check if within burst window
            if t < (base_sec + burst_duration):
                return burst_rate
        return baseline
    return func

def targeted_burst_loss():
    def func(t):
        if 9.0 <= t < 12.0:
            return 0.30  # 30% 高丢包
        else:
            # 1% ～ 5% 随机波动（为保证可重现，用 deterministic hash 或固定 seed）
            # 方法：用 t 的小数部分生成伪随机值
            import math
            # 简单 deterministic noise: sin-based or fractional part
            noise = (math.sin(t * 1000) + 1) / 2  # [0, 1]
            return 0.01 + 0.04 * noise  # => [0.01, 0.05]
    return func


# ======================
# Main
# ======================
def main():
    random.seed(42)
    flows = generate_bursty_flows()
    # flows = generate_mixed_flows()

     # ✅ 定义动态丢包率函数
    loss_a = targeted_burst_loss()
    loss_b = constant_loss(0.1)  # 备链路仍固定

    runner = MultiFlowRunner(loss_rate_a_func=loss_a, loss_rate_b_func=loss_b, flows=flows)
    results = runner.run()

    print("\n=== DETAILED FLOW RESULTS ===")
    total_delivered_bytes = 0
    max_end_time = 0

    for fid, arr_time, dur, res in results:
        flow = next(f for f in flows if f.flow_id == fid)
        end_time = arr_time + dur
        max_end_time = max(max_end_time, end_time)
        total_delivered_bytes += res['delivered_packets'] * Config().PKT_SIZE

        eff = res['efficiency_bps_per_byte']
        print(f"\nFlow {fid} ({flow.sendMode if hasattr(flow, 'sendMode') else 'N/A'}):")
        print(f"  Throughput: {res['throughput_mbps']:.2f} Mbps")
        print(f"  Avg Buffer: {res['avg_buffer_bytes']/1024:.1f} KB")
        print(f"  Efficiency: {eff:.2f} bps/byte  (or {eff*8:.1f} bps/bit)")
        print(f"  Retransmissions: {res['retransmit_count']}")
        print(f"  Avg Queue: {res['avg_queue_length']:.1f} pkts")

    overall_throughput = (total_delivered_bytes * 8) / max_end_time / 1e6
    print(f"\nOverall Throughput: {overall_throughput:.2f} Mbps over {max_end_time:.2f}s")


if __name__ == '__main__':
    main()