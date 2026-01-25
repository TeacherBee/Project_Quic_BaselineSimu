import heapq
import random
import sys
import datetime
from redundancy import RedundancyManager

useLog = False

class Config:
    B = 100e6               # 100 Mbps
    RTT = 0.6               # 600 ms
    PKT_SIZE = 1250         # bytes
    FLOW_SIZE = 100 * 1024 * 1024  # 100 MB
    RTO = 1.0               # Timeout

# --- 重构后的 FastSim 类 ---
class FastSim:
    def __init__(self, loss_rate, redundancy_mode):
        self.loss_rate = loss_rate
        
        # 初始化冗余管理器
        self.redundancy_manager = RedundancyManager(redundancy_mode)
        
        cfg = Config()
        self.B = cfg.B
        self.RTT = cfg.RTT
        self.PKT_SIZE = cfg.PKT_SIZE
        self.FLOW_SIZE = cfg.FLOW_SIZE
        self.RTO = cfg.RTO
        
        self.num_pkts = (self.FLOW_SIZE + self.PKT_SIZE - 1) // self.PKT_SIZE
        base_interval = self.PKT_SIZE * 8 / self.B  # time to send one packet

        # 告知冗余管理器总包数
        self.redundancy_manager.set_total_packets(self.num_pkts)

        # 🔧 使用冗余管理器计算有效发送间隔
        interval_factor = self.redundancy_manager.get_effective_send_interval_factor(base_interval, self.num_pkts)
        self.send_interval = base_interval * interval_factor

        # 用于管理当前恢复组的状态变量 (现在属于FastSim，因为它处理接收逻辑)
        if self.redundancy_manager.mode == 'xor_k_1':
            self.current_group_start = None
            self.current_group_status = set() # 存储当前组中已收到的元素 (sn or -sn)
            self.group_size = self.redundancy_manager.k_rep + 1 # k_xor 个数据包 + 1 个 XOR 包
        
        # Sender
        self.next_sn = 0
        self.unacked = {}
        
        # Receiver
        self.next_expected = 0
        self.delivered = 0
        self.receiver_buffer = {}
        self.buffer_history = []
        
        # Event queue
        self.events = []
        self.event_id = 0
        self.max_window = 100

        # --- Logging Setup - Separated ---
        if useLog:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            redundancy = redundancy_mode # Use the original string for filename
            self.log_tx_filename = f"./log/log_tx_{redundancy}_loss_{loss_rate:.2f}_{timestamp}.txt"
            self.log_rx_filename = f"./log/log_rx_{redundancy}_loss_{loss_rate:.2f}_{timestamp}.txt"
            
            try:
                self.log_tx_handle = open(self.log_tx_filename, 'w')
                self.log_rx_handle = open(self.log_rx_filename, 'w')
                
                header_info = f"--- Simulation Log Started at {datetime.datetime.now()} ---\n"
                config_info = f"Mode: {self.redundancy_manager.mode}, k: {self.redundancy_manager.k_rep}, Loss Rate: {self.loss_rate}\nTotal Packets to Send: {self.num_pkts}\n\n"
                
                self.log_tx_handle.write(header_info + "LOG TYPE: TRANSMISSION EVENTS\n" + config_info)
                self.log_rx_handle.write(header_info + "LOG TYPE: RECEPTION EVENTS\n" + config_info)
                
                self.log_tx_handle.flush()
                self.log_rx_handle.flush()
                
            except IOError as e:
                print(f"Error opening log files: {e}", file=sys.stderr)
                raise

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

    def schedule(self, t, ev_type, data=None):
        heapq.heappush(self.events, (t, self.event_id, ev_type, data))
        self.event_id += 1

    def run(self):
        self.log_rx_event("--- Simulation Run Started ---")
        for sn in range(min(self.max_window, self.num_pkts)):
            send_time = sn * self.send_interval
            self._send_packet(sn, send_time)

        try:
            while self.events and (self.next_sn < self.num_pkts or self.unacked):
                t, _, ev_type, data = heapq.heappop(self.events)
                
                if ev_type == 'arrive':
                    self._on_arrival(t, data['sn'])
                elif ev_type == 'ack':
                    self._on_ack(t, data['cumulative'], data['selective'])
                elif ev_type == 'timeout':
                    if data['sn'] in self.unacked:
                        self._retransmit(t, data['sn'])

            self.log_rx_event("\n--- Simulation Run Ended ---")
            self.log_rx_event(f"Final next_expected: {self.next_expected}")
            self.log_rx_event(f"Final delivered: {self.delivered}")
            self.log_rx_event(f"Final receiver_buffer: {sorted(list(self.receiver_buffer))}")
        finally:
            self.close_logs()

        last_time = max((t for t, _ in self.buffer_history), default=0.1)
        throughput_mbps = (self.delivered * self.PKT_SIZE * 8) / last_time / 1e6

        if len(self.buffer_history) < 2:
            avg_queue = 0.0
        else:
            total = 0.0
            prev_t, prev_q = self.buffer_history[0]
            for t, q in self.buffer_history[1:]:
                total += prev_q * (t - prev_t)
                prev_t, prev_q = t, q
            time_span = prev_t - self.buffer_history[0][0]
            avg_queue = total / time_span if time_span > 0 else 0.0

        results = {
            'throughput_mbps': throughput_mbps,
            'avg_queue_length': avg_queue,
            'delivered_packets': self.delivered
        }
        return results

    def _send_packet(self, original_sn, send_time):
        if original_sn >= self.num_pkts:
            return

        base_pkt_time = self.PKT_SIZE * 8 / self.B

        # 🔧 使用冗余管理器获取要发送的物理包列表
        packets_to_send = self.redundancy_manager.get_packets_to_send(original_sn)
        
        # 发送所有物理包
        current_send_offset = 0.0
        for i, pkt_info in enumerate(packets_to_send):
            sn_to_schedule = pkt_info['sn']
            is_redundant = pkt_info.get('is_redundant', False)
            pkt_type = pkt_info.get('type', 'data')

            # 计算此物理包的发送时间
            physical_send_time = send_time + current_send_offset

            # 决定是否丢失
            if random.random() >= self.loss_rate:
                arrive_time = physical_send_time + self.RTT / 2
                self.schedule(arrive_time, 'arrive', {'sn': sn_to_schedule})
                log_type = "REPLICATE" if is_redundant and pkt_type != 'xor' else \
                           "XOR" if pkt_type == 'xor' else \
                           "REPLICATE_K_1" if is_redundant else \
                           "ORIGINAL"
                self.log_tx_event(f"[SEND] Scheduling {log_type} packet {sn_to_schedule} arrival at {arrive_time:.4f}")
            else:
                log_type = "REPLICATE" if is_redundant and pkt_type != 'xor' else \
                           "XOR" if pkt_type == 'xor' else \
                           "REPLICATE_K_1" if is_redundant else \
                           "ORIGINAL"
                self.log_tx_event(f"[SEND] {log_type} packet {sn_to_schedule} LOST at {physical_send_time:.4f}")

            # 更新下一个物理包的发送偏移
            current_send_offset += base_pkt_time

        # 只为原始包设置超时和未确认状态
        self.unacked[original_sn] = send_time
        self.schedule(send_time + self.RTO, 'timeout', {'sn': original_sn})

    def _on_arrival(self, t, pkt_identifier):
        sn = pkt_identifier
        self.log_rx_event(f"[{t:.4f}] RX: Packet {sn} arrived.")

        if self.redundancy_manager.mode == 'xor_k_1':
            if sn == 0:
                self.current_group_start = 0
                self.current_group_status = set()
                self.log_rx_event(f"[{t:.4f}]  -> Packet {sn} starts the first XOR group at {self.current_group_start}.")

            if self.current_group_start is not None:
                group_end = self.current_group_start + self.redundancy_manager.k_rep - 1
                expected_xor_sn = -group_end
                relevant_group_elements = set(range(self.current_group_start, group_end + 1))
                relevant_group_elements.add(expected_xor_sn)

                if sn in relevant_group_elements:
                    if sn not in self.current_group_status:
                        self.current_group_status.add(sn)
                        self.log_rx_event(f"[{t:.4f}]     Added {sn} to current group [{self.current_group_start}, {group_end}, {expected_xor_sn}] status. Current status: {sorted(list(self.current_group_status))} / {self.group_size}")
                    else:
                        self.log_rx_event(f"[{t:.4f}]     {sn} was already in current group status. Skipping.")

                    if len(self.current_group_status) >= self.group_size - 1:
                        self.log_rx_event(f"[{t:.4f}]     *** Recovery Triggered! Group [{self.current_group_start}, {group_end}, {expected_xor_sn}] has {len(self.current_group_status)}/{self.group_size} elements. Attempting recovery.")
                        self._attempt_recovery(t, t)
        
        if sn < 0: # It's an XOR packet
            pass # Logic handled via group management above

        else: # Handle regular data packet (sn >= 0)
            if sn < self.next_expected:
                self.log_rx_event(f"[{t:.4f}]  -> Packet {sn} is old/duplicate (expected >= {self.next_expected}). Dropping.")
                self.buffer_history.append((t, len(self.receiver_buffer)))
                return

            if sn == self.next_expected:
                self.delivered += 1
                self.next_expected += 1
                self.log_rx_event(f"[{t:.4f}]  -> Packet {sn} is expected. Delivering. New delivered count: {self.delivered}")

                while self.next_expected in self.receiver_buffer:
                    self.delivered += 1
                    self.log_rx_event(f"[{t:.4f}]  -> Packet {self.next_expected} is consecutive. Delivering. New delivered count: {self.delivered}")
                    del self.receiver_buffer[self.next_expected]
                    self.next_expected += 1

                self.log_rx_event(f"[{t:.4f}]  -> Updated next_expected to {self.next_expected}.")
                cumulative = self.next_expected - 1
                selective = [s for s in self.receiver_buffer if s > cumulative]
                self.log_rx_event(f"[{t:.4f}]  -> Sending ACK: Cumulative={cumulative}, Selective={selective}")
                self.schedule(t + 1e-9, 'ack', {'cumulative': cumulative, 'selective': selective})

                while (self.next_sn < self.num_pkts and
                       (self.next_sn - (self.next_expected - 1)) < self.max_window):
                    send_time = self.next_sn * self.send_interval
                    if self.next_sn < 10:
                        self.log_rx_event(f"[{t:.4f}]  -> Scheduling next packet {self.next_sn} at {send_time:.4f}")
                    self._send_packet(self.next_sn, send_time)
                    self.next_sn += 1

            else: # sn > next_expected
                if sn not in self.receiver_buffer:
                    self.receiver_buffer[sn] = t
                    self.log_rx_event(f"[{t:.4f}]  -> Packet {sn} is not expected (expected {self.next_expected}). Adding to buffer. Buffer now: {sorted(list(self.receiver_buffer.keys()))}")
                else:
                    self.log_rx_event(f"[{t:.4f}]  -> Packet {sn} is not expected and already in buffer. Ignoring duplicate arrival.")

                cumulative = self.next_expected - 1
                selective = [s for s in self.receiver_buffer if s > cumulative]
                self.log_rx_event(f"[{t:.4f}]  -> Sending ACK: Cumulative={cumulative}, Selective={selective}")
                self.schedule(t + 1e-9, 'ack', {'cumulative': cumulative, 'selective': selective})

        if self.redundancy_manager.mode == 'xor_k_1':
            if (-sn + 1) % self.redundancy_manager.k_rep == 0 and sn != 0:
                new_group_start = -sn + 1
                if new_group_start != self.current_group_start:
                    self.log_rx_event(f"[{t:.4f}]  -> Packet {sn} triggers a new XOR group starting at {new_group_start}. Clearing old group status.")
                    self.current_group_start = new_group_start
                    self.current_group_status = set()

        self.buffer_history.append((t, len(self.receiver_buffer)))

    def _attempt_recovery(self, t, event_time):
        if self.current_group_start is None:
            self.log_rx_event(f"[{event_time:.4f}]  *** ERROR: _attempt_recovery called but no active group.")
            return

        group_start = self.current_group_start
        group_end = group_start + self.redundancy_manager.k_rep - 1
        expected_xor_sn = -group_end
        all_group_elements = set(range(group_start, group_end + 1))
        all_group_elements.add(expected_xor_sn)

        missing_elements = all_group_elements - self.current_group_status
        self.log_rx_event(f"[{event_time:.4f}]     *** Identifying missing elements: {missing_elements}")

        missing_data_packets = [elem for elem in missing_elements if elem >= 0]

        if len(missing_data_packets) == 0:
            self.log_rx_event(f"[{event_time:.4f}]     *** All data packets in group are present. No recovery needed.")
            self.current_group_start = None
            self.current_group_status = set()
            return
        elif len(missing_data_packets) > 1:
            self.log_rx_event(f"[{event_time:.4f}]     *** WARNING: More than one data packet missing ({missing_data_packets}), cannot recover reliably.")
            self.current_group_start = None
            self.current_group_status = set()
            return
        else:
            recovered_sn = missing_data_packets[0]
            self.log_rx_event(f"[{event_time:.4f}]     *** Successfully identified missing data packet: {recovered_sn}")

            if recovered_sn == self.next_expected:
                self.delivered += 1
                self.next_expected += 1
                self.log_rx_event(f"[{event_time:.4f}]     *** Recovered packet {recovered_sn} is expected. Delivered. New delivered: {self.delivered}, next_expected: {self.next_expected}")

                while self.next_expected in self.receiver_buffer:
                    self.delivered += 1
                    self.log_rx_event(f"[{event_time:.4f}]  *** Delivered buffered packet {self.next_expected}. New delivered: {self.delivered}")
                    del self.receiver_buffer[self.next_expected]
                    self.next_expected += 1
                self.log_rx_event(f"[{event_time:.4f}]     *** Final next_expected after recovery delivery: {self.next_expected}")

                cumulative_ack = self.next_expected - 1
                selective_ack = [s for s in self.receiver_buffer if s > cumulative_ack]
                self.log_rx_event(f"[{event_time:.4f}]     *** Sending ACK after recovery: Cumulative={cumulative_ack}, Selective={selective_ack}")
                self.schedule(event_time + 1e-9, 'ack', {'cumulative': cumulative_ack, 'selective': selective_ack})

                while (self.next_sn < self.num_pkts and
                       (self.next_sn - (self.next_expected - 1)) < self.max_window):
                    send_time = self.next_sn * self.send_interval
                    if self.next_sn < 10:
                        self.log_rx_event(f"[{event_time:.4f}]     *** Scheduling next packet {self.next_sn} at {send_time:.4f} (due to recovery)")
                    self._send_packet(self.next_sn, send_time)
                    self.next_sn += 1

            else:
                if recovered_sn not in self.receiver_buffer:
                    self.receiver_buffer[recovered_sn] = event_time
                    self.log_rx_event(f"[{event_time:.4f}]     *** Recovered packet {recovered_sn} is not expected ({self.next_expected}). Added to buffer. Buffer now: {sorted(list(self.receiver_buffer.keys()))}")

                    cumulative_ack = self.next_expected - 1
                    selective_ack = [s for s in self.receiver_buffer if s > cumulative_ack]
                    self.log_rx_event(f"[{event_time:.4f}]     *** Sending ACK after recovery (to buffer): Cumulative={cumulative_ack}, Selective={selective_ack}")
                    self.schedule(event_time + 1e-9, 'ack', {'cumulative': cumulative_ack, 'selective': selective_ack})
                else:
                    self.log_rx_event(f"[{event_time:.4f}]     *** Recovered packet {recovered_sn} was already in buffer. Skipping.")

            self.log_rx_event(f"[{event_time:.4f}]     *** Recovery attempt finished. Clearing group status for next group.")
            self.current_group_start = None
            self.current_group_status = set()

    def _on_ack(self, t, cumulative, selective):
        to_remove = [sn for sn in self.unacked if sn <= cumulative]
        for sn in to_remove:
            del self.unacked[sn]
        for sn in selective:
            if sn in self.unacked:
                del self.unacked[sn]


    def _retransmit(self, t, sn):
        if sn not in self.unacked:
            return
        if random.random() >= self.loss_rate:
            arrive_time = t + self.RTT / 2
            self.schedule(arrive_time, 'arrive', {'sn': sn})
            self.log_tx_event(f"[RETRANS] Scheduling RETRANSMISSION packet {sn} arrival at {arrive_time:.4f}")
        else:
            self.log_tx_event(f"[RETRANS] RETRANSMISSION packet {sn} LOST at {t:.4f}")
            
        self.unacked[sn] = t
        self.schedule(t + self.RTO, 'timeout', {'sn': sn})


# ======================
# CLI
# ======================
def simulate(loss_rate, redundancy_mode):
    sim = FastSim(loss_rate, redundancy_mode)
    result = sim.run()
    return result

def main():
    if len(sys.argv) != 3:
        print("Usage: python fast_sim.py <loss_rate> <redundancy_mode>")
        print("  loss_rate: e.g., 0.05")
        print("  redundancy_mode options:")
        print("    - 'none'")
        print("    - 'replicate' (1:1 redundancy)")
        print("    - 'replicate_k_1' (e.g., 'replicate_4_1' → every 4 packets, add 1 redundant copy)")
        print("    - 'xor_k_1' (e.g., 'xor_4_1' → every 4 packets, add 1 XOR redundant copy)")
        sys.exit(1)
    
    loss_rate = float(sys.argv[1])
    redundancy_mode = sys.argv[2]
    
    random.seed(42)
    result = simulate(loss_rate, redundancy_mode)
    
    total_pkts = (Config().FLOW_SIZE + Config().PKT_SIZE - 1) // Config().PKT_SIZE
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"TX Log saved to file: log_tx_{redundancy_mode}_loss_{loss_rate:.2f}_{timestamp}.txt")
    print(f"RX Log saved to file: log_rx_{redundancy_mode}_loss_{loss_rate:.2f}_{timestamp}.txt")
    print(f"Loss={loss_rate:.1%}, Redundancy={redundancy_mode}")
    print(f"Throughput: {result['throughput_mbps']:.2f} Mbps")
    print(f"Avg Queue Length: {result['avg_queue_length']:.1f} packets")
    print(f"Delivered Packets: {result['delivered_packets']} / {total_pkts}")

if __name__ == '__main__':
    main()