import heapq
import random
import sys
import datetime
from redundancy import RedundancyManager

useLog = False

class Config:
    PKT_SIZE = 1250         # bytes
    FLOW_SIZE = 1 * 1024 * 1024  # 100 MB
    RTO = 0.5               # Timeout

    # 链路 A (快路径)
    B_A = 100e6               # 100 Mbps
    RTT_A = 0.05              # 300 ms
    LOSS_RATE_A = 0.05       # 5% 丢包率
    # 链路 B (慢路径)
    B_B = 20e6               # 20 Mbps
    RTT_B = 0.15              # 900 ms
    LOSS_RATE_B = 0.2        # 10% 丢包率

    # FLOW_SIZE_THRESHOLD = B_A * RTT_A / 8  # 如果流量小于此值，则不使用慢路径
    FLOW_SIZE_THRESHOLD = B_A / 2 / 8  # 如果流量小于此值，则不使用慢路径

def is_large_flow(flow, threshold_bytes=1024*1024):
        return flow.total_bytes >= threshold_bytes # > 6.25MB?  (B_A/2)

class Flow:
    def __init__(self, flow_id, total_bytes, bandwidth_demand=None):
        self.flow_id = flow_id
        self.total_bytes = total_bytes
        self.bandwidth_demand = bandwidth_demand

# --- 重构后的 FastSim 类 ---
class FastSim:
    def __init__(self, loss_rate_a, loss_rate_b, redundancy_mode, flow):     
        # 初始化冗余管理器
        self.redundancy_manager = RedundancyManager(redundancy_mode)
        
        cfg = Config()
        # 链路 A 参数 (快路径)
        self.B_A = cfg.B_A
        self.RTT_A = cfg.RTT_A
        self.loss_rate_a = loss_rate_a
        # 链路 B 参数 (慢路径)
        self.B_B = cfg.B_B
        self.RTT_B = cfg.RTT_B
        self.loss_rate_b = loss_rate_b
        print(f"Initialized FastSim with Loss Rates: A={self.loss_rate_a}, B={self.loss_rate_b}, Redundancy Mode: {redundancy_mode}")
        # 其他参数
        self.PKT_SIZE = cfg.PKT_SIZE
        # self.FLOW_SIZE = cfg.FLOW_SIZE
        self.flow_total_bytes = flow.total_bytes
        self.RTO = cfg.RTO
        self.threshold = cfg.FLOW_SIZE_THRESHOLD
        self.sendMode = self.mode_determine(flow)  # 'FPS', 'SP', or 'RSR'

        # FPS初始化，即慢路需要等快路先发的时间差和包数
        self.delay_time_diff = self.RTT_B / 2 - self.RTT_A / 2  # 0.45 - 0.15 = 0.3 seconds
        self.delay_packet_diff = self.delay_time_diff * (self.B_A / 8) / self.PKT_SIZE  # Convert bits to packets
        
        self.num_pkts = (self.flow_total_bytes + self.PKT_SIZE - 1) // self.PKT_SIZE
        # Note: Original base_interval calculation used single 'B'. Now we calculate per path.
        # self.base_interval_A = self.PKT_SIZE * 8 / self.B_A
        # self.base_interval_B = self.PKT_SIZE * 8 / self.B_B
        # The effective interval logic for MPS might differ significantly from single-path.

        # 告知冗余管理器总包数
        self.redundancy_manager.set_total_packets(self.num_pkts)

        # 用于管理当前恢复组的状态变量 (现在属于FastSim，因为它处理接收逻辑)
        if self.redundancy_manager.mode == 'xor_k_1':
            self.current_group_start = None
            self.current_group_status = set() # 存储当前组中已收到的元素 (sn or -sn)
            self.group_size = self.redundancy_manager.k_rep + 1 # k_xor 个数据包 + 1 个 XOR 包

        # --- FPS Algorithm Specific Variables ---
        # Define paths
        self.fast_path = 'A'
        self.slow_path = 'B'
        # Track sequence numbers sent on each path
        self.last_fast_sn = -1
        self.last_slow_sn = -1
        # Track the start of the current fast path batch
        self.current_batch_start_sn = 0
        # Size of the batch to send on the fast path before sending one on slow
        self.batch_size = self.B_A // self.B_B  # Example ratio-based batch size

        # Sender
        self.next_sn = 0
        self.unacked = {} # Store {original_sn: (send_time, path)}
        
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
            # Fixed filename to include both loss rates
            self.log_tx_filename = f"./log/log_tx_{redundancy}_loss_{loss_rate_a:.2f}_{loss_rate_b:.2f}_{timestamp}.txt"
            self.log_rx_filename = f"./log/log_rx_{redundancy}_loss_{loss_rate_a:.2f}_{loss_rate_b:.2f}_{timestamp}.txt"
            
            try:
                self.log_tx_handle = open(self.log_tx_filename, 'w')
                self.log_rx_handle = open(self.log_rx_filename, 'w')
                
                header_info = f"--- Simulation Log Started at {datetime.datetime.now()} ---\n"
                # Updated config info to reflect dual paths
                config_info = f"Mode: {self.redundancy_manager.mode}, k: {self.redundancy_manager.k_rep}, Loss Rates: A={self.loss_rate_a}, B={self.loss_rate_b}\nTotal Packets to Send: {self.num_pkts}\n\n"
                
                self.log_tx_handle.write(header_info + "LOG TYPE: TRANSMISSION EVENTS\n" + config_info)
                self.log_rx_handle.write(header_info + "LOG TYPE: RECEPTION EVENTS\n" + config_info)
                
                self.log_tx_handle.flush()
                self.log_rx_handle.flush()
                
            except IOError as e:
                print(f"Error opening log files: {e}", file=sys.stderr)
                raise

        # --- Add variables to track path-specific send times ---
        # Initialize to a time before the simulation starts to ensure the first packet is sent at time 0 or base_time
        self.last_send_time_A = - (self.PKT_SIZE * 8 / self.B_A)
        self.last_send_time_B = - (self.PKT_SIZE * 8 / self.B_B)

    def mode_determine(self, flow):
        if is_large_flow(flow, self.threshold):
            actual_send_mode = 'FPS'    # 大流：强制使用 FPS 多路径
        else:
            # 小流：使用主路径
            if self._should_send_redundancy_on_bad_path():
                actual_send_mode = 'RSR'  # 使用备份路径发送冗余包
            else:
                actual_send_mode = 'SP'  # 或根据 send_mode_config 决定
        print(f"Determined send mode: {actual_send_mode} for flow size {flow.total_bytes} bytes")
        return actual_send_mode

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
        """
        Enhanced decision: compare expected recovery time.
        Returns True if sending redundancy on bad path (B) yields lower expected recovery time.
        """
        L_A = self.loss_rate_a
        L_B = self.loss_rate_b
        T_A = self.RTT_A
        T_B = self.RTT_B
        RTO = self.RTO

        # Avoid division by zero or degenerate cases
        if L_A <= 0:
            return False  # No loss on A, no need for redundancy on B
        if L_B >= 10.0:
            return False  # B always drops, useless

        # Expected recovery time if redundancy stays on A
        E1 = T_A + (L_A ** 2) * RTO

        # Expected recovery time if redundancy goes to B
        term1 = (1 - L_A) * T_A
        term2 = L_A * (1 - L_B) * (T_A + T_B) / 2.0
        term3 = L_A * L_B * (RTO + T_A)
        E2 = term1 + term2 + term3

        # print(f"Expected Recovery Times: E1 (A)={E1:.4f}, E2 (B)={E2:.4f}")

        return E2 < E1
    
    def schedule(self, t, ev_type, data=None):
        heapq.heappush(self.events, (t, self.event_id, ev_type, data))
        self.event_id += 1

    def run(self):
        self.log_rx_event("--- Simulation Run Started ---")
        # Initial sending using the new modular approach
        # Base time can be 0 or calculated based on MPS logic (e.g., batch timing)
        base_initial_time = 0.0 
        for sn in range(min(self.max_window, self.num_pkts)):
            self._send_packet(sn, base_initial_time)

        try:
            while self.events and (self.next_sn < self.num_pkts or self.unacked):
                t, _, ev_type, data = heapq.heappop(self.events)
                
                if ev_type == 'arrive':
                    # Data now contains 'sn' and 'path'
                    self._on_arrival(t, data['sn'], data.get('path', 'N/A'))
                elif ev_type == 'ack':
                    self._on_ack(t, data['cumulative'], data['selective'])
                elif ev_type == 'timeout':
                    # Data now contains 'sn' and 'path'
                    if data['sn'] in self.unacked:
                        self._retransmit(t, data['sn'], data.get('path', 'N/A'))

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

    # --- Module for sending packets ---
    def _send_packet(self, original_sn, base_time):
        """
        Generic send packet function.
        This function calls the specific multi-path algorithm's send logic.
        """
        # Currently, we only have FPS implemented.
        if self.sendMode == 'FPS':
            self.fps_send_packet(original_sn, base_time)
        elif self.sendMode == 'SP':
            self.sp_send_packet(original_sn, base_time)
        elif self.sendMode == 'RSR':
            self.rsr_send_packet(original_sn, base_time)
        else:
            raise ValueError(f"Unsupported sendMode: {self.sendMode}")

    def _send_physical_packet_on_path(self, sn, send_time, path, is_redundant=False, pkt_type='data'):
        """
        Send a single physical packet (no redundancy expansion).
        Used by RSR and other advanced modes that pre-expand packets.
        """
        if path == 'A':
            loss_rate = self.loss_rate_a
            rtt = self.RTT_A
        elif path == 'B':
            loss_rate = self.loss_rate_b
            rtt = self.RTT_B
        else:
            raise ValueError(f"Unknown path: {path}")

        # Simulate loss
        if random.random() >= loss_rate:
            arrive_time = send_time + rtt / 2
            self.schedule(arrive_time, 'arrive', {'sn': sn, 'path': path})
            log_type = "REPLICATE" if is_redundant and pkt_type != 'xor' else \
                    "XOR" if pkt_type == 'xor' else \
                    "REPLICATE_K_1" if is_redundant else \
                    "ORIGINAL"
            self.log_tx_event(f"[SEND] Scheduling {log_type} packet {sn} on Path {path} arrival at {arrive_time:.4f}")
        else:
            log_type = "REPLICATE" if is_redundant and pkt_type != 'xor' else \
                    "XOR" if pkt_type == 'xor' else \
                    "REPLICATE_K_1" if is_redundant else \
                    "ORIGINAL"
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
                # Original packet → path A
                path = 'A'
                interval = self.PKT_SIZE * 8 / self.B_A
                send_time = max(base_time, self.last_send_time_A + interval)
                self.last_send_time_A = send_time
            else:
                # Redundancy packet
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

        # Set timeout only for the original logical packet
        self.unacked[original_sn] = (base_time, 'A')
        self.schedule(base_time + self.RTO, 'timeout', {'sn': original_sn, 'path': 'A'})
       

    def sp_send_packet(self, original_sn, base_time):
        """ Implements Single Path (SP) sending: all packets go through fast path A. """
        if original_sn >= self.num_pkts:
            return

        path = 'A'  # Use only the fast path

        # Calculate send time based on last packet sent on path A
        path_interval_A = self.PKT_SIZE * 8 / self.B_A
        send_time = max(base_time, self.last_send_time_A + path_interval_A)
        self.last_send_time_A = send_time

        self._send_packet_on_path(original_sn, send_time, path)
    
    def fps_send_packet(self, original_sn, base_time):
        """
        Implements the FPS (Fast Path Slow Path) algorithm logic.
        """
        if original_sn >= self.num_pkts:
            return

        # self.log_tx_event(f"[FPS111] now original_sn :  {original_sn}; last_fast_sn : {self.last_fast_sn}")

        # Determine if this packet should go on fast path or slow path
        # According to FPS: Fast path sends a batch (s1 to s2), then slow path sends s2+1
        # Check if this sn is part of the current fast path batch
        # Use the variable name 'next_expected_fast_start_sn' for clarity if needed, but using existing var

        if original_sn < self.delay_packet_diff:
            path = self.fast_path
            self.last_fast_sn = original_sn

        else:
            if self.last_fast_sn < self.current_batch_start_sn + self.batch_size - 1:
                # self.log_tx_event(f"[FPS222] now original_sn :  {original_sn}; last_fast_sn : {self.last_fast_sn}; self.current_batch_start_sn + self.batch_size - 1 : {self.current_batch_start_sn + self.batch_size - 1}")
                # Send on fast path A
                path = self.fast_path
                self.last_fast_sn = original_sn
                # If this was the last packet of the current fast batch
                if self.last_fast_sn == self.current_batch_start_sn + self.batch_size - 1:
                    # DO NOT update self.current_batch_start_sn here yet.
                    # It will be updated when the slow packet (last_fast_sn + 1) is processed.
                    # The slow packet to send is last_fast_sn + 1
                    self.log_tx_event(f"[FPS] Fast batch [{self.last_fast_sn - self.batch_size + 1}, {self.last_fast_sn}] completed. Waiting for slow packet {self.last_fast_sn + 1} before updating next fast batch start.")
            else:
                # This packet should be the slow packet (s2+1) following the fast batch
                # Or it might be the start of a new fast batch if it follows the previous slow packet
                expected_slow_sn = self.last_fast_sn + 1
                # self.log_tx_event(f"[FPS111] now original_sn :  {original_sn}; last_fast_sn : {self.last_fast_sn}; expected_slow_sn : {expected_slow_sn}")
                if original_sn == expected_slow_sn:
                    # Send on slow path B
                    path = self.slow_path
                    self.last_slow_sn = original_sn
                    # NOW update the start of the next fast batch, as the slow packet has been handled.
                    self.current_batch_start_sn = self.last_slow_sn + 1
                    self.log_tx_event(f"[FPS] Sent slow packet {original_sn}. Next fast batch starts at {self.current_batch_start_sn}.")
                else:
                    # This case handles packets that don't strictly follow the FPS s1-s2-s2+1 pattern.
                    # This might occur if the initial window fills beyond the first batch,
                    # or if redundancy creates extra packets out of the strict sequence.
                    # For simplicity in FPS, we'll still route them according to the pattern logic.
                    # If the sequence number is exactly one more than the last slow packet, it's the next fast batch starter
                    if original_sn == self.last_slow_sn + 1:
                        path = self.fast_path
                        self.last_fast_sn = original_sn
                        # This packet becomes the start of a new fast batch
                        # Update current_batch_start_sn only if necessary, maybe redundant if already set correctly after prev slow pkt
                        # self.current_batch_start_sn = original_sn # Keep commented or remove, as it should already be correct
                        # It's possible that after a slow packet, the next fast batch start is already set correctly by the slow packet handler.
                        # Only update if the logic requires it differently.
                        # Let's re-evaluate: if last_slow_sn was N, then the slow packet handler sets current_batch_start_sn = N + 1.
                        # If original_sn is N+1, it matches the already set value. No update needed here.
                        self.log_tx_event(f"[FPS] Packet {original_sn} continues fast path after slow packet (starts at {self.current_batch_start_sn}).")
                    else:
                        # Default to fast path if it doesn't fit the immediate slow/fast pattern
                        # This covers cases like initial window packets beyond the first batch
                        # Note: This default might still be problematic if packets arrive out of order significantly
                        path = self.fast_path
                        self.last_fast_sn = original_sn
                        self.log_tx_event(f"[FPS] Packet {original_sn} routed to fast path (default/initial window).")

        # --- CORRECTED Send Time Calculation Logic ---
        path_intervals = {'A': self.PKT_SIZE * 8 / self.B_A, 'B': self.PKT_SIZE * 8 / self.B_B}

        # Calculate send time based on the last packet sent on the target path
        # This ensures packets are spaced according to the path's bandwidth
        if path == 'A':
            # New send time is the last send time on A plus the transmission time of one packet on A
            # Use max to ensure send time doesn't go back in time relative to base_time if needed
            # self.log_tx_event(f"[aaaaaa] base_time :  {base_time}; last_send_time_A : {self.last_send_time_A}; path_intervals['A'] : {path_intervals['A']}")
            send_time = max(base_time, self.last_send_time_A + path_intervals['A'])
            self.last_send_time_A = send_time # Update the last send time for path A
        elif path == 'B':
            # New send time is the last send time on B plus the transmission time of one packet on B
            # Use max to ensure send time doesn't go back in time relative to base_time if needed
            send_time = max(base_time, self.last_send_time_B + path_intervals['B'])
            self.last_send_time_B = send_time # Update the last send time for path B
        else:
            # Handle unexpected path, fallback to original base_time
            send_time = base_time
            self.log_tx_event(f"WARNING: Unexpected path {path} for packet {original_sn}")

        # self.log_tx_event(f"[bbbbbb] now original_sn :  {original_sn}; last_fast_sn : {self.last_fast_sn}; send_time : {send_time:.4f}; path : {path}")

        # Call the path-specific sending function
        self._send_packet_on_path(original_sn, send_time, path)


    def _send_packet_on_path(self, original_sn, send_time, path):
        """ Legacy function for FPS/SP: expands redundancy internally. """
        if original_sn >= self.num_pkts:
            return

        packets_to_send = self.redundancy_manager.get_packets_to_send(original_sn)
        current_offset = 0.0

        if path == 'A':
            base_interval = self.PKT_SIZE * 8 / self.B_A
        else:
            base_interval = self.PKT_SIZE * 8 / self.B_B

        for pkt_info in packets_to_send:
            sn = pkt_info['sn']
            is_redundant = pkt_info.get('is_redundant', False)
            pkt_type = pkt_info.get('type', 'data')
            physical_time = send_time + current_offset
            self._send_physical_packet_on_path(sn, physical_time, path, is_redundant, pkt_type)
            current_offset += base_interval

        # Set timeout for original packet
        self.unacked[original_sn] = (send_time, path)
        self.schedule(send_time + self.RTO, 'timeout', {'sn': original_sn, 'path': path})

    def _on_arrival(self, t, pkt_identifier, path_received='N/A'):
        sn = pkt_identifier
        self.log_rx_event(f"[{t:.4f}] RX: Packet {sn} arrived on Path {path_received}.")

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

                ack_arrival_time = t + self.RTT_A / 2
                self.schedule(ack_arrival_time, 'ack', {'cumulative': cumulative, 'selective': selective})

                while (self.next_sn < self.num_pkts and
                       (self.next_sn - (self.next_expected - 1)) < self.max_window):
                    # Use a placeholder base time; actual MPS timing is handled by the specific algorithm function
                    send_time = self.next_sn * self.PKT_SIZE * 8 / self.B_A
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
            if (-sn + 1) % self.redundancy_manager.k_rep == 0 and sn != 0 and sn < 0:
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
                    # Use a placeholder base time; actual MPS timing is handled by the specific algorithm function
                    self._send_packet(self.next_sn, event_time)
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
        # Simplified ACK processing - removes acknowledged packets from unacked map
        # In a real multi-path scenario, this could be more complex
        to_remove = [sn for sn in self.unacked if sn <= cumulative]
        for sn in to_remove:
            del self.unacked[sn]
        # Note: Selective ACKs are noted but not fully processed here
        # for loop over selective might be needed depending on impl detail

    def _retransmit(self, t, sn, path_for_retransmit):
        if sn not in self.unacked:
            return # Already acknowledged

        # Get parameters for the path to retransmit on
        if path_for_retransmit == 'A':
             loss_rate = self.loss_rate_a
             rtt = self.RTT_A
        elif path_for_retransmit == 'B':
             loss_rate = self.loss_rate_b
             rtt = self.RTT_B
        else:
             # Fallback if path is unknown, maybe use the original path stored in unacked
             # For this version, we assume the path is correctly passed down.
             # Retrieve path from unacked dict if not provided explicitly in timeout event data
             _, stored_path = self.unacked.get(sn, (None, 'A')) # Default to A if somehow missing
             path_for_retransmit = stored_path
             loss_rate = self.loss_rate_a if stored_path == 'A' else self.loss_rate_b
             rtt = self.RTT_A if stored_path == 'A' else self.RTT_B
             self.log_tx_event(f"[RETRANS] WARNING: Path for retransmit {sn} not specified, using stored path {stored_path}.")

        if random.random() >= loss_rate:
            arrive_time = t + rtt / 2
            # Pass the path information for the retransmitted packet
            self.schedule(arrive_time, 'arrive', {'sn': sn, 'path': path_for_retransmit})
            self.log_tx_event(f"[RETRANS] Scheduling RETRANSMISSION packet {sn} on {path_for_retransmit} arrival at {arrive_time:.4f}")
        else:
            self.log_tx_event(f"[RETRANS] RETRANSMISSION packet {sn} on {path_for_retransmit} LOST at {t:.4f}")
            
        # Update unacked time and reschedule timeout, keeping the path
        self.unacked[sn] = (t, path_for_retransmit)
        self.schedule(t + self.RTO, 'timeout', {'sn': sn, 'path': path_for_retransmit})

def determine_redundancy_mode(loss_rate_a):
    """根据链路A的丢包率自动选择冗余模式。"""
    if loss_rate_a < 0.04:
        return 'none'
    elif loss_rate_a <= 0.12:
        return 'xor_4_1'
    else: # loss_rate_a > 0.12
        return 'replicate_4_1'

# ======================
# CLI
# ======================
def simulate(loss_rate_a, loss_rate_b, flow=None):
    """模拟函数，冗余模式由链路A的丢包率自动决定。"""
    redundancy_mode = determine_redundancy_mode(loss_rate_a)
    if flow is None:
        # 默认向后兼容：使用 Config.FLOW_SIZE 创建一个 Flow
        flow = Flow(flow_id=0, total_bytes=Config().FLOW_SIZE)
    sim = FastSim(loss_rate_a, loss_rate_b, redundancy_mode, flow)
    result = sim.run()
    return result

def run_multi_flow_test():
    """运行多流混合场景，但是是串行不是并行"""
    flows = [
        Flow(0, 500 * 1024),          # 小流
        Flow(1, 5 * 1024 * 1024),    # 小流（<5MB?）
        Flow(2, 50 * 1024 * 1024),    # 大流
        Flow(3, 200 * 1024 * 1024),   # 大流
    ]
    results = []
    for flow in flows:
        print(f"\n--- Running flow {flow.flow_id} ({flow.total_bytes / 1e6:.1f} MB) ---")
        res = simulate(0.03, 0.2, flow=flow)
        results.append((flow.flow_id, res))
    
    for fid, res in results:
        print(f"Flow {fid}: {res['throughput_mbps']:.2f} Mbps")

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <loss_rate_a> <loss_rate_b> <redundancy_mode>")
        print("  loss_rate_a: e.g., 0.05")
        print("  loss_rate_b: e.g., 0.10")
        sys.exit(1)
    
    loss_rate_a = float(sys.argv[1])
    loss_rate_b = float(sys.argv[2])
    
    random.seed(42)
    result = simulate(loss_rate_a, loss_rate_b)
    
    total_pkts = (Config().FLOW_SIZE + Config().PKT_SIZE - 1) // Config().PKT_SIZE
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    redundancy_mode = determine_redundancy_mode(loss_rate_a) # 获取用于打印的模式
    print(f"TX Log saved to file: log_tx_{redundancy_mode}_loss_{loss_rate_a:.2f}_{loss_rate_b:.2f}_{timestamp}.txt")
    print(f"RX Log saved to file: log_rx_{redundancy_mode}_loss_{loss_rate_a:.2f}_{loss_rate_b:.2f}_{timestamp}.txt")
    print(f"Loss Rates A/B={loss_rate_a:.1%}/{loss_rate_b:.1%}, Auto-selected Redundancy={redundancy_mode}")
    print(f"Throughput: {result['throughput_mbps']:.2f} Mbps")
    print(f"Avg Queue Length: {result['avg_queue_length']:.1f} packets")
    print(f"Delivered Packets: {result['delivered_packets']} / {total_pkts}")

if __name__ == '__main__':
    # main()  # 单流
    run_multi_flow_test()  # 临时测试多流