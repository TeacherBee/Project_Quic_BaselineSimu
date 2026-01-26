import heapq
import random
import sys
import re
from collections import defaultdict
import datetime

useLog = False

class Config:
    B = 100e6               # 100 Mbps
    RTT = 0.6               # 600 ms
    PKT_SIZE = 1250         # bytes
    FLOW_SIZE = 100 * 1024 * 1024  # 100 MB
    RTO = 1.0               # Timeout

class FastSim:
    def __init__(self, loss_rate, redundancy):
        self.loss_rate = loss_rate
        self.redundancy = redundancy
        
        # Parse redundancy mode
        self.mode = 'none'
        self.k_rep = None  # for replicate_k_1 and xor_k_1
        self.xor_groups = {} # For xor_k_1: maps group_id -> {pkts_sent, pkts_arrived}
        
        if redundancy == 'none':
            self.mode = 'none'
        elif redundancy == 'replicate':
            self.mode = 'replicate'
            self.k_rep = 1
        elif redundancy.startswith('replicate_'):
            match = re.match(r'replicate_(\d+)_1', redundancy)
            if not match:
                raise ValueError("Use 'replicate_k_1', e.g., 'replicate_4_1'")
            self.k_rep = int(match.group(1))
            if self.k_rep <= 0:
                raise ValueError("k must be positive")
            self.mode = 'replicate_k_1'
        elif redundancy.startswith('xor_'):
            match = re.match(r'xor_(\d+)_1', redundancy)
            if not match:
                raise ValueError("Use 'xor_k_1', e.g., 'xor_4_1'")
            self.k_rep = int(match.group(1))
            if self.k_rep <= 0:
                raise ValueError("k must be positive")
            self.mode = 'xor_k_1'
        else:
            raise ValueError("Redundancy must be: 'none', 'replicate', 'replicate_k_1' (e.g., 'replicate_4_1'), or 'xor_k_1' (e.g., 'xor_4_1')")

        cfg = Config()
        self.B = cfg.B
        self.RTT = cfg.RTT
        self.PKT_SIZE = cfg.PKT_SIZE
        self.FLOW_SIZE = cfg.FLOW_SIZE
        self.RTO = cfg.RTO
        
        self.num_pkts = (self.FLOW_SIZE + self.PKT_SIZE - 1) // self.PKT_SIZE
        base_interval = self.PKT_SIZE * 8 / self.B  # time to send one packet
        
        # 🔧 Compute effective send interval based on redundancy overhead
        if self.mode == 'none':
            self.send_interval = base_interval
        elif self.mode == 'replicate':
            self.send_interval = base_interval * 2  # 1 orig + 1 red per packet
        elif self.mode == 'replicate_k_1' or self.mode == 'xor_k_1':
            # Every k original packets -> k + 1 physical packets
            self.send_interval = base_interval * (self.k_rep + 1) / self.k_rep
        else:
            self.send_interval = base_interval

        # 用于管理当前恢复组的状态变量
        if self.mode == 'xor_k_1':
            self.current_group_start = None
            self.current_group_status = set() # 存储当前组中已收到的元素 (sn or -sn)
            self.group_size = self.k_rep + 1 # k_xor 个数据包 + 1 个 XOR 包
        
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
            # Define separate log file names with './log/' prefix
            self.log_tx_filename = f"./log/log_tx_{redundancy}_loss_{loss_rate:.2f}_{timestamp}.txt"
            self.log_rx_filename = f"./log/log_rx_{redundancy}_loss_{loss_rate:.2f}_{timestamp}.txt"
            
            # Open separate file handles
            try:
                self.log_tx_handle = open(self.log_tx_filename, 'w')
                self.log_rx_handle = open(self.log_rx_filename, 'w')
                
                # Write initial headers
                header_info = f"--- Simulation Log Started at {datetime.datetime.now()} ---\n"
                config_info = f"Mode: {self.mode}, k: {self.k_rep}, Loss Rate: {self.loss_rate}\nTotal Packets to Send: {self.num_pkts}\n\n"
                
                self.log_tx_handle.write(header_info + "LOG TYPE: TRANSMISSION EVENTS\n" + config_info)
                self.log_rx_handle.write(header_info + "LOG TYPE: RECEPTION EVENTS\n" + config_info)
                
                # Flush headers immediately
                self.log_tx_handle.flush()
                self.log_rx_handle.flush()
                
            except IOError as e:
                print(f"Error opening log files: {e}", file=sys.stderr)
                raise # Re-raise the exception to stop execution if files can't be opened


    # Define separate logging methods
    def log_tx_event(self, message):
        """Helper function to write transmission-related messages to the log file."""
        if hasattr(self, 'log_tx_handle') and self.log_tx_handle and useLog:
            self.log_tx_handle.write(message + "\n")
            self.log_tx_handle.flush() # Ensure immediate write

    def log_rx_event(self, message):
        """Helper function to write reception-related messages to the log file."""
        if hasattr(self, 'log_rx_handle') and self.log_rx_handle and useLog:
            self.log_rx_handle.write(message + "\n")
            self.log_rx_handle.flush() # Ensure immediate write

    # Define separate closing methods
    def close_log_tx(self):
        """Close the transmission log file handle."""
        if hasattr(self, 'log_tx_handle') and self.log_tx_handle and not self.log_tx_handle.closed and useLog:
            self.log_tx_handle.close()
            self.log_tx_handle = None

    def close_log_rx(self):
        """Close the reception log file handle."""
        if hasattr(self, 'log_rx_handle') and self.log_rx_handle and not self.log_rx_handle.closed and useLog:
            self.log_rx_handle.close()
            self.log_rx_handle = None

    # Convenience method to close both
    def close_logs(self):
        """Close both log file handles."""
        self.close_log_tx()
        self.close_log_rx()

    def schedule(self, t, ev_type, data=None):
        heapq.heappush(self.events, (t, self.event_id, ev_type, data))
        self.event_id += 1

    def run(self):
        # Log initial setup to RX log (as it's about the receiver's state)
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

            # Final logging before calculations - goes to RX log
            self.log_rx_event("\n--- Simulation Run Ended ---")
            self.log_rx_event(f"Final next_expected: {self.next_expected}")
            self.log_rx_event(f"Final delivered: {self.delivered}")
            self.log_rx_event(f"Final receiver_buffer: {sorted(list(self.receiver_buffer))}")
        finally:
            # Ensure both log files are closed regardless of how run() exits
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
        # Logs are already closed by the 'finally' block
        return results

    def _send_packet(self, sn, send_time):
        if sn >= self.num_pkts:
            return

        # Calculate base time for sending one packet
        base_pkt_time = self.PKT_SIZE * 8 / self.B

        # Send original packet
        if random.random() >= self.loss_rate:
            arrive_time = send_time + self.RTT / 2
            self.schedule(arrive_time, 'arrive', {'sn': sn})
            # --- ADDED LOGGING FOR ORIGINAL PACKET TO TX LOG ---
            self.log_tx_event(f"[SEND] Scheduling ORIGINAL packet {sn} arrival at {arrive_time:.4f}")
            # -----------------------------------------
        else:
            # Log if the original packet is lost to TX log
            self.log_tx_event(f"[SEND] ORIGINAL packet {sn} LOST at {send_time:.4f}")


        # Send redundant packet based on mode
        if self.mode == 'replicate':
            # Send redundant copy right after
            red_time = send_time + base_pkt_time
            if random.random() >= self.loss_rate:
                arrive_time = red_time + self.RTT / 2
                self.schedule(arrive_time, 'arrive', {'sn': sn})
                # --- ADDED LOGGING FOR REDUNDANT PACKET TO TX LOG ---
                self.log_tx_event(f"[SEND] Scheduling REPLICATE packet {sn} arrival at {arrive_time:.4f}")
                # -----------------------------------------
            else:
                self.log_tx_event(f"[SEND] REPLICATE packet {sn} LOST at {red_time:.4f}")

        elif self.mode == 'replicate_k_1':
            # Check if this is the k-th packet in a group of k
            if (sn + 1) % self.k_rep == 0 or sn == self.num_pkts - 1:
                red_time = send_time + base_pkt_time
                if random.random() >= self.loss_rate:
                    arrive_time = red_time + self.RTT / 2
                    self.schedule(arrive_time, 'arrive', {'sn': sn})
                    # --- ADDED LOGGING FOR REDUNDANT PACKET TO TX LOG ---
                    self.log_tx_event(f"[SEND] Scheduling REPLICATE_K_1 packet {sn} arrival at {arrive_time:.4f}")
                    # -----------------------------------------
                else:
                    self.log_tx_event(f"[SEND] REPLICATE_K_1 packet {sn} LOST at {red_time:.4f}")

        elif self.mode == 'xor':
            # XOR packet is sent after base_pkt_time
            xor_time = send_time + base_pkt_time
            if random.random() >= self.loss_rate:
                arrive_time = xor_time + self.RTT / 2
                self.schedule(arrive_time, 'arrive', {'sn': -sn})
                # --- ADDED LOGGING FOR REDUNDANT PACKET TO TX LOG ---
                self.log_tx_event(f"[SEND] Scheduling XOR packet {-sn} arrival at {arrive_time:.4f}")
                # -----------------------------------------
            else:
                self.log_tx_event(f"[SEND] XOR packet {-sn} LOST at {xor_time:.4f}")

        elif self.mode == 'xor_k_1':
            # Check if this is the k-th packet in a group of k
            # FIXED: The condition should be (sn + 1) % self.k_rep == 0 for k=4, group is [0,1,2,3,-3], [4,5,6,7,-7]...
            if (sn + 1) % self.k_rep == 0 or sn == self.num_pkts - 1: 
                # XOR result of the group
                xor_time = send_time + base_pkt_time
                if random.random() >= self.loss_rate:
                    arrive_time = xor_time + self.RTT / 2
                    self.schedule(arrive_time, 'arrive', {'sn': -sn})
                    # --- ADDED LOGGING FOR REDUNDANT PACKET TO TX LOG ---
                    self.log_tx_event(f"[SEND] Scheduling XOR_K_1 packet {-sn} arrival at {arrive_time:.4f}")
                    # -----------------------------------------
                else:
                    self.log_tx_event(f"[SEND] XOR_K_1 packet {-sn} LOST at {xor_time:.4f}")

        # Set timeout for original packet
        self.unacked[sn] = send_time
        self.schedule(send_time + self.RTO, 'timeout', {'sn': sn})

    def _on_arrival(self, t, pkt_identifier):
        sn = pkt_identifier
        self.log_rx_event(f"[{t:.4f}] RX: Packet {sn} arrived.")

        # --- Check if this packet triggers a new group ---
        if self.mode == 'xor_k_1':
            if sn == 0:
                # First packet always starts the first group
                self.current_group_start = 0
                self.current_group_status = set()
                self.log_rx_event(f"[{t:.4f}]  -> Packet {sn} starts the first XOR group at {self.current_group_start}.")

            # --- Add the current packet to the relevant group status ---
            if self.current_group_start is not None:
                group_end = self.current_group_start + self.k_rep - 1 # e.g., if start is 4, end is 7
                expected_xor_sn = -group_end                  # e.g., if end is 7, xor_sn is -7
                relevant_group_elements = set(range(self.current_group_start, group_end + 1))
                relevant_group_elements.add(expected_xor_sn)

                if sn in relevant_group_elements:
                    if sn not in self.current_group_status:
                        self.current_group_status.add(sn)
                        self.log_rx_event(f"[{t:.4f}]     Added {sn} to current group [{self.current_group_start}, {group_end}, {expected_xor_sn}] status. Current status: {sorted(list(self.current_group_status))} / {self.group_size}")
                    else:
                        self.log_rx_event(f"[{t:.4f}]     {sn} was already in current group status. Skipping.")

                    # --- Check for recovery condition ---
                    if len(self.current_group_status) >= self.group_size - 1: # >= 4 out of 5
                        self.log_rx_event(f"[{t:.4f}]     *** Recovery Triggered! Group [{self.current_group_start}, {group_end}, {expected_xor_sn}] has {len(self.current_group_status)}/{self.group_size} elements. Attempting recovery.")
                        self._attempt_recovery(t, t) # Pass arrival time for recovery logs
                    # else:
                    #     self.log_event(f"     Group has {len(self.current_group_status)}/{self.group_size} elements. Waiting for more.")


        # --- Process the packet normally (data or XOR) ---

        if sn < 0: # It's an XOR packet
            # The XOR packet itself is now tracked in self.current_group_status if it belongs to the active group.
            # Its content is used for recovery, which happens above if condition is met.
            # We don't process it as a "regular" old/duplicate packet here anymore under this new logic,
            # because its role is specifically tied to the group it belongs to.
            # However, if it doesn't belong to the *current* group, maybe log it separately.
            # The main logic assumes it belongs to the current group if added to status.
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

                # Check buffer for consecutive packets
                while self.next_expected in self.receiver_buffer:
                    self.delivered += 1
                    self.log_rx_event(f"[{t:.4f}]  -> Packet {self.next_expected} is consecutive. Delivering. New delivered count: {self.delivered}")
                    del self.receiver_buffer[self.next_expected]
                    self.next_expected += 1

                self.log_rx_event(f"[{t:.4f}]  -> Updated next_expected to {self.next_expected}. Checking buffer for consecutive packets...")
                self.log_rx_event(f"[{t:.4f}]  -> Final next_expected after delivery: {self.next_expected}")
                cumulative = self.next_expected - 1
                selective = [s for s in self.receiver_buffer if s > cumulative]
                self.log_rx_event(f"[{t:.4f}]  -> Sending ACK: Cumulative={cumulative}, Selective={selective}")
                self.schedule(t + 1e-9, 'ack', {'cumulative': cumulative, 'selective': selective})

                # Schedule new packets based on current state (window control)
                while (self.next_sn < self.num_pkts and
                       (self.next_sn - (self.next_expected - 1)) < self.max_window):
                    send_time = self.next_sn * self.send_interval
                    if self.next_sn < 10:
                        # Scheduling new packets is a sender action, could go to TX log
                        # But it's triggered by reception, so RX log is also fine. Let's keep it in RX for now.
                        self.log_rx_event(f"[{t:.4f}]  -> Scheduling next packet {self.next_sn} at {send_time:.4f}")
                    self._send_packet(self.next_sn, send_time)
                    self.next_sn += 1

            else: # sn > next_expected
                # Add to buffer
                if sn not in self.receiver_buffer: # Avoid adding duplicates to buffer if somehow scheduled twice
                    self.receiver_buffer[sn] = t
                    self.log_rx_event(f"[{t:.4f}]  -> Packet {sn} is not expected (expected {self.next_expected}). Adding to buffer. Buffer now: {sorted(list(self.receiver_buffer.keys()))}")
                else:
                    self.log_rx_event(f"[{t:.4f}]  -> Packet {sn} is not expected and already in buffer. Ignoring duplicate arrival.")

                cumulative = self.next_expected - 1
                selective = [s for s in self.receiver_buffer if s > cumulative]
                self.log_rx_event(f"[{t:.4f}]  -> Sending ACK: Cumulative={cumulative}, Selective={selective}")
                self.schedule(t + 1e-9, 'ack', {'cumulative': cumulative, 'selective': selective})

        if self.mode == 'xor_k_1':
            if (-sn + 1) % self.k_rep == 0 and sn != 0 and sn < 0:
                new_group_start = -sn + 1
                if new_group_start != self.current_group_start:
                    self.log_rx_event(f"[{t:.4f}]  -> Packet {sn} triggers a new XOR group starting at {new_group_start}. Clearing old group status.")
                    self.current_group_start = new_group_start
                    self.current_group_status = set()
                    # No need to explicitly add sn itself to the status here, it's the trigger packet.


        self.buffer_history.append((t, len(self.receiver_buffer)))


    def _attempt_recovery(self, t, event_time):
        """Attempts to recover missing packets in the current group."""
        if self.current_group_start is None:
            self.log_rx_event(f"[{event_time:.4f}]  *** ERROR: _attempt_recovery called but no active group.")
            return

        group_start = self.current_group_start
        group_end = group_start + self.k_rep - 1 # e.g., 4, 7
        expected_xor_sn = -group_end              # e.g., -7
        all_group_elements = set(range(group_start, group_end + 1))
        all_group_elements.add(expected_xor_sn)

        # Find which elements are missing from the group
        missing_elements = all_group_elements - self.current_group_status
        self.log_rx_event(f"[{event_time:.4f}]     *** Identifying missing elements: {missing_elements}")

        # Identify which *data* packets are missing
        missing_data_packets = [elem for elem in missing_elements if elem >= 0]

        if len(missing_data_packets) == 0:
            self.log_rx_event(f"[{event_time:.4f}]     *** All data packets in group are present. No recovery needed.")
            # Reset group tracking even if no recovery was needed
            self.current_group_start = None
            self.current_group_status = set()
            return
        elif len(missing_data_packets) > 1:
            # This shouldn't happen if the >=4 check passed, but let's handle it.
            self.log_rx_event(f"[{event_time:.4f}]     *** WARNING: More than one data packet missing ({missing_data_packets}), cannot recover reliably. Recovery condition check might be incorrect.")
            # Reset group tracking on failure too
            self.current_group_start = None
            self.current_group_status = set()
            return
        else: # len(missing_data_packets) == 1
            recovered_sn = missing_data_packets[0]
            self.log_rx_event(f"[{event_time:.4f}]     *** Successfully identified missing data packet: {recovered_sn}")

            # Now, handle the recovered packet like a normal arrival
            # Case 1: Recovered packet is the one we are expecting
            if recovered_sn == self.next_expected:
                self.delivered += 1
                self.next_expected += 1
                self.log_rx_event(f"[{event_time:.4f}]     *** Recovered packet {recovered_sn} is expected. Delivered. New delivered: {self.delivered}, next_expected: {self.next_expected}")

                # Check buffer for consecutive packets that might now be deliverable
                while self.next_expected in self.receiver_buffer:
                    self.delivered += 1
                    self.log_rx_event(f"[{event_time:.4f}]  *** Delivered buffered packet {self.next_expected}. New delivered: {self.delivered}")
                    del self.receiver_buffer[self.next_expected]
                    self.next_expected += 1
                self.log_rx_event(f"[{event_time:.4f}]     *** Final next_expected after recovery delivery: {self.next_expected}")

                # Send updated ACK
                cumulative_ack = self.next_expected - 1
                selective_ack = [s for s in self.receiver_buffer if s > cumulative_ack]
                self.log_rx_event(f"[{event_time:.4f}]     *** Sending ACK after recovery: Cumulative={cumulative_ack}, Selective={selective_ack}")
                self.schedule(event_time + 1e-9, 'ack', {'cumulative': cumulative_ack, 'selective': selective_ack})

                # Potentially schedule more packets due to advancement
                while (self.next_sn < self.num_pkts and
                       (self.next_sn - (self.next_expected - 1)) < self.max_window):
                    send_time = self.next_sn * self.send_interval
                    if self.next_sn < 10:
                        # Scheduling due to recovery is a sender action, could go to TX log
                        # But it's triggered by reception, so RX log is also fine. Let's keep it in RX.
                        self.log_rx_event(f"[{event_time:.4f}]     *** Scheduling next packet {self.next_sn} at {send_time:.4f} (due to recovery)")
                    self._send_packet(self.next_sn, send_time)
                    self.next_sn += 1

            else: # Case 2: Recovered packet is not the one we are expecting, add to buffer
                # Check if it's already in the buffer (shouldn't happen if logic is correct, but good to be safe)
                if recovered_sn not in self.receiver_buffer:
                    self.receiver_buffer[recovered_sn] = event_time # Store arrival time or placeholder
                    self.log_rx_event(f"[{event_time:.4f}]     *** Recovered packet {recovered_sn} is not expected ({self.next_expected}). Added to buffer. Buffer now: {sorted(list(self.receiver_buffer.keys()))}")

                    # Send ACK reflecting the new buffered packet
                    cumulative_ack = self.next_expected - 1
                    selective_ack = [s for s in self.receiver_buffer if s > cumulative_ack]
                    self.log_rx_event(f"[{event_time:.4f}]     *** Sending ACK after recovery (to buffer): Cumulative={cumulative_ack}, Selective={selective_ack}")
                    self.schedule(event_time + 1e-9, 'ack', {'cumulative': cumulative_ack, 'selective': selective_ack})
                else:
                    self.log_rx_event(f"[{event_time:.4f}]     *** Recovered packet {recovered_sn} was already in buffer. Skipping.")

            # Reset group tracking after successful recovery attempt (whether packet was delivered or buffered)
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
        # ACK processing is part of the sender's state update, could be logged in TX log.
        # But it's triggered by events potentially related to reception. For now, keeping in RX log context.
        # self.log_rx_event(f"[{t:.4f}] ACK processed. Cumulative={cumulative}, Selective={selective}. Unacked now: {sorted(list(self.unacked.keys()))}")

    def _retransmit(self, t, sn):
        if sn not in self.unacked:
            return
        if random.random() >= self.loss_rate:
            arrive_time = t + self.RTT / 2
            self.schedule(arrive_time, 'arrive', {'sn': sn})
            # Retransmission scheduling is a sender action, could go to TX log
            self.log_tx_event(f"[RETRANS] Scheduling RETRANSMISSION packet {sn} arrival at {arrive_time:.4f}")
        else:
            # Retransmission loss, goes to TX log
            self.log_tx_event(f"[RETRANS] RETRANSMISSION packet {sn} LOST at {t:.4f}")
            
        self.unacked[sn] = t
        self.schedule(t + self.RTO, 'timeout', {'sn': sn})


# ======================
# CLI
# ======================
def simulate(loss_rate, redundancy):
    sim = FastSim(loss_rate, redundancy)
    # The 'finally' block in sim.run() ensures logs are closed
    result = sim.run()
    return result

def main():
    if len(sys.argv) != 3:
        print("Usage: python fast_sim.py <loss_rate> <redundancy>")
        print("  loss_rate: e.g., 0.05")
        print("  redundancy options:")
        print("    - 'none'")
        print("    - 'replicate' (1:1 redundancy)")
        print("    - 'replicate_k_1' (e.g., 'replicate_4_1' → every 4 packets, add 1 redundant copy)")
        print("    - 'xor_k_1' (e.g., 'xor_4_1' → every 4 packets, add 1 XOR redundant copy)")
        sys.exit(1)
    
    loss_rate = float(sys.argv[1])
    redundancy = sys.argv[2]
    
    random.seed(42)
    result = simulate(loss_rate, redundancy)
    
    total_pkts = (Config().FLOW_SIZE + Config().PKT_SIZE - 1) // Config().PKT_SIZE
    # Print the names of the generated log files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Get current timestamp again for display
    print(f"TX Log saved to file: log_tx_{redundancy}_loss_{loss_rate:.2f}_{timestamp}.txt")
    print(f"RX Log saved to file: log_rx_{redundancy}_loss_{loss_rate:.2f}_{timestamp}.txt")
    print(f"Loss={loss_rate:.1%}, Redundancy={redundancy}")
    print(f"Throughput: {result['throughput_mbps']:.2f} Mbps")
    print(f"Avg Queue Length: {result['avg_queue_length']:.1f} packets")
    print(f"Delivered Packets: {result['delivered_packets']} / {total_pkts}")

if __name__ == '__main__':
    main()