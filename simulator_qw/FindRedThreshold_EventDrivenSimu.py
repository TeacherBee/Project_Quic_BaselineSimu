# fast_sim.py
import heapq
import random
import sys
from collections import defaultdict

class Config:
    B = 100e6               # 100 Mbps
    RTT = 0.6               # 600 ms
    PKT_SIZE = 1250         # bytes
    FLOW_SIZE = 100 * 1024 * 1024  # 100 MB
    RTO = 1.0               # 超时重传时间 (seconds)

class FastSim:
    def __init__(self, loss_rate, redundancy):
        self.loss_rate = loss_rate
        self.redundancy = redundancy
        
        cfg = Config()
        self.B = cfg.B
        self.RTT = cfg.RTT
        self.PKT_SIZE = cfg.PKT_SIZE
        self.FLOW_SIZE = cfg.FLOW_SIZE
        self.RTO = cfg.RTO
        
        self.num_pkts = (self.FLOW_SIZE + self.PKT_SIZE - 1) // self.PKT_SIZE
        self.pkt_interval = self.PKT_SIZE * 8 / self.B  # seconds per packet
        
        # Sender state
        self.next_sn = 0
        self.unacked = {}  # sn -> send_time
        
        # Receiver state
        self.next_expected = 0
        self.delivered = 0
        self.receiver_buffer = set()
        self.buffer_history = []  # list of (time, buffer_size)
        
        # Event queue: (time, event_id, event_type, data)
        self.events = []
        self.event_id = 0
        self.max_window = 100  # max in-flight packets

    def schedule(self, t, ev_type, data=None):
        heapq.heappush(self.events, (t, self.event_id, ev_type, data))
        self.event_id += 1

    def run(self):
        # Send initial window
        for sn in range(min(self.max_window, self.num_pkts)):
            self._send_packet(sn, sn * self.pkt_interval)

        while self.events and (self.next_sn < self.num_pkts or self.unacked):
            t, _, ev_type, data = heapq.heappop(self.events)
            
            if ev_type == 'arrive':
                self._on_arrival(t, data['sn'])
            elif ev_type == 'ack':
                self._on_ack(t, data['cumulative'], data['selective'])
            elif ev_type == 'timeout':
                if data['sn'] in self.unacked:
                    self._retransmit(t, data['sn'])

        # Compute metrics
        if self.delivered == 0:
            last_time = 0.1
        else:
            last_time = max(t for t, _ in self.buffer_history) if self.buffer_history else 0.1

        throughput_mbps = (self.delivered * self.PKT_SIZE * 8) / last_time / 1e6

        # Time-weighted average queue length
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

        return {
            'throughput_mbps': throughput_mbps,
            'avg_queue_length': avg_queue,
            'delivered_packets': self.delivered
        }

    def _send_packet(self, sn, send_time):
        if sn >= self.num_pkts:
            return
            
        # Primary packet
        if random.random() >= self.loss_rate:
            arrive_time = send_time + self.RTT / 2
            self.schedule(arrive_time, 'arrive', {'sn': sn})
            
        # Redundant packet (replicate)
        if self.redundancy != 'none':
            if random.random() >= self.loss_rate:
                arrive_time = send_time + self.RTT / 2 + 0.001  # slight jitter
                self.schedule(arrive_time, 'arrive', {'sn': sn})
                
        self.unacked[sn] = send_time
        self.schedule(send_time + self.RTO, 'timeout', {'sn': sn})

    def _on_arrival(self, t, sn):
        if sn < self.next_expected:
            return  # duplicate
            
        if sn == self.next_expected:
            self.delivered += 1
            self.next_expected += 1
            # Deliver contiguous buffered packets
            while self.next_expected in self.receiver_buffer:
                self.receiver_buffer.remove(self.next_expected)
                self.delivered += 1
                self.next_expected += 1
        else:
            self.receiver_buffer.add(sn)
            
        self.buffer_history.append((t, len(self.receiver_buffer)))
        
        # Send ACK (cumulative + selective)
        cumulative = self.next_expected - 1
        selective = [s for s in self.receiver_buffer if s > cumulative]
        self.schedule(t + 1e-5, 'ack', {'cumulative': cumulative, 'selective': selective})
        
        # Pipeline new packets
        while (self.next_sn < self.num_pkts and 
               (self.next_sn - (self.next_expected - 1)) < self.max_window):
            self._send_packet(self.next_sn, self.next_sn * self.pkt_interval)
            self.next_sn += 1

    def _on_ack(self, t, cumulative, selective):
        # Remove acknowledged packets from unacked
        to_remove = [sn for sn in self.unacked if sn <= cumulative]
        for sn in to_remove:
            del self.unacked[sn]
        for sn in selective:
            if sn in self.unacked:
                del self.unacked[sn]

    def _retransmit(self, t, sn):
        if sn not in self.unacked:
            return
        # Retransmit
        if random.random() >= self.loss_rate:
            arrive_time = t + self.RTT / 2
            self.schedule(arrive_time, 'arrive', {'sn': sn})
        # Update send time and reset timeout
        self.unacked[sn] = t
        self.schedule(t + self.RTO, 'timeout', {'sn': sn})


# ======================
# CLI Interface
# ======================
def simulate(loss_rate, redundancy):
    """Public API for sweep scripts."""
    sim = FastSim(loss_rate, redundancy)
    return sim.run()

def main():
    if len(sys.argv) != 3:
        print("Usage: python fast_sim.py <loss_rate> <redundancy>")
        print("  loss_rate: e.g., 0.05 for 5%")
        print("  redundancy: 'none' or 'replicate'")
        sys.exit(1)
    
    loss_rate = float(sys.argv[1])
    redundancy = sys.argv[2]
    
    if redundancy not in ['none', 'replicate']:
        print("Redundancy must be 'none' or 'replicate'")
        sys.exit(1)
    
    random.seed(42)  # for reproducibility
    result = simulate(loss_rate, redundancy)
    
    print(f"Loss={loss_rate:.1%}, Redundancy={redundancy}")
    print(f"Throughput: {result['throughput_mbps']:.2f} Mbps")
    print(f"Avg Queue Length: {result['avg_queue_length']:.1f} packets")
    print(f"Delivered Packets: {result['delivered_packets']} / {Config().FLOW_SIZE // Config().PKT_SIZE + 1}")

if __name__ == '__main__':
    main()