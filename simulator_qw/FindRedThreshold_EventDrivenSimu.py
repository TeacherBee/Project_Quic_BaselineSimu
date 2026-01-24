# fast_sim.py —— Supports: none, replicate, replicate_k_1
import heapq
import random
import sys
import re
from collections import defaultdict

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
        self.k_rep = None  # for replicate_k_1
        
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
        else:
            raise ValueError("Redundancy must be: 'none', 'replicate', or 'replicate_k_1' (e.g., 'replicate_4_1')")
        
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
        elif self.mode == 'replicate_k_1':
            # Every k original packets → k + 1 physical packets
            self.send_interval = base_interval * (self.k_rep + 1) / self.k_rep
        else:
            self.send_interval = base_interval
        
        # Sender
        self.next_sn = 0
        self.unacked = {}
        
        # Receiver
        self.next_expected = 0
        self.delivered = 0
        self.receiver_buffer = set()
        self.buffer_history = []
        
        # Event queue
        self.events = []
        self.event_id = 0
        self.max_window = 100

    def schedule(self, t, ev_type, data=None):
        heapq.heappush(self.events, (t, self.event_id, ev_type, data))
        self.event_id += 1

    def run(self):
        for sn in range(min(self.max_window, self.num_pkts)):
            send_time = sn * self.send_interval
            self._send_packet(sn, send_time)

        while self.events and (self.next_sn < self.num_pkts or self.unacked):
            t, _, ev_type, data = heapq.heappop(self.events)
            
            if ev_type == 'arrive':
                self._on_arrival(t, data['sn'])
            elif ev_type == 'ack':
                self._on_ack(t, data['cumulative'], data['selective'])
            elif ev_type == 'timeout':
                if data['sn'] in self.unacked:
                    self._retransmit(t, data['sn'])

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

        return {
            'throughput_mbps': throughput_mbps,
            'avg_queue_length': avg_queue,
            'delivered_packets': self.delivered
        }

    def _send_packet(self, sn, send_time):
        if sn >= self.num_pkts:
            return
            
        # Send original packet
        if random.random() >= self.loss_rate:
            arrive_time = send_time + self.RTT / 2
            self.schedule(arrive_time, 'arrive', {'sn': sn})
            
        # Send redundant packet(s) based on mode
        base_pkt_time = self.PKT_SIZE * 8 / self.B
        
        if self.mode == 'replicate':
            # Send redundant copy right after
            red_time = send_time + base_pkt_time
            if random.random() >= self.loss_rate:
                arrive_time = red_time + self.RTT / 2
                self.schedule(arrive_time, 'arrive', {'sn': sn})
                
        elif self.mode == 'replicate_k_1':
            # Check if this is the last packet in a group of k
            if (sn + 1) % self.k_rep == 0 or sn == self.num_pkts - 1:
                # Send redundant copy of THIS packet (sn)
                red_time = send_time + base_pkt_time
                if random.random() >= self.loss_rate:
                    arrive_time = red_time + self.RTT / 2
                    self.schedule(arrive_time, 'arrive', {'sn': sn})
        
        self.unacked[sn] = send_time
        self.schedule(send_time + self.RTO, 'timeout', {'sn': sn})

    def _on_arrival(self, t, sn):
        if sn < self.next_expected:
            return
            
        if sn == self.next_expected:
            self.delivered += 1
            self.next_expected += 1
            while self.next_expected in self.receiver_buffer:
                self.receiver_buffer.remove(self.next_expected)
                self.delivered += 1
                self.next_expected += 1
        else:
            self.receiver_buffer.add(sn)
            
        self.buffer_history.append((t, len(self.receiver_buffer)))
        
        cumulative = self.next_expected - 1
        selective = [s for s in self.receiver_buffer if s > cumulative]
        self.schedule(t + 1e-5, 'ack', {'cumulative': cumulative, 'selective': selective})
        
        while (self.next_sn < self.num_pkts and 
               (self.next_sn - (self.next_expected - 1)) < self.max_window):
            send_time = self.next_sn * self.send_interval
            self._send_packet(self.next_sn, send_time)
            self.next_sn += 1

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
        self.unacked[sn] = t
        self.schedule(t + self.RTO, 'timeout', {'sn': sn})


# ======================
# CLI
# ======================
def simulate(loss_rate, redundancy):
    sim = FastSim(loss_rate, redundancy)
    return sim.run()

def main():
    if len(sys.argv) != 3:
        print("Usage: python fast_sim.py <loss_rate> <redundancy>")
        print("  loss_rate: e.g., 0.05")
        print("  redundancy options:")
        print("    - 'none'")
        print("    - 'replicate' (1:1 redundancy)")
        print("    - 'replicate_k_1' (e.g., 'replicate_4_1' → every 4 packets, add 1 redundant copy)")
        sys.exit(1)
    
    loss_rate = float(sys.argv[1])
    redundancy = sys.argv[2]
    
    random.seed(42)
    result = simulate(loss_rate, redundancy)
    
    total_pkts = (Config().FLOW_SIZE + Config().PKT_SIZE - 1) // Config().PKT_SIZE
    print(f"Loss={loss_rate:.1%}, Redundancy={redundancy}")
    print(f"Throughput: {result['throughput_mbps']:.2f} Mbps")
    print(f"Avg Queue Length: {result['avg_queue_length']:.1f} packets")
    print(f"Delivered Packets: {result['delivered_packets']} / {total_pkts}")

if __name__ == '__main__':
    main()