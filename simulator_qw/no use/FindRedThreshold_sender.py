# sender.py
import queue
import threading
import random
import time
from FindRedThreshold_receiver import start_receiver

class Config:
    B = 100 * 1e6          # 100 Mbps
    RTT = 600 * 0.001      # 600 ms
    PKT_SIZE = 1250
    FLOW_SIZE = 100 * 1024 * 1024
    RTO = 1.0             # 超时重传时间
    LOSS_RATE = 0.05
    REDUNDANCY = 'none'   # 'none', 'replicate'

def simulate(loss_rate, redundancy):
    cfg = Config()
    cfg.LOSS_RATE = loss_rate
    cfg.REDUNDANCY = redundancy

    # 通信队列
    net_to_receiver = queue.PriorityQueue()  # (time, sn)
    net_to_sender = queue.Queue()           # ACK messages

    # 启动 receiver 线程
    receiver_done = threading.Event()
    receiver_result = {}
    def run_receiver():
        res = start_receiver(net_to_sender, net_to_receiver)
        receiver_result.update(res)
        receiver_done.set()

    recv_thread = threading.Thread(target=run_receiver)
    recv_thread.start()

    # Sender logic
    next_sn = 0
    unacked = {}  # sn -> send_time
    delivered_bytes = 0
    start_time = time.time()

    num_pkts = (cfg.FLOW_SIZE + cfg.PKT_SIZE - 1) // cfg.PKT_SIZE
    pkt_interval = cfg.PKT_SIZE * 8 / cfg.B

    while next_sn < num_pkts or unacked:
        # print("Sender loop: next_sn =", next_sn, "unacked =", len(unacked))
        current_time = time.time() - start_time

        # 发送新包（如果还有）
        if next_sn < num_pkts:
            send_time = next_sn * pkt_interval
            if current_time >= send_time:
                # 发主包
                if random.random() >= cfg.LOSS_RATE:
                    net_to_receiver.put((current_time + cfg.RTT/2, next_sn))
                # 发冗余包？
                if cfg.REDUNDANCY != 'none':
                    if random.random() >= cfg.LOSS_RATE:
                        net_to_receiver.put((current_time + cfg.RTT/2 + 0.001, next_sn))
                unacked[next_sn] = current_time
                next_sn += 1

        # 处理 ACK
        while not net_to_sender.empty():
            ack = net_to_sender.get()
            ack_time = ack['time']
            cumulative = ack['cumulative']
            selective = set(ack['selective'])

            # 确认 cumulative 及之前所有包
            to_remove = [sn for sn in unacked if sn <= cumulative]
            for sn in to_remove:
                del unacked[sn]

            # 确认 selective 包
            for sn in selective:
                if sn in unacked:
                    del unacked[sn]

        # 超时重传
        to_retransmit = []
        for sn, send_t in unacked.items():
            if current_time - send_t > cfg.RTO:
                to_retransmit.append(sn)

        for sn in to_retransmit:
            if random.random() >= cfg.LOSS_RATE:
                net_to_receiver.put((current_time + cfg.RTT/2, sn))
            unacked[sn] = current_time  # 更新发送时间

        time.sleep(0.001)  # 避免 busy-wait

    # 停止 receiver
    net_to_receiver.put((None, None))
    receiver_done.wait(timeout=5)

    # 计算指标
    res = receiver_result
    time_elapsed = max(res['last_time'], 0.001)
    throughput = (res['delivered'] * cfg.PKT_SIZE * 8) / time_elapsed / 1e6
    avg_queue = res['avg_queue_length']

    recv_thread.join(timeout=1)
    return {'throughput_mbps': throughput, 'avg_queue_length': avg_queue}

# ===== 单独运行 sender =====
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python sender.py <loss_rate> <redundancy>")
        print("  redundancy: none, replicate")
        sys.exit(1)

    loss = float(sys.argv[1])
    red = sys.argv[2]
    random.seed(42)

    result = simulate(loss, red)
    print(f"Loss={loss:.1%}, Redundancy={red}")
    print(f"Throughput: {result['throughput_mbps']:.2f} Mbps")
    print(f"Avg Queue Length: {result['avg_queue_length']:.1f} packets")