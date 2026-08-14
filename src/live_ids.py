import pyshark
import torch
import numpy as np
import time
from collections import deque, defaultdict
from model_attn import CNN_BiLSTM_Attn_IDS
from explainer import explain_alert
import subprocess
from ai_analyzer import analyze_packet
import json
import os
# -------------------------
# CONFIG
# -------------------------
SEQ_LEN = 50
INPUT_DIM = 42
NUM_CLASSES = 2
DEVICE = torch.device("cpu")

PACKET_DELAY = 0.01
CONF_BUFFER_LEN = 8
ALERT_THRESHOLD = 0.80
MIN_ALERT_OCCURRENCES = 2
ALERT_WINDOW_SEC = 8
ALERT_COOLDOWN = 30

WHITELIST_IPS = {
    "192.168.236.1",
    "140.82.112.21"
}
WHITELIST_PREFIXES = ()

# -------------------------
# CAPTURE
# -------------------------
capture = pyshark.LiveCapture(
    interface='\\Device\\NPF_{FF4599CE-7414-43D0-943C-56E2BF6C8F50}',
    tshark_path=r'D:\Wireshark\tshark.exe',
    display_filter="ip"
)

# -------------------------
# MODEL LOAD
# -------------------------
model = CNN_BiLSTM_Attn_IDS(
    input_dim=INPUT_DIM,
    seq_len=SEQ_LEN,
    num_classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(torch.load("best_ids_model.pth", map_location=DEVICE))
model.eval()

# -------------------------
# HELPERS
# -------------------------
def is_whitelisted(ip):
    if not ip:
        return False
    if ip in WHITELIST_IPS:
        return True
    for p in WHITELIST_PREFIXES:
        if ip.startswith(p):
            return True
    return False


def extract_features(packet):
    try:
        if not hasattr(packet, "ip"):
            return None

        features = []

        features.append(int(packet.ip.src.split('.')[-1]))
        features.append(int(packet.ip.dst.split('.')[-1]))
        # IP-based features
        features.append(int(packet.ip.src.split('.')[-1]))
        features.append(int(packet.ip.dst.split('.')[-1]))

        # Internal vs external flags
        features.append(1 if packet.ip.src.startswith("10.") else 0)
        features.append(1 if packet.ip.dst.startswith("10.") else 0)

        features.append(int(packet.ip.ttl))

        if hasattr(packet, "tcp"):
            features.append(int(packet.tcp.srcport))
            features.append(int(packet.tcp.dstport))
            flags = int(packet.tcp.flags, 16)
            features.append(flags)
        elif hasattr(packet, "udp"):
            features.append(int(packet.udp.srcport))
            features.append(int(packet.udp.dstport))
            features.append(0)
        else:
            features.extend([0, 0, 0])

        features.append(int(packet.length) if hasattr(packet, "length") else 0)

        proto_map = {"ICMP": 1, "TCP": 6, "UDP": 17}
        proto_num = proto_map.get(packet.highest_layer.upper(), 255)
        features.append(proto_num)

        if len(features) < INPUT_DIM:
            features.extend([0] * (INPUT_DIM - len(features)))
        else:
            features = features[:INPUT_DIM]

        return features

    except Exception:
        return None

# -------------------------
# RUNTIME STATE
# -------------------------
seq_buffer = deque(maxlen=SEQ_LEN)
conf_buffer = deque(maxlen=CONF_BUFFER_LEN)
pair_occurrences = defaultdict(deque)
last_alert_time = {}

# -------------------------
# MAIN LOOP
# -------------------------
try:
    for packet in capture.sniff_continuously():

        feat = extract_features(packet)
        if feat is None:
            continue

        try:
            frame_no = packet.frame_info.number
        except Exception:
            frame_no = None

        src = packet.ip.src if hasattr(packet, "ip") else None
        dst = packet.ip.dst if hasattr(packet, "ip") else None

        if is_whitelisted(src) or is_whitelisted(dst):
            continue

        seq_buffer.append(feat)
        if len(seq_buffer) < SEQ_LEN:
            continue

        seq_input = torch.tensor(
            np.array(seq_buffer).reshape(1, SEQ_LEN, INPUT_DIM),
            dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(seq_input)
            prob = float(torch.sigmoid(logits).item())

        conf_buffer.append(prob)
        avg_conf = sum(conf_buffer) / len(conf_buffer)

        now = time.time()
        pair = (src, dst)

        if avg_conf >= ALERT_THRESHOLD:
            pair_occurrences[pair].append(now)

            while pair_occurrences[pair] and now - pair_occurrences[pair][0] > ALERT_WINDOW_SEC:
                pair_occurrences[pair].popleft()

            if len(pair_occurrences[pair]) >= MIN_ALERT_OCCURRENCES:
                last = last_alert_time.get(pair, 0)

                info = analyze_packet(
                    src_ip=src,
                    dst_ip=dst,
                    protocol=packet.highest_layer,
                    confidence=avg_conf,
                    frame_no=frame_no
                )
                os.makedirs("dashboard", exist_ok=True)

                with open("dashboard/latest_alert.json", "w") as f:
                    json.dump(info, f, indent=4)

                print("\n[🚨 AI INTRUSION ALERT 🚨]")
                print("=" * 60)

                print(f"Time         : {info['timestamp']}")
                print(f"Frame        : {info['frame']}")
                print(f"Source       : {info['source']}")
                print(f"Destination  : {info['destination']}")
                print(f"Protocol     : {info['protocol']}")
                print(f"Confidence   : {info['confidence']}%")
                print(f"Severity     : {info['severity']}")

                print("\nAI Analysis")
                print("-" * 60)
                print(info["reason"])

                print("\nRecommendation")
                print("-" * 60)
                print(info["action"])

                print("=" * 60)

#                if frame_no:
#                   subprocess.Popen([
#                            r"C:\Program Files\Wireshark\Wireshark.exe",
#                           "-Y",
#                           f"frame.number == {frame_no}"
#                   ])

                last_alert_time[pair] = now
                pair_occurrences[pair].clear()

        time.sleep(PACKET_DELAY)

except KeyboardInterrupt:
    print("\n[INFO] Capture stopped by user.")

finally:
    capture.close()
