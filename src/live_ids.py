# live_ids_hardend.py
import pyshark
import torch
import numpy as np
import time
from collections import deque, defaultdict
from model_attn import CNN_BiLSTM_Attn_IDS

# -------------------------
# CONFIG (tune these)
# -------------------------
SEQ_LEN = 50
INPUT_DIM = 42
NUM_CLASSES = 2
DEVICE = torch.device("cpu")

# Inference smoothing + thresholds
PACKET_DELAY = 0.01        # pause between handling packets
CONF_BUFFER_LEN = 8        # average over last N predictions
ALERT_THRESHOLD = 0.85     # raise alert only if avg prob >= this
MIN_ALERT_OCCURRENCES = 3  # require this many high-confidence sequences within window
ALERT_WINDOW_SEC = 8       # time window to count occurrences
ALERT_COOLDOWN = 20        # seconds before repeating same alert for same pair

# Whitelist known-good services/IPs (edit to your network)
WHITELIST_IPS = {
    "192.168.236.1",       # gateway, example
    "140.82.112.21",       # (if you know these are benign) remove as needed
    # add IPs or subnets you trust
}
WHITELIST_PREFIXES = ("10.", "192.168.", "172.16.")  # local networks

# -------------------------
# CAPTURE (point to tshark if on Windows)
# -------------------------
capture = pyshark.LiveCapture(
    interface='\\Device\\NPF_{FF4599CE-7414-43D0-943C-56E2BF6C8F50}',
    tshark_path=r'D:\Wireshark\tshark.exe',
    display_filter="ip"  # only capture IP packets
)


for packet in capture.sniff_continuously(packet_count=5):
    try:
        layer = packet.highest_layer
        src = packet.ip.src if hasattr(packet, "ip") else "Unknown"
        dst = packet.ip.dst if hasattr(packet, "ip") else "Unknown"
        print(f"[CAPTURED] {src} → {dst} ({layer})")
    except Exception:
        pass

# -------------------------
# model load (make sure model architecture & INPUT_DIM match training)
# -------------------------
model = CNN_BiLSTM_Attn_IDS(input_dim=INPUT_DIM, seq_len=SEQ_LEN, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("best_ids_model.pth", map_location=DEVICE))
model.eval()

# -------------------------
# helper functions
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
    """
    Extract numeric features from TCP, UDP, or QUIC packets.
    Returns a fixed-length vector of size INPUT_DIM or None if packet invalid.
    """
    try:
        # Only process packets with IP layer
        if not hasattr(packet, "ip"):
            return None

        features = []

        # Source / Destination IP last octet
        features.append(int(packet.ip.src.split('.')[-1]))
        features.append(int(packet.ip.dst.split('.')[-1]))

        # TTL
        features.append(int(packet.ip.ttl))

        # Source / Destination Ports
        if hasattr(packet, "tcp"):
            features.append(int(packet.tcp.srcport))
            features.append(int(packet.tcp.dstport))
        elif hasattr(packet, "udp"):
            features.append(int(packet.udp.srcport))
            features.append(int(packet.udp.dstport))
        else:
            # Non-TCP/UDP packet
            features.extend([0, 0])

        # Packet length
        if hasattr(packet, "length"):
            features.append(int(packet.length))
        else:
            features.append(0)

        # Flags (TCP only, else 0)
        if hasattr(packet, "tcp"):
            flags = int(packet.tcp.flags, 16)
            features.append(flags)
        else:
            features.append(0)

        # Protocol (1=ICMP, 6=TCP, 17=UDP, 255=others)
        proto_map = {"ICMP": 1, "TCP": 6, "UDP": 17}
        proto_num = proto_map.get(packet.highest_layer.upper(), 255)
        features.append(proto_num)

        # Padding/truncating to INPUT_DIM
        if len(features) < INPUT_DIM:
            features.extend([0] * (INPUT_DIM - len(features)))
        else:
            features = features[:INPUT_DIM]

        return features

    except Exception:
        return None

# -------------------------
# runtime state
# -------------------------

def human_alert_message(src, dst, avg_conf, reason):
    """
    Build a human-friendly alert message for given source/destination and confidence.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return f"[ALERT] {ts} {src} -> {dst} avg_conf={avg_conf:.3f}: {reason}"

seq_buffer = deque(maxlen=SEQ_LEN)
# For each IP pair track timestamps when high-confidence observed
pair_occurrences = defaultdict(deque)
# last alert time per pair
last_alert_time = {}

conf_buffer = deque(maxlen=CONF_BUFFER_LEN)

try:
    for packet in capture.sniff_continuously():
        feat = extract_features(packet)
        if feat is None:
            continue

        # get readable IPs (safe guards)
        src = packet.ip.src if hasattr(packet, "ip") else "Unknown"
        dst = packet.ip.dst if hasattr(packet, "ip") else "Unknown"

        # skip whitelisted endpoints early
        if is_whitelisted(src) or is_whitelisted(dst):
            continue

        seq_buffer.append(feat)

        if len(seq_buffer) < SEQ_LEN:
            continue

        # build model input (most recent SEQ_LEN)
        seq_input = np.array(list(seq_buffer)).reshape(1, SEQ_LEN, INPUT_DIM)
        seq_input = torch.tensor(seq_input, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model(seq_input)
            prob = float(torch.sigmoid(logits).item())

        # smoothing buffer
        conf_buffer.append(prob)
        avg_conf = float(sum(conf_buffer) / len(conf_buffer))

        # If average confidence high enough, record occurrence for the IP pair
        now = time.time()
        pair = (src, dst)

        if avg_conf >= ALERT_THRESHOLD:
            # record timestamp for this pair
            pair_occurrences[pair].append(now)

            # drop old occurrences outside ALERT_WINDOW_SEC
            while pair_occurrences[pair] and now - pair_occurrences[pair][0] > ALERT_WINDOW_SEC:
                pair_occurrences[pair].popleft()

            # check if we have enough occurrences in the window
            if len(pair_occurrences[pair]) >= MIN_ALERT_OCCURRENCES:
                # check cooldown
                last = last_alert_time.get(pair, 0)
                if now - last >= ALERT_COOLDOWN:
                    # produce a human-friendly reason hint (simple heuristics)
                    reason = "high-confidence pattern detected"
                    # print human message
                    print(human_alert_message(src, dst, avg_conf, reason))
                    last_alert_time[pair] = now
                    # optional: write to file or call teammate webhook here

                # clear occurrences to avoid immediate repeat
                pair_occurrences[pair].clear()

        # small sleep to reduce CPU hogging
        time.sleep(PACKET_DELAY)

except KeyboardInterrupt:
    print("\nLive capture stopped by user.")
