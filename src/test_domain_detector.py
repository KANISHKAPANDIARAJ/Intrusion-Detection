import pyshark
import time
from domain_detector import extract_domain


capture = pyshark.LiveCapture(
    interface=r'\Device\NPF_{FF4599CE-7414-43D0-943C-56E2BF6C8F50}',
    tshark_path=r'D:\Wireshark\tshark.exe',
    display_filter="dns"
)

seen_domains = {}

DOMAIN_COOLDOWN = 30

print("🌐 Monitoring DNS queries...")
print("-" * 60)

try:
    for packet in capture.sniff_continuously():

        domain = extract_domain(packet)

        if not domain:
            continue

        domain = domain.lower()

        now = time.time()

        # Ignore the same domain for 30 seconds
        if domain in seen_domains:
            if now - seen_domains[domain] < DOMAIN_COOLDOWN:
                continue

        seen_domains[domain] = now

        print(f"🌐 Website requested: {domain}")

except KeyboardInterrupt:
    print("\n[INFO] DNS monitoring stopped.")

finally:
    capture.close()