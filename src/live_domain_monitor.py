import pyshark
import time

from domain_detector import extract_domain
from reputation_checker import check_domain
from website_alert import save_website_alert

INTERFACE = r'\Device\NPF_{FF4599CE-7414-43D0-943C-56E2BF6C8F50}'
TSHARK_PATH = r'C:\Program Files\Wireshark\Wireshark.exe'

DOMAIN_COOLDOWN = 3600  # Don't check same domain again for 1 hour
TEST_MODE = True
checked_domains = {}


capture = pyshark.LiveCapture(
    interface=INTERFACE,
    tshark_path=TSHARK_PATH,
    display_filter="dns"
)


print("🛡️ AI WEBSITE THREAT MONITOR")
print("=" * 60)
print("Monitoring DNS traffic...")
        # --------------------------------
import pyshark
import time

from domain_detector import extract_domain
from reputation_checker import check_domain
from website_alert import save_website_alert

INTERFACE = r'\Device\NPF_{FF4599CE-7414-43D0-943C-56E2BF6C8F50}'
TSHARK_PATH = r'C:\Program Files\Wireshark\Wireshark.exe'

DOMAIN_COOLDOWN = 3600  # Don't check same domain again for 1 hour
TEST_MODE = False
checked_domains = {}


capture = pyshark.LiveCapture(
    interface=INTERFACE,
    tshark_path=TSHARK_PATH,
    display_filter="dns",
)


print("🛡️ AI WEBSITE THREAT MONITOR")
print("=" * 60)
print("Monitoring DNS traffic...")
print("=" * 60)


try:
    for packet in capture.sniff_continuously():

        domain = extract_domain(packet)

        if not domain:
            continue

        domain = domain.lower().rstrip(".")

        now = time.time()

        # --------------------------------
        # CACHE
        # --------------------------------

        if domain in checked_domains:
            if now - checked_domains[domain] < DOMAIN_COOLDOWN:
                continue

        checked_domains[domain] = now

        print(f"\n🌐 Website: {domain}")

        # --------------------------------
        # REPUTATION CHECK
        # --------------------------------

        result = check_domain(domain)

        # --------------------------------
        # TEST MODE
        # --------------------------------

        if TEST_MODE and domain == "chatgpt.com":
            result = {
                "status": "SUSPICIOUS",
                "domain": domain,
                "malicious": 3,
                "suspicious": 2,
                "harmless": 50,
                "undetected": 36,
                "reputation": -20,
                "malicious_ratio": 3.3,
            }

        status = result["status"]

        print(f"🔎 Reputation: {status}")

        # --------------------------------
        # SAFE
        # --------------------------------

        if status == "SAFE":
            print("🟢 Safe website")
            continue

        # --------------------------------
        # UNKNOWN
        # --------------------------------

        if status == "UNKNOWN":
            print("🟡 Domain not sufficiently known")
            continue

        # --------------------------------
        # REVIEW
        # --------------------------------

        if status == "REVIEW":
            print("🟡 LOW-CONFIDENCE THREAT")
            print(
                f"Malicious detections : {result.get('malicious', 0)}"
            )
            print(
                f"Suspicious detections: {result.get('suspicious', 0)}"
            )
            print(
                f"Malicious ratio      : {result.get('malicious_ratio', 0)}%"
            )
            print(
                "ℹ️ Only one security engine reported malicious activity."
            )
            continue

        # --------------------------------
        # SUSPICIOUS
        # --------------------------------

        if status == "SUSPICIOUS":
            print("🟠 SUSPICIOUS WEBSITE")
            print(
                f"Malicious detections : {result.get('malicious', 0)}"
            )
            print(
                f"Suspicious detections: {result.get('suspicious', 0)}"
            )
            save_website_alert(domain, result)
            continue

        # --------------------------------
        # MALICIOUS
        # --------------------------------

        if status == "MALICIOUS":
            print("\n🚨 MALICIOUS WEBSITE DETECTED 🚨")
            print("-" * 60)
            print(f"Domain      : {domain}")
            print(
                f"Malicious   : {result.get('malicious', 0)}"
            )
            print(
                f"Suspicious  : {result.get('suspicious', 0)}"
            )
            print(
                f"Harmless    : {result.get('harmless', 0)}"
            )
            save_website_alert(domain, result)
            print("-" * 60)
            print(
                "⚠️ Recommendation: Do not continue to this website."
            )

except KeyboardInterrupt:
    print("\n[INFO] Website monitoring stopped.")

finally:
    try:
        capture.close()
    except Exception:
        pass

    print("[INFO] Capture closed.")