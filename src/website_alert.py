import json
import os
from datetime import datetime


ALERT_FILE = "dashboard/latest_alert.json"


def save_website_alert(domain, result):

    os.makedirs("dashboard", exist_ok=True)

    alert = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "domain": domain,
        "status": result.get("status"),
        "malicious": result.get("malicious", 0),
        "suspicious": result.get("suspicious", 0),
        "harmless": result.get("harmless", 0),
        "undetected": result.get("undetected", 0),
        "reputation": result.get("reputation", 0)
    }

    with open(ALERT_FILE, "w") as f:
        json.dump(alert, f, indent=4)

    return alert