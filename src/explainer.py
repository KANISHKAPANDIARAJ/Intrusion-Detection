# src/explainer.py

def explain_alert(src_ip, dst_ip, protocol, confidence):
    explanation = ""
    action = ""
    severity = ""

    if confidence >= 0.9:
        severity = "HIGH"
        explanation = (
            "This packet shows highly abnormal behavior that strongly deviates "
            "from normal network traffic patterns. It may indicate scanning, "
            "spoofing, or misconfigured broadcast activity."
        )
        action = "Immediately inspect the source device and consider isolating it."
    
    elif confidence >= 0.75:
        severity = "MEDIUM"
        explanation = (
            "This packet exhibits suspicious characteristics that are uncommon "
            "in regular traffic. While not confirmed malicious, it requires attention."
        )
        action = "Monitor the traffic and verify the device configuration."
    
    else:
        severity = "LOW"
        explanation = (
            "This packet slightly deviates from normal behavior but does not "
            "pose an immediate threat."
        )
        action = "No immediate action required. Continue monitoring."

    return {
        "severity": severity,
        "explanation": explanation,
        "action": action
    }
