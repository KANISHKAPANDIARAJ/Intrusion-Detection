from datetime import datetime


def analyze_packet(
    src_ip,
    dst_ip,
    protocol,
    confidence,
    frame_no=None
):
    """
    Basic AI-style threat analysis layer.

    CNN-BiLSTM-Attention:
        Detects suspicious traffic.

    This module:
        Converts the detection into a structured
        security explanation for the dashboard.
    """

    confidence_percent = confidence * 100

    # -------------------------
    # Severity
    # -------------------------
    if confidence >= 0.95:
        severity = "CRITICAL"
    elif confidence >= 0.90:
        severity = "HIGH"
    elif confidence >= 0.80:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    # -------------------------
    # Basic explanation
    # -------------------------
    if protocol.upper() == "TCP":
        reason = (
            "The CNN-BiLSTM model detected an abnormal TCP "
            "traffic pattern in the recent packet sequence."
        )

    elif protocol.upper() == "TLS":
        reason = (
            "The model detected an abnormal pattern in encrypted "
            "TLS traffic. Payload inspection is unavailable."
        )

    elif protocol.upper() == "DNS":
        reason = (
            "The model detected an unusual DNS traffic pattern "
            "that differs from the learned normal behavior."
        )

    elif protocol.upper() == "DHCP":
        reason = (
            "The model detected an unusual DHCP traffic pattern. "
            "DHCP broadcast traffic should be validated before "
            "classifying it as malicious."
        )

    else:
        reason = (
            f"The model detected an abnormal {protocol} "
            "traffic pattern."
        )

    # -------------------------
    # Recommendation
    # -------------------------
    if severity == "CRITICAL":
        action = (
            "Immediately investigate the source host and "
            "consider isolating it."
        )

    elif severity == "HIGH":
        action = (
            "Investigate the source and review recent "
            "network activity."
        )

    else:
        action = (
            "Continue monitoring the source and compare "
            "future traffic against its baseline."
        )

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame": frame_no,
        "source": src_ip,
        "destination": dst_ip,
        "protocol": protocol,
        "confidence": round(confidence_percent, 2),
        "severity": severity,
        "reason": reason,
        "action": action
    }