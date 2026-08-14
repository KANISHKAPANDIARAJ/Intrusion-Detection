import streamlit as st
import json
import os
import time


st.set_page_config(
    page_title="AI Website Threat Monitor",
    page_icon="🛡️",
    layout="wide"
)


ALERT_FILE = "dashboard/latest_alert.json"


def load_alert():

    if not os.path.exists(ALERT_FILE):
        return None

    try:

        with open(ALERT_FILE, "r") as f:
            return json.load(f)

    except Exception:
        return None


st.title("🛡️ AI Website Threat Monitor")

st.caption(
    "DNS-based website reputation monitoring"
)


alert = load_alert()


if alert:

    status = alert.get("status", "UNKNOWN")

    if status == "MALICIOUS":
        st.error("🚨 MALICIOUS WEBSITE DETECTED")

    elif status == "SUSPICIOUS":
        st.warning("🟠 SUSPICIOUS WEBSITE DETECTED")

    else:
        st.info("🟡 UNKNOWN WEBSITE")


else:

    st.success("🟢 SYSTEM MONITORING")


st.divider()


# -------------------------
# METRICS
# -------------------------

col1, col2, col3, col4 = st.columns(4)


with col1:

    if alert:
        st.metric(
            "Threat Status",
            alert.get("status", "UNKNOWN")
        )
    else:
        st.metric(
            "Threat Status",
            "NORMAL"
        )


with col2:

    if alert:
        st.metric(
            "Malicious Detections",
            alert.get("malicious", 0)
        )
    else:
        st.metric(
            "Malicious Detections",
            0
        )


with col3:

    if alert:
        st.metric(
            "Suspicious Detections",
            alert.get("suspicious", 0)
        )
    else:
        st.metric(
            "Suspicious Detections",
            0
        )


with col4:

    if alert:
        st.metric(
            "Reputation",
            alert.get("reputation", 0)
        )
    else:
        st.metric(
            "Reputation",
            0
        )


st.divider()


# -------------------------
# WEBSITE INFORMATION
# -------------------------

st.subheader("🌐 Latest Website Analysis")


if alert:

    st.write(
        f"**Domain:** `{alert.get('domain', 'Unknown')}`"
    )

    st.write(
        f"**Time:** `{alert.get('timestamp', 'Unknown')}`"
    )

    st.write(
        f"**Status:** `{alert.get('status', 'Unknown')}`"
    )

    st.divider()

    st.subheader("🔎 Threat Intelligence")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            f"**Malicious:** "
            f"`{alert.get('malicious', 0)}`"
        )

        st.write(
            f"**Suspicious:** "
            f"`{alert.get('suspicious', 0)}`"
        )

    with col2:

        st.write(
            f"**Harmless:** "
            f"`{alert.get('harmless', 0)}`"
        )

        st.write(
            f"**Undetected:** "
            f"`{alert.get('undetected', 0)}`"
        )


    st.divider()


    if alert["status"] == "MALICIOUS":

        st.error(
            "⚠️ This domain has been identified as "
            "malicious by multiple security engines. "
            "Avoid continuing to the website."
        )

    elif alert["status"] == "SUSPICIOUS":

        st.warning(
            "⚠️ This domain has suspicious reputation "
            "indicators. Further investigation is recommended."
        )


else:

    st.info(
        "No suspicious or malicious websites detected yet."
    )


time.sleep(2)
st.rerun()