import pyshark


def extract_domain(packet):
    """
    Extract the queried domain from a DNS packet.

    Returns:
        str | None: queried domain, or None if the packet
        does not contain a DNS query.
    """

    try:
        if not hasattr(packet, "dns"):
            return None

        # DNS query
        if hasattr(packet.dns, "qry_name"):
            domain = packet.dns.qry_name

            if domain:
                return str(domain).rstrip(".")

    except Exception:
        pass

    return None