import os
import time
import requests
from dotenv import load_dotenv


# -------------------------
# CONFIG
# -------------------------

load_dotenv()

API_KEY = os.getenv("VIRUSTOTAL_API_KEY")

API_URL = "https://www.virustotal.com/api/v3/domains/{}"

CACHE_DURATION = 3600  # 1 hour

_cache = {}


# -------------------------
# DOMAIN NORMALIZATION
# -------------------------

def normalize_domain(domain):
    """
    Normalize a domain name.

    Example:
        www.google.com -> google.com
    """

    domain = domain.lower().strip().rstrip(".")

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


# -------------------------
# DOMAIN REPUTATION
# -------------------------

def check_domain(domain):

    # -------------------------
    # API KEY CHECK
    # -------------------------

    if not API_KEY:

        return {
            "status": "ERROR",
            "domain": domain,
            "reason": "VirusTotal API key not configured."
        }

    domain = normalize_domain(domain)

    # -------------------------
    # CACHE CHECK
    # -------------------------

    if domain in _cache:

        cached_time = _cache[domain]["time"]

        if time.time() - cached_time < CACHE_DURATION:

            return _cache[domain]["result"]

    # -------------------------
    # API REQUEST
    # -------------------------

    headers = {
        "x-apikey": API_KEY
    }

    url = API_URL.format(domain)

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        # -------------------------
        # DOMAIN NOT FOUND
        # -------------------------

        if response.status_code == 404:

            result = {
                "status": "UNKNOWN",
                "domain": domain,
                "malicious": 0,
                "suspicious": 0,
                "harmless": 0,
                "undetected": 0,
                "reputation": 0,
                "reason": "Domain not found in VirusTotal."
            }

            _cache[domain] = {
                "time": time.time(),
                "result": result
            }

            return result

        # -------------------------
        # RATE LIMIT
        # -------------------------

        if response.status_code == 429:

            return {
                "status": "RATE_LIMITED",
                "domain": domain,
                "reason": "VirusTotal API rate limit reached."
            }

        # -------------------------
        # OTHER HTTP ERRORS
        # -------------------------

        response.raise_for_status()

        # -------------------------
        # PARSE RESPONSE
        # -------------------------

        data = response.json()

        attributes = data["data"]["attributes"]

        stats = attributes.get(
            "last_analysis_stats",
            {}
        )

        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        harmless = stats.get("harmless", 0)
        undetected = stats.get("undetected", 0)

        reputation = attributes.get(
            "reputation",
            0
        )

        # -------------------------
        # TOTAL ENGINES
        # -------------------------

        total_engines = (
            malicious
            + suspicious
            + harmless
            + undetected
        )

        # -------------------------
        # MALICIOUS RATIO
        # -------------------------

        if total_engines > 0:

            malicious_ratio = (
                malicious / total_engines
            )

        else:

            malicious_ratio = 0.0

        # -------------------------
        # CLASSIFICATION
        # -------------------------

        if malicious >= 4:

            status = "MALICIOUS"

        elif malicious >= 2:

            status = "SUSPICIOUS"

        elif suspicious >= 3:

            status = "SUSPICIOUS"

        elif malicious == 1:

            status = "REVIEW"

        elif harmless >= 3:

            status = "SAFE"

        else:

            status = "UNKNOWN"

        # -------------------------
        # RESULT
        # -------------------------

        result = {

            "status": status,

            "domain": domain,

            "malicious": malicious,

            "suspicious": suspicious,

            "harmless": harmless,

            "undetected": undetected,

            "total_engines": total_engines,

            "malicious_ratio": round(
                malicious_ratio * 100,
                2
            ),

            "reputation": reputation
        }

        # -------------------------
        # CACHE RESULT
        # -------------------------

        _cache[domain] = {

            "time": time.time(),

            "result": result
        }

        return result

    # -------------------------
    # REQUEST ERROR
    # -------------------------

    except requests.RequestException as e:

        return {

            "status": "ERROR",

            "domain": domain,

            "reason": str(e)
        }