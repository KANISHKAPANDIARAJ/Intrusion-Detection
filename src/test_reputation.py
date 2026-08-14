from reputation_checker import check_domain


domains = [
    "google.com",
    "github.com",
    "chatgpt.com",
    "example.com"
]


for domain in domains:

    print("\n" + "=" * 60)

    print(f"Checking: {domain}")

    result = check_domain(domain)

    print(f"Status      : {result['status']}")

    if "malicious" in result:
        print(f"Malicious   : {result['malicious']}")
        print(f"Suspicious  : {result['suspicious']}")
        print(f"Harmless    : {result['harmless']}")
        print(f"Undetected  : {result['undetected']}")

    if "reputation" in result:
        print(f"Reputation  : {result['reputation']}")

    if "reason" in result:
        print(f"Reason      : {result['reason']}")