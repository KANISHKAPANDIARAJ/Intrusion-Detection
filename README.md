<h1><b>AI-Based Hybrid Real-time threat  detection</b></h1>

<h2><b>Overview</b></h2>

This project is a real-time hybrid cybersecurity monitoring system designed to detect suspicious network activity and potentially malicious websites.

The system combines two complementary security mechanisms:

1. AI-based Network Intrusion Detection using a CNN-BiLSTM-Attention deep learning architecture.
2. DNS-based Website Threat Detection using VirusTotal threat intelligence.

The network intrusion detection component captures live network packets using PyShark/TShark, extracts network traffic features, forms packet sequences, and analyzes them using a trained CNN-BiLSTM-Attention model.

The website threat detection component monitors DNS requests, extracts requested domains, and checks their reputation using the VirusTotal API.

The final results are classified into threat categories and displayed through a Streamlit-based monitoring dashboard.

---

<h2><b>Problem Statement</b></h2>

Traditional intrusion detection systems often depend on static rules or signatures and may struggle to identify abnormal network behavior that does not match previously known attack patterns.

Similarly, website security based only on domain reputation may not provide sufficient information about the actual behavior of network traffic.

There is a need for a hybrid real-time security system that can:

- Monitor live network traffic.
- Detect abnormal traffic behavior using deep learning.
- Monitor DNS requests.
- Identify potentially malicious or suspicious domains.
- Use external threat intelligence for website reputation.
- Reduce unnecessary alerts through confidence thresholds and repeated detection logic.
- Present security information through an understandable dashboard.

---

<h2><b>Solution Approach</b></h2>

The system processes live network and DNS traffic through two parallel detection pipelines.


<h3><b>1. Network Intrusion Detection Pipeline</b></h3>

Live Packet Capture

The system captures live network packets using PyShark and TShark.

Feature Extraction

Important packet-level features are extracted, including:

- Source IP
- Destination IP
- Source port
- Destination port
- Internal/external network indicators
- TTL
- TCP flags
- Packet length
- Protocol information

Sequence Formation

Individual packet feature vectors are collected into a sequence.

Current configuration:

- Sequence Length: 50 packets
- Input Features: 42

The resulting input can be represented as:

50 × 42

CNN Feature Extraction

A Convolutional Neural Network extracts local patterns from consecutive network traffic features.

BiLSTM Processing

A Bidirectional Long Short-Term Memory network learns sequential relationships in the packet sequence.

Attention Mechanism

The attention layer assigns greater importance to relevant portions of the sequence before the final classification.

Confidence Calculation

The model produces an intrusion confidence score.

Current alert threshold:

0.80

If:

confidence >= 0.80

the traffic sequence becomes a potential intrusion alert candidate.

Additional alert logic is applied using repeated detections, time windows, and cooldown periods to reduce repeated alerts.


<h3><b>2. Website Threat Detection Pipeline</b></h3>

DNS Monitoring

The system monitors DNS packets in real time.

Domain Extraction

The requested hostname is extracted from DNS queries.

Examples:

- google.com
- github.com
- chatgpt.com

Domain Normalization

Hostnames are normalized before reputation analysis.

For example:

www.google.com

is normalized to:

google.com

VirusTotal Threat Intelligence

The normalized domain is sent to the VirusTotal API.

VirusTotal provides aggregated security analysis results including:

- Malicious detections
- Suspicious detections
- Harmless detections
- Undetected results
- Reputation score

Threat Classification

The system applies predefined classification rules to the returned threat intelligence.

Possible classifications include:

- SAFE
- REVIEW
- SUSPICIOUS
- MALICIOUS
- UNKNOWN

The classification logic is designed to avoid treating a single security-engine detection as automatically malicious.

For example:

Malicious >= 4
→ MALICIOUS

Malicious = 2 or 3
→ SUSPICIOUS

Malicious = 1
→ REVIEW

Multiple harmless detections
→ SAFE

---

<h2><b>System Architecture</b></h2>

                         +-----------------------------+
                         |       LIVE NETWORK          |
                         |          TRAFFIC            |
                         +-------------+---------------+
                                       |
                                       v
                         +-----------------------------+
                         |       PyShark / TShark      |
                         |     Live Packet Capture     |
                         +-------------+---------------+
                                       |
                    +------------------+------------------+
                    |                                     |
                    v                                     v
          +-------------------+                  +-------------------+
          | Packet Monitoring |                  |  DNS Monitoring   |
          +---------+---------+                  +---------+---------+
                    |                                      |
                    v                                      v
          +-------------------+                  +-------------------+
          | Feature Extraction|                  | Domain Extraction|
          +---------+---------+                  +---------+---------+
                    |                                      |
                    v                                      v
          +-------------------+                  +-------------------+
          | Sequence Buffer   |                  | Domain Normalize |
          | 50 × 42 Features  |                  +---------+---------+
          +---------+---------+                            |
                    |                                      v
                    v                            +-------------------+
          +-------------------+                  |  VirusTotal API   |
          |       CNN         |                  | Threat Intelligence|
          | Local Features    |                  +---------+---------+
          +---------+---------+                            |
                    |                                      v
                    v                            +-------------------+
          +-------------------+                  | Reputation Rules |
          |      BiLSTM       |                  +---------+---------+
          | Sequential        |                            |
          | Dependencies      |                            |
          +---------+---------+                            |
                    |                                      |
                    v                                      |
          +-------------------+                             |
          |    Attention     |                             |
          | Important Pattern|                             |
          | Identification   |                             |
          +---------+---------+                             |
                    |                                      |
                    v                                      |
          +-------------------+                             |
          | Intrusion         |                             |
          | Confidence Score  |                             |
          +---------+---------+                             |
                    |                                      |
                    v                                      |
          +-------------------+                             |
          | Threshold Check   |                             |
          |      >= 0.80      |                             |
          +---------+---------+                             |
                    |                                      |
                    +------------------+-------------------+
                                       |
                                       v
                         +-----------------------------+
                         |      Threat Classification  |
                         |                             |
                         | SAFE                        |
                         | REVIEW                      |
                         | SUSPICIOUS                  |
                         | MALICIOUS                   |
                         +-------------+---------------+
                                       |
                                       v
                         +-----------------------------+
                         |     Alert Generation        |
                         |                             |
                         | latest_alert.json           |
                         +-------------+---------------+
                                       |
                                       v
                         +-----------------------------+
                         |      Streamlit Dashboard    |
                         +-----------------------------+
                                       |
                                       v
                         +-----------------------------+
                         | Real-Time Threat Monitoring |
                         +-----------------------------+

---

<h2><b>Threat Detection Architecture</b></h2>

The network intrusion detection model follows the architecture:

Input Packet Sequence
        |
        v
50 × 42 Feature Matrix
        |
        v
CNN
        |
        | Local Traffic Patterns
        v
BiLSTM
        |
        | Forward + Backward
        | Sequential Dependencies
        v
Attention
        |
        | Important Sequence Information
        v
Classification Layer
        |
        v
Intrusion Confidence
        |
        v
Threshold = 0.80
        |
   +----+----+
   |         |
 < 0.80   >= 0.80
   |         |
 Normal    Alert

---

<h2><b>CNN-BiLSTM-Attention Model</b></h2>

<h3><b>CNN Layer</b></h3>

The CNN component extracts local patterns from consecutive packet features.

It can learn combinations of features such as:

- Packet size patterns
- TCP flag combinations
- Port patterns
- Protocol-related patterns
- Short-term traffic behavior


<h3><b>BiLSTM Layer</b></h3>

The Bidirectional LSTM learns sequential relationships between packets.

The forward LSTM processes the sequence from the beginning to the end, while the backward LSTM processes the sequence in the opposite direction.

This allows the model to learn richer temporal relationships within the packet sequence.


<h3><b>Attention Layer</b></h3>

The attention mechanism identifies the most relevant portions of the processed sequence.

Instead of treating every packet representation equally, the model can assign higher importance to portions that contribute more strongly to the final prediction.


<h3><b>Threshold Detection</b></h3>

The current intrusion alert threshold is:

ALERT_THRESHOLD = 0.80

The threshold converts the model's confidence into an operational detection decision.

Example:

Confidence = 0.35
→ Normal

Confidence = 0.65
→ Normal

Confidence = 0.79
→ Normal

Confidence = 0.80
→ Alert Candidate

Confidence = 0.95
→ Alert Candidate

The threshold is configurable and can be optimized using validation data in future versions.

---

<h2><b>Website Reputation Detection</b></h2>

The website monitoring component uses DNS traffic to identify websites requested by the local machine.

The process is:

DNS Packet
    ↓
Extract Domain
    ↓
Normalize Domain
    ↓
Check Cache
    ↓
VirusTotal API
    ↓
Retrieve Threat Statistics
    ↓
Apply Classification Rules
    ↓
Generate Website Alert
    ↓
Update Dashboard


<h3><b>VirusTotal Security Engines</b></h3>

VirusTotal aggregates results from multiple security vendors and analysis engines.

The project uses these aggregated results to determine the reputation of a domain.

For example:

Malicious: 0
Suspicious: 0
Harmless: 60
Undetected: 31

would normally result in:

SAFE


Whereas:

Malicious: 3
Suspicious: 2
Harmless: 50
Undetected: 36

can result in:

SUSPICIOUS

---

<h2><b>Tech Stack</b></h2>

<h3><b>Programming Language</b></h3>

Python


<h3><b>Deep Learning</b></h3>

PyTorch
CNN
Bidirectional LSTM
Attention Mechanism


<h3><b>Network Monitoring</b></h3>

PyShark
TShark
Wireshark


<h3><b>Threat Intelligence</b></h3>

VirusTotal API
REST API
Requests


<h3><b>Data Processing</b></h3>

NumPy
JSON
Python Collections


<h3><b>Environment and Configuration</b></h3>

Python Virtual Environment
python-dotenv
.env configuration


<h3><b>Dashboard</b></h3>

Streamlit

---

<h2><b>Project Structure</b></h2>

| Folder / File | Description |
| --------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **src/** | Contains the core monitoring, detection, and model components. |
| **src/live_ids.py** | Captures live network packets and performs CNN-BiLSTM-Attention intrusion detection. |
| **src/model_attn.py** | Contains the CNN-BiLSTM-Attention IDS model architecture. |
| **src/explainer.py** | Generates explanations for detected network intrusion alerts. |
| **src/live_domain_monitor.py** | Monitors live DNS traffic and performs website threat detection. |
| **src/domain_detector.py** | Extracts and normalizes domains from DNS traffic. |
| **src/reputation_checker.py** | Communicates with VirusTotal API and classifies domain reputation. |
| **src/website_alert.py** | Saves website threat information for dashboard consumption. |
| **src/test_reputation.py** | Tests domain reputation classification using sample domains. |
| **best_ids_model.pth** | Trained CNN-BiLSTM-Attention model weights. |
| **dashboard/** | Contains the Streamlit dashboard. |
| **dashboard/app.py** | Main Streamlit dashboard application. |
| **dashboard/latest_alert.json** | Stores the latest detected website threat information. |
| **requirements.txt** | Lists project dependencies. |
| **.env** | Stores sensitive configuration such as the VirusTotal API key. |
| **venv/** | Python virtual environment containing project dependencies. |


<h2><b>Features</b></h2>

- Real-time network packet monitoring
- AI-based intrusion detection
- CNN-BiLSTM-Attention architecture
- 50-packet sequence analysis
- 42 network traffic features
- Configurable intrusion confidence threshold
- Current threshold of 0.80
- Repeated detection filtering
- Alert cooldown mechanism
- Real-time DNS monitoring
- Automatic domain extraction
- Domain normalization
- VirusTotal threat intelligence
- Malicious/Suspicious/Review/Safe classification
- Domain caching to reduce repeated API requests
- JSON-based alert storage
- Streamlit threat dashboard
- Human-readable security explanations

---

<h2><b>Detection Threshold and Alert Control</b></h2>

The IDS uses a confidence threshold to control when an AI prediction becomes an operational alert.

Current configuration:

ALERT_THRESHOLD = 0.80

Additional controls are used to reduce false alerts:

MIN_ALERT_OCCURRENCES = 2

ALERT_WINDOW_SEC = 8

ALERT_COOLDOWN = 30 seconds

This means the system does not necessarily generate an alert immediately after one high-confidence prediction.

Instead, the system checks whether high-confidence detections occur repeatedly within the configured time window.

This helps reduce noisy one-off detections.

---

<h2><b>Website Detection Classification</b></h2>

The website detection component uses VirusTotal analysis statistics.

The current classification approach is:

| Condition | Classification |
| --------------------------- | ---------------- |
| Malicious >= 4 | MALICIOUS |
| Malicious = 2 or 3 | SUSPICIOUS |
| Suspicious >= 3 | SUSPICIOUS |
| Malicious = 1 | REVIEW |
| Harmless >= 3 | SAFE |
| Otherwise | UNKNOWN |

A single malicious detection is therefore treated as a low-confidence review case rather than automatically declaring the website malicious.

This is important because a single security engine can produce a false positive.

---

<h2><b>Data Flow</b></h2>

Network Traffic:

Live Packet
    ↓
Packet Feature Extraction
    ↓
Sequence Buffer
    ↓
CNN
    ↓
BiLSTM
    ↓
Attention
    ↓
Confidence Score
    ↓
Threshold
    ↓
Intrusion Alert


Website Traffic:

DNS Query
    ↓
Domain Extraction
    ↓
Domain Normalization
    ↓
VirusTotal API
    ↓
Threat Statistics
    ↓
Classification
    ↓
Website Alert

---

<h2><b>Dashboard</b></h2>

The Streamlit dashboard provides a visual interface for monitoring detected website threats.

Dashboard information can include:

- Current threat status
- Latest analyzed domain
- Malicious detections
- Suspicious detections
- Harmless detections
- Undetected detections
- Reputation score
- Detection timestamp
- Threat explanation
- Recommended investigation status

---

<h2><b>How to Run the Project</b></h2>

<h3><b>1. Create and activate virtual environment</b></h3>

python -m venv venv

venv\Scripts\activate


<h3><b>2. Install dependencies</b></h3>

pip install -r requirements.txt


<h3><b>3. Configure VirusTotal API</b></h3>

Create a `.env` file:

VIRUSTOTAL_API_KEY=your_api_key_here


<h3><b>4. Verify domain reputation</b></h3>

python src\test_reputation.py


<h3><b>5. Start live website monitoring</b></h3>

python src\live_domain_monitor.py


<h3><b>6. Start the Streamlit dashboard</b></h3>

streamlit run dashboard\app.py


<h3><b>7. Open the dashboard</b></h3>

http://localhost:8501

---

<h2><b>Example Output</b></h2>

A safe domain may produce:

🌐 Website: google.com

🔎 Reputation: SAFE

🟢 Safe website


A suspicious domain may produce:

🌐 Website: example-domain.com

🔎 Reputation: SUSPICIOUS

🟠 SUSPICIOUS WEBSITE


The dashboard can display:

🟠 SUSPICIOUS WEBSITE DETECTED

Threat Status:
SUSPICIOUS

Malicious Detections:
3

Suspicious Detections:
2

Reputation:
-20

---

<h2><b>Alert Handling</b></h2>

When a potential network intrusion is detected, the system records:

- Timestamp
- Frame number
- Source IP
- Destination IP
- Protocol
- Severity
- Confidence
- Explanation
- Recommended action

For website threats, the system records:

- Domain
- Timestamp
- Threat status
- Malicious detections
- Suspicious detections
- Harmless detections
- Undetected detections
- Reputation score

---

<h2><b>Current Limitations</b></h2>

- The quality of intrusion detection depends on the training dataset and feature representation.
- The current packet feature extraction is based on a limited set of packet-level attributes.
- The 0.80 threshold is configurable and has not been established as a universal cybersecurity threshold.
- VirusTotal API availability depends on API limits and network connectivity.
- A single security engine detection does not necessarily indicate a confirmed malicious domain.
- DNS monitoring identifies requested domains but does not inspect the complete encrypted HTTPS content.
- The current system focuses on detection and alerting rather than automatic threat prevention.
- Real-world deployment would require additional security controls, logging, authentication, and testing.

---

<h2><b>Project Objective</b></h2>

The ultimate objective of this project is to provide a unified real-time security monitoring platform that combines deep-learning-based network anomaly detection with external website threat intelligence.

---

By combining network behavior analysis and domain reputation analysis, the system provides multiple sources of security evidence instead of depending on a single detection mechanism.
