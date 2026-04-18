# Automated Taxonomy of APT Research Literature

Derived from hierarchical clustering of 100 papers.

Each cluster below represents a thematic group of papers that use
similar vocabulary in their abstracts. The **suggested label** is
a heuristic match to MITRE ATT&CK tactics - verify by inspection.

---

## Cluster 0: Unclassified (inspect top terms)
- **Size:** 60 papers
- **Top characteristic terms:** `digital` (0.028), `risk` (0.019), `social` (0.017), `frameworks` (0.016), `assessment` (0.016), `challenges` (0.015), `financial` (0.015), `legal` (0.014)
- **Representative papers (closest to centroid):**
  - Inaugural Editorial of the Inspire Smart Systems First Issue Publication
  - Harnessing Artificial Intelligence to Strengthen Intrusion Detection in Modern Network System
  - Risk Management in it Program Management: Balancing Cybersecurity, AI Integration and Infrastructure Stability

## Cluster 1: Unclassified (inspect top terms)
- **Size:** 3 papers
- **Top characteristic terms:** `job` (0.104), `measure` (0.098), `realistic` (0.093), `roles` (0.093), `scores` (0.068), `generative models` (0.067), `severity` (0.066), `generative` (0.064)
- **Representative papers (closest to centroid):**
  - Estimating Attackers’ Profiles Results in More Realistic Vulnerability Severity Scores
  - Generating Synthetic Tabular Data for DDoS Detection Using Generative Models
  - A Cyber Counterintelligence Competence Framework: Developing the Job Roles

## Cluster 2: Unclassified (inspect top terms)
- **Size:** 24 papers
- **Top characteristic terms:** `cloud` (0.038), `high` (0.026), `network` (0.023), `cyberattacks` (0.020), `graph` (0.020), `layer` (0.018), `hook` (0.017), `low` (0.017)
- **Representative papers (closest to centroid):**
  - An Integrated Cybersecurity Defense Framework for Attack Intelligence Analysis, Counteraction, and Traceability in Complex Network Architectures
  - LOW-SPEED HTTP DDOS ATTACK PREVENTION MODEL FOR END USERS
  - Bridging Observability and Security: A Graph-Based Arbitration and Adaptive Sensing Approach via eBPF

## Cluster 3: Unclassified (inspect top terms)
- **Size:** 2 papers
- **Top characteristic terms:** `dengan` (0.180), `dan` (0.165), `yang` (0.160), `serangan` (0.146), `untuk` (0.138), `ini` (0.129), `klasifikasi` (0.127), `menggunakan` (0.117)
- **Representative papers (closest to centroid):**
  - Machine Learning for Cybersecurity: Web Attack Detection (Brute Force, XSS, SQL Injection)
  - Analisis Kinerja Intrusion Detection System Berbasis Algoritma Random Forest Menggunakan Dataset Unbalanced Honeynet BSSN

## Cluster 4: Unclassified (inspect top terms)
- **Size:** 3 papers
- **Top characteristic terms:** `noise` (0.144), `et` (0.142), `non` (0.092), `label` (0.088), `density` (0.086), `machines` (0.085), `limit` (0.080), `biomolecules` (0.071)
- **Representative papers (closest to centroid):**
  - Junctionless negative capacitance FinFET-based dielectric modulated biosensor with strain silicon integration at different FE thickness
  - Apprentissage automatique non supervisé pour la détection de trafics illégitimes. (Unsupervised machine learning for illegitimate traffic detection)
  - Rethinking Online Smart Contract Diagnosis in Blockchains: A Diffusion Perspective

## Cluster 5: Unclassified (inspect top terms)
- **Size:** 3 papers
- **Top characteristic terms:** `abnormal` (0.159), `log` (0.106), `behaviors` (0.086), `ce` (0.075), `databases` (0.070), `detecting analyzing` (0.068), `ii` (0.068), `matching` (0.066)
- **Representative papers (closest to centroid):**
  - Intrusion Detection Based on Sequential Information Preserving Log Embedding Methods and Anomaly Detection Algorithms
  - A multi-layer approach for advanced persistent threat detection using machine learning based on network traffic
  - Artificial intelligence, capsule endoscopy, databases, and the Sword of Damocles

## Cluster 6: Unclassified (inspect top terms)
- **Size:** 5 papers
- **Top characteristic terms:** `number` (0.084), `grid` (0.058), `wind` (0.055), `data sets` (0.054), `sets` (0.053), `campaign` (0.051), `command` (0.048), `node` (0.046)
- **Representative papers (closest to centroid):**
  - Smart Grid and wind generators: an overview of cyber threats and vulnerabilities of power supply networks
  - Probabilistic Deming Deep Recursive Network and Schmidt-Samoa Cryptosystem for Cyber Attack Detection and Secure Transmission in Wireless Networks
  - Doctrine-to-Deployment: Role of Advanced Persistent Threats in Russia’s “Information Confrontation” Doctrine

---
## Run Metadata
- Corpus size: **100** papers
- TF-IDF features (full): **3964**
- GA-selected features: **139**
- Baseline silhouette (k=7): **+0.0443**
- GA-optimized silhouette (k=7): **+0.1175**
- Delta: **+0.0732**
- GA runtime: **0.7s**
- Random seed: **42** (reproducible)