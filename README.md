# An Ensemble Machine Learning Approach to Insider Threat Detection Using Cultural Metrics and System Behaviors

[![Monorepo](https://img.shields.io/badge/Repository-Monorepo-9333ea.svg)]()
[![Datasets](https://img.shields.io/badge/Datasets-CMU%20CERT%20r4.2%20%7C%20r5.2%20%7C%20r6.2-blue.svg)](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099)
[![Scale](https://img.shields.io/badge/Scale-3.424M%20Total%20Sessions-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2016%20App%20Router-black.svg)](https://nextjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Styling-Tailwind%20CSS%20v4-38B2AC.svg)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)](CERT-r4-2-Insider-Threat-Project/license.txt)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

> **Final Year Project (FYP)** — Sukkur IBA University  
> **Domain**: Cybersecurity · User & Entity Behavior Analytics (UEBA) · Enterprise Machine Learning  
> **Project Members**:  
> - **Muhammad Muteeb** (`F23CSC048`) — *Lead Architect & Security Modeling*  
> - **Shahjahan Vighio** (`F23CSC021`) — *Data Pipelines & Feature Engineering*  
> - **Bilal Naeem** (`F23CSC008`) — *Evaluation Metrics & Frontend Analytics*  
>  
> **Benchmark Datasets**: Carnegie Mellon University Software Engineering Institute (CMU SEI) CERT Insider Threat Synthetic Test Suite (Releases 4.2, 5.2 & 6.2)  
> **Scale**: Over **`3,424,144` combined extracted user sessions** across 53 cumulative monitoring months, 7,000 enterprise employees, and all 5 canonical threat scenarios (including zero-shot Dropbox cloud exfiltration).

---

## 📌 Executive Summary & Abstract

Detecting malicious insider threats within enterprise IT networks is one of the most intractable challenges in modern cybersecurity. Malicious actions—ranging from data exfiltration and intellectual property theft to IT sabotage—occur hidden within immense volumes of routine employee activity. In authentic security logs, malicious sessions represent **less than 0.25% of total enterprise activity** in dense benchmarks, dropping to an extreme **0.0050% (20,515:1 needle-in-a-haystack sparsity)** in CERT r6.2. Standard machine learning classifiers trained naively on such telemetry suffer from crippling false alarm floods or catastrophic detection blind spots.

This monorepo houses a comprehensive, publication-grade research framework and operational Security Operations Center (SOC) dashboard that addresses this challenge through:
1. **Out-of-Core DuckDB Streaming & Vectorized Sessionization**: Processing over 154 GB of multi-source audit logs (logon/logoff, USB devices, file actions, email metadata, HTTP browsing, monthly LDAP rosters, and psychometric Big Five OCEAN scores) into structured, behavioral sessions under strictly **`<1 GB` peak RAM**.
2. **Leakage-Free Dual-Criteria Feature Selection**: Selecting the Top 50 non-redundant signals using combined linear (ANOVA $F$-statistic) and non-linear (Mutual Information) criteria strictly on training partitions.
3. **Rigorous Empirical Imbalance Benchmark**: Proving across **3.424 million sessions** that **Cost-Sensitive Class Weighting decisively outperforms synthetic oversampling (SMOTE + RUS)** on tabular security logs.
4. **Out-of-Fold (OOF) Stacking Meta-Learner**: Fusing diverse algorithmic strengths—**Random Forest** (Bagging variance reducer), **XGBoost** (Precision anchor), and **CatBoost** (Recall anchor)—via an enriched, calibrated meta-learner to achieve state-of-the-art PR-AUC with microscopic false positive rates (`0.019%` on r4.2, `0.037%` on r5.2, and `0.0015%` on r6.2).
5. **Human-Crafted Enterprise SOC Analytics Platform**: A production-ready Next.js 16 + Tailwind CSS frontend interface designed with the aesthetic of Linear, Stripe Dashboard, and Datadog, with live scenario triage, threshold calibration, and 3-release benchmark switching.

---

## 🗂️ Monorepo Structure & Quick Links

This repository is organized into three clean, self-contained subprojects:

| Subproject Directory | Focus & Benchmark | Total Sessions | Ground-Truth Threats | Key Outcome | Link |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **`CERT-r4-2-Insider-Threat-Project/`** | CMU CERT Release 4.2 (17 Months, 1,000 Employees) | `470,611` | `1,114` (421:1 ratio) | **PR-AUC: `0.9238`**, F1: `84.49%`, 188/223 threats caught, 18 FPs | [Explore r4.2 →](CERT-r4-2-Insider-Threat-Project/README.md) |
| **`CERT-r5-2-Insider-Threat-Project/`** | CMU CERT Release 5.2 (18 Months, 2,000 Employees, Scenarios 1–4) | `1,004,460` | `1,846` (543:1 ratio) | **PR-AUC: `0.8683`**, F1: `78.52%`, 287/369 threats caught, 75 FPs | [Explore r5.2 →](CERT-r5-2-Insider-Threat-Project/README.md) |
| **`CERT-r6-2-Insider-Threat-Project/`** | CMU CERT Release 6.2 (18 Months, 4,000 Employees, Scenarios 1–5) | `1,949,073` | `98` (19,888:1 ratio) | **PR-AUC: `0.3322`**, ROC: `0.9696`, 6 threats caught, only 6 FPs across 389k test sessions | [Explore r6.2 →](CERT-r6-2-Insider-Threat-Project/README.md) |
| **`Insider-Threat-Project-frontend/`** | Enterprise SOC Command Center & Live Threat Simulator | *Multi-Dataset Ready* | *Plug-in Ready* | Next.js 16, Recharts, Real-time scenario injection, 3-Release Switcher | [Explore Frontend →](Insider-Threat-Project-frontend/README.md) |

---

## 📊 Cross-Release Master Comparison: CERT r4.2 vs. r5.2 vs. r6.2

The table below summarizes dataset characteristics and empirical test results across all three benchmarks on strictly untouched test sets:

| Dimension / Metric | CMU CERT Release 4.2 | CMU CERT Release 5.2 | CMU CERT Release 6.2 | Cross-Release Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Duration & Timeline** | 17 Months (Jan 2010 – May 2011) | 18 Months (Jan 2010 – Jun 2011) | 18 Months (Jan 2010 – Jun 2011) | Full multi-year telemetry |
| **Monitored Employees** | 1,000 Users | 2,000 Users | 4,000 Users | 4× enterprise employee scale |
| **Raw Telemetry Volume** | 16.5 GB | 38.2 GB | 100.1 GB | Massive log scaling (`http.csv`: 90.2 GB) |
| **Total Extracted Sessions** | **`470,611`** | **`1,004,460`** | **`1,949,073`** | Nearly 2M sessions in r6.2 |
| **Active Threat Scenarios** | Scenarios 1–3 | Scenarios 1–4 | Scenarios 1–5 | Includes Zero-Shot Dropbox Exfiltration |
| **Ground-Truth Threats** | 1,114 Sessions (30 Users) | 1,846 Sessions (99 Users) | 98 Sessions (5 Users) | Ultra-sparse needle-in-haystack |
| **Class Imbalance Ratio** | **421.5 : 1** (0.2367% threats) | **543.1 : 1** (0.1838% threats) | **19,887.5 : 1** (0.0050% threats) | Extreme real-world sparsity |
| **Train / Test Partition** | 376,488 / 94,123 | 803,568 / 200,892 | 1,559,256 / 389,817 | 80/20 Stratified Partitioning |
| **Isolated Test Threats** | 223 Sessions | 369 Sessions | 22 Sessions | Strictly isolated during training |
| **Peak RAM Consumption** | `<0.5 GB` | `<0.8 GB` | `<1.0 GB` | DuckDB out-of-core streaming |
| **Random Forest PR-AUC** | `0.8086` (ROC: `0.9859`) | `0.6962` (ROC: `0.9763`) | `0.2864` (ROC: `0.9081`) | Precision anchor (only 4 FPs on r6.2) |
| **XGBoost PR-AUC** | `0.9160` (ROC: `0.9986`) | `0.8378` (ROC: `0.9987`) | `0.3236` (ROC: `0.9794`) | High ROC-AUC, 50% Scenario 2 recall |
| **CatBoost PR-AUC** | `0.9082` (ROC: `0.9984`) | `0.8532` (ROC: `0.9984`) | `0.1276` (ROC: `0.9621`) | **Recall Leader** (54.55% recall, 12 threats) |
| **Stacking Ensemble PR-AUC** | **`0.9238`** (ROC: `0.9987`) | **`0.8683`** (ROC: `0.9989`) | **`0.3322`** (ROC: `0.9696`) | **Overall Winner on all 3 benchmarks** |
| **Ensemble F1-Score ($T^*$)** | **`84.49%`** ($T^* = 0.34$) | **`78.52%`** ($T^* = 0.42$) | **`35.29%`** ($T^* = 0.35$) | Optimal harmonic cutoff |
| **False Alarms (FP / Test)** | **18 / 93,900 (0.019%)** | **75 / 200,523 (0.037%)** | **6 / 389,795 (0.0015%)** | Ultra-low SOC noise |

---

## 🔬 Core Empirical Findings & Scientific Contributions

### 1. Cost-Sensitive Loss Weighting Consistently Dominates Resampling
Across all three benchmarks, Stratified 5-Fold Cross-Validation demonstrates that **Cost-Sensitive Class Weighting decisively outperforms SMOTE + Random Undersampling on tabular security logs**:
- On CERT r4.2: Cost-Sensitive beats SMOTE by **+21.5%** PR-AUC on Random Forest.
- On CERT r5.2: Cost-Sensitive beats SMOTE by **+11.7%** PR-AUC on CatBoost.
- On CERT r6.2: Cost-Sensitive beats SMOTE by **+31.4%** PR-AUC on Random Forest and **+98.2%** on XGBoost (`0.4096` vs. `0.2067`).

**Scientific Explanation**: In dense security telemetry, benign activities outnumber threats by `400:1` to `20,000:1`. Synthetic interpolation (SMOTE) across 50 heterogeneous features corrupts discrete indicator distributions (e.g., USB connection flags, count distributions), degrading generalization. Direct cost-sensitive class weighting preserves real-world distribution density while heavily penalizing threat misses.

### 2. Multi-Model Algorithmic Complementarity
- **Random Forest (Bagging)**: Acts as a variance stabilizer across thousands of low-correlation decision trees (only 4 false alarms across 389,795 test sessions in r6.2).
- **XGBoost (Histogram Boosting)**: Maximizes **Precision & ROC-AUC** (`0.9794` on r6.2, leading to only 10 false alarms and 50% Scenario 2 recall).
- **CatBoost (Symmetric Oblivious Trees)**: Maximizes **Recall & Sensitive Detection** (intercepting 12/22 threats on r6.2, 90% recall on Scenario 2 Sabotage, and the sole detector of Scenario 3 Keylogger attack).
- **Stacking Meta-Learner**: Combines these complementary decision boundaries using out-of-fold probability vectors to achieve the highest overall PR-AUC across every single benchmark release.

---

## 🚀 Quickstart & Reproducibility Guide

### 1. Global Setup
Clone the monorepo:
```bash
git clone https://github.com/ShahjahanVighio/FYP_1.git
cd FYP_1
```

### 2. Run the Next.js SOC Dashboard
```bash
cd Insider-Threat-Project-frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to explore the interactive SOC platform, model benchmark matrices, and threat simulator.

---

## 👥 Authors & Project Attribution

This project is submitted in partial fulfillment of the requirements for the Degree of **Bachelor of Science in Computer Science** at **Sukkur IBA University**:
- **Muhammad Muteeb** (`F23CSC048`) — *Lead Architect & Security Modeling*
- **Shahjahan Vighio** (`F23CSC021`) — *Data Pipelines & Feature Engineering*
- **Bilal Naeem** (`F23CSC008`) — *Evaluation Metrics & Frontend Analytics*
