# SENTINEL.UEBA — Executive SOC Dashboard & Live Threat Simulator

[![Framework](https://img.shields.io/badge/Framework-Next.js%2016%20(App%20Router)-black.svg)](https://nextjs.org/)
[![UI Library](https://img.shields.io/badge/Styling-Tailwind%20CSS%20v4-38B2AC.svg)](https://tailwindcss.com/)
[![Charts](https://img.shields.io/badge/Charts-Recharts-22c55e.svg)](https://recharts.org/)
[![Icons](https://img.shields.io/badge/Icons-Lucide%20React-06b6d4.svg)](https://lucide.dev/)
[![TypeScript](https://img.shields.io/badge/Language-TypeScript-blue.svg)](https://www.typescriptlang.org/)

SENTINEL.UEBA is an executive-grade Security Operations Center (SOC) dashboard and live insider risk inference interface designed to visualize the complete machine learning pipeline, statistical methodology, experimental leaderboards, and real-time session risk scoring across the **CMU CERT Insider Threat Benchmarks (Release 4.2 & Release 5.2)**.

---

## 📸 Key Capabilities & Page Structure

```
Insider-Threat-Project-frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx             # Root layout with dark theme, Navbar, and Footer
│   │   ├── page.tsx               # / (Command Center & Executive KPI Summary)
│   │   ├── methodology/page.tsx   # /methodology (5-Stage Architecture & SMOTE Comparison)
│   │   ├── results/page.tsx       # /results (Interactive Leaderboard & Plot Lightbox)
│   │   ├── features/page.tsx      # /features (Top 50 Signals & ANOVA+MI Table)
│   │   ├── simulator/page.tsx     # /simulator (Interactive Live Threat Simulator)
│   │   └── api/simulate/route.ts  # /api/simulate (Microservice Proxy + Fallback Engine)
│   ├── components/
│   │   ├── Navbar.tsx             # Navigation + Multi-Dataset Selector (r4.2 vs r5.2)
│   │   └── Footer.tsx             # Academic Attribution, System Specs & GitHub Links
│   └── data/
│       ├── r4_2_metrics.json      # Structured benchmark metrics & feature stats for CERT r4.2
│       └── r5_2_metrics.json      # Structured benchmark metrics & feature stats for CERT r5.2
└── public/
    └── plots/                     # High-resolution 300 DPI evaluation figures
        ├── r4_2/                  # CERT r4.2 evaluation charts
        └── r5_2/                  # CERT r5.2 evaluation charts
```

### 1. Command Center (`/`)
- **Executive KPIs**: Real-time display of Benchmark Scale (Sessions, Employees, Months), Threat Detection Recall (77.8%–84.3%), Ultra-Low False Positive Rate (`0.019%`–`0.037%`), and Ensemble PR-AUC (`0.868`–`0.924`).
- **Comparative Radar & Bar Analytics**: Recharts multi-model visualization contrasting Random Forest, XGBoost, CatBoost, and the Stacking Ensemble across Precision, Recall, F1, and PR-AUC.
- **SOC Recommendation Cards**: Quick status view showing optimal operational thresholds ($T^*$) and threat triage rules.

### 2. Methodology & Science (`/methodology`)
- **5-Stage Pipeline Breakdown**: Interactive cards describing Raw Telemetry Ingestion, Vectorized Sessionization, Leakage-Free Preprocessing, Cross-Validated Base Classifiers, and Out-of-Fold Stacking.
- **Empirical Breakthrough Visualizer**: In-depth analysis comparing **Cost-Sensitive Loss Weighting** against **SMOTE + Random Undersampling**, detailing why synthetic oversampling corrupts discrete security telemetry manifolds.

### 3. Model Results & Plot Lightbox (`/results`)
- **Interactive Threshold Switcher**: Instant switching between Default Threshold ($T = 0.50$) and Tuned Optimal Threshold ($T^*$), updating precision, recall, F1, and confusion matrix cell values in real time.
- **Full-Resolution Plot Inspector**: Click-to-zoom modal for high-resolution 300 DPI confusion matrices, PR/ROC curves, and Gini feature importances.

### 4. Threat Feature Explorer (`/features`)
- **Search & Filter**: Real-time search across the Top 50 features with category filters: *Removable Media / USB*, *Data Leak Sites / Cloud*, *Job Search Activity*, *Hacking / Keylogger*, *File Exfiltration*, *Psychometric (OCEAN)*, and *Organizational / LDAP*.
- **Statistical Metric Attribution**: Sortable tables showing ANOVA $F$-statistic, Mutual Information score, individual ranks, and combined Borda-count rankings.

### 5. Live Insider Risk Simulator (`/simulator`)
- **Interactive Risk Sliders**: Dynamically manipulate session attributes (USB devices, cloud leak visits, job board queries, hacking URLs, file counts, after-hours flags, and psychometrics).
- **Preset Attack Scenarios**: One-click preset injection:
  - *Normal Employee Shift* (Benign office hours baseline)
  - *Scenario 1: After-Hours USB Exfiltration*
  - *Scenario 2: Job Hunting + Cloud Leak*
  - *Scenario 3: Keylogger & Admin Reconnaissance*
- **Real-Time Risk Scoring**: Connects to the FastAPI backend microservice (`api_server.py`) or falls back to a vectorized client-side logistic risk evaluator to output model probabilities ($P_{\text{RF}}, P_{\text{XGB}}, P_{\text{CB}}, P_{\text{Ensemble}}$), threat severity badge, and actionable SOC next steps.

---

## 🔄 Multi-Dataset Switcher Architecture

The dashboard is built with plug-in dataset readiness:
- The top navigation bar includes a dataset selector (`CMU CERT r4.2` and `CMU CERT r5.2`).
- Dataset schemas and benchmark stats adhere to a standard JSON contract (`r4_2_metrics.json` and `r5_2_metrics.json`), enabling drop-in visualization of newly trained models or future releases (e.g., r6.2) without altering frontend components.

---

## 🛠️ Installation & Development Setup

### Prerequisites
- Node.js 18.x or 20.x
- npm, yarn, or pnpm

### Quickstart

1. **Install Dependencies:**
   ```bash
   cd Insider-Threat-Project-frontend
   npm install
   ```

2. **Run Development Server:**
   ```bash
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) in your browser.

3. **Build for Production:**
   ```bash
   npm run build
   npm run start
   ```

---

## 🔌 Connecting Live Inference API (Optional)

To enable live model inference using trained weights:
1. Start the FastAPI microservice from either the r4.2 or r5.2 project directory:
   ```bash
   python api_server.py
   ```
   *(Listens on `http://localhost:8000`)*
2. The frontend Route Handler (`src/app/api/simulate/route.ts`) will automatically proxy requests to the FastAPI engine and provide live probability estimates.
