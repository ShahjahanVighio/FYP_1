"use client";

import {
  Database,
  GitFork,
  Cpu,
  Layers,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  TrendingUp,
  Award,
  Zap,
  Target,
  Shield,
  FileText
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";
import { useDataset } from "@/context/DatasetContext";

export default function MethodologyPage() {
  const { data, dataset, benchmarks } = useDataset();
  const models = data.models;
  const stats = data.dataset.stats;

  // Imbalance comparison data: Cost-Sensitive vs. SMOTE+RUS
  const imbalanceComparison = [
    {
      model: "Random Forest",
      "Cost-Sensitive (Raw Train)": Number((models.Random_Forest.cross_validation.cost_sensitive.cv_mean_ap * 100).toFixed(1)),
      "SMOTE + RUS (Resampled)": Number((models.Random_Forest.cross_validation.resampled.cv_mean_ap * 100).toFixed(1)),
      gain: `+${((models.Random_Forest.cross_validation.cost_sensitive.cv_mean_ap - models.Random_Forest.cross_validation.resampled.cv_mean_ap) * 100).toFixed(1)}% PR-AUC`,
    },
    {
      model: "XGBoost",
      "Cost-Sensitive (Raw Train)": Number((models.XGBoost.cross_validation.cost_sensitive.cv_mean_ap * 100).toFixed(1)),
      "SMOTE + RUS (Resampled)": Number((models.XGBoost.cross_validation.resampled.cv_mean_ap * 100).toFixed(1)),
      gain: `+${((models.XGBoost.cross_validation.cost_sensitive.cv_mean_ap - models.XGBoost.cross_validation.resampled.cv_mean_ap) * 100).toFixed(1)}% PR-AUC`,
    },
    {
      model: "CatBoost",
      "Cost-Sensitive (Raw Train)": Number((models.CatBoost.cross_validation.cost_sensitive.cv_mean_ap * 100).toFixed(1)),
      "SMOTE + RUS (Resampled)": Number((models.CatBoost.cross_validation.resampled.cv_mean_ap * 100).toFixed(1)),
      gain: `+${((models.CatBoost.cross_validation.cost_sensitive.cv_mean_ap - models.CatBoost.cross_validation.resampled.cv_mean_ap) * 100).toFixed(1)}% PR-AUC`,
    },
  ];

  const stages = [
    {
      step: "01",
      title: "Out-of-Core DuckDB Streaming & Vectorization",
      icon: Database,
      badge: `${stats.raw_log_size_gb} GB Telemetry (<1 GB RAM)`,
      description:
        `Parsed ${stats.duration_months} continuous months of heterogeneous audit telemetry across ${stats.employees.toLocaleString()} corporate employees in CERT ${dataset}. Employs streaming DuckDB chunking and vectorized NumPy processors to eliminate memory bottlenecks, processing up to 100 GB logs under <1 GB peak RAM.`,
      bullets: [
        "logon.csv: User logon/logoff hour distributions, after-hours activity, unauthorized workstation hops.",
        "device.csv: USB mass storage connection counts, connect/disconnect timestamps, mean dwell times.",
        "file.csv: File transfers to removable drives, document word count statistics, extension tagging.",
        "email.csv: Inbound/outbound email volumes, external recipient counts, attachment sizes.",
        "http.csv: Web browsing logs, data leak domain visits, keylogger lookups, job hunting activity.",
        "psychometric.csv: Big Five personality scores (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).",
        "LDAP/: Monthly employee role changes, organizational hierarchy, department and project team transfers.",
      ],
    },
    {
      step: "02",
      title: "Logon/Logoff Delimited Sessionization & Rolling Baselines",
      icon: Layers,
      badge: `${stats.total_sessions.toLocaleString()} Sessions Reconstructed`,
      description:
        "Engineered discrete behavioral user sessions bounded by logon and logoff events on specific workstations. Applied vectorized 7-day rolling baseline calculation to quantify behavioral deviation:",
      bullets: [
        "Workhour: Standard weekday business hours (Monday–Friday, 08:00–18:00).",
        "Afterhour: Weekday evenings and early mornings (18:00–08:00).",
        "Weekend Daytime: Saturday and Sunday business hours (08:00–18:00).",
        "Weekend Nighttime: Saturday and Sunday overnight sessions (18:00–08:00).",
        "Extracted 128 multi-domain behavioral, content, and organizational features per session.",
      ],
    },
    {
      step: "03",
      title: "Leakage-Free 80/20 Stratified Partitioning & Feature Selection",
      icon: Sliders,
      badge: "Zero Data Snooping",
      description:
        `Segregated an untouched test partition (${stats.test_set_sessions.toLocaleString()} sessions, ${stats.test_threats} ground-truth threats) strictly before any statistical imputation or feature scaling:`,
      bullets: [
        "Fitted median imputer and standardizer strictly on training folds to eliminate data snooping.",
        "Ranked all 128 features using linear ANOVA F-statistic and non-linear Mutual Information.",
        "Extracted the Top 50 optimal non-redundant features using Borda-count consensus ranking.",
        "Empirically benchmarked raw Cost-Sensitive training against SMOTE + Random Undersampling.",
      ],
    },
    {
      step: "04",
      title: "Base Classifiers Trained via 5-Fold Stratified Cross-Validation",
      icon: Cpu,
      badge: "Bagging + Boosting Diversity",
      description:
        "Trained three structurally complementary tree-based architectures with 50-round early stopping and hyperparameter optimization calibrated on PR-AUC:",
      bullets: [
        "Random Forest: 200 bagging trees with balanced_subsample weighting for variance stabilization.",
        "XGBoost: Histogram gradient boosting with scale_pos_weight optimization for maximal precision.",
        "CatBoost: Symmetric oblivious decision trees with SqrtBalanced/SMOTE weighting for maximal recall.",
      ],
    },
    {
      step: "05",
      title: "Out-of-Fold Stacking Meta-Learner & Threshold Calibration",
      icon: GitFork,
      badge: "Leakage-Free Meta-Learning",
      description:
        "Generated out-of-fold probability vectors strictly on the training partition via 5-fold CV to train the meta-learner:",
      bullets: [
        dataset === "r4.2"
          ? "Convex soft-voting simplex optimization: w_XGB = 0.70, w_CatBoost = 0.30, w_RF = 0.00."
          : "Calibrated Logistic Regression on base probabilities [P_RF, P_XGB, P_CB], logit transforms, and dispersion statistics.",
        `Tuned decision threshold on OOF validation curves to maximize F1-score (T* = ${models.Stacking_Ensemble.test_metrics_optimal_threshold.threshold}).`,
        "Evaluated final generalization on the untouched test partition without any retraining.",
      ],
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">
          Scientific Methodology & Engineering Architecture
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">
          End-to-end mathematical, statistical, and algorithmic design for sparse tabular security log classification in CMU CERT {dataset}.
        </p>
      </div>

      {/* 5-Stage Scientific Pipeline Diagram */}
      <div className="space-y-4">
        <h2 className="text-sm font-bold text-slate-900">
          5-Stage Pipeline Architecture
        </h2>

        <div className="space-y-4">
          {stages.map((stage) => {
            const Icon = stage.icon;
            return (
              <div
                key={stage.step}
                className="enterprise-card p-5 border-l-4 border-l-blue-600 space-y-3"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div className="flex items-center space-x-3">
                    <span className="font-mono text-xs font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                      STAGE {stage.step}
                    </span>
                    <h3 className="text-sm font-bold text-slate-900">
                      {stage.title}
                    </h3>
                  </div>
                  <span className="text-[11px] font-semibold text-slate-600 bg-slate-100 px-2.5 py-0.5 rounded">
                    {stage.badge}
                  </span>
                </div>

                <p className="text-xs text-slate-600 leading-relaxed">
                  {stage.description}
                </p>

                <ul className="grid grid-cols-1 md:grid-cols-2 gap-2 pt-1">
                  {stage.bullets.map((bullet, idx) => (
                    <li
                      key={idx}
                      className="text-xs text-slate-600 flex items-start space-x-2 bg-slate-50 p-2.5 rounded border border-slate-100"
                    >
                      <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 mt-0.5 shrink-0" />
                      <span className="leading-snug">{bullet}</span>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>
      </div>

      {/* Imbalance Strategy Empirical Comparison Section */}
      <div className="enterprise-card p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-bold text-slate-900">
              Imbalance Strategy Empirical Benchmark: Cost-Sensitive vs. SMOTE+RUS
            </h2>
            <p className="text-xs text-slate-500">
              Comparing 5-fold cross-validation Mean Average Precision (PR-AUC) under extreme class imbalance.
            </p>
          </div>
          <span className="text-[11px] font-bold text-emerald-700 bg-emerald-50 px-2 py-1 rounded border border-emerald-200">
            Cost-Sensitive Outperforms Resampling
          </span>
        </div>

        <div className="h-64 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={imbalanceComparison}
              margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
              <XAxis dataKey="model" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
              <Tooltip
                formatter={(val: any) => [`${val}% PR-AUC`]}
                contentStyle={{
                  backgroundColor: "#FFFFFF",
                  border: "1px solid #E2E8F0",
                  borderRadius: "6px",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
              <Bar
                dataKey="Cost-Sensitive (Raw Train)"
                fill="#1E40AF"
                radius={[3, 3, 0, 0]}
              />
              <Bar
                dataKey="SMOTE + RUS (Resampled)"
                fill="#94A3B8"
                radius={[3, 3, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="p-3.5 rounded-lg border border-blue-200 bg-blue-50/40 text-xs text-slate-700 space-y-1.5 leading-relaxed">
          <div className="font-bold text-blue-900 flex items-center space-x-1.5">
            <Award className="w-4 h-4 text-blue-700" />
            <span>Key Research Finding on Ultra-Sparse Tabular Cyber Telemetry:</span>
          </div>
          <p>
            Synthetic Minority Over-sampling (SMOTE) with Random Under-sampling (RUS) interpolates between nearest neighbors in continuous space, which creates artificial synthetic sessions that violate discrete organizational role boundaries. In contrast, <strong>Cost-Sensitive Class Weighting</strong> preserves natural enterprise feature distributions while heavily penalizing false negatives, consistently delivering superior PR-AUC across all releases.
          </p>
        </div>
      </div>
    </div>
  );
}
