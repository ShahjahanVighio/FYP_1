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
  Award
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
  const { data, dataset } = useDataset();
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
      title: "Multi-Modal Log Ingestion & Stream Vectorization",
      icon: Database,
      badge: `${stats.raw_log_size_gb} GB Telemetry`,
      description:
        `Parsed ${stats.duration_months} continuous months of heterogeneous audit telemetry across ${stats.employees.toLocaleString()} corporate employees in CERT ${dataset}. Employs streaming DuckDB chunking and vectorized NumPy processors to eliminate memory bottlenecks.`,
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
      title: "Logon/Logoff Delimited Sessionization",
      icon: Layers,
      badge: `${stats.total_sessions.toLocaleString()} Sessions`,
      description:
        "Engineered discrete behavioral user sessions bounded by logon and logoff events on specific workstations. Applied 4 temporal slices per session:",
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
      badge: "Strict Data Isolation",
      description:
        `Segregated an untouched test partition (${stats.test_set_sessions.toLocaleString()} sessions, ${stats.test_threats} ground-truth threats) strictly before any statistical imputation or feature scaling:`,
      bullets: [
        "Fitted median imputer and standardizer strictly on training folds to eliminate data snooping.",
        "Ranked all 128 features using linear ANOVA F-statistic and non-linear Mutual Information.",
        "Extracted the Top 50 optimal non-redundant features using Borda-count consensus ranking.",
        `Empirically benchmarked raw Cost-Sensitive training against SMOTE + Random Undersampling.`,
      ],
    },
    {
      step: "04",
      title: "Base Classifiers Trained via 5-Fold Stratified Cross-Validation",
      icon: Cpu,
      badge: "Bagging + Boosting Diversity",
      description:
        "Trained three structurally complementary tree-based architectures with hyperparameter optimization on PR-AUC:",
      bullets: [
        "Random Forest: 200 bagging trees with balanced_subsample weighting for variance stabilization.",
        "XGBoost: Histogram gradient boosting with scale_pos_weight optimization for maximal precision.",
        "CatBoost: Symmetric oblivious decision trees with SqrtBalanced weighting for maximal recall.",
      ],
    },
    {
      step: "05",
      title: "Out-of-Fold Stacking Meta-Learner & Threshold Calibration",
      icon: GitFork,
      badge: "Zero-Leakage Stacking",
      description:
        "Generated out-of-fold probability vectors strictly on the training partition via 5-fold CV to train the meta-learner:",
      bullets: [
        dataset === "r4.2"
          ? "Convex soft-voting simplex optimization: w_XGB = 0.70, w_CatBoost = 0.30, w_RF = 0.00."
          : "Calibrated Logistic Regression on base probabilities, logit transforms, and dispersion statistics.",
        `Tuned decision threshold on OOF validation curves to maximize F1-score (T* = ${models.Stacking_Ensemble.test_metrics_optimal_threshold.threshold}).`,
        `Evaluated final generalization on the untouched test partition without any retraining.`,
      ],
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">
          Scientific Methodology & Architecture
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">
          End-to-end mathematical, statistical, and algorithmic design for sparse tabular security log classification in CERT {dataset}.
        </p>
      </div>

      {/* Breakthrough Card: Cost-Sensitive vs SMOTE */}
      <div className="enterprise-card p-6 space-y-6">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-blue-100 text-blue-800 border border-blue-200">
              Key Empirical Breakthrough
            </span>
          </div>
          <h2 className="mt-2 text-base font-bold text-slate-900">
            Cost-Sensitive Loss Weighting Outperforms Synthetic Oversampling (SMOTE + RUS)
          </h2>
          <p className="text-xs text-slate-500 mt-1 leading-relaxed max-w-4xl">
            Across both CERT r4.2 and r5.2, training base models directly on authentic imbalanced logs ({stats.imbalance_ratio}) using cost-sensitive class weighting consistently beat SMOTE + Random Undersampling on Stratified 5-Fold Cross-Validation.
          </p>
        </div>

        {/* Comparison Bar Chart */}
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={imbalanceComparison}
              margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
              <XAxis
                dataKey="model"
                stroke="#64748B"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: "#CBD5E1" }}
              />
              <YAxis
                stroke="#64748B"
                fontSize={11}
                tickLine={false}
                domain={[0, 100]}
                axisLine={{ stroke: "#CBD5E1" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#FFFFFF",
                  border: "1px solid #CBD5E1",
                  borderRadius: "6px",
                  fontSize: "11px",
                  boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.08)",
                }}
              />
              <Legend
                verticalAlign="top"
                align="right"
                wrapperStyle={{ fontSize: "11px", paddingBottom: "10px" }}
              />
              <Bar dataKey="Cost-Sensitive (Raw Train)" fill="#1E40AF" radius={[3, 3, 0, 0]} />
              <Bar dataKey="SMOTE + RUS (Resampled)" fill="#94A3B8" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Explanation Card */}
        <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 leading-relaxed">
          <span className="font-bold text-slate-900">Theoretical Reason: </span>
          Synthetic interpolation (SMOTE) between sparse positive outliers in 50-dimensional tabular space creates artificial feature combinations that distort the true manifold of normal employee behavior, introducing excessive false positives. Cost-sensitive weighting preserves authentic negative sample density while heavily penalizing threat misclassifications.
        </div>
      </div>

      {/* 5-Stage Architecture Cards */}
      <div className="space-y-4">
        <h2 className="text-sm font-bold text-slate-900">
          5-Stage End-to-End Pipeline
        </h2>

        <div className="space-y-4">
          {stages.map((st) => {
            const Icon = st.icon;
            return (
              <div key={st.step} className="enterprise-card p-5 space-y-3">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
                  <div className="flex items-center space-x-3">
                    <span className="font-mono text-xs font-bold text-slate-400 bg-slate-100 px-2 py-0.5 rounded">
                      STAGE {st.step}
                    </span>
                    <h3 className="text-sm font-bold text-slate-900 flex items-center space-x-2">
                      <Icon className="w-4 h-4 text-blue-700" />
                      <span>{st.title}</span>
                    </h3>
                  </div>

                  <span className="text-[11px] font-semibold text-slate-600 bg-slate-100 px-2.5 py-0.5 rounded border border-slate-200 w-fit">
                    {st.badge}
                  </span>
                </div>

                <p className="text-xs text-slate-600 leading-relaxed">
                  {st.description}
                </p>

                <ul className="space-y-1 text-xs text-slate-600 pt-1">
                  {st.bullets.map((b, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-blue-700 font-bold">•</span>
                      <span>{b}</span>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
