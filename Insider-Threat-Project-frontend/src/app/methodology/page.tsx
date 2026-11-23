"use client";

import {
  GitFork,
  Database,
  Layers,
  Filter,
  Cpu,
  ShieldCheck,
  CheckCircle2,
  AlertOctagon,
  ArrowRight,
  TrendingUp,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
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
      title: "Multi-Modal Log Ingestion & High-Speed Vectorization",
      icon: Database,
      badge: `${stats.raw_log_size_gb} GB Raw Data`,
      description:
        `Parsed ${stats.duration_months} continuous months of heterogeneous audit telemetry across ${stats.employees.toLocaleString()} corporate employees in CERT ${dataset}. Employs streaming DuckDB chunking and vectorized NumPy processors to eliminate memory bottlenecks on Windows.`,
      bullets: [
        "logon.csv (58.5 MB, 854k rows): Logon/logoff hour distributions, after-hours activity, unauthorized workstation hops.",
        "device.csv (29.0 MB, 405k rows): Removable USB flash drive connection/disconnection events.",
        "file.csv (193 MB, 445k rows): File access, document word counts, copy actions to external storage media.",
        "email.csv (1.36 GB): Internal vs. external destination analysis, blind carbon copies (BCC), file attachments.",
        "http.csv (14.5 GB): Categorization into cloud leak sites, job hunting portals, and hacker/keylogger lookups.",
        "psychometric.csv & LDAP/: Dynamic organizational hierarchy and Big Five OCEAN psychological profiles.",
      ],
    },
    {
      step: "02",
      title: "Sessionization & Behavioral Feature Engineering",
      icon: Layers,
      badge: "128 Features / Session",
      description:
        "Aggregated continuous event streams into bounded user sessions delimited by logon and logoff boundaries. Reconstructed 470,611 discrete operational sessions mapped to exact CMU ground-truth scenario action IDs.",
      bullets: [
        "Temporal Slicing: Workhour weekday, Afterhour weekday, Weekend daytime, Weekend night.",
        "Disambiguation: Primary assigned workstation vs. shared laboratory PCs vs. supervisor machines.",
        "Content Vectorization: Mean URL length, URL depth, document word counts, file extensions, and disk targets.",
        "Ground Truth Alignment: 469,497 benign sessions vs. 1,114 malicious insider threat sessions (0.237% prevalence).",
      ],
    },
    {
      step: "03",
      title: "Leakage-Free Partitioning & Statistical Feature Selection",
      icon: Filter,
      badge: "Top 50 Features",
      description:
        "Strict methodological isolation: An 80/20 stratified split locked away 94,123 test sessions untouched until final evaluation. Statistical feature selection was fitted exclusively on the 80% training partition.",
      bullets: [
        "Dual-Metric Scoring: Combines linear separation (ANOVA F-statistic) with non-linear information gain (Mutual Information).",
        "Harmonic Combined Ranking: Ranks all 128 candidate features and selects the optimal Top 50 features.",
        "Eliminated Noisy Attributes: Suppressed volatile counters and high-cardinality metadata to prevent identity memorization.",
      ],
    },
    {
      step: "04",
      title: "Empirical Imbalance Strategy Discovery: Cost-Sensitive vs. SMOTE",
      icon: TrendingUp,
      badge: "Key Finding",
      description:
        "Systematic comparison of two fundamental handling strategies across 5-Fold Stratified Cross-Validation on all tree architectures: Cost-Sensitive loss penalties vs. synthetic oversampling (SMOTE + RUS).",
      bullets: [
        "Cost-Sensitive Weighting won decisively across Random Forest (+21.5% PR-AUC), XGBoost (+5.96%), and CatBoost (+8.44%).",
        "Synthetic SMOTE oversampling distorted the continuous multi-modal distribution of human enterprise actions.",
        "Preserving genuine imbalanced distributions allowed tree algorithms to learn authentic negative boundary density.",
      ],
    },
    {
      step: "05",
      title: "Out-Of-Fold (OOF) Stacking Ensemble Meta-Learner",
      icon: Cpu,
      badge: "0.9238 PR-AUC",
      description:
        "Trained base classifiers on 5-fold cross-validation to produce strictly out-of-fold probability predictions. Built an optimal soft-voting ensemble leveraging complementary algorithmic strengths.",
      bullets: [
        "Precision Anchor (XGBoost, weight = 0.70): Delivers ultra-low false alarms (0.0160% FPR, 15 false alarms).",
        "Recall Anchor (CatBoost, weight = 0.30): Symmetric oblivious trees detect subtle anomalous shifts (187 threats detected).",
        "Threshold Calibration: Optimal decision threshold T* = 0.34 tuned strictly on OOF validation (never on X_test).",
        "Final Test Evaluation: Intercepts 188 of 223 threats (84.30% Recall) with peak 0.9238 PR-AUC and 0.9987 ROC-AUC.",
      ],
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="space-y-2">
        <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-md text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
          <GitFork className="w-3.5 h-3.5" />
          <span>Leakage-Free Research Methodology</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-100 tracking-tight">
          System Architecture & Pipeline Engineering
        </h1>
        <p className="text-sm text-slate-400 max-w-3xl leading-relaxed">
          How 16.5 GB of unstructured enterprise event logs were transformed into high-accuracy insider threat detection without data leakage.
        </p>
      </section>

      {/* 5-Stage Interactive Timeline */}
      <section className="space-y-6">
        {stages.map((stage) => {
          const Icon = stage.icon;
          return (
            <div
              key={stage.step}
              className="rounded-xl bg-slate-900/80 border border-slate-800 p-6 relative overflow-hidden hover:border-slate-700 transition-all"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-slate-800/80">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center font-mono font-bold text-cyan-400">
                    {stage.step}
                  </div>
                  <div>
                    <h2 className="text-base sm:text-lg font-bold text-slate-100 flex items-center space-x-2">
                      <span>{stage.title}</span>
                    </h2>
                    <p className="text-xs text-slate-400">{stage.description}</p>
                  </div>
                </div>
                <span className="self-start sm:self-center text-xs font-semibold px-2.5 py-1 rounded-md bg-cyan-950/60 text-cyan-300 border border-cyan-500/30">
                  {stage.badge}
                </span>
              </div>

              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                {stage.bullets.map((bullet, idx) => (
                  <div key={idx} className="flex items-start space-x-2 text-slate-300">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span className="leading-relaxed">{bullet}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </section>

      {/* Feature Discovery: Cost-Sensitive vs. SMOTE Chart */}
      <section className="rounded-xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
              <span>Cross-Validation Discovery: Cost-Sensitive vs. SMOTE+RUS</span>
            </h2>
            <p className="text-xs text-slate-400">
              PR-AUC (Average Precision %) comparison on 5-Fold Stratified Cross-Validation
            </p>
          </div>
          <span className="text-xs font-medium text-emerald-400">Cost-Sensitive Wins Universally</span>
        </div>

        <div className="h-64 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={imbalanceComparison} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
              <XAxis dataKey="model" stroke="#64748B" fontSize={12} tickLine={false} />
              <YAxis stroke="#64748B" fontSize={12} domain={[40, 100]} tickLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: "8px", fontSize: "12px" }}
              />
              <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />
              <Bar dataKey="Cost-Sensitive (Raw Train)" fill="#10B981" radius={[3, 3, 0, 0]} />
              <Bar dataKey="SMOTE + RUS (Resampled)" fill="#64748B" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-slate-800 text-xs">
          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
            <span className="text-slate-400 block">Random Forest Imbalance Gain</span>
            <div className="text-base font-bold text-emerald-400 mt-1">+21.50% PR-AUC</div>
            <p className="text-[11px] text-slate-500 mt-1">Cost-Sensitive 75.39% vs. SMOTE 53.89%</p>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
            <span className="text-slate-400 block">XGBoost Imbalance Gain</span>
            <div className="text-base font-bold text-emerald-400 mt-1">+5.96% PR-AUC</div>
            <p className="text-[11px] text-slate-500 mt-1">Cost-Sensitive 89.81% vs. SMOTE 83.85%</p>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
            <span className="text-slate-400 block">CatBoost Imbalance Gain</span>
            <div className="text-base font-bold text-emerald-400 mt-1">+8.44% PR-AUC</div>
            <p className="text-[11px] text-slate-500 mt-1">Cost-Sensitive 88.55% vs. SMOTE 80.11%</p>
          </div>
        </div>
      </section>
    </div>
  );
}
