"use client";

import Link from "next/link";
import {
  ShieldAlert,
  TrendingUp,
  Cpu,
  Layers,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Database,
  BarChart2,
  Activity,
  Terminal,
  GitFork,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import data from "@/data/r4_2_metrics.json";

export default function OverviewPage() {
  const stats = data.dataset.stats;
  const models = data.models;
  const ensemble = models.Stacking_Ensemble.test_metrics_optimal_threshold;

  // Recharts 4-model comparison data
  const comparisonData = [
    {
      metric: "Precision",
      "Random Forest": Number((models.Random_Forest.test_metrics_optimal_threshold.precision * 100).toFixed(1)),
      XGBoost: Number((models.XGBoost.test_metrics_optimal_threshold.precision * 100).toFixed(1)),
      CatBoost: Number((models.CatBoost.test_metrics_optimal_threshold.precision * 100).toFixed(1)),
      "Stacking Ensemble": Number((models.Stacking_Ensemble.test_metrics_optimal_threshold.precision * 100).toFixed(1)),
    },
    {
      metric: "Recall",
      "Random Forest": Number((models.Random_Forest.test_metrics_optimal_threshold.recall * 100).toFixed(1)),
      XGBoost: Number((models.XGBoost.test_metrics_optimal_threshold.recall * 100).toFixed(1)),
      CatBoost: Number((models.CatBoost.test_metrics_optimal_threshold.recall * 100).toFixed(1)),
      "Stacking Ensemble": Number((models.Stacking_Ensemble.test_metrics_optimal_threshold.recall * 100).toFixed(1)),
    },
    {
      metric: "F1-Score",
      "Random Forest": Number((models.Random_Forest.test_metrics_optimal_threshold.f1_score * 100).toFixed(1)),
      XGBoost: Number((models.XGBoost.test_metrics_optimal_threshold.f1_score * 100).toFixed(1)),
      CatBoost: Number((models.CatBoost.test_metrics_optimal_threshold.f1_score * 100).toFixed(1)),
      "Stacking Ensemble": Number((models.Stacking_Ensemble.test_metrics_optimal_threshold.f1_score * 100).toFixed(1)),
    },
    {
      metric: "PR-AUC",
      "Random Forest": Number((models.Random_Forest.test_metrics_optimal_threshold.pr_auc * 100).toFixed(1)),
      XGBoost: Number((models.XGBoost.test_metrics_optimal_threshold.pr_auc * 100).toFixed(1)),
      CatBoost: Number((models.CatBoost.test_metrics_optimal_threshold.pr_auc * 100).toFixed(1)),
      "Stacking Ensemble": Number((models.Stacking_Ensemble.test_metrics_optimal_threshold.pr_auc * 100).toFixed(1)),
    },
  ];

  return (
    <div className="space-y-8">
      {/* Hero Header */}
      <section className="relative overflow-hidden rounded-2xl bg-gradient-to-b from-slate-900 via-[#0B132B]/80 to-[#090D16] border border-slate-800 p-6 md:p-8">
        <div className="absolute top-0 right-0 -mt-8 -mr-8 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 -mb-12 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-4xl space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-md text-xs font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
              <Database className="w-3.5 h-3.5" />
              <span>CMU CERT r4.2 Benchmark</span>
            </span>
            <span className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-md text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              <Activity className="w-3.5 h-3.5" />
              <span>17 Months Telemetry</span>
            </span>
            <span className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-md text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
              <Cpu className="w-3.5 h-3.5" />
              <span>Stacking Meta-Learner</span>
            </span>
          </div>

          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-slate-100 tracking-tight leading-tight">
            An Ensemble Machine Learning Approach to Insider Risk Classification
          </h1>
          <p className="text-sm sm:text-base text-slate-400 leading-relaxed max-w-3xl">
            Detecting malicious enterprise activity on sparse audit logs (0.237% positive class).
            Combines high-speed sessionization across 16.5 GB of heterogeneous logs with leakage-free feature selection
            and a calibrated soft-voting stacking ensemble of Random Forest, XGBoost, and CatBoost.
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-3">
            <Link
              href="/simulator"
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-sm font-semibold transition-colors shadow-lg shadow-cyan-500/20"
            >
              <Terminal className="w-4 h-4" />
              <span>Live Threat Simulator</span>
              <ArrowRight className="w-4 h-4" />
            </Link>

            <Link
              href="/results"
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium border border-slate-700 transition-colors"
            >
              <BarChart2 className="w-4 h-4 text-cyan-400" />
              <span>View Leaderboard</span>
            </Link>

            <Link
              href="/methodology"
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium border border-slate-700 transition-colors"
            >
              <Layers className="w-4 h-4 text-emerald-400" />
              <span>Pipeline Architecture</span>
            </Link>
          </div>
        </div>
      </section>

      {/* KPI Cards Grid */}
      <section className="grid grid-cols-2 lg:grid-cols-6 gap-4">
        {/* Total Sessions */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all col-span-1">
          <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Sessions</div>
          <div className="mt-2 text-xl md:text-2xl font-bold text-slate-100">{stats.total_sessions.toLocaleString()}</div>
          <div className="mt-1 text-[11px] text-slate-500">1,000 users · 17 months</div>
        </div>

        {/* Threat Prevalence */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all col-span-1">
          <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">Threat Rate</div>
          <div className="mt-2 text-xl md:text-2xl font-bold text-amber-400">{stats.threat_percentage}%</div>
          <div className="mt-1 text-[11px] text-slate-500">1,114 threats ({stats.imbalance_ratio})</div>
        </div>

        {/* Best PR-AUC */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-cyan-500/30 hover:border-cyan-500/50 transition-all col-span-1 glow-cyan">
          <div className="text-xs font-medium text-cyan-400 uppercase tracking-wider flex items-center justify-between">
            <span>Peak PR-AUC</span>
            <TrendingUp className="w-3.5 h-3.5" />
          </div>
          <div className="mt-2 text-xl md:text-2xl font-extrabold text-cyan-300">
            {ensemble.pr_auc.toFixed(4)}
          </div>
          <div className="mt-1 text-[11px] text-cyan-400/80">Stacking Ensemble</div>
        </div>

        {/* Best ROC-AUC */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all col-span-1">
          <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">Peak ROC-AUC</div>
          <div className="mt-2 text-xl md:text-2xl font-bold text-slate-100">
            {ensemble.roc_auc.toFixed(4)}
          </div>
          <div className="mt-1 text-[11px] text-slate-500">True vs. False Positive Rate</div>
        </div>

        {/* Threats Intercepted */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-emerald-500/30 hover:border-emerald-500/50 transition-all col-span-1 glow-emerald">
          <div className="text-xs font-medium text-emerald-400 uppercase tracking-wider flex items-center justify-between">
            <span>Threats Caught</span>
            <CheckCircle2 className="w-3.5 h-3.5" />
          </div>
          <div className="mt-2 text-xl md:text-2xl font-extrabold text-emerald-300">
            {ensemble.confusion_matrix.tp} / 223
          </div>
          <div className="mt-1 text-[11px] text-emerald-400/80">
            {(ensemble.recall * 100).toFixed(1)}% Test Recall
          </div>
        </div>

        {/* False Positive Rate */}
        <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all col-span-1">
          <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">False Alarms</div>
          <div className="mt-2 text-xl md:text-2xl font-bold text-slate-100">
            {ensemble.confusion_matrix.fp} <span className="text-xs font-normal text-slate-400">/ 93.9k</span>
          </div>
          <div className="mt-1 text-[11px] text-slate-500">
            {(ensemble.false_positive_rate * 100).toFixed(4)}% FPR
          </div>
        </div>
      </section>

      {/* Main Charts & Highlights */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 4-Model Comparison Bar Chart */}
        <div className="lg:col-span-2 rounded-xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                <span>4-Model Benchmark Performance on Untouched Test Set</span>
              </h2>
              <p className="text-xs text-slate-400">
                Evaluation on X_test (94,123 sessions, 223 threat cases) at optimal OOF threshold T*
              </p>
            </div>
            <span className="text-xs text-slate-500 font-mono">Metric Scale: %</span>
          </div>

          <div className="h-72 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                <XAxis dataKey="metric" stroke="#64748B" fontSize={12} tickLine={false} />
                <YAxis stroke="#64748B" fontSize={12} domain={[60, 100]} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: "8px", fontSize: "12px" }}
                />
                <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "8px" }} />
                <Bar dataKey="Random Forest" fill="#3B82F6" radius={[3, 3, 0, 0]} />
                <Bar dataKey="XGBoost" fill="#F97316" radius={[3, 3, 0, 0]} />
                <Bar dataKey="CatBoost" fill="#10B981" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Stacking Ensemble" fill="#06B6D4" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="pt-2 border-t border-slate-800 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400 block text-[10px]">RF F1-Score</span>
              <span className="font-semibold text-blue-400">74.57%</span>
            </div>
            <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400 block text-[10px]">XGB Precision</span>
              <span className="font-semibold text-orange-400">92.54%</span>
            </div>
            <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
              <span className="text-slate-400 block text-[10px]">CatBoost Recall</span>
              <span className="font-semibold text-emerald-400">83.86%</span>
            </div>
            <div className="p-2 rounded bg-slate-950/60 border border-cyan-500/20">
              <span className="text-cyan-400 block text-[10px]">Ensemble PR-AUC</span>
              <span className="font-bold text-cyan-300">0.9238</span>
            </div>
          </div>
        </div>

        {/* Model Synergy & Findings Box */}
        <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-6 flex flex-col justify-between space-y-4">
          <div>
            <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
              <Layers className="w-4 h-4 text-emerald-400" />
              <span>Core Empirical Discoveries</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Key takeaways validating the ensemble architecture
            </p>

            <div className="mt-4 space-y-3 text-xs">
              <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 space-y-1">
                <div className="font-semibold text-emerald-400 flex items-center space-x-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Cost-Sensitive Weighting Wins</span>
                </div>
                <p className="text-slate-400 leading-relaxed text-[11px]">
                  Directly weighting loss functions beat SMOTE + Undersampling by <strong className="text-slate-200">+21.5%</strong> on RF and <strong className="text-slate-200">+8.44%</strong> on CatBoost. Synthetic oversampling distorted real behavioral feature geometry.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 space-y-1">
                <div className="font-semibold text-cyan-400 flex items-center space-x-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Complementary Tree Architectures</span>
                </div>
                <p className="text-slate-400 leading-relaxed text-[11px]">
                  XGBoost acts as the <strong className="text-slate-200">Precision Anchor</strong> (92.54% Prec, 15 FPs) while CatBoost acts as the <strong className="text-slate-200">Recall Anchor</strong> (83.86% Rec, 187 threats).
                </p>
              </div>

              <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 space-y-1">
                <div className="font-semibold text-indigo-400 flex items-center space-x-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Optimal Soft-Voting Blend</span>
                </div>
                <p className="text-slate-400 leading-relaxed text-[11px]">
                  Stacking simplex weights: <strong className="text-slate-200">0.70 XGB + 0.30 CatBoost</strong> at optimal threshold <strong className="text-slate-200">T* = 0.34</strong> intercepts 188 threats with peak 0.9238 PR-AUC.
                </p>
              </div>
            </div>
          </div>

          <Link
            href="/results"
            className="w-full text-center py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-cyan-400 border border-slate-700 transition-colors"
          >
            Explore Detailed Curves & Confusion Matrices →
          </Link>
        </div>
      </section>

      {/* Quick Navigation Cards */}
      <section className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Link
          href="/methodology"
          className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 hover:bg-slate-800/40 transition-all group"
        >
          <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/20 flex items-center justify-center text-cyan-400 mb-3 group-hover:scale-110 transition-transform">
            <GitFork className="w-4 h-4" />
          </div>
          <div className="font-semibold text-sm text-slate-100 group-hover:text-cyan-400 transition-colors">
            5-Stage Pipeline
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Data ingestion, 128 feature sessionization, ANOVA+MI selection, and OOF stacking.
          </div>
        </Link>

        <Link
          href="/results"
          className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 hover:bg-slate-800/40 transition-all group"
        >
          <div className="w-8 h-8 rounded-lg bg-orange-950/80 border border-orange-500/20 flex items-center justify-center text-orange-400 mb-3 group-hover:scale-110 transition-transform">
            <Cpu className="w-4 h-4" />
          </div>
          <div className="font-semibold text-sm text-slate-100 group-hover:text-orange-400 transition-colors">
            Master Leaderboard
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Sortable metrics table, threshold toggles (0.50 vs T*), and Cost-Sensitive vs. SMOTE cards.
          </div>
        </Link>

        <Link
          href="/features"
          className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 hover:bg-slate-800/40 transition-all group"
        >
          <div className="w-8 h-8 rounded-lg bg-emerald-950/80 border border-emerald-500/20 flex items-center justify-center text-emerald-400 mb-3 group-hover:scale-110 transition-transform">
            <Sliders className="w-4 h-4" />
          </div>
          <div className="font-semibold text-sm text-slate-100 group-hover:text-emerald-400 transition-colors">
            Top 50 Threat Signals
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Searchable ANOVA F and Mutual Information rankings across USB, leak URLs, and psychology.
          </div>
        </Link>

        <Link
          href="/simulator"
          className="p-4 rounded-xl bg-slate-900/60 border border-cyan-500/30 hover:border-cyan-500/60 hover:bg-slate-800/40 transition-all group"
        >
          <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-500/20 flex items-center justify-center text-cyan-400 mb-3 group-hover:scale-110 transition-transform">
            <ShieldAlert className="w-4 h-4" />
          </div>
          <div className="font-semibold text-sm text-slate-100 group-hover:text-cyan-400 transition-colors">
            Live Risk Simulator
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Interactive sliders and one-click presets simulating real-time threat probability scoring.
          </div>
        </Link>
      </section>
    </div>
  );
}
