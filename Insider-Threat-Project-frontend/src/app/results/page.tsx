"use client";

import { useState } from "react";
import Image from "next/image";
import {
  Cpu,
  Sliders,
  TrendingUp,
  Maximize2,
  X,
  CheckCircle2,
  AlertTriangle,
  Award,
  Layers,
  BarChart2,
  Shield,
  Target,
  Zap,
  Info,
  ChevronRight,
  Database
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

export default function ResultsPage() {
  const { data, dataset, plotPrefix, benchmarks } = useDataset();
  const [thresholdMode, setThresholdMode] = useState<"optimal" | "default">("optimal");
  const [selectedPlot, setSelectedPlot] = useState<string | null>(null);
  const [activePlotTab, setActivePlotTab] = useState<"all" | "ensemble" | "xgb" | "cb" | "rf">("all");
  const [selectedModelCM, setSelectedModelCM] = useState<string>("Stacking Ensemble");

  const models = data.models;
  const stats = data.dataset.stats;
  const isCrossRelease = dataset === "cross_release";

  // Helper to safely extract metrics for active threshold mode
  const getModelMetrics = (modelObj: any) => {
    const opt = modelObj.test_metrics_optimal_threshold;
    const def = modelObj.test_metrics_default_threshold;
    const active = thresholdMode === "optimal" ? opt : def;
    return {
      data: active,
      pr_auc: opt.pr_auc,
      roc_auc: opt.roc_auc,
      cm: active.confusion_matrix || opt.confusion_matrix,
      fpr: active.false_positive_rate !== undefined ? active.false_positive_rate : opt.false_positive_rate,
    };
  };

  const rfMetrics = getModelMetrics(models.Random_Forest);
  const xgbMetrics = getModelMetrics(models.XGBoost);
  const cbMetrics = getModelMetrics(models.CatBoost);
  const ensembleMetrics = getModelMetrics(models.Stacking_Ensemble);

  // Data for single-dataset model rows
  const rows = [
    {
      name: "Random Forest",
      key: "Random_Forest",
      role: "Bagging Variance Anchor",
      strategy: dataset === "r6.2" ? "Cost-Sensitive (balanced_subsample)" : "Cost-Sensitive (balanced_subsample)",
      data: rfMetrics.data,
      pr_auc: rfMetrics.pr_auc,
      roc_auc: rfMetrics.roc_auc,
      cm: rfMetrics.cm,
      fpr: rfMetrics.fpr,
      isWinner: false,
      badge: dataset === "r6.2" ? "Lowest FPs (4)" : "High Specificity",
    },
    {
      name: "XGBoost",
      key: "XGBoost",
      role: "Precision Anchor (Lowest False Alarms)",
      strategy: dataset === "r4.2" ? "Cost-Sensitive (scale_pos_weight=1.0)" : (dataset === "r5.2" ? "Cost-Sensitive (scale_pos_weight=23.3)" : "Cost-Sensitive (scale_pos_weight=25.0)"),
      data: xgbMetrics.data,
      pr_auc: xgbMetrics.pr_auc,
      roc_auc: xgbMetrics.roc_auc,
      cm: xgbMetrics.cm,
      fpr: xgbMetrics.fpr,
      isWinner: false,
      badge: dataset === "r6.2" ? "ROC-AUC 0.979" : "Top Tree Booster",
    },
    {
      name: "CatBoost",
      key: "CatBoost",
      role: "Recall Anchor (Highest Single Threat Capture)",
      strategy: dataset === "r6.2" ? "SMOTE+RUS (Fold-level balanced)" : "Cost-Sensitive (SqrtBalanced)",
      data: cbMetrics.data,
      pr_auc: cbMetrics.pr_auc,
      roc_auc: cbMetrics.roc_auc,
      cm: cbMetrics.cm,
      fpr: cbMetrics.fpr,
      isWinner: false,
      badge: dataset === "r6.2" ? "Top Recall (54.5%)" : "High Sensitivity",
    },
    {
      name: "Stacking Ensemble",
      key: "Stacking_Ensemble",
      role: "Multi-Model Meta-Learner (Overall Champion)",
      strategy: dataset === "r4.2" ? "Convex Soft-Voting (0.70 XGB + 0.30 CB)" : "Calibrated Logistic Regression (OOF Meta-Features)",
      data: ensembleMetrics.data,
      pr_auc: ensembleMetrics.pr_auc,
      roc_auc: ensembleMetrics.roc_auc,
      cm: ensembleMetrics.cm,
      fpr: ensembleMetrics.fpr,
      isWinner: true,
      badge: "Overall Winner",
    },
  ];

  // Recharts Bar chart data
  const chartData = rows.map((r) => ({
    name: r.name === "Stacking Ensemble" ? "Ensemble" : r.name,
    Precision: Number((r.data.precision * 100).toFixed(1)),
    Recall: Number((r.data.recall * 100).toFixed(1)),
    F1: Number((r.data.f1_score * 100).toFixed(1)),
    PRAUC: Number((r.pr_auc * 100).toFixed(1)),
  }));

  // Selected Model Confusion Matrix
  const activeCMRow = rows.find((r) => r.name === selectedModelCM) || rows[3];

  const gallery = [
    {
      title: "All Models PR & ROC Comparison",
      category: "all",
      path: `${plotPrefix}/ensemble/all_models_pr_roc_comparison.png`,
      caption: `Precision-Recall and ROC comparative evaluation across all architectures on CERT ${dataset}.`,
    },
    {
      title: "Stacking Ensemble Confusion Matrix",
      category: "ensemble",
      path: `${plotPrefix}/ensemble/ensemble_confusion_matrix.png`,
      caption: `Evaluated on ${stats.test_set_sessions.toLocaleString()} untouched test sessions (${stats.test_threats} ground-truth threats).`,
    },
    {
      title: "Meta-Learner Optimal Model Weights",
      category: "ensemble",
      path: `${plotPrefix}/ensemble/meta_learner_weights.png`,
      caption: "Learned stacking meta-learner feature weights and algorithmic coefficients.",
    },
    {
      title: "XGBoost Feature Importance",
      category: "xgb",
      path: `${plotPrefix}/xgb/xgb_feature_importance.png`,
      caption: "Top behavioral signals: Removable media, data leak URLs, and document word count.",
    },
    {
      title: "XGBoost PR & ROC Curves",
      category: "xgb",
      path: `${plotPrefix}/xgb/xgb_pr_roc_curves.png`,
      caption: `Histogram gradient tree boosting curves on CERT ${dataset}.`,
    },
    {
      title: "XGBoost Confusion Matrix",
      category: "xgb",
      path: `${plotPrefix}/xgb/xgb_confusion_matrix.png`,
      caption: `Evaluated on ${stats.test_set_sessions.toLocaleString()} test sessions.`,
    },
    {
      title: "CatBoost Feature Importance",
      category: "cb",
      path: `${plotPrefix}/cb/cb_feature_importance.png`,
      caption: "Top predictive attributes driving CatBoost oblivious tree splits.",
    },
    {
      title: "CatBoost PR & ROC Curves",
      category: "cb",
      path: `${plotPrefix}/cb/cb_pr_roc_curves.png`,
      caption: `Symmetric tree gradient boosting curves on CERT ${dataset}.`,
    },
    {
      title: "CatBoost Confusion Matrix",
      category: "cb",
      path: `${plotPrefix}/cb/cb_confusion_matrix.png`,
      caption: `Evaluated on ${stats.test_set_sessions.toLocaleString()} test sessions.`,
    },
    {
      title: "Random Forest Feature Importance",
      category: "rf",
      path: `${plotPrefix}/rf/rf_feature_importance.png`,
      caption: "Mean decrease in impurity across bagging trees.",
    },
    {
      title: "Random Forest PR & ROC Curves",
      category: "rf",
      path: `${plotPrefix}/rf/rf_pr_roc_curves.png`,
      caption: `Subsample-balanced bagging trees on CERT ${dataset}.`,
    },
    {
      title: "Random Forest Confusion Matrix",
      category: "rf",
      path: `${plotPrefix}/rf/rf_confusion_matrix.png`,
      caption: `Evaluated on ${stats.test_set_sessions.toLocaleString()} test sessions.`,
    },
  ];

  const filteredGallery = gallery.filter((item) => {
    if (activePlotTab === "all") return true;
    return item.category === activePlotTab;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-slate-900 tracking-tight">
              Model Performance Benchmarks & CV Analysis
            </h1>
            <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
              {dataset === "cross_release" ? "Cross-Release Tri-Evaluation" : `CERT ${dataset}`}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Rigorous evaluation on strictly untouched test partition (
            <span className="font-mono font-medium text-slate-700">
              {stats.test_set_sessions.toLocaleString()}
            </span>{" "}
            sessions,{" "}
            <span className="font-mono font-medium text-slate-700">
              {stats.test_threats}
            </span>{" "}
            ground-truth threats) in CMU CERT {dataset}.
          </p>
        </div>

        {/* Threshold Toggle */}
        <div className="flex items-center space-x-2 bg-white p-1 rounded-lg border border-slate-200 shadow-2xs">
          <span className="text-[11px] font-semibold text-slate-500 px-2">Decision Cutoff:</span>
          <button
            onClick={() => setThresholdMode("optimal")}
            className={`px-3 py-1 rounded text-xs font-semibold transition-colors cursor-pointer ${
              thresholdMode === "optimal"
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Optimal (T*)
          </button>
          <button
            onClick={() => setThresholdMode("default")}
            className={`px-3 py-1 rounded text-xs font-semibold transition-colors cursor-pointer ${
              thresholdMode === "default"
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Default (0.50)
          </button>
        </div>
      </div>

      {/* Smart Model Recommendation Cards (Which Model is Best & Why?) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Card 1: Overall Champion */}
        <div className="enterprise-card p-4 border-l-4 border-l-blue-600 bg-gradient-to-br from-white to-blue-50/20 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-blue-700 bg-blue-100/70 px-2 py-0.5 rounded">
                Overall Champion
              </span>
              <Award className="w-4 h-4 text-blue-600" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 mt-2">
              Stacking Ensemble Meta-Learner
            </h3>
            <p className="text-xs text-slate-600 mt-1 leading-relaxed">
              Consistently achieves the highest PR-AUC across all releases (
              <span className="font-mono font-bold text-slate-900">
                {dataset === "r6.2" ? "0.3322" : dataset === "r5.2" ? "0.8683" : "0.9238"}
              </span>
              ). Uses out-of-fold calibrated logistic regression to prune 97% of CatBoost false alarms while retaining critical recall.
            </p>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px]">
            <span className="text-slate-500 font-medium">SOC Role:</span>
            <span className="font-semibold text-blue-700">Primary Autonomous Defense</span>
          </div>
        </div>

        {/* Card 2: Precision Champion */}
        <div className="enterprise-card p-4 border-l-4 border-l-emerald-600 bg-gradient-to-br from-white to-emerald-50/20 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-700 bg-emerald-100/70 px-2 py-0.5 rounded">
                Precision & Low-Fatigue
              </span>
              <Target className="w-4 h-4 text-emerald-600" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 mt-2">
              {dataset === "r6.2" ? "Random Forest" : "XGBoost"}
            </h3>
            <p className="text-xs text-slate-600 mt-1 leading-relaxed">
              {dataset === "r6.2"
                ? "Generated only 4 false alarms across 389,795 untouched benign sessions (FPR = 0.00103%), achieving 50.0% precision under 20,515:1 extreme sparsity."
                : "Delivers maximum precision (82.5% on r5.2, 91.2% on r4.2) with near-zero false alarms for resource-constrained analyst teams."}
            </p>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px]">
            <span className="text-slate-500 font-medium">SOC Role:</span>
            <span className="font-semibold text-emerald-700">Zero-Fatigue Auto-Escalation</span>
          </div>
        </div>

        {/* Card 3: Sensitivity Champion */}
        <div className="enterprise-card p-4 border-l-4 border-l-amber-600 bg-gradient-to-br from-white to-amber-50/20 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-amber-700 bg-amber-100/70 px-2 py-0.5 rounded">
                High-Sensitivity Hunter
              </span>
              <Zap className="w-4 h-4 text-amber-600" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 mt-2">
              CatBoost
            </h3>
            <p className="text-xs text-slate-600 mt-1 leading-relaxed">
              Achieved highest individual threat capture (
              <span className="font-mono font-bold text-slate-900">
                {dataset === "r6.2" ? "54.55% recall (12/22 threats)" : "90.0% scenario recall"}
              </span>
              ). On r6.2, CatBoost caught 9/10 USB sabotage threats and was the single architecture capable of identifying the Scenario 3 keylogger attack.
            </p>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px]">
            <span className="text-slate-500 font-medium">SOC Role:</span>
            <span className="font-semibold text-amber-700">Proactive Threat Hunting Sweep</span>
          </div>
        </div>
      </div>

      {/* Cross-Release Comparison Table (Rendered when cross_release is selected or as an overview) */}
      {isCrossRelease && (
        <div className="enterprise-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold text-slate-900">
                Tri-Release Cross-Benchmark Comparison (CERT r4.2 vs. r5.2 vs. r6.2)
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Evaluation across three orders of corporate employee scale and class imbalance severity.
              </p>
            </div>
            <span className="text-[11px] font-mono font-semibold px-2 py-0.5 bg-slate-100 rounded text-slate-700">
              3,424,144 Total Sessions
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left soc-table">
              <thead>
                <tr>
                  <th>Benchmark Dataset</th>
                  <th className="text-right">Employees</th>
                  <th className="text-right">Log Volume</th>
                  <th className="text-right">Sessions</th>
                  <th className="text-right">Threats</th>
                  <th className="text-right">Imbalance</th>
                  <th className="text-right">Test Threats</th>
                  <th className="text-right">Peak RAM</th>
                  <th className="text-right">Best PR-AUC</th>
                  <th className="text-right">Best ROC-AUC</th>
                  <th className="text-right">Ensemble F1</th>
                </tr>
              </thead>
              <tbody>
                {benchmarks.cross_release_summary.map((rel) => (
                  <tr key={rel.release} className="hover:bg-slate-50">
                    <td>
                      <div className="font-bold text-slate-900">{rel.name}</div>
                      <div className="text-[11px] text-slate-500">{rel.description}</div>
                    </td>
                    <td className="text-right font-mono text-xs tabular-nums text-slate-800">
                      {rel.employees.toLocaleString()}
                    </td>
                    <td className="text-right font-mono text-xs tabular-nums text-slate-700">
                      {rel.raw_log_gb} GB
                    </td>
                    <td className="text-right font-mono text-xs tabular-nums text-slate-900 font-medium">
                      {rel.total_sessions.toLocaleString()}
                    </td>
                    <td className="text-right font-mono text-xs tabular-nums text-rose-700 font-bold">
                      {rel.threat_sessions}
                    </td>
                    <td className="text-right font-mono text-xs tabular-nums text-slate-800 font-semibold">
                      {rel.threat_ratio}
                    </td>
                    <td className="text-right font-mono text-xs tabular-nums text-slate-700">
                      {rel.test_threats}
                    </td>
                    <td className="text-right font-mono text-xs tabular-nums text-emerald-700 font-bold">
                      {rel.peak_ram}
                    </td>
                    <td className="text-right font-mono text-xs tabular-nums text-blue-700 font-bold">
                      {rel.best_pr_auc.toFixed(4)}
                    </td>
                    <td className="text-right font-mono text-xs tabular-nums text-slate-800">
                      {rel.best_roc_auc.toFixed(4)}
                    </td>
                    <td className="text-right font-mono text-xs tabular-nums text-slate-900 font-bold">
                      {rel.best_f1.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Main Leaderboard Table */}
      <div className="enterprise-card overflow-hidden">
        <div className="p-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-bold text-slate-900">
              Model Architecture Comparison Matrix — CERT {dataset}
            </h2>
            <p className="text-xs text-slate-500">
              Metrics computed at {thresholdMode === "optimal" ? "optimal cutoff (T* maximizing F1)" : "default standard cutoff (T = 0.50)"}.
            </p>
          </div>
          <div className="text-xs font-mono text-slate-500">
            Split: 80% Train ({stats.train_set_sessions.toLocaleString()}) / 20% Test ({stats.test_set_sessions.toLocaleString()})
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left soc-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Training Strategy</th>
                <th className="text-right">Threshold (T)</th>
                <th className="text-right">PR-AUC</th>
                <th className="text-right">ROC-AUC</th>
                <th className="text-right">Precision</th>
                <th className="text-right">Recall</th>
                <th className="text-right">F1-Score</th>
                <th className="text-right">Threats Caught</th>
                <th className="text-right">False Alarms</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.name}
                  className={row.isWinner ? "bg-blue-50/40 font-medium" : "hover:bg-slate-50/60"}
                >
                  <td>
                    <div className="flex items-center space-x-2">
                      <span className="font-bold text-slate-900">{row.name}</span>
                      <span
                        className={`text-[9px] font-bold uppercase px-1.5 py-0.2 rounded border ${
                          row.isWinner
                            ? "bg-blue-100 text-blue-800 border-blue-200"
                            : "bg-slate-100 text-slate-700 border-slate-200"
                        }`}
                      >
                        {row.badge}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-500">{row.role}</div>
                  </td>

                  <td className="text-xs text-slate-700">
                    {row.strategy}
                  </td>

                  <td className="text-right font-mono text-xs text-slate-800 font-bold tabular-nums">
                    {row.data.threshold}
                  </td>

                  <td className="text-right font-mono text-xs font-bold text-blue-800 tabular-nums">
                    {row.pr_auc.toFixed(4)}
                  </td>

                  <td className="text-right font-mono text-xs text-slate-700 tabular-nums">
                    {row.roc_auc.toFixed(4)}
                  </td>

                  <td className="text-right font-mono text-xs text-slate-900 tabular-nums">
                    {(row.data.precision * 100).toFixed(1)}%
                  </td>

                  <td className="text-right font-mono text-xs text-slate-900 tabular-nums">
                    {(row.data.recall * 100).toFixed(1)}%
                  </td>

                  <td className="text-right font-mono text-xs font-bold text-slate-900 tabular-nums">
                    {(row.data.f1_score * 100).toFixed(1)}%
                  </td>

                  <td className="text-right font-mono text-xs font-bold text-emerald-700 tabular-nums">
                    {row.cm.tp} / {stats.test_threats}
                  </td>

                  <td className="text-right font-mono text-xs text-slate-600 tabular-nums">
                    {row.cm.fp}{" "}
                    <span className="text-[10px] text-slate-400">
                      ({(row.fpr * 100).toFixed(4)}%)
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Two-Column Analytics: Recharts Metric Comparison & Confusion Matrix Visualizer */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Metric Bar Chart */}
        <div className="enterprise-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold text-slate-900">
                Metric Comparison Across Architectures
              </h2>
              <p className="text-xs text-slate-500">
                Precision, Recall, F1-Score, and PR-AUC (%) on test split
              </p>
            </div>
            <span className="text-xs font-mono font-semibold text-slate-600">
              {thresholdMode === "optimal" ? "Cutoff: T*" : "Cutoff: 0.50"}
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(value: any) => [`${value}%`]}
                  contentStyle={{
                    backgroundColor: "#FFFFFF",
                    borderRadius: "6px",
                    border: "1px solid #E2E8F0",
                    fontSize: "12px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
                <Bar dataKey="Precision" fill="#1E40AF" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Recall" fill="#059669" radius={[3, 3, 0, 0]} />
                <Bar dataKey="F1" fill="#D97706" radius={[3, 3, 0, 0]} />
                <Bar dataKey="PRAUC" fill="#475569" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix Visualizer */}
        <div className="enterprise-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold text-slate-900">
                Interactive Confusion Matrix Visualizer
              </h2>
              <p className="text-xs text-slate-500">
                {stats.test_set_sessions.toLocaleString()} total test sessions evaluated
              </p>
            </div>

            {/* Model Selector Pill */}
            <select
              value={selectedModelCM}
              onChange={(e) => setSelectedModelCM(e.target.value)}
              className="text-xs font-semibold px-2.5 py-1 rounded bg-slate-50 border border-slate-200 text-slate-800"
            >
              {rows.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.name}
                </option>
              ))}
            </select>
          </div>

          {/* Matrix Grid */}
          <div className="grid grid-cols-2 gap-3 pt-2">
            {/* True Negative */}
            <div className="p-3.5 rounded-lg border border-slate-200 bg-slate-50/60">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                True Negatives (TN)
              </div>
              <div className="text-xl font-bold font-mono text-slate-900 mt-1 tabular-nums">
                {activeCMRow.cm.tn.toLocaleString()}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">
                Benign sessions correctly cleared
              </div>
            </div>

            {/* False Positive */}
            <div className="p-3.5 rounded-lg border border-rose-200 bg-rose-50/30">
              <div className="text-[10px] font-bold uppercase tracking-wider text-rose-600">
                False Positives (FP)
              </div>
              <div className="text-xl font-bold font-mono text-rose-700 mt-1 tabular-nums">
                {activeCMRow.cm.fp.toLocaleString()}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">
                False alarms (FPR: {(activeCMRow.fpr * 100).toFixed(4)}%)
              </div>
            </div>

            {/* False Negative */}
            <div className="p-3.5 rounded-lg border border-amber-200 bg-amber-50/30">
              <div className="text-[10px] font-bold uppercase tracking-wider text-amber-700">
                False Negatives (FN)
              </div>
              <div className="text-xl font-bold font-mono text-amber-800 mt-1 tabular-nums">
                {activeCMRow.cm.fn.toLocaleString()}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">
                Missed insider threat sessions
              </div>
            </div>

            {/* True Positive */}
            <div className="p-3.5 rounded-lg border border-emerald-200 bg-emerald-50/40">
              <div className="text-[10px] font-bold uppercase tracking-wider text-emerald-700">
                True Positives (TP)
              </div>
              <div className="text-xl font-bold font-mono text-emerald-800 mt-1 tabular-nums">
                {activeCMRow.cm.tp.toLocaleString()}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">
                Threat sessions intercepted (Recall: {(activeCMRow.data.recall * 100).toFixed(1)}%)
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Visual Artifacts Gallery */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-bold text-slate-900">
              High-Resolution Evaluation Artifacts (300 DPI)
            </h2>
            <p className="text-[11px] text-slate-500">
              Verified PNG plots generated directly by evaluation scripts for CERT {dataset}. Click to inspect in high resolution.
            </p>
          </div>

          {/* Gallery Category Tabs */}
          <div className="flex items-center space-x-1 bg-white p-1 rounded-md border border-slate-200">
            {[
              { id: "all", label: "All" },
              { id: "ensemble", label: "Ensemble" },
              { id: "xgb", label: "XGBoost" },
              { id: "cb", label: "CatBoost" },
              { id: "rf", label: "Random Forest" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActivePlotTab(tab.id as any)}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors cursor-pointer ${
                  activePlotTab === tab.id
                    ? "bg-slate-900 text-white font-semibold"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Gallery Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredGallery.map((item, idx) => (
            <div
              key={idx}
              onClick={() => setSelectedPlot(item.path)}
              className="enterprise-card overflow-hidden hover:border-slate-300 transition-all cursor-pointer group flex flex-col"
            >
              <div className="relative h-56 bg-slate-100 border-b border-slate-200 flex items-center justify-center overflow-hidden">
                <Image
                  src={item.path}
                  alt={item.title}
                  fill
                  className="object-contain p-2 group-hover:scale-[1.02] transition-transform duration-200"
                  sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                />
                <div className="absolute top-2 right-2 p-1.5 rounded bg-white/90 border border-slate-200 shadow-xs opacity-0 group-hover:opacity-100 transition-opacity">
                  <Maximize2 className="w-3.5 h-3.5 text-slate-700" />
                </div>
              </div>

              <div className="p-3.5 flex-1 flex flex-col justify-between">
                <div>
                  <h3 className="text-xs font-bold text-slate-900">{item.title}</h3>
                  <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
                    {item.caption}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Lightbox Modal */}
      {selectedPlot && (
        <div
          onClick={() => setSelectedPlot(null)}
          className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="relative bg-white rounded-xl shadow-2xl border border-slate-200 max-w-5xl w-full p-4 flex flex-col"
          >
            <div className="flex items-center justify-between pb-3 border-b border-slate-200">
              <span className="font-mono text-xs text-slate-500">{selectedPlot}</span>
              <button
                onClick={() => setSelectedPlot(null)}
                className="p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="relative h-[75vh] w-full my-2 bg-slate-50 rounded-lg flex items-center justify-center">
              <Image
                src={selectedPlot}
                alt="Full Resolution Plot"
                fill
                className="object-contain p-4"
                sizes="100vw"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
