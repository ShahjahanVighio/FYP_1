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
  BarChart2
} from "lucide-react";
import { useDataset } from "@/context/DatasetContext";

export default function ResultsPage() {
  const { data, dataset, plotPrefix } = useDataset();
  const [thresholdMode, setThresholdMode] = useState<"optimal" | "default">("optimal");
  const [selectedPlot, setSelectedPlot] = useState<string | null>(null);
  const [activePlotTab, setActivePlotTab] = useState<"all" | "ensemble" | "xgb" | "cb" | "rf">("all");

  const models = data.models;
  const stats = data.dataset.stats;

  const rows = [
    {
      name: "Random Forest",
      role: "Bagging Variance Anchor",
      strategy: "Cost-Sensitive (balanced_subsample)",
      data: thresholdMode === "optimal" ? models.Random_Forest.test_metrics_optimal_threshold : models.Random_Forest.test_metrics_default_threshold,
      pr_auc: models.Random_Forest.test_metrics_optimal_threshold.pr_auc,
      roc_auc: models.Random_Forest.test_metrics_optimal_threshold.roc_auc,
      cm: thresholdMode === "optimal" ? models.Random_Forest.test_metrics_optimal_threshold.confusion_matrix : models.Random_Forest.test_metrics_default_threshold.confusion_matrix,
      fpr: thresholdMode === "optimal" ? models.Random_Forest.test_metrics_optimal_threshold.false_positive_rate : models.Random_Forest.test_metrics_default_threshold.false_positive_rate,
      isWinner: false,
    },
    {
      name: "XGBoost",
      role: "Precision Anchor (Lowest False Alarms)",
      strategy: dataset === "r4.2" ? "Cost-Sensitive (scale_pos_weight=1.0)" : "Cost-Sensitive (scale_pos_weight=23.3)",
      data: thresholdMode === "optimal" ? models.XGBoost.test_metrics_optimal_threshold : models.XGBoost.test_metrics_default_threshold,
      pr_auc: models.XGBoost.test_metrics_optimal_threshold.pr_auc,
      roc_auc: models.XGBoost.test_metrics_optimal_threshold.roc_auc,
      cm: thresholdMode === "optimal" ? models.XGBoost.test_metrics_optimal_threshold.confusion_matrix : models.XGBoost.test_metrics_default_threshold.confusion_matrix,
      fpr: thresholdMode === "optimal" ? models.XGBoost.test_metrics_optimal_threshold.false_positive_rate : models.XGBoost.test_metrics_default_threshold.false_positive_rate,
      isWinner: false,
    },
    {
      name: "CatBoost",
      role: "Recall Anchor (Highest Single Threat Capture)",
      strategy: "Cost-Sensitive (SqrtBalanced)",
      data: thresholdMode === "optimal" ? models.CatBoost.test_metrics_optimal_threshold : models.CatBoost.test_metrics_default_threshold,
      pr_auc: models.CatBoost.test_metrics_optimal_threshold.pr_auc,
      roc_auc: models.CatBoost.test_metrics_optimal_threshold.roc_auc,
      cm: thresholdMode === "optimal" ? models.CatBoost.test_metrics_optimal_threshold.confusion_matrix : models.CatBoost.test_metrics_default_threshold.confusion_matrix,
      fpr: thresholdMode === "optimal" ? models.CatBoost.test_metrics_optimal_threshold.false_positive_rate : models.CatBoost.test_metrics_default_threshold.false_positive_rate,
      isWinner: false,
    },
    {
      name: "Stacking Ensemble",
      role: "Multi-Model Meta-Learner (Overall Champion)",
      strategy: dataset === "r4.2" ? "Convex Soft-Voting (0.70 XGB + 0.30 CB)" : "Calibrated Logistic Regression (Enriched)",
      data: thresholdMode === "optimal" ? models.Stacking_Ensemble.test_metrics_optimal_threshold : models.Stacking_Ensemble.test_metrics_default_threshold,
      pr_auc: models.Stacking_Ensemble.test_metrics_optimal_threshold.pr_auc,
      roc_auc: models.Stacking_Ensemble.test_metrics_optimal_threshold.roc_auc,
      cm: thresholdMode === "optimal" ? models.Stacking_Ensemble.test_metrics_optimal_threshold.confusion_matrix : models.Stacking_Ensemble.test_metrics_default_threshold.confusion_matrix,
      fpr: thresholdMode === "optimal" ? models.Stacking_Ensemble.test_metrics_optimal_threshold.false_positive_rate : models.Stacking_Ensemble.test_metrics_default_threshold.false_positive_rate,
      isWinner: true,
    },
  ];

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
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            Model Performance Benchmarks
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Rigorous evaluation on strictly untouched test partition ({stats.test_set_sessions.toLocaleString()} sessions, {stats.test_threats} ground-truth threats) in CERT {dataset}.
          </p>
        </div>

        {/* Threshold Toggle */}
        <div className="flex items-center space-x-2 bg-white p-1 rounded-lg border border-slate-200">
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

      {/* Main Leaderboard Table */}
      <div className="enterprise-card overflow-hidden">
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
                  className={row.isWinner ? "bg-blue-50/40 font-medium" : ""}
                >
                  <td>
                    <div className="flex items-center space-x-2">
                      <span className="font-bold text-slate-900">{row.name}</span>
                      {row.isWinner && (
                        <span className="text-[9px] font-bold uppercase px-1.5 py-0.2 rounded bg-blue-100 text-blue-800 border border-blue-200">
                          Champion
                        </span>
                      )}
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
                    {row.cm.fp} <span className="text-[10px] text-slate-400">({(row.fpr * 100).toFixed(4)}%)</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
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
              Click any visualization to zoom in full resolution.
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
