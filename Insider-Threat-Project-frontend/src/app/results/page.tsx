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
      type: "Cost-Sensitive (balanced_subsample)",
      data: thresholdMode === "optimal" ? models.Random_Forest.test_metrics_optimal_threshold : models.Random_Forest.test_metrics_default_threshold,
      pr_auc: models.Random_Forest.test_metrics_optimal_threshold.pr_auc,
      roc_auc: models.Random_Forest.test_metrics_optimal_threshold.roc_auc,
      cm: thresholdMode === "optimal" ? models.Random_Forest.test_metrics_optimal_threshold.confusion_matrix : models.Random_Forest.test_metrics_default_threshold.confusion_matrix,
      fpr: thresholdMode === "optimal" ? models.Random_Forest.test_metrics_optimal_threshold.false_positive_rate : models.Random_Forest.test_metrics_default_threshold.false_positive_rate,
      badge: "Bagging Variance Anchor",
    },
    {
      name: "XGBoost",
      type: dataset === "r4.2" ? "Cost-Sensitive (scale_pos_weight=1.0)" : "Cost-Sensitive (scale_pos_weight=23.3)",
      data: thresholdMode === "optimal" ? models.XGBoost.test_metrics_optimal_threshold : models.XGBoost.test_metrics_default_threshold,
      pr_auc: models.XGBoost.test_metrics_optimal_threshold.pr_auc,
      roc_auc: models.XGBoost.test_metrics_optimal_threshold.roc_auc,
      cm: thresholdMode === "optimal" ? models.XGBoost.test_metrics_optimal_threshold.confusion_matrix : models.XGBoost.test_metrics_default_threshold.confusion_matrix,
      fpr: thresholdMode === "optimal" ? models.XGBoost.test_metrics_optimal_threshold.false_positive_rate : models.XGBoost.test_metrics_default_threshold.false_positive_rate,
      badge: "Highest Precision Anchor",
    },
    {
      name: "CatBoost",
      type: "Cost-Sensitive (SqrtBalanced)",
      data: thresholdMode === "optimal" ? models.CatBoost.test_metrics_optimal_threshold : models.CatBoost.test_metrics_default_threshold,
      pr_auc: models.CatBoost.test_metrics_optimal_threshold.pr_auc,
      roc_auc: models.CatBoost.test_metrics_optimal_threshold.roc_auc,
      cm: thresholdMode === "optimal" ? models.CatBoost.test_metrics_optimal_threshold.confusion_matrix : models.CatBoost.test_metrics_default_threshold.confusion_matrix,
      fpr: thresholdMode === "optimal" ? models.CatBoost.test_metrics_optimal_threshold.false_positive_rate : models.CatBoost.test_metrics_default_threshold.false_positive_rate,
      badge: "Highest Single-Model Recall",
    },
    {
      name: "Stacking Ensemble",
      type: dataset === "r4.2" ? "Optimal Soft-Voting (0.70 XGB + 0.30 CB)" : "Calibrated Logistic Regression (Enriched)",
      data: thresholdMode === "optimal" ? models.Stacking_Ensemble.test_metrics_optimal_threshold : models.Stacking_Ensemble.test_metrics_default_threshold,
      pr_auc: models.Stacking_Ensemble.test_metrics_optimal_threshold.pr_auc,
      roc_auc: models.Stacking_Ensemble.test_metrics_optimal_threshold.roc_auc,
      cm: thresholdMode === "optimal" ? models.Stacking_Ensemble.test_metrics_optimal_threshold.confusion_matrix : models.Stacking_Ensemble.test_metrics_default_threshold.confusion_matrix,
      fpr: thresholdMode === "optimal" ? models.Stacking_Ensemble.test_metrics_optimal_threshold.false_positive_rate : models.Stacking_Ensemble.test_metrics_default_threshold.false_positive_rate,
      badge: "Winner: Peak PR-AUC & Coverage",
      isWinner: true,
    },
  ];

  const gallery = [
    {
      title: "All Models PR & ROC Comparison",
      category: "all",
      path: `${plotPrefix}/ensemble/all_models_pr_roc_comparison.png`,
      caption: `Master comparative Precision-Recall and ROC curves on CERT ${dataset}.`,
    },
    {
      title: "Stacking Ensemble Confusion Matrix",
      category: "ensemble",
      path: `${plotPrefix}/ensemble/ensemble_confusion_matrix.png`,
      caption: `Evaluated on ${stats.test_set_sessions.toLocaleString()} untouched test sessions (${stats.test_threats} threats).`,
    },
    {
      title: "Meta-Learner Optimal Model Weights",
      category: "ensemble",
      path: `${plotPrefix}/ensemble/meta_learner_weights.png`,
      caption: "Learned stacking meta-learner feature weights and importance coefficients.",
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
      caption: `XGBoost Precision-Recall and ROC curves on CERT ${dataset}.`,
    },
    {
      title: "XGBoost Confusion Matrix",
      category: "xgb",
      path: `${plotPrefix}/xgb/xgb_confusion_matrix.png`,
      caption: `XGBoost evaluation on ${stats.test_set_sessions.toLocaleString()} untouched test sessions.`,
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
      caption: `CatBoost Precision-Recall and ROC curves on CERT ${dataset}.`,
    },
    {
      title: "CatBoost Confusion Matrix",
      category: "cb",
      path: `${plotPrefix}/cb/cb_confusion_matrix.png`,
      caption: `CatBoost evaluation on ${stats.test_set_sessions.toLocaleString()} untouched test sessions.`,
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
      caption: `Random Forest Precision-Recall and ROC curves on CERT ${dataset}.`,
    },
    {
      title: "Random Forest Confusion Matrix",
      category: "rf",
      path: `${plotPrefix}/rf/rf_confusion_matrix.png`,
      caption: `Random Forest evaluation on ${stats.test_set_sessions.toLocaleString()} untouched test sessions.`,
    },
  ];

  const filteredGallery = activePlotTab === "all"
    ? gallery
    : gallery.filter((p) => p.category === activePlotTab);

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="space-y-2">
        <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-md text-xs font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
          <Cpu className="w-3.5 h-3.5" />
          <span>Evaluation Benchmark</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-100 tracking-tight">
          Model Leaderboard & Empirical Curves
        </h1>
        <p className="text-sm text-slate-400 max-w-3xl leading-relaxed">
          Head-to-head performance across all 4 machine learning pipelines evaluated on the exact same untouched test partition (94,123 sessions, 223 threat cases).
        </p>
      </section>

      {/* Threshold Toggle & Leaderboard Table */}
      <section className="rounded-xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
              <Award className="w-4 h-4 text-cyan-400" />
              <span>Master 4-Model Test Set Leaderboard</span>
            </h2>
            <p className="text-xs text-slate-400">
              Evaluated on X_test (94,123 rows). Thresholds optimized strictly on out-of-fold cross-validation.
            </p>
          </div>

          {/* Toggle Button */}
          <div className="flex items-center p-1 rounded-lg bg-slate-950 border border-slate-800 text-xs">
            <button
              onClick={() => setThresholdMode("optimal")}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                thresholdMode === "optimal"
                  ? "bg-cyan-500 text-slate-950 font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Optimal Threshold (T*)
            </button>
            <button
              onClick={() => setThresholdMode("default")}
              className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                thresholdMode === "default"
                  ? "bg-cyan-500 text-slate-950 font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Default (T = 0.50)
            </button>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
                <th className="py-3 px-3">Model</th>
                <th className="py-3 px-3">Strategy</th>
                <th className="py-3 px-2 text-center">Threshold</th>
                <th className="py-3 px-3 text-right">Precision</th>
                <th className="py-3 px-3 text-right">Recall</th>
                <th className="py-3 px-3 text-right">F1-Score</th>
                <th className="py-3 px-3 text-right">PR-AUC</th>
                <th className="py-3 px-3 text-right">ROC-AUC</th>
                <th className="py-3 px-3 text-right">FPR</th>
                <th className="py-3 px-3 text-right">Threats (TP)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {rows.map((row) => (
                <tr
                  key={row.name}
                  className={`hover:bg-slate-800/40 transition-colors ${
                    row.isWinner ? "bg-cyan-950/20" : ""
                  }`}
                >
                  <td className="py-3 px-3 font-sans font-bold text-slate-100 flex items-center space-x-2">
                    <span>{row.name}</span>
                    {row.isWinner && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                        WINNER
                      </span>
                    )}
                  </td>
                  <td className="py-3 px-3 font-sans text-slate-400 text-[11px]">{row.type}</td>
                  <td className="py-3 px-2 text-center text-cyan-400 font-bold">
                    {row.data.threshold.toFixed(2)}
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-slate-200">
                    {(row.data.precision * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-slate-200">
                    {(row.data.recall * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-emerald-400">
                    {(row.data.f1_score * 100).toFixed(2)}%
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-cyan-300">
                    {row.pr_auc.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {row.roc_auc.toFixed(4)}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-400">
                    {(row.fpr * 100).toFixed(4)}%
                  </td>
                  <td className="py-3 px-3 text-right font-bold text-emerald-400">
                    {row.cm.tp} <span className="text-slate-500 font-normal">/ 223</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="pt-2 text-[11px] text-slate-500 flex flex-wrap items-center justify-between gap-2">
          <span>* Evaluated on untouched test partition: 94,123 total sessions (93,900 normal, 223 threat).</span>
          <span className="text-cyan-400">
            Stacking Ensemble intercepts 188 threats (84.30% Recall) with only 18 false alarms.
          </span>
        </div>
      </section>

      {/* Plot Gallery & Lightbox */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-emerald-400" />
              <span>Publication-Grade Evaluation Curves & Figures (300 DPI)</span>
            </h2>
            <p className="text-xs text-slate-400">
              Click any visualization to expand into full-resolution inspector view.
            </p>
          </div>

          {/* Filter Tabs */}
          <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <button
              onClick={() => setActivePlotTab("all")}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activePlotTab === "all" ? "bg-slate-800 text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              All Plots
            </button>
            <button
              onClick={() => setActivePlotTab("ensemble")}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activePlotTab === "ensemble" ? "bg-slate-800 text-cyan-400 font-bold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Stacking Ensemble
            </button>
            <button
              onClick={() => setActivePlotTab("xgb")}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activePlotTab === "xgb" ? "bg-slate-800 text-orange-400 font-bold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              XGBoost
            </button>
            <button
              onClick={() => setActivePlotTab("cb")}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activePlotTab === "cb" ? "bg-slate-800 text-emerald-400 font-bold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              CatBoost
            </button>
            <button
              onClick={() => setActivePlotTab("rf")}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                activePlotTab === "rf" ? "bg-slate-800 text-blue-400 font-bold" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Random Forest
            </button>
          </div>
        </div>

        {/* Gallery Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredGallery.map((item, idx) => (
            <div
              key={idx}
              onClick={() => setSelectedPlot(item.path)}
              className="group cursor-pointer rounded-xl bg-slate-900/80 border border-slate-800 overflow-hidden hover:border-cyan-500/50 transition-all flex flex-col justify-between"
            >
              <div className="relative h-56 w-full bg-slate-950 flex items-center justify-center p-2 overflow-hidden">
                <img
                  src={item.path}
                  alt={item.title}
                  className="max-h-full max-w-full object-contain group-hover:scale-105 transition-transform duration-200"
                />
                <div className="absolute top-2 right-2 p-1.5 rounded-lg bg-slate-900/80 text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Maximize2 className="w-4 h-4 text-cyan-400" />
                </div>
              </div>

              <div className="p-3 border-t border-slate-800/80 bg-slate-900">
                <div className="font-semibold text-xs text-slate-200 group-hover:text-cyan-400 transition-colors">
                  {item.title}
                </div>
                <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                  {item.caption}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Lightbox Modal */}
      {selectedPlot && (
        <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="relative max-w-5xl w-full max-h-[90vh] bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden p-4 flex flex-col items-center">
            <button
              onClick={() => setSelectedPlot(null)}
              className="absolute top-4 right-4 p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors z-10"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="w-full flex-1 flex items-center justify-center overflow-auto max-h-[80vh]">
              <img
                src={selectedPlot}
                alt="Enlarged Plot"
                className="max-h-full max-w-full object-contain rounded-lg"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
