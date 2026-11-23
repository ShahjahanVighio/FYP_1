"use client";

import Link from "next/link";
import { useState } from "react";
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
  Inbox,
  Filter,
  BarChart2,
  ExternalLink
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
import triageData from "@/data/triage_queue.json";
import SessionDrawer, { TriageSession } from "@/components/SessionDrawer";

export default function OverviewPage() {
  const { data, dataset } = useDataset();
  const stats = data.dataset.stats;
  const models = data.models;
  const ensemble = models.Stacking_Ensemble.test_metrics_optimal_threshold;

  const [selectedSession, setSelectedSession] = useState<TriageSession | null>(null);
  const queueSnippet: TriageSession[] = ((triageData as any)[dataset] || []).slice(0, 4);

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
    <div className="space-y-6">
      {/* Top Header Card */}
      <div className="enterprise-card p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-slate-900 tracking-tight">
              Security Operations Command Center
            </h1>
            <span className="text-[11px] font-semibold uppercase px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
              CERT {dataset} Active
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1 max-w-2xl">
            Enterprise User & Entity Behavior Analytics (UEBA) telemetry tracking {stats.employees.toLocaleString()} corporate accounts
            across {stats.duration_months} continuous monitoring months ({stats.total_sessions.toLocaleString()} sessions).
          </p>
        </div>

        <div className="flex items-center space-x-2.5">
          <Link
            href="/queue"
            className="px-3.5 py-2 rounded-md bg-blue-700 hover:bg-blue-800 text-white text-xs font-semibold flex items-center space-x-1.5 transition-colors shadow-2xs"
          >
            <Inbox className="w-3.5 h-3.5" />
            <span>Triage Queue</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>

          <Link
            href="/simulator"
            className="px-3.5 py-2 rounded-md border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold transition-colors"
          >
            Threat Simulator
          </Link>
        </div>
      </div>

      {/* KPI Cards Grid (6 Clean Cards) */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3.5">
        {/* Total Sessions */}
        <div className="enterprise-card p-3.5">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Total Sessions
          </div>
          <div className="mt-1 text-xl font-bold text-slate-900 font-mono tabular-nums">
            {stats.total_sessions.toLocaleString()}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            {stats.employees.toLocaleString()} employees · {stats.duration_months} mo
          </div>
        </div>

        {/* Threat Prevalence */}
        <div className="enterprise-card p-3.5">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Threat Rate
          </div>
          <div className="mt-1 text-xl font-bold text-amber-600 font-mono tabular-nums">
            {stats.threat_percentage}%
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            {stats.imbalance_ratio} sparsity
          </div>
        </div>

        {/* Peak PR-AUC */}
        <div className="enterprise-card p-3.5 border-l-4 border-l-blue-600">
          <div className="text-[11px] font-semibold text-blue-700 uppercase tracking-wider">
            Peak PR-AUC
          </div>
          <div className="mt-1 text-xl font-bold text-blue-900 font-mono tabular-nums">
            {ensemble.pr_auc.toFixed(4)}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            Stacking Ensemble
          </div>
        </div>

        {/* Threats Intercepted */}
        <div className="enterprise-card p-3.5 border-l-4 border-l-emerald-600">
          <div className="text-[11px] font-semibold text-emerald-700 uppercase tracking-wider">
            Threats Caught
          </div>
          <div className="mt-1 text-xl font-bold text-emerald-900 font-mono tabular-nums">
            {ensemble.confusion_matrix.tp} / {stats.test_threats}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            {(ensemble.recall * 100).toFixed(1)}% test recall
          </div>
        </div>

        {/* False Positive Rate */}
        <div className="enterprise-card p-3.5">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            False Positive Rate
          </div>
          <div className="mt-1 text-xl font-bold text-slate-900 font-mono tabular-nums">
            {(ensemble.false_positive_rate * 100).toFixed(4)}%
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            {ensemble.confusion_matrix.fp} alarms in test
          </div>
        </div>

        {/* Optimal Decision Threshold */}
        <div className="enterprise-card p-3.5">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Decision Threshold
          </div>
          <div className="mt-1 text-xl font-bold text-slate-900 font-mono tabular-nums">
            T* = {ensemble.threshold}
          </div>
          <div className="mt-0.5 text-[11px] text-slate-500">
            OOF calibrated
          </div>
        </div>
      </div>

      {/* Main Section: 2 Columns */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: 4-Model Comparison & Triage Queue Snippet (2/3 width) */}
        <div className="lg:col-span-2 space-y-6">
          {/* 4-Model Comparative Analytics Bar Chart */}
          <div className="enterprise-card">
            <div className="enterprise-card-header flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold text-slate-900">
                  Model Efficacy Comparison (Untouched Test Partition)
                </h2>
                <p className="text-[11px] text-slate-500">
                  Benchmarking Random Forest, XGBoost, CatBoost, and Stacking Ensemble on {stats.test_set_sessions.toLocaleString()} sessions.
                </p>
              </div>

              <Link
                href="/results"
                className="text-xs text-blue-700 hover:text-blue-800 font-medium flex items-center space-x-1"
              >
                <span>Full Leaderboard</span>
                <ArrowRight className="w-3 h-3" />
              </Link>
            </div>

            <div className="p-5">
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={comparisonData}
                    margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                    <XAxis
                      dataKey="metric"
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
                    <Bar dataKey="Random Forest" fill="#94A3B8" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="XGBoost" fill="#3B82F6" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="CatBoost" fill="#6366F1" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="Stacking Ensemble" fill="#1E40AF" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Active Triage Queue Snippet */}
          <div className="enterprise-card">
            <div className="enterprise-card-header flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold text-slate-900">
                  Recent High-Priority Flagged Sessions
                </h2>
                <p className="text-[11px] text-slate-500">
                  Showing urgent threats intercepted by the meta-learner requiring analyst disposition.
                </p>
              </div>

              <Link
                href="/queue"
                className="text-xs text-blue-700 hover:text-blue-800 font-medium flex items-center space-x-1"
              >
                <span>View All ({triageData[dataset]?.length || 0})</span>
                <ArrowRight className="w-3 h-3" />
              </Link>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left soc-table">
                <thead>
                  <tr>
                    <th>Session ID</th>
                    <th>User & PC</th>
                    <th>Primary Trigger</th>
                    <th className="text-right">Ensemble Risk</th>
                    <th>Status</th>
                    <th className="text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {queueSnippet.map((s) => (
                    <tr
                      key={s.id}
                      onClick={() => setSelectedSession(s)}
                      className="cursor-pointer"
                    >
                      <td className="font-mono text-xs font-semibold text-blue-700">
                        {s.id}
                      </td>
                      <td>
                        <div className="font-medium text-slate-900 text-xs">{s.name}</div>
                        <div className="text-[10px] text-slate-400 font-mono">{s.workstation}</div>
                      </td>
                      <td className="text-xs text-slate-800 font-medium">
                        {s.trigger}
                      </td>
                      <td className="text-right font-mono text-xs font-bold text-slate-900 tabular-nums">
                        {(s.ensemble_prob * 100).toFixed(1)}%
                      </td>
                      <td>
                        <span
                          className={`text-[9px] font-semibold uppercase px-2 py-0.5 rounded border ${
                            s.disposition === "Escalated"
                              ? "bg-red-50 text-red-700 border-red-200"
                              : s.disposition === "Unreviewed"
                              ? "bg-amber-50 text-amber-700 border-amber-200"
                              : "bg-slate-100 text-slate-700 border-slate-200"
                          }`}
                        >
                          {s.disposition}
                        </span>
                      </td>
                      <td className="text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedSession(s);
                          }}
                          className="px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium"
                        >
                          Inspect →
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right Column: SOC Policy & Baseline Profile (1/3 width) */}
        <div className="space-y-6">
          {/* Operational Triage Tiers */}
          <div className="enterprise-card p-5 space-y-4">
            <div>
              <h2 className="text-sm font-bold text-slate-900">
                SOC Threat Escalation Tiers
              </h2>
              <p className="text-[11px] text-slate-500">
                Standard operating procedures based on out-of-fold calibrated threshold bounds.
              </p>
            </div>

            <div className="space-y-3 text-xs">
              {/* Critical Tier */}
              <div className="p-3 rounded-lg border border-red-200 bg-red-50/50 space-y-1">
                <div className="flex items-center justify-between font-bold text-red-900">
                  <span>Critical Alert (P ≥ 0.70)</span>
                  <span className="text-[10px] uppercase px-1.5 py-0.5 rounded bg-red-100 text-red-800">
                    Tier 1 Action
                  </span>
                </div>
                <p className="text-[11px] text-red-700">
                  Mass USB transfer or external cloud archive detected. Immediate workstation credential revocation.
                </p>
              </div>

              {/* Elevated Tier */}
              <div className="p-3 rounded-lg border border-amber-200 bg-amber-50/50 space-y-1">
                <div className="flex items-center justify-between font-bold text-amber-900">
                  <span>Elevated Risk (T* ≤ P &lt; 0.70)</span>
                  <span className="text-[10px] uppercase px-1.5 py-0.5 rounded bg-amber-100 text-amber-800">
                    Investigate
                  </span>
                </div>
                <p className="text-[11px] text-amber-700">
                  Session exceeded optimal threshold (T* = {ensemble.threshold}). Assigned to L2 analyst for forensic audit.
                </p>
              </div>

              {/* Benign Baseline */}
              <div className="p-3 rounded-lg border border-emerald-200 bg-emerald-50/50 space-y-1">
                <div className="flex items-center justify-between font-bold text-emerald-900">
                  <span>Routine Enterprise (P &lt; T*)</span>
                  <span className="text-[10px] uppercase px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800">
                    Pass
                  </span>
                </div>
                <p className="text-[11px] text-emerald-700">
                  Normal employee workflow consistent with organizational role baselines. Suppressed from alert queue.
                </p>
              </div>
            </div>
          </div>

          {/* Dataset Specifications Card */}
          <div className="enterprise-card p-5 space-y-3">
            <h2 className="text-sm font-bold text-slate-900">
              Benchmark Specifications
            </h2>

            <div className="divide-y divide-slate-100 text-xs">
              <div className="py-2 flex items-center justify-between">
                <span className="text-slate-500">Benchmark Source:</span>
                <span className="font-medium text-slate-800">CMU SEI CERT {dataset}</span>
              </div>
              <div className="py-2 flex items-center justify-between">
                <span className="text-slate-500">Telemetry Size:</span>
                <span className="font-mono font-medium text-slate-800">{stats.raw_log_size_gb} GB</span>
              </div>
              <div className="py-2 flex items-center justify-between">
                <span className="text-slate-500">Total User Sessions:</span>
                <span className="font-mono font-medium text-slate-800">{stats.total_sessions.toLocaleString()}</span>
              </div>
              <div className="py-2 flex items-center justify-between">
                <span className="text-slate-500">Engineered Features:</span>
                <span className="font-mono font-medium text-slate-800">128 Total Signals</span>
              </div>
              <div className="py-2 flex items-center justify-between">
                <span className="text-slate-500">Selected Optimal:</span>
                <span className="font-mono font-medium text-slate-800">Top 50 (ANOVA + MI)</span>
              </div>
              <div className="py-2 flex items-center justify-between">
                <span className="text-slate-500">Test Partition:</span>
                <span className="font-mono font-medium text-slate-800">{stats.test_set_sessions.toLocaleString()} sessions</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Slide-over Session Inspector Drawer */}
      <SessionDrawer
        session={selectedSession}
        onClose={() => setSelectedSession(null)}
      />
    </div>
  );
}
