"use client";

import { useState, useMemo } from "react";
import {
  Inbox,
  Search,
  Filter,
  AlertTriangle,
  CheckCircle2,
  Clock,
  ArrowUpDown,
  ChevronRight,
  ShieldAlert,
  Download,
  RefreshCw,
  ExternalLink,
  Layers,
  Shield
} from "lucide-react";
import { useDataset } from "@/context/DatasetContext";
import triageData from "@/data/triage_queue.json";
import SessionDrawer, { TriageSession } from "@/components/SessionDrawer";

export default function QueuePage() {
  const { dataset } = useDataset();
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [scenarioFilter, setScenarioFilter] = useState("All");
  const [selectedSession, setSelectedSession] = useState<TriageSession | null>(null);

  // Active key for queue data
  const activeQueueKey = dataset === "cross_release" ? "r6.2" : dataset;
  const rawSessions = (triageData as any)[activeQueueKey] || (triageData as any)["r6.2"] || [];
  const [sessions, setSessions] = useState<TriageSession[]>(rawSessions);

  // Sync when dataset changes
  useMemo(() => {
    const key = dataset === "cross_release" ? "r6.2" : dataset;
    setSessions((triageData as any)[key] || (triageData as any)["r6.2"] || []);
  }, [dataset]);

  const filteredSessions = useMemo(() => {
    return sessions.filter((s) => {
      const matchStatus = statusFilter === "All" || s.disposition === statusFilter;
      const matchScenario =
        scenarioFilter === "All" ||
        (scenarioFilter === "Scenario 1" && s.ground_truth.includes("Scenario 1")) ||
        (scenarioFilter === "Scenario 2" && s.ground_truth.includes("Scenario 2")) ||
        (scenarioFilter === "Scenario 3" && s.ground_truth.includes("Scenario 3")) ||
        (scenarioFilter === "Scenario 4" && s.ground_truth.includes("Scenario 4")) ||
        (scenarioFilter === "Scenario 5" && s.ground_truth.includes("Scenario 5")) ||
        (scenarioFilter === "Benign" && s.ground_truth.includes("Benign"));

      const matchSearch =
        s.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        s.user_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        s.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        s.workstation.toLowerCase().includes(searchTerm.toLowerCase()) ||
        s.trigger.toLowerCase().includes(searchTerm.toLowerCase());
      return matchStatus && matchScenario && matchSearch;
    });
  }, [sessions, statusFilter, scenarioFilter, searchTerm]);

  const handleUpdateDisposition = (id: string, newDisp: string) => {
    setSessions((prev) =>
      prev.map((s) => (s.id === id ? { ...s, disposition: newDisp } : s))
    );
    if (selectedSession && selectedSession.id === id) {
      setSelectedSession((prev) => (prev ? { ...prev, disposition: newDisp } : null));
    }
  };

  const counts = useMemo(() => {
    return {
      total: sessions.length,
      unreviewed: sessions.filter((s) => s.disposition === "Unreviewed").length,
      escalated: sessions.filter((s) => s.disposition === "Escalated").length,
      falsePositives: sessions.filter((s) => s.disposition === "False Positive").length,
      resolved: sessions.filter((s) => s.disposition === "Resolved").length,
    };
  }, [sessions]);

  return (
    <div className="space-y-6">
      {/* Page Title & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold text-slate-900 tracking-tight">
              Security Triage Queue
            </h1>
            <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
              {dataset === "cross_release" ? "CERT r6.2 Active Stream" : `CERT ${dataset}`}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Real-time flagged session queue for SOC Level-1 & Level-2 analyst triage. Click any session to inspect individual model probability contributions.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => {
              const key = dataset === "cross_release" ? "r6.2" : dataset;
              setSessions((triageData as any)[key] || []);
            }}
            className="p-1.5 rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 text-xs flex items-center space-x-1 cursor-pointer"
            title="Refresh queue"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Refresh Queue</span>
          </button>
        </div>
      </div>

      {/* Metric Counters */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
        <div className="enterprise-card p-3.5">
          <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
            Total In Queue
          </div>
          <div className="mt-1 text-2xl font-bold text-slate-900 font-mono tabular-nums">
            {counts.total}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Automated ML detections</div>
        </div>

        <div className="enterprise-card p-3.5 border-l-4 border-l-amber-500">
          <div className="text-[11px] font-semibold text-amber-700 uppercase tracking-wider">
            Pending Review
          </div>
          <div className="mt-1 text-2xl font-bold text-amber-800 font-mono tabular-nums">
            {counts.unreviewed}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Awaiting triage disposition</div>
        </div>

        <div className="enterprise-card p-3.5 border-l-4 border-l-red-500">
          <div className="text-[11px] font-semibold text-red-700 uppercase tracking-wider">
            Escalated Cases
          </div>
          <div className="mt-1 text-2xl font-bold text-red-800 font-mono tabular-nums">
            {counts.escalated}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Assigned to IR specialists</div>
        </div>

        <div className="enterprise-card p-3.5 border-l-4 border-l-emerald-500">
          <div className="text-[11px] font-semibold text-emerald-700 uppercase tracking-wider">
            Resolved / False Positives
          </div>
          <div className="mt-1 text-2xl font-bold text-emerald-800 font-mono tabular-nums">
            {counts.falsePositives + counts.resolved}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">Audited & closed</div>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="enterprise-card p-3 flex flex-col sm:flex-row items-center justify-between gap-3">
        {/* Status Filter Tabs */}
        <div className="flex items-center space-x-1 w-full sm:w-auto overflow-x-auto">
          {["All", "Unreviewed", "Escalated", "Resolved"].map((status) => (
            <button
              key={status}
              onClick={() => setStatusFilter(status)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                statusFilter === status
                  ? "bg-slate-900 text-white font-semibold"
                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
              }`}
            >
              {status}
            </button>
          ))}
        </div>

        {/* Scenario Filter & Search */}
        <div className="flex items-center space-x-2 w-full sm:w-auto">
          <select
            value={scenarioFilter}
            onChange={(e) => setScenarioFilter(e.target.value)}
            className="text-xs px-2.5 py-1.5 rounded-md border border-slate-200 bg-white text-slate-700 font-medium"
          >
            <option value="All">All Scenarios</option>
            <option value="Scenario 1">Scenario 1 (USB Theft)</option>
            <option value="Scenario 2">Scenario 2 (Sabotage)</option>
            <option value="Scenario 3">Scenario 3 (Keylogger)</option>
            <option value="Scenario 4">Scenario 4 (Email/Cloud)</option>
            <option value="Scenario 5">Scenario 5 (Dropbox Zero-Shot)</option>
            <option value="Benign">Benign Baseline</option>
          </select>

          {/* Search Input */}
          <div className="relative w-full sm:w-64">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search User, PC, ID, Trigger..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 rounded-md border border-slate-200 bg-white text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-1 focus:ring-blue-600"
            />
          </div>
        </div>
      </div>

      {/* Main Dense Table */}
      <div className="enterprise-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left soc-table">
            <thead>
              <tr>
                <th>Session ID</th>
                <th>Employee & Unit</th>
                <th>Workstation</th>
                <th>Primary Threat Trigger</th>
                <th className="text-right">RF / XGB / CB</th>
                <th className="text-right">Ensemble Risk</th>
                <th>Disposition</th>
                <th className="text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredSessions.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-slate-400 text-xs">
                    No sessions match current filter criteria.
                  </td>
                </tr>
              ) : (
                filteredSessions.map((s) => {
                  const isCritical = s.ensemble_prob >= 0.70;
                  const isHigh = s.ensemble_prob >= 0.34 && s.ensemble_prob < 0.70;

                  return (
                    <tr
                      key={s.id}
                      onClick={() => setSelectedSession(s)}
                      className="cursor-pointer transition-colors hover:bg-slate-50"
                    >
                      <td className="font-mono text-xs font-semibold text-blue-700">
                        {s.id}
                      </td>

                      <td>
                        <div className="font-medium text-slate-900">{s.name}</div>
                        <div className="text-[11px] text-slate-500">{s.department} · {s.user_id}</div>
                      </td>

                      <td className="font-mono text-xs text-slate-700">
                        {s.workstation}
                      </td>

                      <td>
                        <div className="font-medium text-slate-800">{s.trigger}</div>
                        <div className="text-[11px] text-slate-400 font-mono">{s.timestamp}</div>
                      </td>

                      <td className="text-right font-mono text-[11px] text-slate-500 tabular-nums">
                        {s.rf_prob.toFixed(2)} · {s.xgb_prob.toFixed(2)} · {s.cb_prob.toFixed(2)}
                      </td>

                      <td className="text-right">
                        <span
                          className={`font-mono text-xs font-bold px-2 py-0.5 rounded border tabular-nums ${
                            isCritical
                              ? "bg-red-50 text-red-700 border-red-200"
                              : isHigh
                              ? "bg-amber-50 text-amber-700 border-amber-200"
                              : "bg-emerald-50 text-emerald-700 border-emerald-200"
                          }`}
                        >
                          {(s.ensemble_prob * 100).toFixed(1)}%
                        </span>
                      </td>

                      <td>
                        <span
                          className={`inline-block text-[10px] font-semibold uppercase px-2 py-0.5 rounded border ${
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
                          className="px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium transition-colors cursor-pointer"
                        >
                          Inspect →
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Slide-over Session Inspector Drawer */}
      <SessionDrawer
        session={selectedSession}
        onClose={() => setSelectedSession(null)}
        onUpdateDisposition={handleUpdateDisposition}
      />
    </div>
  );
}
