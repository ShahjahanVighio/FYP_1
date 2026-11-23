"use client";

import { X, ShieldAlert, AlertTriangle, CheckCircle2, Laptop, User, Clock, Building, ArrowUpRight } from "lucide-react";

export interface TriageSession {
  id: string;
  user_id: string;
  name: string;
  department: string;
  workstation: string;
  timestamp: string;
  is_afterhours: boolean;
  trigger: string;
  rf_prob: number;
  xgb_prob: number;
  cb_prob: number;
  ensemble_prob: number;
  disposition: string;
  ground_truth: string;
  factors: {
    name: string;
    val: string;
    baseline: string;
    risk: string;
  }[];
}

interface SessionDrawerProps {
  session: TriageSession | null;
  onClose: () => void;
  onUpdateDisposition?: (sessionId: string, newDisposition: string) => void;
}

export default function SessionDrawer({ session, onClose, onUpdateDisposition }: SessionDrawerProps) {
  if (!session) return null;

  const isCritical = session.ensemble_prob >= 0.70;
  const isHigh = session.ensemble_prob >= 0.34 && session.ensemble_prob < 0.70;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-slate-900/30 flex justify-end transition-opacity">
      <div className="w-full max-w-lg bg-white h-full shadow-2xl border-l border-slate-200 flex flex-col animate-in slide-in-from-right duration-200">
        {/* Drawer Header */}
        <div className="p-5 border-b border-slate-200 flex items-start justify-between bg-slate-50/60">
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-mono text-xs font-bold text-slate-800 bg-white px-2 py-0.5 rounded border border-slate-200">
                {session.id}
              </span>
              <span
                className={`text-[10px] font-semibold uppercase px-2 py-0.5 rounded border ${
                  isCritical
                    ? "bg-red-50 text-red-700 border-red-200"
                    : isHigh
                    ? "bg-amber-50 text-amber-700 border-amber-200"
                    : "bg-emerald-50 text-emerald-700 border-emerald-200"
                }`}
              >
                {session.disposition}
              </span>
            </div>
            <h2 className="mt-2 text-base font-bold text-slate-900">
              {session.trigger}
            </h2>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Drawer Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-6">
          {/* Risk Score Master Card */}
          <div
            className={`p-4 rounded-lg border ${
              isCritical
                ? "bg-red-50/50 border-red-200"
                : isHigh
                ? "bg-amber-50/50 border-amber-200"
                : "bg-emerald-50/50 border-emerald-200"
            }`}
          >
            <div className="flex items-center justify-between">
              <div>
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-600">
                  Ensemble Threat Probability
                </span>
                <div className="mt-1 flex items-baseline space-x-2">
                  <span className="font-mono text-2xl font-extrabold text-slate-900 tabular-nums">
                    {(session.ensemble_prob * 100).toFixed(1)}%
                  </span>
                  <span className="text-xs font-medium text-slate-500 font-mono">
                    (Score: {session.ensemble_prob.toFixed(4)})
                  </span>
                </div>
              </div>

              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center ${
                  isCritical
                    ? "bg-red-100 text-red-700"
                    : isHigh
                    ? "bg-amber-100 text-amber-700"
                    : "bg-emerald-100 text-emerald-700"
                }`}
              >
                <ShieldAlert className="w-5 h-5" />
              </div>
            </div>

            <div className="mt-3 pt-3 border-t border-slate-200/60 text-xs text-slate-600 flex items-center justify-between">
              <span>Ground Truth Validation:</span>
              <span className="font-semibold text-slate-800">{session.ground_truth}</span>
            </div>
          </div>

          {/* User & Workstation Context */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">
              Entity Telemetry Context
            </h3>
            <div className="enterprise-card p-3.5 grid grid-cols-2 gap-3 text-xs">
              <div>
                <div className="text-slate-400 text-[11px] flex items-center space-x-1">
                  <User className="w-3 h-3 text-slate-400" />
                  <span>Employee</span>
                </div>
                <div className="mt-0.5 font-semibold text-slate-900">{session.name}</div>
                <div className="text-[11px] text-slate-500 font-mono">{session.user_id}</div>
              </div>

              <div>
                <div className="text-slate-400 text-[11px] flex items-center space-x-1">
                  <Building className="w-3 h-3 text-slate-400" />
                  <span>Department</span>
                </div>
                <div className="mt-0.5 font-medium text-slate-800">{session.department}</div>
              </div>

              <div>
                <div className="text-slate-400 text-[11px] flex items-center space-x-1">
                  <Laptop className="w-3 h-3 text-slate-400" />
                  <span>Workstation</span>
                </div>
                <div className="mt-0.5 font-mono font-medium text-slate-800">{session.workstation}</div>
              </div>

              <div>
                <div className="text-slate-400 text-[11px] flex items-center space-x-1">
                  <Clock className="w-3 h-3 text-slate-400" />
                  <span>Session Start</span>
                </div>
                <div className="mt-0.5 font-mono text-[11px] text-slate-700">{session.timestamp}</div>
              </div>
            </div>
          </div>

          {/* Base Model Attribution Consensus */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">
              Tri-Model Algorithmic Consensus
            </h3>
            <div className="enterprise-card p-3.5 space-y-2.5">
              {/* XGBoost */}
              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-medium text-slate-700">XGBoost (Histogram Boosting)</span>
                  <span className="font-mono font-bold text-slate-900 tabular-nums">
                    {(session.xgb_prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-blue-600 h-1.5 rounded-full"
                    style={{ width: `${session.xgb_prob * 100}%` }}
                  />
                </div>
              </div>

              {/* CatBoost */}
              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-medium text-slate-700">CatBoost (Oblivious Trees)</span>
                  <span className="font-mono font-bold text-slate-900 tabular-nums">
                    {(session.cb_prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-indigo-600 h-1.5 rounded-full"
                    style={{ width: `${session.cb_prob * 100}%` }}
                  />
                </div>
              </div>

              {/* Random Forest */}
              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-medium text-slate-700">Random Forest (Bagging Trees)</span>
                  <span className="font-mono font-bold text-slate-900 tabular-nums">
                    {(session.rf_prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-slate-500 h-1.5 rounded-full"
                    style={{ width: `${session.rf_prob * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Key Behavioral Contributing Factors */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">
              Anomalous Contributing Factors
            </h3>
            <div className="enterprise-card divide-y divide-slate-100">
              {session.factors.map((f, idx) => (
                <div key={idx} className="p-3 flex items-start justify-between text-xs">
                  <div>
                    <div className="font-medium text-slate-800">{f.name}</div>
                    <div className="text-[11px] text-slate-500 mt-0.5">
                      Baseline: <span className="font-mono text-slate-600">{f.baseline}</span>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="font-mono font-bold text-slate-900">{f.val}</div>
                    <span
                      className={`inline-block mt-0.5 text-[9px] font-semibold uppercase px-1.5 py-0.2 rounded border ${
                        f.risk === "CRITICAL"
                          ? "bg-red-50 text-red-700 border-red-200"
                          : f.risk === "HIGH"
                          ? "bg-amber-50 text-amber-700 border-amber-200"
                          : "bg-slate-50 text-slate-600 border-slate-200"
                      }`}
                    >
                      {f.risk}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Drawer Action Dispositions */}
        <div className="p-4 border-t border-slate-200 bg-slate-50 flex items-center justify-between gap-2">
          <button
            onClick={() => onUpdateDisposition?.(session.id, "False Positive")}
            className="flex-1 py-2 px-3 rounded-md border border-slate-300 bg-white hover:bg-slate-50 text-xs font-medium text-slate-700 transition-colors cursor-pointer"
          >
            Mark False Positive
          </button>
          <button
            onClick={() => onUpdateDisposition?.(session.id, "Escalated")}
            className="flex-1 py-2 px-3 rounded-md bg-blue-700 hover:bg-blue-800 text-white text-xs font-semibold transition-colors shadow-xs cursor-pointer"
          >
            Escalate Incident
          </button>
        </div>
      </div>
    </div>
  );
}
