"use client";

import { useState, useEffect } from "react";
import {
  ShieldAlert,
  Play,
  RotateCcw,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Info,
  Terminal,
  Cpu,
  Layers,
  Sparkles
} from "lucide-react";
import { useDataset } from "@/context/DatasetContext";

interface SimulationResult {
  ensemble_threat_probability: number;
  optimal_threshold: number;
  dataset?: string;
  is_threat: boolean;
  risk_level: string;
  risk_color: string;
  recommendation: string;
  base_model_probabilities: {
    random_forest: number;
    xgboost: number;
    catboost: number;
  };
  model_weights: {
    random_forest: number;
    xgboost: number;
    catboost: number;
  };
  key_signals: string[];
  source?: string;
}

export default function SimulatorPage() {
  const { data, dataset } = useDataset();
  const presets = data.presets;

  // Active form state
  const [features, setFeatures] = useState<{ [key: string]: number }>({
    n_usb: 0,
    usb_mean_usb_dur: 0,
    http_n_leakf: 0,
    http_leakf_mean_url_len: 0,
    http_n_jobf: 0,
    http_n_hackf: 0,
    n_file: 3,
    file_docf_mean_file_nwords: 450,
    isworkhour: 1,
    duration: 28800,
    C: 35,
    E: 28,
  });

  const [activePreset, setActivePreset] = useState<string>("normal_day");
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState(false);

  // Run simulation whenever features change
  const runSimulation = async (currentFeatures = features) => {
    setLoading(true);
    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: currentFeatures, dataset }),
      });
      const resData = await res.json();
      setResult(resData);
    } catch (err) {
      console.error("Simulation error", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runSimulation(features);
  }, [dataset]);

  const handlePresetSelect = (presetKey: string) => {
    setActivePreset(presetKey);
    const p = (presets as any)[presetKey];
    if (p) {
      const newFeats = {
        ...features,
        ...p.features,
      };
      setFeatures(newFeats);
      runSimulation(newFeats);
    }
  };

  const handleSliderChange = (key: string, value: number) => {
    setActivePreset("custom");
    const updated = { ...features, [key]: value };
    setFeatures(updated);
    runSimulation(updated);
  };

  const resetToNormal = () => {
    handlePresetSelect("normal_day");
  };

  const probPercent = result ? Math.round(result.ensemble_threat_probability * 100) : 0;
  const isCritical = result?.risk_level.includes("CRITICAL");
  const isElevated = result?.risk_level.includes("HIGH") || result?.risk_level.includes("ELEVATED");

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            Live Session Insider Risk Simulator
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Evaluate real-time session inference using the Stacking Ensemble Meta-Learner calibrated for CERT {dataset}.
          </p>
        </div>

        <button
          onClick={resetToNormal}
          className="px-3 py-1.5 rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-xs font-medium text-slate-600 flex items-center space-x-1.5 transition-colors cursor-pointer"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span>Reset to Normal Baseline</span>
        </button>
      </div>

      {/* Preset Scenario Selector */}
      <div className="space-y-2">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center space-x-1.5">
          <Sparkles className="w-3.5 h-3.5 text-blue-600" />
          <span>One-Click Ground-Truth Attack Presets</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {Object.entries(presets).map(([key, p]: [string, any]) => {
            const isSelected = activePreset === key;
            return (
              <button
                key={key}
                onClick={() => handlePresetSelect(key)}
                className={`p-3.5 rounded-lg border text-left transition-all cursor-pointer ${
                  isSelected
                    ? "bg-blue-50/70 border-blue-400 ring-1 ring-blue-400 shadow-xs"
                    : "bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-xs font-bold ${isSelected ? "text-blue-900" : "text-slate-900"}`}>
                    {p.title || p.name}
                  </span>
                  {isSelected && (
                    <span className="w-2 h-2 rounded-full bg-blue-600" />
                  )}
                </div>
                <p className="text-[11px] text-slate-500 mt-1 leading-relaxed line-clamp-2">
                  {p.description}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Grid: Controls (2/3) & Live Evaluation Drawer (1/3) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sliders Input Area */}
        <div className="lg:col-span-2 space-y-4">
          {/* Section 1: Removable Media & File System */}
          <div className="enterprise-card p-4 space-y-4">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2 border-b border-slate-100 pb-2">
              <span>Removable Devices & File Exfiltration</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* USB Count */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">USB Storage Mounts (n_usb)</span>
                  <span className="font-mono font-bold text-slate-900">{features.n_usb}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="12"
                  step="1"
                  value={features.n_usb}
                  onChange={(e) => handleSliderChange("n_usb", Number(e.target.value))}
                  className="w-full accent-blue-700 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0 (None)</span>
                  <span>12 (Mass Mounts)</span>
                </div>
              </div>

              {/* USB Dwell Duration */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">USB Dwell Time (seconds)</span>
                  <span className="font-mono font-bold text-slate-900">{features.usb_mean_usb_dur}s</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="7200"
                  step="100"
                  value={features.usb_mean_usb_dur}
                  onChange={(e) => handleSliderChange("usb_mean_usb_dur", Number(e.target.value))}
                  className="w-full accent-blue-700 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0s</span>
                  <span>7,200s (2 hrs)</span>
                </div>
              </div>

              {/* Copied File Count */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">Total File Actions (n_file)</span>
                  <span className="font-mono font-bold text-slate-900">{features.n_file}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="60"
                  step="1"
                  value={features.n_file}
                  onChange={(e) => handleSliderChange("n_file", Number(e.target.value))}
                  className="w-full accent-blue-700 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0 files</span>
                  <span>60 files</span>
                </div>
              </div>

              {/* Document Mean Word Count */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">Doc Word Count (words)</span>
                  <span className="font-mono font-bold text-slate-900">{features.file_docf_mean_file_nwords}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="8000"
                  step="200"
                  value={features.file_docf_mean_file_nwords}
                  onChange={(e) => handleSliderChange("file_docf_mean_file_nwords", Number(e.target.value))}
                  className="w-full accent-blue-700 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0 words</span>
                  <span>8,000 words</span>
                </div>
              </div>
            </div>
          </div>

          {/* Section 2: Web Navigation & HTTP Telemetry */}
          <div className="enterprise-card p-4 space-y-4">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center space-x-2 border-b border-slate-100 pb-2">
              <span>Web Navigation & Online Intelligence</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Cloud Leak Visits */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">Data Leak Sites (http_n_leakf)</span>
                  <span className="font-mono font-bold text-slate-900">{features.http_n_leakf}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="15"
                  step="1"
                  value={features.http_n_leakf}
                  onChange={(e) => handleSliderChange("http_n_leakf", Number(e.target.value))}
                  className="w-full accent-blue-700 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0</span>
                  <span>15 requests</span>
                </div>
              </div>

              {/* Job Search Queries */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">Job Board Queries (http_n_jobf)</span>
                  <span className="font-mono font-bold text-slate-900">{features.http_n_jobf}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="25"
                  step="1"
                  value={features.http_n_jobf}
                  onChange={(e) => handleSliderChange("http_n_jobf", Number(e.target.value))}
                  className="w-full accent-blue-700 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0</span>
                  <span>25 queries</span>
                </div>
              </div>

              {/* Hacking / Keylogger Sites */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">Hacking Sites (http_n_hackf)</span>
                  <span className="font-mono font-bold text-slate-900">{features.http_n_hackf}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="15"
                  step="1"
                  value={features.http_n_hackf}
                  onChange={(e) => handleSliderChange("http_n_hackf", Number(e.target.value))}
                  className="w-full accent-blue-700 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0</span>
                  <span>15 lookups</span>
                </div>
              </div>

              {/* Workhour vs Afterhour Shift */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">Operational Timing</span>
                  <span className="font-semibold text-slate-900">
                    {features.isworkhour === 1 ? "Normal Business Hours" : "Off-Hours / Weekend"}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-1">
                  <button
                    onClick={() => handleSliderChange("isworkhour", 1)}
                    className={`py-1.5 rounded-md text-xs font-semibold border transition-colors cursor-pointer ${
                      features.isworkhour === 1
                        ? "bg-slate-900 text-white border-slate-900"
                        : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    Business Shift (1)
                  </button>
                  <button
                    onClick={() => handleSliderChange("isworkhour", 0)}
                    className={`py-1.5 rounded-md text-xs font-semibold border transition-colors cursor-pointer ${
                      features.isworkhour === 0
                        ? "bg-red-700 text-white border-red-700"
                        : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    After-Hours (0)
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Live Evaluation Right Panel */}
        <div className="space-y-4">
          <div className="enterprise-card p-5 space-y-5 sticky top-20">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                  Real-Time Threat Score
                </span>
                <span className="text-[10px] font-mono text-slate-400">
                  CERT {dataset}
                </span>
              </div>

              <div className="mt-2 flex items-baseline space-x-2">
                <span className="font-mono text-3xl font-extrabold text-slate-900 tabular-nums">
                  {probPercent}%
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  (P = {result?.ensemble_threat_probability.toFixed(4) || "0.0000"})
                </span>
              </div>

              {/* Progress bar */}
              <div className="mt-2 w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${
                    isCritical ? "bg-red-600" : isElevated ? "bg-amber-500" : "bg-emerald-600"
                  }`}
                  style={{ width: `${Math.min(100, Math.max(2, probPercent))}%` }}
                />
              </div>

              <div className="mt-2 flex items-center justify-between text-[11px]">
                <span className="text-slate-400">Optimal Cutoff:</span>
                <span className="font-mono font-semibold text-slate-700">
                  T* = {result?.optimal_threshold || 0.34}
                </span>
              </div>
            </div>

            {/* Severity Pill */}
            <div
              className={`p-3 rounded-lg border text-xs space-y-1 ${
                isCritical
                  ? "bg-red-50 border-red-200 text-red-900"
                  : isElevated
                  ? "bg-amber-50 border-amber-200 text-amber-900"
                  : "bg-emerald-50 border-emerald-200 text-emerald-900"
              }`}
            >
              <div className="font-bold flex items-center space-x-1.5">
                <ShieldAlert className="w-3.5 h-3.5" />
                <span>{result?.risk_level}</span>
              </div>
              <p className="text-[11px] leading-relaxed">
                {result?.recommendation}
              </p>
            </div>

            {/* Tri-Model Scores Breakdown */}
            <div className="space-y-2 pt-2 border-t border-slate-100">
              <h4 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                Base Model Attributions
              </h4>

              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">XGBoost (Histogram)</span>
                  <span className="font-mono font-bold text-slate-900 tabular-nums">
                    {result ? (result.base_model_probabilities.xgboost * 100).toFixed(1) : 0}%
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-slate-600">CatBoost (Oblivious)</span>
                  <span className="font-mono font-bold text-slate-900 tabular-nums">
                    {result ? (result.base_model_probabilities.catboost * 100).toFixed(1) : 0}%
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-slate-600">Random Forest (Bagging)</span>
                  <span className="font-mono font-bold text-slate-900 tabular-nums">
                    {result ? (result.base_model_probabilities.random_forest * 100).toFixed(1) : 0}%
                  </span>
                </div>
              </div>
            </div>

            {/* Key Triggered Signals */}
            <div className="space-y-2 pt-2 border-t border-slate-100">
              <h4 className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                Triggered Anomaly Signals
              </h4>
              <ul className="space-y-1 text-xs">
                {result?.key_signals.map((sig, idx) => (
                  <li key={idx} className="flex items-start space-x-1.5 text-slate-700">
                    <span className="text-blue-600 font-bold">•</span>
                    <span className="text-[11px]">{sig}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
