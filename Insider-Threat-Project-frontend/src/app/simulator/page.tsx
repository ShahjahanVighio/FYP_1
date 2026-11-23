"use client";

import { useState, useEffect } from "react";
import {
  ShieldAlert,
  Sliders,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  Flame,
  Activity,
  Cpu,
  Layers,
  Sparkles,
} from "lucide-react";
import data from "@/data/r4_2_metrics.json";

interface SimulationResult {
  ensemble_threat_probability: number;
  optimal_threshold: number;
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
        body: JSON.stringify({ features: currentFeatures }),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Simulation error", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runSimulation(features);
  }, []);

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

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="space-y-2">
        <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-md text-xs font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
          <ShieldAlert className="w-3.5 h-3.5" />
          <span>Interactive UEBA Security Operations Center</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-100 tracking-tight">
          Live Session Insider Threat Simulator
        </h1>
        <p className="text-sm text-slate-400 max-w-3xl leading-relaxed">
          Test real-time session inference using our Stacking Ensemble (XGBoost + CatBoost). Adjust behavioral activity sliders or trigger real-world scenario presets to see live risk scoring.
        </p>
      </section>

      {/* Preset Scenario Selector */}
      <section className="space-y-3">
        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center space-x-2">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>One-Click Scenario Presets</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {Object.entries(presets).map(([key, p]: [string, any]) => {
            const isSelected = activePreset === key;
            return (
              <button
                key={key}
                onClick={() => handlePresetSelect(key)}
                className={`p-3.5 rounded-xl text-left border transition-all ${
                  isSelected
                    ? "bg-cyan-950/40 border-cyan-500/60 shadow-lg shadow-cyan-500/10 text-slate-100"
                    : "bg-slate-900/80 border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200"
                }`}
              >
                <div className="font-semibold text-xs text-slate-200">{p.title}</div>
                <div className="text-[11px] text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                  {p.description}
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* Main Interactive Grid: Sliders vs. Live Gauge */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Behavioral Feature Sliders (7 cols) */}
        <div className="lg:col-span-7 rounded-xl bg-slate-900/80 border border-slate-800 p-6 space-y-6">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <span>Session Activity Parameters</span>
            </h2>
            <button
              onClick={resetToNormal}
              className="flex items-center space-x-1 text-xs text-slate-400 hover:text-cyan-400 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
          </div>

          <div className="space-y-5 text-xs">
            {/* Shift Timing Toggle */}
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between">
              <div>
                <span className="font-semibold text-slate-200 block">Session Timing</span>
                <span className="text-[11px] text-slate-500">
                  {features.isworkhour === 1 ? "Standard Workhours (07:30 - 17:30)" : "Off-Hours / Weekend Activity"}
                </span>
              </div>
              <div className="flex items-center space-x-1.5 p-1 rounded-md bg-slate-900 border border-slate-800">
                <button
                  onClick={() => handleSliderChange("isworkhour", 1)}
                  className={`px-2.5 py-1 rounded text-xs transition-colors ${
                    features.isworkhour === 1 ? "bg-cyan-500 text-slate-950 font-bold" : "text-slate-400"
                  }`}
                >
                  Workday
                </button>
                <button
                  onClick={() => handleSliderChange("isworkhour", 0)}
                  className={`px-2.5 py-1 rounded text-xs transition-colors ${
                    features.isworkhour === 0 ? "bg-amber-500 text-slate-950 font-bold" : "text-slate-400"
                  }`}
                >
                  Off-Hours
                </button>
              </div>
            </div>

            {/* USB Media Sliders */}
            <div className="space-y-4 pt-1">
              <div className="space-y-1.5">
                <div className="flex justify-between font-medium">
                  <span className="text-slate-300">Removable USB Storage Inserts (n_usb)</span>
                  <span className="text-cyan-400 font-bold font-mono">{features.n_usb} devices</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="1"
                  value={features.n_usb}
                  onChange={(e) => handleSliderChange("n_usb", Number(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>

              {features.n_usb > 0 && (
                <div className="space-y-1.5 pl-3 border-l-2 border-cyan-500/40">
                  <div className="flex justify-between font-medium">
                    <span className="text-slate-300">Mean USB Duration (seconds)</span>
                    <span className="text-cyan-400 font-bold font-mono">{features.usb_mean_usb_dur}s</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="5000"
                    step="100"
                    value={features.usb_mean_usb_dur}
                    onChange={(e) => handleSliderChange("usb_mean_usb_dur", Number(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>
              )}
            </div>

            {/* Cloud & Data Leak Sites */}
            <div className="space-y-4 pt-2 border-t border-slate-800/80">
              <div className="space-y-1.5">
                <div className="flex justify-between font-medium">
                  <span className="text-slate-300">Cloud Storage / Leak Requests (http_n_leakf)</span>
                  <span className="text-cyan-400 font-bold font-mono">{features.http_n_leakf} requests</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="15"
                  step="1"
                  value={features.http_n_leakf}
                  onChange={(e) => handleSliderChange("http_n_leakf", Number(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between font-medium">
                  <span className="text-slate-300">Job Search Queries (http_n_jobf)</span>
                  <span className="text-cyan-400 font-bold font-mono">{features.http_n_jobf} queries</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="20"
                  step="1"
                  value={features.http_n_jobf}
                  onChange={(e) => handleSliderChange("http_n_jobf", Number(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between font-medium">
                  <span className="text-slate-300">Hacking / Keylogger Lookups (http_n_hackf)</span>
                  <span className="text-red-400 font-bold font-mono">{features.http_n_hackf} lookups</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="1"
                  value={features.http_n_hackf}
                  onChange={(e) => handleSliderChange("http_n_hackf", Number(e.target.value))}
                  className="w-full accent-red-400 cursor-pointer"
                />
              </div>
            </div>

            {/* Document Exfiltration */}
            <div className="space-y-4 pt-2 border-t border-slate-800/80">
              <div className="space-y-1.5">
                <div className="flex justify-between font-medium">
                  <span className="text-slate-300">Document Word Volume (file_docf_mean_file_nwords)</span>
                  <span className="text-cyan-400 font-bold font-mono">
                    {features.file_docf_mean_file_nwords.toLocaleString()} words
                  </span>
                </div>
                <input
                  type="range"
                  min="100"
                  max="8000"
                  step="250"
                  value={features.file_docf_mean_file_nwords}
                  onChange={(e) => handleSliderChange("file_docf_mean_file_nwords", Number(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>
            </div>

            {/* Psychometric Traits */}
            <div className="space-y-4 pt-2 border-t border-slate-800/80">
              <div className="space-y-1.5">
                <div className="flex justify-between font-medium">
                  <span className="text-slate-300">Conscientiousness (C - OCEAN Personality Score)</span>
                  <span className="text-slate-400 font-mono">{features.C} / 50</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="50"
                  step="1"
                  value={features.C}
                  onChange={(e) => handleSliderChange("C", Number(e.target.value))}
                  className="w-full accent-slate-400 cursor-pointer"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Live Risk Verdict & Base Models (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Main Verdict Card */}
          <div
            className="rounded-xl p-6 border transition-all"
            style={{
              backgroundColor: "#0E1526",
              borderColor: result ? result.risk_color : "#1E293B",
              boxShadow: result && result.is_threat ? `0 0 30px -8px ${result.risk_color}40` : "none",
            }}
          >
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Stacking Ensemble Score
              </span>
              <span className="text-xs font-mono text-slate-500">
                Threshold: T* = 0.34
              </span>
            </div>

            {/* Probability Gauge */}
            <div className="py-6 text-center space-y-2">
              <div className="text-5xl sm:text-6xl font-black tracking-tight" style={{ color: result?.risk_color || "#F1F5F9" }}>
                {probPercent}%
              </div>
              <div className="text-xs text-slate-400 font-medium uppercase tracking-wider">
                Predicted Malicious Probability
              </div>

              {/* Threshold Progress Bar */}
              <div className="pt-3 max-w-xs mx-auto">
                <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden p-0.5 border border-slate-800 relative">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                      width: `${Math.min(100, Math.max(2, probPercent))}%`,
                      backgroundColor: result?.risk_color || "#10B981",
                    }}
                  />
                  {/* Threshold vertical line indicator at 34% */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-white/70"
                    style={{ left: "34%" }}
                    title="Optimal Threshold T* = 0.34"
                  />
                </div>
                <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>0%</span>
                  <span className="text-cyan-400 font-mono">T* = 34%</span>
                  <span>100%</span>
                </div>
              </div>
            </div>

            {/* Verdict Badge */}
            <div className="pt-2 text-center">
              <div
                className="inline-block px-4 py-1.5 rounded-lg text-xs font-extrabold uppercase tracking-wider"
                style={{
                  backgroundColor: `${result?.risk_color}20`,
                  color: result?.risk_color,
                  border: `1px solid ${result?.risk_color}50`,
                }}
              >
                {result?.risk_level}
              </div>
              <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                {result?.recommendation}
              </p>
            </div>
          </div>

          {/* Base Models Breakdown */}
          <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 space-y-3 text-xs">
            <h3 className="font-bold text-slate-100 flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <span>Base Classifier Risk Vectors</span>
            </h3>

            {result && (
              <div className="space-y-2.5 pt-1">
                {/* XGBoost */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span className="text-slate-300 font-medium">XGBoost (Weight: 70%)</span>
                    <span className="font-mono font-bold text-orange-400">
                      {(result.base_model_probabilities.xgboost * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-orange-500 rounded-full"
                      style={{ width: `${result.base_model_probabilities.xgboost * 100}%` }}
                    />
                  </div>
                </div>

                {/* CatBoost */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span className="text-slate-300 font-medium">CatBoost (Weight: 30%)</span>
                    <span className="font-mono font-bold text-emerald-400">
                      {(result.base_model_probabilities.catboost * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full"
                      style={{ width: `${result.base_model_probabilities.catboost * 100}%` }}
                    />
                  </div>
                </div>

                {/* Random Forest */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span className="text-slate-300 font-medium">Random Forest</span>
                    <span className="font-mono font-bold text-blue-400">
                      {(result.base_model_probabilities.random_forest * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full"
                      style={{ width: `${result.base_model_probabilities.random_forest * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Active Detected Threat Signals */}
          <div className="rounded-xl bg-slate-900/80 border border-slate-800 p-5 space-y-3 text-xs">
            <h3 className="font-bold text-slate-100 flex items-center space-x-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span>Active Behavioral Indicators</span>
            </h3>

            {result && (
              <div className="space-y-2 pt-1">
                {result.key_signals.map((sig, idx) => (
                  <div key={idx} className="flex items-start space-x-2 text-slate-300">
                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 shrink-0" />
                    <span>{sig}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
