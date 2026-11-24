"use client";

import { usePathname } from "next/navigation";
import { ChevronRight, Database, ChevronDown, Check, Filter, Layers } from "lucide-react";
import { useState } from "react";
import { useDataset, DatasetType } from "@/context/DatasetContext";

const ROUTE_NAMES: { [key: string]: string } = {
  "/": "Overview",
  "/queue": "Session Triage Queue",
  "/simulator": "Live Threat Simulator",
  "/results": "Model Benchmarks & CV Analysis",
  "/features": "Threat Behavioral Signals",
  "/methodology": "Scientific Methodology",
};

export default function Header() {
  const pathname = usePathname();
  const { dataset, setDataset } = useDataset();
  const [datasetMenuOpen, setDatasetMenuOpen] = useState(false);

  const currentPage = ROUTE_NAMES[pathname] || "Dashboard";

  const getOptThreshold = () => {
    switch (dataset) {
      case "r4.2":
        return "0.34";
      case "r5.2":
        return "0.42";
      case "r6.2":
        return "0.35";
      case "cross_release":
        return "Ensemble";
      default:
        return "0.34";
    }
  };

  const getDatasetLabel = () => {
    switch (dataset) {
      case "r4.2":
        return "CERT r4.2";
      case "r5.2":
        return "CERT r5.2";
      case "r6.2":
        return "CERT r6.2";
      case "cross_release":
        return "Cross-Release";
      default:
        return "CERT r4.2";
    }
  };

  const options: { id: DatasetType; label: string; desc: string }[] = [
    {
      id: "r4.2",
      label: "CMU CERT r4.2",
      desc: "470,611 sessions · 1,000 employees · 421.5:1",
    },
    {
      id: "r5.2",
      label: "CMU CERT r5.2",
      desc: "1,004,460 sessions · 2,000 employees · 543.1:1",
    },
    {
      id: "r6.2",
      label: "CMU CERT r6.2",
      desc: "1,949,073 sessions · 4,000 employees · 20,515:1",
    },
    {
      id: "cross_release",
      label: "Cross-Release Analysis",
      desc: "Unified 3-release benchmark comparison",
    },
  ];

  return (
    <header className="h-14 border-b border-slate-200 bg-white sticky top-0 z-30 flex items-center justify-between px-6">
      {/* Left: Breadcrumbs */}
      <div className="flex items-center space-x-2 text-xs">
        <span className="font-semibold text-slate-500">SENTINEL</span>
        <ChevronRight className="w-3.5 h-3.5 text-slate-300" />
        <span className="font-semibold text-slate-900">{currentPage}</span>
      </div>

      {/* Right: Operational Controls */}
      <div className="flex items-center space-x-3">
        {/* Operational Threshold Badge */}
        <div className="hidden sm:flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-slate-50 border border-slate-200 text-xs">
          <Filter className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-500 font-medium">Optimal Cutoff:</span>
          <span className="font-mono font-bold text-slate-800">
            {dataset === "cross_release" ? "Multi-Cutoff" : `T* = ${getOptThreshold()}`}
          </span>
        </div>

        {/* Dataset Switcher Dropdown */}
        <div className="relative">
          <button
            onClick={() => setDatasetMenuOpen(!datasetMenuOpen)}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-md bg-white border border-slate-200 hover:border-slate-300 hover:bg-slate-50 text-xs text-slate-700 font-medium transition-all shadow-2xs cursor-pointer"
          >
            {dataset === "cross_release" ? (
              <Layers className="w-3.5 h-3.5 text-indigo-600" />
            ) : (
              <Database className="w-3.5 h-3.5 text-blue-600" />
            )}
            <span>{getDatasetLabel()}</span>
            <ChevronDown className="w-3 h-3 text-slate-400" />
          </button>

          {datasetMenuOpen && (
            <div className="absolute right-0 mt-1.5 w-72 rounded-lg bg-white border border-slate-200 shadow-lg py-1 z-50 text-xs">
              <div className="px-3 py-1.5 border-b border-slate-100 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                Select Benchmark Dataset
              </div>

              {options.map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => {
                    setDataset(opt.id);
                    setDatasetMenuOpen(false);
                  }}
                  className={`w-full text-left px-3.5 py-2.5 flex items-center justify-between hover:bg-slate-50 transition-colors cursor-pointer ${
                    dataset === opt.id
                      ? "bg-blue-50/60 text-blue-700 font-semibold"
                      : "text-slate-700"
                  }`}
                >
                  <div>
                    <div className="font-medium text-slate-900">{opt.label}</div>
                    <div className="text-[10px] text-slate-500 font-mono mt-0.5">{opt.desc}</div>
                  </div>
                  {dataset === opt.id && <Check className="w-4 h-4 text-blue-600 shrink-0 ml-2" />}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
