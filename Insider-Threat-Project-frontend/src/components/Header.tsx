"use client";

import { usePathname } from "next/navigation";
import { ChevronRight, Database, ChevronDown, Check, ShieldCheck, Filter } from "lucide-react";
import { useState } from "react";
import { useDataset } from "@/context/DatasetContext";

const ROUTE_NAMES: { [key: string]: string } = {
  "/": "Overview",
  "/queue": "Session Triage Queue",
  "/simulator": "Live Threat Simulator",
  "/results": "Model Benchmarks & ROC Analysis",
  "/features": "Threat Behavioral Signals",
  "/methodology": "Scientific Methodology",
};

export default function Header() {
  const pathname = usePathname();
  const { dataset, setDataset, data } = useDataset();
  const [datasetMenuOpen, setDatasetMenuOpen] = useState(false);

  const currentPage = ROUTE_NAMES[pathname] || "Dashboard";
  const optThreshold = dataset === "r4.2" ? "0.34" : "0.42";

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
          <span className="font-mono font-bold text-slate-800">T* = {optThreshold}</span>
        </div>

        {/* Dataset Switcher Dropdown */}
        <div className="relative">
          <button
            onClick={() => setDatasetMenuOpen(!datasetMenuOpen)}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-md bg-white border border-slate-200 hover:border-slate-300 hover:bg-slate-50 text-xs text-slate-700 font-medium transition-all shadow-2xs cursor-pointer"
          >
            <Database className="w-3.5 h-3.5 text-blue-600" />
            <span>CMU CERT {dataset}</span>
            <ChevronDown className="w-3 h-3 text-slate-400" />
          </button>

          {datasetMenuOpen && (
            <div className="absolute right-0 mt-1.5 w-60 rounded-lg bg-white border border-slate-200 shadow-lg py-1 z-50 text-xs">
              <div className="px-3 py-1.5 border-b border-slate-100 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                Select Benchmark Dataset
              </div>

              <button
                onClick={() => {
                  setDataset("r4.2");
                  setDatasetMenuOpen(false);
                }}
                className={`w-full text-left px-3.5 py-2 flex items-center justify-between hover:bg-slate-50 transition-colors cursor-pointer ${
                  dataset === "r4.2" ? "bg-blue-50/60 text-blue-700 font-semibold" : "text-slate-700"
                }`}
              >
                <div>
                  <div className="font-medium">CMU CERT r4.2</div>
                  <div className="text-[10px] text-slate-500 font-mono">470k sessions · 1,000 users</div>
                </div>
                {dataset === "r4.2" && <Check className="w-4 h-4 text-blue-600" />}
              </button>

              <button
                onClick={() => {
                  setDataset("r5.2");
                  setDatasetMenuOpen(false);
                }}
                className={`w-full text-left px-3.5 py-2 flex items-center justify-between hover:bg-slate-50 transition-colors cursor-pointer ${
                  dataset === "r5.2" ? "bg-blue-50/60 text-blue-700 font-semibold" : "text-slate-700"
                }`}
              >
                <div>
                  <div className="font-medium">CMU CERT r5.2</div>
                  <div className="text-[10px] text-slate-500 font-mono">1.004M sessions · 2,000 users</div>
                </div>
                {dataset === "r5.2" && <Check className="w-4 h-4 text-blue-600" />}
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
