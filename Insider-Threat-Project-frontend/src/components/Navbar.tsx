"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldAlert, BarChart3, GitFork, Cpu, Sliders, Database, ChevronDown } from "lucide-react";
import { useState } from "react";

const NAV_ITEMS = [
  { name: "Overview", href: "/", icon: BarChart3 },
  { name: "Methodology", href: "/methodology", icon: GitFork },
  { name: "Results & Models", href: "/results", icon: Cpu },
  { name: "Threat Features", href: "/features", icon: Sliders },
  { name: "Live Simulator", href: "/simulator", icon: ShieldAlert },
];

export default function Navbar() {
  const pathname = usePathname();
  const [selectedDataset, setSelectedDataset] = useState<"r4.2" | "r5.2">("r4.2");
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800 bg-[#090D16]/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Left: Brand / Title */}
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-cyan-950/60 border border-cyan-500/30 text-cyan-400">
            <ShieldAlert className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg tracking-tight text-slate-100">
                SENTINEL<span className="text-cyan-400">.UEBA</span>
              </span>
              <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                FYP SOC
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Insider Threat Detection Ensemble
            </p>
          </div>
        </div>

        {/* Center: Navigation Links */}
        <nav className="hidden md:flex items-center space-x-1 lg:space-x-2">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? "bg-slate-800 text-cyan-400 border border-cyan-500/30 shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-cyan-400" : "text-slate-500"}`} />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>

        {/* Right: Multi-Dataset Switcher */}
        <div className="relative">
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 hover:border-slate-600 text-xs text-slate-300 transition-colors"
          >
            <Database className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-medium">
              {selectedDataset === "r4.2" ? "CMU CERT r4.2" : "CMU CERT r5.2"}
            </span>
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </button>

          {dropdownOpen && (
            <div className="absolute right-0 mt-2 w-56 rounded-xl bg-slate-900 border border-slate-700 shadow-xl py-1 z-50 text-xs">
              <button
                onClick={() => {
                  setSelectedDataset("r4.2");
                  setDropdownOpen(false);
                }}
                className={`w-full text-left px-4 py-2.5 flex items-center justify-between hover:bg-slate-800 transition-colors ${
                  selectedDataset === "r4.2" ? "text-cyan-400 font-semibold bg-cyan-950/20" : "text-slate-300"
                }`}
              >
                <div>
                  <div className="font-medium">CMU CERT r4.2</div>
                  <div className="text-[10px] text-slate-500">470k sessions · 50 features</div>
                </div>
                <span className="text-[10px] uppercase px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-bold">
                  ACTIVE
                </span>
              </button>

              <button
                onClick={() => {
                  setSelectedDataset("r5.2");
                  setDropdownOpen(false);
                }}
                className={`w-full text-left px-4 py-2.5 flex items-center justify-between hover:bg-slate-800 transition-colors ${
                  selectedDataset === "r5.2" ? "text-cyan-400 font-semibold bg-cyan-950/20" : "text-slate-300"
                }`}
              >
                <div>
                  <div className="font-medium">CMU CERT r5.2</div>
                  <div className="text-[10px] text-slate-500">2,000 employees · Extended</div>
                </div>
                <span className="text-[10px] uppercase px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 font-bold">
                  PLUG-IN READY
                </span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Mobile Nav row */}
      <div className="md:hidden flex items-center justify-around border-t border-slate-800/80 px-2 py-2 bg-slate-950">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex flex-col items-center px-2 py-1 text-[11px] ${
                isActive ? "text-cyan-400 font-semibold" : "text-slate-400"
              }`}
            >
              <Icon className="w-4 h-4 mb-0.5" />
              <span>{item.name}</span>
            </Link>
          );
        })}
      </div>
    </header>
  );
}
