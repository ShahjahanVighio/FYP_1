"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Inbox,
  BarChart3,
  Sliders,
  Terminal,
  BookOpen,
  Shield,
  Layers,
  Database,
  CheckCircle2,
  Users
} from "lucide-react";
import { useDataset, DatasetType } from "@/context/DatasetContext";

const NAV_OPERATIONS = [
  { name: "Overview", href: "/", icon: LayoutDashboard },
  { name: "Triage Queue", href: "/queue", icon: Inbox, badge: "Live" },
  { name: "Threat Simulator", href: "/simulator", icon: Terminal },
];

const NAV_ANALYTICS = [
  { name: "Model Benchmarks", href: "/results", icon: BarChart3 },
  { name: "Behavioral Signals", href: "/features", icon: Sliders },
  { name: "Methodology & Science", href: "/methodology", icon: BookOpen },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { dataset, setDataset } = useDataset();

  const getSubtext = () => {
    switch (dataset) {
      case "r4.2":
        return "470,611 sessions · 1,000 users · 421.5:1";
      case "r5.2":
        return "1,004,460 sessions · 2,000 users · 543.1:1";
      case "r6.2":
        return "1,949,073 sessions · 4,000 users · 20,515:1";
      case "cross_release":
        return "3,424,144 sessions · 3 benchmark releases";
      default:
        return "470,611 sessions · 1,000 users";
    }
  };

  return (
    <aside className="w-60 flex-shrink-0 bg-white border-r border-slate-200 flex flex-col h-screen sticky top-0 z-40 select-none">
      {/* Platform Branding */}
      <div className="h-14 border-b border-slate-200 px-4 flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-slate-900 flex items-center justify-center text-white shadow-sm">
            <Shield className="w-4 h-4 text-blue-400" />
          </div>
          <div>
            <div className="flex items-center space-x-1.5">
              <span className="font-bold text-sm tracking-tight text-slate-900">
                SENTINEL
              </span>
              <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200">
                UEBA
              </span>
            </div>
            <p className="text-[10px] text-slate-500 font-medium">
              Enterprise SOC Platform
            </p>
          </div>
        </div>
      </div>

      {/* Navigation Sections */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
        {/* Operations */}
        <div>
          <div className="px-2 mb-2 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Operations
          </div>
          <nav className="space-y-1">
            {NAV_OPERATIONS.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center justify-between px-2.5 py-2 rounded-md text-xs font-medium transition-colors ${
                    isActive
                      ? "bg-slate-100 text-blue-700 font-semibold"
                      : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                  }`}
                >
                  <div className="flex items-center space-x-2.5">
                    <Icon className={`w-4 h-4 ${isActive ? "text-blue-700" : "text-slate-400"}`} />
                    <span>{item.name}</span>
                  </div>
                  {item.badge && (
                    <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Analytics & Research */}
        <div>
          <div className="px-2 mb-2 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Analytics & Models
          </div>
          <nav className="space-y-1">
            {NAV_ANALYTICS.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center justify-between px-2.5 py-2 rounded-md text-xs font-medium transition-colors ${
                    isActive
                      ? "bg-slate-100 text-blue-700 font-semibold"
                      : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                  }`}
                >
                  <div className="flex items-center space-x-2.5">
                    <Icon className={`w-4 h-4 ${isActive ? "text-blue-700" : "text-slate-400"}`} />
                    <span>{item.name}</span>
                  </div>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Dataset Quick-Switch Card */}
        <div className="p-3 rounded-lg border border-slate-200 bg-slate-50/70 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold text-slate-700">
            <span className="flex items-center space-x-1.5">
              <Database className="w-3.5 h-3.5 text-slate-500" />
              <span>Target Benchmark</span>
            </span>
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-500" />
          </div>

          <div className="grid grid-cols-3 gap-1">
            {(["r4.2", "r5.2", "r6.2"] as DatasetType[]).map((ds) => (
              <button
                key={ds}
                onClick={() => setDataset(ds)}
                className={`py-1 px-1.5 rounded text-[11px] font-semibold text-center border transition-all cursor-pointer ${
                  dataset === ds
                    ? "bg-white text-blue-700 border-blue-300 shadow-xs"
                    : "bg-transparent text-slate-500 border-transparent hover:bg-white/60 hover:text-slate-700"
                }`}
              >
                {ds}
              </button>
            ))}
          </div>

          <button
            onClick={() => setDataset("cross_release")}
            className={`w-full py-1 px-2 rounded text-[11px] font-semibold text-center border transition-all cursor-pointer flex items-center justify-center space-x-1.5 ${
              dataset === "cross_release"
                ? "bg-white text-indigo-700 border-indigo-300 shadow-xs"
                : "bg-transparent text-slate-500 border-transparent hover:bg-white/60 hover:text-slate-700"
            }`}
          >
            <Layers className="w-3 h-3" />
            <span>Cross-Release View</span>
          </button>

          <div className="text-[10px] text-slate-500 font-mono leading-tight">
            {getSubtext()}
          </div>
        </div>
      </div>

      {/* Sidebar Footer */}
      <div className="border-t border-slate-200 p-3 bg-slate-50/50 space-y-1.5">
        <div className="flex items-center justify-between text-[11px] text-slate-500">
          <div className="flex items-center space-x-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            <span className="font-medium text-slate-600">CERT Pipeline v2.4</span>
          </div>
          <span className="font-mono text-[10px] text-slate-400">SOC Ready</span>
        </div>

        <div className="text-[10px] text-slate-500 font-medium">
          Muteeb · Shahjahan · Bilal
        </div>
        <div className="text-[9px] text-slate-400">
          Salim Habib University
        </div>
      </div>
    </aside>
  );
}
