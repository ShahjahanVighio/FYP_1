"use client";

import { useState, useMemo } from "react";
import {
  Sliders,
  Search,
  Filter,
  BarChart2,
  Tag,
  ArrowUpDown,
  ShieldAlert,
  Info,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useDataset } from "@/context/DatasetContext";

export default function FeaturesPage() {
  const { data, dataset } = useDataset();
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [sortField, setSortField] = useState<"combined_rank" | "anova_f" | "mi">("combined_rank");
  const [sortAsc, setSortAsc] = useState(true);

  const features = data.top_50_features;

  // Extract unique categories
  const categories = useMemo(() => {
    const set = new Set(features.map((f) => f.category));
    return ["All", ...Array.from(set)];
  }, [features]);

  // Filter & Sort
  const filteredFeatures = useMemo(() => {
    return features
      .filter((f) => {
        const matchesCat = selectedCategory === "All" || f.category === selectedCategory;
        const matchesSearch =
          f.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          f.category.toLowerCase().includes(searchTerm.toLowerCase());
        return matchesCat && matchesSearch;
      })
      .sort((a, b) => {
        const valA = a[sortField];
        const valB = b[sortField];
        if (sortAsc) {
          return valA > valB ? 1 : -1;
        } else {
          return valA < valB ? 1 : -1;
        }
      });
  }, [features, searchTerm, selectedCategory, sortField, sortAsc]);

  // Top 12 for horizontal bar chart
  const top12ChartData = useMemo(() => {
    return features.slice(0, 12).map((f) => ({
      name: f.name.length > 20 ? f.name.slice(0, 18) + ".." : f.name,
      fullName: f.name,
      "ANOVA F-Score": Math.round(f.anova_f),
      category: f.category,
    }));
  }, [features]);

  const toggleSort = (field: "combined_rank" | "anova_f" | "mi") => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(field === "combined_rank"); // rank ascending is better (1, 2, 3...)
    }
  };

  const getCategoryColor = (cat: string) => {
    switch (cat) {
      case "Removable Media / USB":
        return "bg-rose-500/10 text-rose-400 border-rose-500/30";
      case "Data Leak Sites / Cloud":
        return "bg-cyan-500/10 text-cyan-400 border-cyan-500/30";
      case "Job Search Activity":
        return "bg-amber-500/10 text-amber-400 border-amber-500/30";
      case "File Exfiltration":
        return "bg-purple-500/10 text-purple-400 border-purple-500/30";
      case "Hacking / Keylogger":
        return "bg-red-500/10 text-red-400 border-red-500/30";
      case "Psychometric (OCEAN)":
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
      default:
        return "bg-slate-500/10 text-slate-400 border-slate-500/30";
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="space-y-2">
        <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-md text-xs font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
          <Sliders className="w-3.5 h-3.5" />
          <span>Statistical Feature Selection</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-100 tracking-tight">
          Top 50 Behavioral Threat Signals
        </h1>
        <p className="text-sm text-slate-400 max-w-3xl leading-relaxed">
          Dual-criterion ranking combining linear ANOVA F-statistic and non-linear Mutual Information fitted strictly on the training partition to eliminate noisy predictors and prevent user identity memorization.
        </p>
      </section>

      {/* Top 12 Bar Chart */}
      <section className="rounded-xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
              <BarChart2 className="w-4 h-4 text-cyan-400" />
              <span>Highest Ranked Predictors (ANOVA F-Statistic Separation)</span>
            </h2>
            <p className="text-xs text-slate-400">
              Top 12 engineered features showing highest discriminative power between benign sessions and insider threats
            </p>
          </div>
          <span className="text-xs text-slate-500 font-mono">F-Score Scale: Logarithmic Impact</span>
        </div>

        <div className="h-72 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={top12ChartData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
            >
              <XAxis type="number" stroke="#64748B" fontSize={11} tickLine={false} />
              <YAxis
                type="category"
                dataKey="name"
                stroke="#94A3B8"
                fontSize={11}
                tickLine={false}
                width={120}
              />
              <Tooltip
                contentStyle={{ backgroundColor: "#0F172A", borderColor: "#334155", borderRadius: "8px", fontSize: "12px" }}
                formatter={(val: any) => [val.toLocaleString(), "ANOVA F-Score"]}
              />
              <Bar dataKey="ANOVA F-Score" fill="#06B6D4" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Interactive Table Section */}
      <section className="rounded-xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
        {/* Search & Filter Bar */}
        <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-4">
          {/* Search Input */}
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search features (e.g. usb, leak, job, file, OCEAN)..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
          </div>

          {/* Category Filter Pills */}
          <div className="flex flex-wrap items-center gap-1.5 overflow-x-auto text-xs">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors whitespace-nowrap ${
                  selectedCategory === cat
                    ? "bg-cyan-500 text-slate-950 font-bold"
                    : "bg-slate-950 border border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Features Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
                <th
                  onClick={() => toggleSort("combined_rank")}
                  className="py-3 px-3 cursor-pointer hover:text-cyan-400 select-none"
                >
                  <div className="flex items-center space-x-1">
                    <span>Rank</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-3">Feature Name</th>
                <th className="py-3 px-3">Behavioral Domain</th>
                <th
                  onClick={() => toggleSort("anova_f")}
                  className="py-3 px-3 text-right cursor-pointer hover:text-cyan-400 select-none"
                >
                  <div className="flex items-center justify-end space-x-1">
                    <span>ANOVA F-Score</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => toggleSort("mi")}
                  className="py-3 px-3 text-right cursor-pointer hover:text-cyan-400 select-none"
                >
                  <div className="flex items-center justify-end space-x-1">
                    <span>Mutual Info (MI)</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-3 text-center">F Rank</th>
                <th className="py-3 px-3 text-center">MI Rank</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredFeatures.map((f) => (
                <tr key={f.name} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-2.5 px-3 font-bold text-cyan-400">
                    #{f.combined_rank.toFixed(1)}
                  </td>
                  <td className="py-2.5 px-3 font-bold text-slate-100">
                    {f.name}
                  </td>
                  <td className="py-2.5 px-3 font-sans">
                    <span
                      className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold border ${getCategoryColor(
                        f.category
                      )}`}
                    >
                      {f.category}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right text-slate-200">
                    {f.anova_f.toLocaleString(undefined, { maximumFractionDigits: 1 })}
                  </td>
                  <td className="py-2.5 px-3 text-right text-emerald-400">
                    {f.mi.toFixed(6)}
                  </td>
                  <td className="py-2.5 px-3 text-center text-slate-400">
                    #{f.anova_rank}
                  </td>
                  <td className="py-2.5 px-3 text-center text-slate-400">
                    #{f.mi_rank}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="text-[11px] text-slate-500 pt-2 flex items-center justify-between">
          <span>Showing {filteredFeatures.length} of {features.length} selected features.</span>
          <span>F-Score measures linear separation; Mutual Information captures non-linear dependency.</span>
        </div>
      </section>
    </div>
  );
}
