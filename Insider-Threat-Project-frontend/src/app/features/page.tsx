"use client";

import { useState, useMemo } from "react";
import {
  Sliders,
  Search,
  Filter,
  ArrowUpDown,
  Download,
  Info,
  Layers,
  Sparkles
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
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
    const set = new Set(features.map((f: any) => f.category));
    return ["All", ...Array.from(set)];
  }, [features]);

  // Filter & Sort
  const filteredFeatures = useMemo(() => {
    return features
      .filter((f: any) => {
        const matchesCat = selectedCategory === "All" || f.category === selectedCategory;
        const matchesSearch =
          f.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          f.category.toLowerCase().includes(searchTerm.toLowerCase());
        return matchesCat && matchesSearch;
      })
      .sort((a: any, b: any) => {
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
    return features.slice(0, 12).map((f: any) => ({
      name: f.name.length > 22 ? f.name.slice(0, 20) + ".." : f.name,
      fullName: f.name,
      "ANOVA F-Score": Math.round(f.anova_f),
      category: f.category,
    }));
  }, [features]);

  const handleSort = (field: "combined_rank" | "anova_f" | "mi") => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(field === "combined_rank");
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">
            Threat Behavioral Signals (Top 50 Selected)
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Statistical dual-criteria ranking: Linear discriminative power (ANOVA F-statistic) and non-linear information gain (Mutual Information) in CERT {dataset}.
          </p>
        </div>
      </div>

      {/* Top 12 Discriminative Signals Chart */}
      <div className="enterprise-card">
        <div className="enterprise-card-header flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold text-slate-900">
              Leading Behavioral Predictors (Top 12 by ANOVA F)
            </h2>
            <p className="text-[11px] text-slate-500">
              Measures variance ratio between benign and threat session clusters strictly on the training partition.
            </p>
          </div>
        </div>

        <div className="p-5">
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={top12ChartData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" horizontal={false} />
                <XAxis
                  type="number"
                  stroke="#64748B"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: "#CBD5E1" }}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  stroke="#334155"
                  fontSize={11}
                  tickLine={false}
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
                <Bar dataKey="ANOVA F-Score" fill="#1E40AF" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="enterprise-card p-3 flex flex-col md:flex-row items-center justify-between gap-3">
        {/* Category Filter Chips */}
        <div className="flex items-center space-x-1.5 w-full md:w-auto overflow-x-auto pb-1 md:pb-0">
          {categories.map((cat: any) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                selectedCategory === cat
                  ? "bg-slate-900 text-white font-semibold"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Search Input */}
        <div className="relative w-full md:w-64">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search feature name or domain..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 rounded-md border border-slate-200 bg-white text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-1 focus:ring-blue-600"
          />
        </div>
      </div>

      {/* Top 50 Signals Dense Table */}
      <div className="enterprise-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left soc-table">
            <thead>
              <tr>
                <th
                  onClick={() => handleSort("combined_rank")}
                  className="cursor-pointer hover:text-slate-900"
                >
                  <div className="flex items-center space-x-1">
                    <span>Rank</span>
                    <ArrowUpDown className="w-3 h-3 text-slate-400" />
                  </div>
                </th>
                <th>Feature Identifier</th>
                <th>Behavioral Domain</th>
                <th
                  onClick={() => handleSort("anova_f")}
                  className="text-right cursor-pointer hover:text-slate-900"
                >
                  <div className="flex items-center justify-end space-x-1">
                    <span>ANOVA F-Score</span>
                    <ArrowUpDown className="w-3 h-3 text-slate-400" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("mi")}
                  className="text-right cursor-pointer hover:text-slate-900"
                >
                  <div className="flex items-center justify-end space-x-1">
                    <span>Mutual Info</span>
                    <ArrowUpDown className="w-3 h-3 text-slate-400" />
                  </div>
                </th>
                <th className="text-right">ANOVA Rank</th>
                <th className="text-right">MI Rank</th>
              </tr>
            </thead>
            <tbody>
              {filteredFeatures.map((f: any) => (
                <tr key={f.name}>
                  <td className="font-mono text-xs font-bold text-slate-500">
                    #{f.combined_rank}
                  </td>
                  <td className="font-mono text-xs font-semibold text-blue-700">
                    {f.name}
                  </td>
                  <td>
                    <span className="text-[11px] font-medium text-slate-700 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                      {f.category}
                    </span>
                  </td>
                  <td className="text-right font-mono text-xs text-slate-900 tabular-nums font-medium">
                    {f.anova_f.toLocaleString(undefined, { maximumFractionDigits: 1 })}
                  </td>
                  <td className="text-right font-mono text-xs text-slate-900 tabular-nums font-medium">
                    {f.mi.toFixed(4)}
                  </td>
                  <td className="text-right font-mono text-xs text-slate-500 tabular-nums">
                    #{f.anova_rank}
                  </td>
                  <td className="text-right font-mono text-xs text-slate-500 tabular-nums">
                    #{f.mi_rank}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
