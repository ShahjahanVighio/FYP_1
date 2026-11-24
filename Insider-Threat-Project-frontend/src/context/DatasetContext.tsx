"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import r4Data from "@/data/r4_2_metrics.json";
import r5Data from "@/data/r5_2_metrics.json";
import r6Data from "@/data/r6_2_metrics.json";
import benchmarksData from "@/data/benchmarks.json";

export type DatasetType = "r4.2" | "r5.2" | "r6.2" | "cross_release";

interface DatasetContextType {
  dataset: DatasetType;
  setDataset: (d: DatasetType) => void;
  data: typeof r4Data;
  benchmarks: typeof benchmarksData;
  plotPrefix: string;
  isCrossRelease: boolean;
}

const DatasetContext = createContext<DatasetContextType>({
  dataset: "r4.2",
  setDataset: () => {},
  data: r4Data,
  benchmarks: benchmarksData,
  plotPrefix: "/plots/r4_2",
  isCrossRelease: false,
});

export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const [dataset, setDatasetState] = useState<DatasetType>("r4.2");

  useEffect(() => {
    const saved = localStorage.getItem("selected_dataset") as DatasetType;
    if (saved === "r4.2" || saved === "r5.2" || saved === "r6.2" || saved === "cross_release") {
      setDatasetState(saved);
    }
  }, []);

  const setDataset = (d: DatasetType) => {
    setDatasetState(d);
    try {
      localStorage.setItem("selected_dataset", d);
    } catch {
      // LocalStorage error handling for incognito/restricted mode
    }
  };

  // Determine active metrics object
  let currentData: typeof r4Data = r4Data;
  if (dataset === "r5.2") {
    currentData = r5Data as unknown as typeof r4Data;
  } else if (dataset === "r6.2") {
    currentData = r6Data as unknown as typeof r4Data;
  } else if (dataset === "cross_release") {
    currentData = r6Data as unknown as typeof r4Data; // default fallback for charts
  }

  const plotPrefix =
    dataset === "r4.2"
      ? "/plots/r4_2"
      : dataset === "r5.2"
      ? "/plots/r5_2"
      : "/plots/r6_2";

  return (
    <DatasetContext.Provider
      value={{
        dataset,
        setDataset,
        data: currentData,
        benchmarks: benchmarksData,
        plotPrefix,
        isCrossRelease: dataset === "cross_release",
      }}
    >
      {children}
    </DatasetContext.Provider>
  );
}

export function useDataset() {
  return useContext(DatasetContext);
}
