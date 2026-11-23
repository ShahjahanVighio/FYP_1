"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import r4Data from "@/data/r4_2_metrics.json";
import r5Data from "@/data/r5_2_metrics.json";

export type DatasetType = "r4.2" | "r5.2";

interface DatasetContextType {
  dataset: DatasetType;
  setDataset: (d: DatasetType) => void;
  data: typeof r4Data;
  plotPrefix: string;
}

const DatasetContext = createContext<DatasetContextType>({
  dataset: "r4.2",
  setDataset: () => {},
  data: r4Data,
  plotPrefix: "/plots/r4_2",
});

export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const [dataset, setDatasetState] = useState<DatasetType>("r4.2");

  useEffect(() => {
    const saved = localStorage.getItem("selected_dataset") as DatasetType;
    if (saved === "r4.2" || saved === "r5.2") {
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

  const currentData = dataset === "r4.2" ? r4Data : (r5Data as typeof r4Data);
  const plotPrefix = dataset === "r4.2" ? "/plots/r4_2" : "/plots/r5_2";

  return (
    <DatasetContext.Provider value={{ dataset, setDataset, data: currentData, plotPrefix }}>
      {children}
    </DatasetContext.Provider>
  );
}

export function useDataset() {
  return useContext(DatasetContext);
}
