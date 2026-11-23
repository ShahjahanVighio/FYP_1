import type { Metadata } from "next";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import { DatasetProvider } from "@/context/DatasetContext";
import "./globals.css";

export const metadata: Metadata = {
  title: "SENTINEL UEBA | Enterprise Insider Threat & Risk Analytics",
  description: "An Ensemble Machine Learning Approach to Insider Risk Classification Using Cultural Metrics and System Behaviors",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-slate-50 text-slate-900 antialiased flex min-h-screen">
        <DatasetProvider>
          <Sidebar />
          <div className="flex-1 flex flex-col min-w-0 min-h-screen overflow-x-hidden">
            <Header />
            <main className="flex-1 p-6 md:p-8 max-w-7xl w-full mx-auto">
              {children}
            </main>
          </div>
        </DatasetProvider>
      </body>
    </html>
  );
}
