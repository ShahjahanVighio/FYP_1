import type { Metadata } from "next";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { DatasetProvider } from "@/context/DatasetContext";
import "./globals.css";

export const metadata: Metadata = {
  title: "SENTINEL UEBA | Insider Threat Detection Dashboard",
  description: "An Ensemble Machine Learning Approach to Insider Risk Classification Using Cultural Metrics and System Behaviors",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen flex flex-col bg-[#090D16] text-slate-100 antialiased selection:bg-cyan-500/20 selection:text-cyan-300">
        <DatasetProvider>
          <Navbar />
          <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {children}
          </main>
          <Footer />
        </DatasetProvider>
      </body>
    </html>
  );
}
