import Link from "next/link";
import { Shield, GitBranch, ExternalLink } from "lucide-react";

export default function Footer() {
  return (
    <footer className="w-full border-t border-slate-800 bg-[#060A12] py-8 text-xs text-slate-500">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center space-x-2">
          <Shield className="w-4 h-4 text-cyan-400" />
          <span className="text-slate-300 font-semibold">
            SENTINEL UEBA · Final Year Project
          </span>
          <span className="text-slate-600">|</span>
          <span>Carnegie Mellon University (CMU SEI) CERT r4.2 Benchmark</span>
        </div>

        <div className="flex items-center space-x-6 text-slate-400">
          <Link
            href="https://github.com/ShahjahanVighio/FYP_1"
            target="_blank"
            className="flex items-center space-x-1.5 hover:text-cyan-400 transition-colors"
          >
            <GitBranch className="w-3.5 h-3.5" />
            <span>GitHub Repository</span>
          </Link>

          <Link
            href="https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099"
            target="_blank"
            className="flex items-center space-x-1 hover:text-cyan-400 transition-colors"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            <span>CMU SEI Dataset</span>
          </Link>
        </div>
      </div>
    </footer>
  );
}
