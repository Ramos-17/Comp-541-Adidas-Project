import { useState } from "react";
import Sidebar from "../components/Sidebar";
import MainView from "../components/MainView";

export default function Dashboard() {
  const [activeView, setActiveView] = useState("forecast");
  const [selectedYear, setSelectedYear] = useState(2025);

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(239,131,84,0.16),_transparent_32%),linear-gradient(180deg,#f7f3eb_0%,#fffdf9_100%)] px-4 py-6 md:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 lg:flex-row">
        <Sidebar activeView={activeView} onChange={setActiveView} />

        <div className="flex-1">
          <div className="mb-5 rounded-[28px] bg-white/70 px-5 py-4 shadow-panel backdrop-blur">
            <p className="text-sm text-slate-500">Demo</p>
            <h2 className="mt-1 text-3xl font-semibold text-slateBrand">
              Forecasting and assortment decisions
            </h2>
            <p className="mt-2 max-w-3xl text-sm text-slate-500">
              This dashboard presents a compact story:
              future trend, current inventory health, and product segments.
            </p>
          </div>

          <MainView
            activeView={activeView}
            selectedYear={selectedYear}
            onYearChange={setSelectedYear}
          />
        </div>
      </div>
    </main>
  );
}
