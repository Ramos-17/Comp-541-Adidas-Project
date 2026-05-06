const NAV_ITEMS = [
  { key: "forecast", label: "Forecast" },
  { key: "inventory", label: "Inventory Health" },
  { key: "clusters", label: "Cluster Analysis" },
];

export default function Sidebar({ activeView, onChange }) {
  return (
    <aside className="w-full rounded-[28px] bg-slateBrand px-5 py-6 text-white shadow-panel lg:w-72">
      <div className="mb-8">
        <p className="text-xs uppercase tracking-[0.35em] text-white/60">Adidas Demo</p>
        <h1 className="mt-3 text-2xl font-semibold">Analytics Dashboard</h1>
        <p className="mt-2 text-sm text-white/70">
          Forecasting, assortment signals, and segment views for the presentation.
        </p>
      </div>

      <nav className="space-y-3">
        {NAV_ITEMS.map((item) => {
          const isActive = item.key === activeView;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => onChange(item.key)}
              className={`w-full rounded-2xl px-4 py-3 text-left text-sm transition ${
                isActive
                  ? "bg-white text-slateBrand"
                  : "bg-white/5 text-white/80 hover:bg-white/10 hover:text-white"
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
