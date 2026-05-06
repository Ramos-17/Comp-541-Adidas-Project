import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchClusterAnalysis } from "../services/api";

const COLORS = {
  "Premium Loyalists": "#102a43",
  "Promotion Driven": "#ef8354",
  "Core Value": "#3c6e71",
};

export default function ClusterView() {
  const [points, setPoints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    const loadPoints = async () => {
      try {
        setLoading(true);
        setError("");
        const data = await fetchClusterAnalysis();
        if (active) {
          setPoints(data);
        }
      } catch {
        if (active) {
          setError("Unable to load cluster analysis data.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    loadPoints();
    return () => {
      active = false;
    };
  }, []);

  const legendItems = useMemo(() => Object.keys(COLORS), []);

  return (
    <section className="rounded-[28px] bg-white p-5 shadow-panel md:p-7">
      <div className="mb-6">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Clusters</p>
        <h2 className="mt-2 text-2xl font-semibold text-slateBrand">Segment Analysis</h2>
        <p className="mt-1 text-sm text-slate-500">
          Revenue-per-unit vs profit-per-unit segments for product storytelling.
        </p>
      </div>

      {loading ? (
        <div className="text-slate-500">Loading cluster analysis...</div>
      ) : error ? (
        <div className="rounded-3xl border border-rose-200 bg-rose-50 px-4 py-6 text-rose-700">
          {error}
        </div>
      ) : (
        <>
          <div className="mb-4 flex flex-wrap gap-3">
            {legendItems.map((item) => (
              <div key={item} className="inline-flex items-center gap-2 rounded-full bg-slate-50 px-3 py-2 text-xs text-slate-600">
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: COLORS[item] }} />
                {item}
              </div>
            ))}
          </div>

          <div className="h-[380px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 12, right: 12, bottom: 10, left: 0 }}>
                <CartesianGrid stroke="#d9e2ec" strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Revenue / Unit"
                  stroke="#486581"
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Profit / Unit"
                  stroke="#486581"
                />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Legend />
                <Scatter name="Products" data={points}>
                  {points.map((entry) => (
                    <Cell key={entry.product_id} fill={COLORS[entry.cluster] || "#829ab1"} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </section>
  );
}
