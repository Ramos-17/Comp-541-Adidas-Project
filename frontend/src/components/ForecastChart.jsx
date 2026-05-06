import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchForecast } from "../services/api";

export default function ForecastChart({ selectedYear, onYearChange }) {
  const [data, setData] = useState([]);
  const [modelName, setModelName] = useState("LSTM");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    const loadForecast = async () => {
      try {
        setLoading(true);
        setError("");
        const response = await fetchForecast(selectedYear);
        if (!active) {
          return;
        }
        setData(response.points);
        setModelName(response.model);
      } catch (err) {
        if (!active) {
          return;
        }
        setError(err.response?.data?.detail || "Unable to load forecast data.");
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    loadForecast();
    return () => {
      active = false;
    };
  }, [selectedYear]);

  return (
    <section className="rounded-[28px] bg-white p-5 shadow-panel md:p-7">
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Forecast</p>
          <h2 className="mt-2 text-2xl font-semibold text-slateBrand">
            Yearly Profit Trend
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Demo endpoint for the selected year using the {modelName} forecast track.
          </p>
        </div>

        <label className="flex items-center gap-3 text-sm text-slate-600">
          <span>Year</span>
          <select
            value={selectedYear}
            onChange={(event) => onYearChange(Number(event.target.value))}
            className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 outline-none focus:border-accent"
          >
            {[2023, 2024, 2025, 2026].map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
        </label>
      </div>

      {loading ? (
        <div className="flex h-[360px] items-center justify-center rounded-3xl bg-slate-50 text-slate-500">
          Loading forecast...
        </div>
      ) : error ? (
        <div className="rounded-3xl border border-rose-200 bg-rose-50 px-4 py-6 text-rose-700">
          {error}
        </div>
      ) : (
        <div className="h-[360px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 12, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d9e2ec" />
              <XAxis dataKey="month" stroke="#486581" />
              <YAxis stroke="#486581" />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="forecast"
                name="Forecast"
                stroke="#ef8354"
                strokeWidth={3}
                dot={{ r: 4 }}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey="actual"
                name="Actual"
                stroke="#3c6e71"
                strokeWidth={3}
                dot={{ r: 4 }}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}
