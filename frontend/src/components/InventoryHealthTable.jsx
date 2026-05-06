import { useEffect, useState } from "react";
import { fetchInventoryHealth } from "../services/api";

const ACTION_STYLES = {
  Hold: "bg-emerald-100 text-emerald-800",
  Monitor: "bg-amber-100 text-amber-800",
  Discount: "bg-rose-100 text-rose-800",
};

export default function InventoryHealthTable() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    const loadRows = async () => {
      try {
        setLoading(true);
        setError("");
        const data = await fetchInventoryHealth();
        if (active) {
          setRows(data);
        }
      } catch {
        if (active) {
          setError("Unable to load inventory health data.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    loadRows();
    return () => {
      active = false;
    };
  }, []);

  return (
    <section className="rounded-[28px] bg-white p-5 shadow-panel md:p-7">
      <div className="mb-6">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-400">Inventory</p>
        <h2 className="mt-2 text-2xl font-semibold text-slateBrand">Inventory Health</h2>
        <p className="mt-1 text-sm text-slate-500">
          Tiering products for action-oriented assortment decisions.
        </p>
      </div>

      {loading ? (
        <div className="text-slate-500">Loading inventory health...</div>
      ) : error ? (
        <div className="rounded-3xl border border-rose-200 bg-rose-50 px-4 py-6 text-rose-700">
          {error}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead>
              <tr className="text-left text-slate-500">
                <th className="pb-3 pr-4 font-medium">Product</th>
                <th className="pb-3 pr-4 font-medium">Category</th>
                <th className="pb-3 pr-4 font-medium">Turnover</th>
                <th className="pb-3 pr-4 font-medium">Velocity</th>
                <th className="pb-3 pr-4 font-medium">Tier</th>
                <th className="pb-3 font-medium">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rows.map((row) => (
                <tr key={row.product_id} className="align-top">
                  <td className="py-4 pr-4">
                    <div className="font-medium text-slateBrand">{row.product_name}</div>
                    <div className="text-xs text-slate-400">{row.product_id}</div>
                  </td>
                  <td className="py-4 pr-4 text-slate-600">{row.category}</td>
                  <td className="py-4 pr-4 text-slate-600">{row.inventory_turnover}</td>
                  <td className="py-4 pr-4 text-slate-600">{row.sales_velocity}</td>
                  <td className="py-4 pr-4 text-slate-600">{row.tier}</td>
                  <td className="py-4">
                    <span
                      className={`inline-flex rounded-full px-3 py-1 text-xs font-medium ${
                        ACTION_STYLES[row.action]
                      }`}
                    >
                      {row.action}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
