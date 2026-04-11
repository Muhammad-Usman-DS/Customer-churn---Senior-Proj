import { useState, useEffect } from "react";

const RISK_COLORS = {
  High: "bg-red-100 text-red-700",
  Medium: "bg-amber-100 text-amber-700",
  Low: "bg-green-100 text-green-700",
};

function exportCSV(rows) {
  if (!rows.length) return;
  const keys = Object.keys(rows[0]);
  const csv = [keys.join(","), ...rows.map((r) => keys.map((k) => r[k]).join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "churn_predictions.csv";
  a.click();
}

export default function History() {
  const [records, setRecords] = useState([]);
  const [filter, setFilter] = useState("All");

  useEffect(() => {
    const raw = JSON.parse(localStorage.getItem("churn_history") || "[]");
    setRecords(raw);
  }, []);

  function clearHistory() {
    if (window.confirm("Clear all prediction history?")) {
      localStorage.removeItem("churn_history");
      setRecords([]);
    }
  }

  const filtered =
    filter === "All" ? records : records.filter((r) => r.risk_level === filter);

  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">Prediction History</h1>
          <p className="text-slate-500 text-sm mt-1">
            All past predictions stored locally in your browser.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => exportCSV(filtered)}
            disabled={!filtered.length}
            className="px-4 py-2 text-sm font-medium bg-white border border-gray-200 rounded-lg hover:bg-gray-50 disabled:opacity-40 transition-colors"
          >
            Export CSV
          </button>
          <button
            onClick={clearHistory}
            disabled={!records.length}
            className="px-4 py-2 text-sm font-medium text-red-600 bg-white border border-red-200 rounded-lg hover:bg-red-50 disabled:opacity-40 transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-5">
        {["All", "High", "Medium", "Low"].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              filter === f
                ? "bg-slate-800 text-white"
                : "bg-white border border-gray-200 text-slate-600 hover:bg-gray-50"
            }`}
          >
            {f}
          </button>
        ))}
        <span className="ml-auto text-sm text-slate-400 self-center">
          {filtered.length} record{filtered.length !== 1 ? "s" : ""}
        </span>
      </div>

      {filtered.length === 0 ? (
        <div className="bg-white rounded-xl border border-dashed border-gray-300 p-12 text-center">
          <p className="text-slate-400">
            {records.length === 0
              ? "No predictions yet. Go to Predict to get started."
              : "No records match this filter."}
          </p>
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100 bg-gray-50">
                  <th className="text-left px-4 py-3 font-medium text-slate-500">Date</th>
                  <th className="text-left px-4 py-3 font-medium text-slate-500">Contract</th>
                  <th className="text-left px-4 py-3 font-medium text-slate-500">Internet</th>
                  <th className="text-left px-4 py-3 font-medium text-slate-500">Tenure</th>
                  <th className="text-left px-4 py-3 font-medium text-slate-500">Monthly $</th>
                  <th className="text-left px-4 py-3 font-medium text-slate-500">Probability</th>
                  <th className="text-left px-4 py-3 font-medium text-slate-500">Risk</th>
                  <th className="text-left px-4 py-3 font-medium text-slate-500">Verdict</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr key={r.id} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-3 text-slate-500 text-xs">
                      {new Date(r.date).toLocaleDateString()}{" "}
                      {new Date(r.date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </td>
                    <td className="px-4 py-3 text-slate-700">{r.Contract}</td>
                    <td className="px-4 py-3 text-slate-700">{r.InternetService}</td>
                    <td className="px-4 py-3 text-slate-700">{r.tenure} mo</td>
                    <td className="px-4 py-3 text-slate-700">${r.MonthlyCharges}</td>
                    <td className="px-4 py-3 font-semibold text-slate-800">{r.probability}%</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${RISK_COLORS[r.risk_level]}`}>
                        {r.risk_level}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-700">{r.prediction}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
