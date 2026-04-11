import { useState, useRef } from "react";
import { predictBatch } from "../api";

const RISK_COLORS = {
  High: "bg-red-100 text-red-700",
  Medium: "bg-amber-100 text-amber-700",
  Low: "bg-green-100 text-green-700",
};

function exportCSV(rows) {
  if (!rows.length) return;
  const keys = Object.keys(rows[0]);
  const csv = [keys.join(","), ...rows.map((r) => keys.map((k) => `"${r[k] ?? ""}"`).join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "bulk_predictions.csv";
  a.click();
}

export default function BulkPredict() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef();

  async function handleFile(file) {
    if (!file || !file.name.endsWith(".csv")) {
      setError("Please upload a .csv file.");
      return;
    }
    setError(null);
    setResults(null);
    setLoading(true);
    try {
      const res = await predictBatch(file);
      setResults(res.data);
    } catch {
      setError("Upload failed. Make sure the backend is running and the CSV has the correct columns.");
    } finally {
      setLoading(false);
    }
  }

  const summary = results
    ? {
        total: results.total,
        high: results.predictions.filter((r) => r.risk_level === "High").length,
        medium: results.predictions.filter((r) => r.risk_level === "Medium").length,
        low: results.predictions.filter((r) => r.risk_level === "Low").length,
      }
    : null;

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-slate-800">Bulk Predict</h1>
        <p className="text-slate-500 text-sm mt-1">
          Upload a CSV with customer data to predict churn for multiple customers at once.
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
          dragOver ? "border-slate-400 bg-slate-50" : "border-gray-300 bg-white"
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
        onClick={() => fileRef.current.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div className="text-4xl mb-3">📂</div>
        <p className="text-slate-700 font-medium">Drop your CSV here, or click to browse</p>
        <p className="text-slate-400 text-sm mt-1">
          Must contain the same 19 columns as the single predict form
        </p>
      </div>

      {loading && (
        <div className="mt-6 text-center text-slate-500 text-sm animate-pulse">
          Processing rows...
        </div>
      )}

      {error && (
        <div className="mt-4 bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-600">
          {error}
        </div>
      )}

      {results && (
        <div className="mt-6 space-y-5">
          {/* Summary cards */}
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: "Total", value: summary.total, color: "text-slate-800" },
              { label: "High Risk", value: summary.high, color: "text-red-600" },
              { label: "Medium Risk", value: summary.medium, color: "text-amber-600" },
              { label: "Low Risk", value: summary.low, color: "text-green-600" },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-white rounded-xl border border-gray-200 p-4 text-center">
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
                <p className="text-xs text-slate-500 mt-1">{label}</p>
              </div>
            ))}
          </div>

          {/* Results table */}
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100">
              <p className="text-sm font-medium text-slate-700">Results</p>
              <button
                onClick={() => exportCSV(results.predictions)}
                className="px-3 py-1.5 text-xs font-medium bg-slate-800 text-white rounded-lg hover:bg-slate-700 transition-colors"
              >
                Download CSV
              </button>
            </div>
            <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-gray-50">
                  <tr className="border-b border-gray-100">
                    <th className="text-left px-4 py-3 font-medium text-slate-500">#</th>
                    <th className="text-left px-4 py-3 font-medium text-slate-500">Contract</th>
                    <th className="text-left px-4 py-3 font-medium text-slate-500">Internet</th>
                    <th className="text-left px-4 py-3 font-medium text-slate-500">Tenure</th>
                    <th className="text-left px-4 py-3 font-medium text-slate-500">Monthly $</th>
                    <th className="text-left px-4 py-3 font-medium text-slate-500">Probability</th>
                    <th className="text-left px-4 py-3 font-medium text-slate-500">Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {results.predictions.map((r, i) => (
                    <tr key={i} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                      <td className="px-4 py-2.5 text-slate-400">{i + 1}</td>
                      <td className="px-4 py-2.5 text-slate-700">{r.Contract}</td>
                      <td className="px-4 py-2.5 text-slate-700">{r.InternetService}</td>
                      <td className="px-4 py-2.5 text-slate-700">{r.tenure} mo</td>
                      <td className="px-4 py-2.5 text-slate-700">${r.MonthlyCharges}</td>
                      <td className="px-4 py-2.5 font-semibold text-slate-800">{r.probability}%</td>
                      <td className="px-4 py-2.5">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${RISK_COLORS[r.risk_level]}`}>
                          {r.risk_level}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
