import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { getFeatureImportance, getModelStats } from "../api";

function MetricCard({ label, value, description, color }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">{label}</p>
      <p className={`text-3xl font-bold ${color}`}>{value}</p>
      <p className="text-xs text-slate-400 mt-1">{description}</p>
    </div>
  );
}

export default function Insights() {
  const [features, setFeatures] = useState([]);
  const [stats, setStats] = useState(null);
  const [topN, setTopN] = useState(15);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getFeatureImportance(), getModelStats()])
      .then(([fi, ms]) => {
        setFeatures(fi.data.feature_importance || []);
        setStats(ms.data);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const displayed = features.slice(0, topN);
  const maxImp = displayed[0]?.importance || 1;

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-slate-800">Model Insights</h1>
        <p className="text-slate-500 text-sm mt-1">
          Performance metrics and feature importance from the trained XGBoost model.
        </p>
      </div>

      {loading ? (
        <div className="text-center text-slate-400 py-20 text-sm animate-pulse">Loading...</div>
      ) : (
        <div className="space-y-6">
          {/* Metric cards */}
          {stats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard
                label="Recall"
                value={`${(stats.recall * 100).toFixed(1)}%`}
                description="Churners correctly identified"
                color="text-green-600"
              />
              <MetricCard
                label="ROC-AUC"
                value={`${(stats.roc_auc * 100).toFixed(1)}%`}
                description="Overall discrimination ability"
                color="text-blue-600"
              />
              <MetricCard
                label="F1 Score"
                value={`${(stats.f1 * 100).toFixed(1)}%`}
                description="Precision-recall balance"
                color="text-purple-600"
              />
              <MetricCard
                label="Precision"
                value={`${(stats.precision * 100).toFixed(1)}%`}
                description="Predicted churners that actually churn"
                color="text-amber-600"
              />
            </div>
          )}

          {/* Model info */}
          {stats && (
            <div className="bg-white rounded-xl border border-gray-200 p-5">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Model Info</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-slate-400">Algorithm</p>
                  <p className="font-medium text-slate-700 mt-0.5">{stats.model}</p>
                </div>
                <div>
                  <p className="text-slate-400">Threshold</p>
                  <p className="font-medium text-slate-700 mt-0.5">{stats.threshold} (recall-optimised)</p>
                </div>
                <div>
                  <p className="text-slate-400">Training Samples</p>
                  <p className="font-medium text-slate-700 mt-0.5">{stats.training_samples.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-slate-400">Churn Rate (train)</p>
                  <p className="font-medium text-slate-700 mt-0.5">{(stats.churn_rate * 100).toFixed(1)}%</p>
                </div>
              </div>
              <p className="text-xs text-slate-400 mt-4 italic">{stats.note}</p>
            </div>
          )}

          {/* Feature importance chart */}
          {features.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-5">
              <div className="flex items-center justify-between mb-4">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Feature Importance (Top {topN})
                </p>
                <select
                  value={topN}
                  onChange={(e) => setTopN(Number(e.target.value))}
                  className="text-xs border border-gray-200 rounded-lg px-2 py-1 focus:outline-none"
                >
                  {[10, 15, 20, features.length].map((n) => (
                    <option key={n} value={n}>{n === features.length ? "All" : `Top ${n}`}</option>
                  ))}
                </select>
              </div>
              <ResponsiveContainer width="100%" height={topN * 28 + 20}>
                <BarChart
                  layout="vertical"
                  data={displayed}
                  margin={{ left: 160, right: 30, top: 0, bottom: 0 }}
                >
                  <XAxis type="number" hide />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    tick={{ fontSize: 12, fill: "#64748b" }}
                    width={155}
                  />
                  <Tooltip
                    formatter={(v) => [v.toFixed(4), "Importance"]}
                    contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e5e7eb" }}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {displayed.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={`hsl(${215 - (i / displayed.length) * 40}, ${60 - i * 1}%, ${45 + i * 1}%)`}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
