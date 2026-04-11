import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from "recharts";

const CHURN_BY_CONTRACT = [
  { name: "Month-to-month", churn: 42.7, retain: 57.3 },
  { name: "One year", churn: 11.3, retain: 88.7 },
  { name: "Two year", churn: 2.8, retain: 97.2 },
];

const CHURN_BY_INTERNET = [
  { name: "Fiber optic", churn: 41.9, retain: 58.1 },
  { name: "DSL", churn: 19.0, retain: 81.0 },
  { name: "No internet", churn: 7.4, retain: 92.6 },
];

const CHURN_PIE = [
  { name: "Retained", value: 73.5 },
  { name: "Churned", value: 26.5 },
];
const PIE_COLORS = ["#94a3b8", "#ef4444"];

const STATS = [
  { label: "Total customers", value: "7,043" },
  { label: "Features", value: "21" },
  { label: "Churn rate", value: "26.5%" },
  { label: "Avg. tenure", value: "32 mo" },
  { label: "Avg. monthly charge", value: "$64.76" },
  { label: "Senior citizens", value: "16.2%" },
];

function StatCard({ label, value }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <p className="text-2xl font-bold text-slate-800">{value}</p>
      <p className="text-xs text-slate-400 mt-1">{label}</p>
    </div>
  );
}

export default function Dataset() {
  return (
    <div>
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">Dataset Overview</h1>
          <p className="text-slate-500 text-sm mt-1">
            Training data — IBM Telco Customer Churn dataset from Kaggle.
          </p>
        </div>
        <a
          href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
        >
          <span>View on Kaggle</span>
          <span className="text-slate-400">↗</span>
        </a>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-6">
        {STATS.map(({ label, value }) => (
          <StatCard key={label} label={label} value={value} />
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-6">
        {/* Pie */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Churn Distribution
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={CHURN_PIE} cx="50%" cy="50%" outerRadius={70} dataKey="value" label={({ name, value }) => `${name} ${value}%`} labelLine={false}>
                {CHURN_PIE.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
              </Pie>
              <Tooltip formatter={(v) => `${v}%`} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Churn by contract */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Churn Rate by Contract
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={CHURN_BY_CONTRACT} layout="vertical" margin={{ left: 90, right: 20 }}>
              <XAxis type="number" unit="%" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={85} />
              <Tooltip formatter={(v) => `${v}%`} />
              <Bar dataKey="churn" fill="#ef4444" radius={[0, 4, 4, 0]} name="Churn %" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Churn by internet */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Churn Rate by Internet Service
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={CHURN_BY_INTERNET} layout="vertical" margin={{ left: 80, right: 20 }}>
              <XAxis type="number" unit="%" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={75} />
              <Tooltip formatter={(v) => `${v}%`} />
              <Bar dataKey="churn" fill="#f59e0b" radius={[0, 4, 4, 0]} name="Churn %" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Key findings */}
      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
          Key Findings
        </p>
        <ul className="space-y-2 text-sm text-slate-600">
          <li className="flex gap-2"><span className="text-red-500 font-bold">→</span> Month-to-month contracts churn at <strong>42.7%</strong> vs 2.8% for two-year contracts</li>
          <li className="flex gap-2"><span className="text-red-500 font-bold">→</span> Fiber optic customers churn at <strong>41.9%</strong> — possibly due to high monthly charges</li>
          <li className="flex gap-2"><span className="text-amber-500 font-bold">→</span> Customers without online security or tech support are significantly more likely to churn</li>
          <li className="flex gap-2"><span className="text-amber-500 font-bold">→</span> Electronic check is the most common payment method among churners</li>
          <li className="flex gap-2"><span className="text-green-500 font-bold">→</span> Long-tenure customers (60+ months) have very low churn rates — loyalty builds retention</li>
          <li className="flex gap-2"><span className="text-green-500 font-bold">→</span> Adding a partner or dependents correlates with lower churn probability</li>
        </ul>
      </div>
    </div>
  );
}
