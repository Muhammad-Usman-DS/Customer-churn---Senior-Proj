import { useState } from "react";
import { predictSingle } from "../api";

const DEFAULTS = {
  gender: "Male",
  SeniorCitizen: 0,
  Partner: "No",
  Dependents: "No",
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  tenure: 5,
  MonthlyCharges: 70.35,
  TotalCharges: 350.75,
};

const FIELDS = [
  {
    section: "Demographics",
    fields: [
      { key: "gender", label: "Gender", type: "select", options: ["Male", "Female"] },
      { key: "SeniorCitizen", label: "Senior Citizen", type: "select", options: [{ label: "No", value: 0 }, { label: "Yes", value: 1 }] },
      { key: "Partner", label: "Partner", type: "select", options: ["Yes", "No"] },
      { key: "Dependents", label: "Dependents", type: "select", options: ["Yes", "No"] },
    ],
  },
  {
    section: "Phone Services",
    fields: [
      { key: "PhoneService", label: "Phone Service", type: "select", options: ["Yes", "No"] },
      { key: "MultipleLines", label: "Multiple Lines", type: "select", options: ["Yes", "No", "No phone service"] },
    ],
  },
  {
    section: "Internet Services",
    fields: [
      { key: "InternetService", label: "Internet Service", type: "select", options: ["DSL", "Fiber optic", "No"] },
      { key: "OnlineSecurity", label: "Online Security", type: "select", options: ["Yes", "No", "No internet service"] },
      { key: "OnlineBackup", label: "Online Backup", type: "select", options: ["Yes", "No", "No internet service"] },
      { key: "DeviceProtection", label: "Device Protection", type: "select", options: ["Yes", "No", "No internet service"] },
      { key: "TechSupport", label: "Tech Support", type: "select", options: ["Yes", "No", "No internet service"] },
      { key: "StreamingTV", label: "Streaming TV", type: "select", options: ["Yes", "No", "No internet service"] },
      { key: "StreamingMovies", label: "Streaming Movies", type: "select", options: ["Yes", "No", "No internet service"] },
    ],
  },
  {
    section: "Account",
    fields: [
      { key: "Contract", label: "Contract", type: "select", options: ["Month-to-month", "One year", "Two year"] },
      { key: "PaperlessBilling", label: "Paperless Billing", type: "select", options: ["Yes", "No"] },
      { key: "PaymentMethod", label: "Payment Method", type: "select", options: ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] },
      { key: "tenure", label: "Tenure (months)", type: "number", min: 0, max: 100 },
      { key: "MonthlyCharges", label: "Monthly Charges ($)", type: "number", min: 0, max: 200, step: 0.01 },
      { key: "TotalCharges", label: "Total Charges ($)", type: "number", min: 0, max: 10000, step: 0.01 },
    ],
  },
];

function RiskGauge({ probability, riskLevel }) {
  const color =
    riskLevel === "High" ? "#ef4444" : riskLevel === "Medium" ? "#f59e0b" : "#22c55e";
  const label =
    riskLevel === "High" ? "High Risk" : riskLevel === "Medium" ? "Medium Risk" : "Low Risk";
  const bg =
    riskLevel === "High" ? "bg-red-50 border-red-200" : riskLevel === "Medium" ? "bg-amber-50 border-amber-200" : "bg-green-50 border-green-200";

  return (
    <div className={`rounded-xl border p-6 text-center ${bg}`}>
      {/* Gauge arc */}
      <div className="relative w-36 h-20 mx-auto mb-4">
        <svg viewBox="0 0 120 65" className="w-full">
          {/* Background arc */}
          <path d="M 10 60 A 50 50 0 0 1 110 60" fill="none" stroke="#e5e7eb" strokeWidth="10" strokeLinecap="round" />
          {/* Filled arc */}
          <path
            d="M 10 60 A 50 50 0 0 1 110 60"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${(probability / 100) * 157} 157`}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-end pb-1">
          <span className="text-2xl font-bold" style={{ color }}>{probability}%</span>
        </div>
      </div>
      <p className="text-lg font-semibold" style={{ color }}>{label}</p>
      <p className="text-sm text-gray-500 mt-1">Churn probability</p>
    </div>
  );
}

function saveToHistory(form, result) {
  const history = JSON.parse(localStorage.getItem("churn_history") || "[]");
  history.unshift({
    id: Date.now(),
    date: new Date().toISOString(),
    ...form,
    ...result,
  });
  localStorage.setItem("churn_history", JSON.stringify(history.slice(0, 200)));
}

export default function Predict() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [whatIfMode, setWhatIfMode] = useState(false);

  function handleChange(key, value) {
    const updated = { ...form, [key]: value };
    setForm(updated);
    // In what-if mode, re-run prediction on every change
    if (whatIfMode && result) {
      runPredict(updated);
    }
  }

  async function runPredict(data = form) {
    setLoading(true);
    setError(null);
    try {
      const res = await predictSingle(data);
      setResult(res.data);
      saveToHistory(data, res.data);
    } catch {
      setError("Could not reach the API. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-slate-800">Churn Prediction</h1>
        <p className="text-slate-500 text-sm mt-1">
          Enter customer details to predict churn risk.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form */}
        <div className="lg:col-span-2 space-y-5">
          {FIELDS.map(({ section, fields }) => (
            <div key={section} className="bg-white rounded-xl border border-gray-200 p-5">
              <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                {section}
              </h2>
              <div className="grid grid-cols-2 gap-4">
                {fields.map((f) => (
                  <div key={f.key}>
                    <label className="block text-sm font-medium text-slate-700 mb-1">
                      {f.label}
                    </label>
                    {f.type === "select" ? (
                      <select
                        value={form[f.key]}
                        onChange={(e) => {
                          const val =
                            typeof f.options[0] === "object"
                              ? Number(e.target.value)
                              : e.target.value;
                          handleChange(f.key, val);
                        }}
                        className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-slate-300"
                      >
                        {f.options.map((o) =>
                          typeof o === "object" ? (
                            <option key={o.value} value={o.value}>{o.label}</option>
                          ) : (
                            <option key={o} value={o}>{o}</option>
                          )
                        )}
                      </select>
                    ) : (
                      <input
                        type="number"
                        min={f.min}
                        max={f.max}
                        step={f.step || 1}
                        value={form[f.key]}
                        onChange={(e) => handleChange(f.key, Number(e.target.value))}
                        className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-slate-300"
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}

          <button
            onClick={() => runPredict()}
            disabled={loading}
            className="w-full bg-slate-800 hover:bg-slate-700 text-white font-medium py-3 rounded-xl transition-colors disabled:opacity-50"
          >
            {loading ? "Predicting..." : "Predict Churn"}
          </button>
        </div>

        {/* Result panel */}
        <div className="space-y-4">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-600">
              {error}
            </div>
          )}

          {result ? (
            <>
              <RiskGauge probability={result.probability ?? 0} riskLevel={result.risk_level} />

              <div className="bg-white rounded-xl border border-gray-200 p-5">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                  Verdict
                </p>
                <p className="text-base font-semibold text-slate-800">{result.prediction}</p>
              </div>

              {/* What-if toggle */}
              <div className="bg-white rounded-xl border border-gray-200 p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-700">What-if Mode</p>
                    <p className="text-xs text-slate-400 mt-0.5">Re-predicts as you change inputs</p>
                  </div>
                  <button
                    onClick={() => setWhatIfMode(!whatIfMode)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      whatIfMode ? "bg-slate-800" : "bg-gray-200"
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        whatIfMode ? "translate-x-6" : "translate-x-1"
                      }`}
                    />
                  </button>
                </div>
                {whatIfMode && (
                  <p className="text-xs text-amber-600 mt-3 bg-amber-50 rounded-lg px-3 py-2">
                    Live mode on — adjust any field and the prediction updates instantly.
                  </p>
                )}
              </div>
            </>
          ) : (
            <div className="bg-white rounded-xl border border-dashed border-gray-300 p-8 text-center">
              <p className="text-slate-400 text-sm">
                Fill in the customer details and click <strong>Predict Churn</strong> to see the result here.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
