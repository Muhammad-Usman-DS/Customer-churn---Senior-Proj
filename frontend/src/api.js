import axios from "axios";

// Switch this to your Cloud Run URL when deploying to production
const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: BASE_URL });

export const predictSingle = (data) => api.post("/predict", data);
export const predictBatch = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/predict-batch", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};
export const getFeatureImportance = () => api.get("/feature-importance");
export const getModelStats = () => api.get("/model-stats");

// Turn an Axios error into a message for the user. The API sends `detail` as a string,
// a list of strings (our input checks), or a list of objects (FastAPI's own schema checks).
export function errorMessage(err, fallback) {
  const detail = err?.response?.data?.detail;
  if (!err?.response) return "Could not reach the API. It may be waking up — try again in a minute.";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((d) => (typeof d === "string" ? d : `${d.loc?.slice(-1)[0] ?? "input"}: ${d.msg}`))
      .join(" · ");
  }
  return fallback;
}
