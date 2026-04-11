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
