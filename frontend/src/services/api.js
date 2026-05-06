import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8000",
});

export const fetchForecast = async (year) => {
  const { data } = await api.get(`/forecast/${year}`);
  return data;
};

export const fetchInventoryHealth = async () => {
  const { data } = await api.get("/inventory-health");
  return data;
};

export const fetchClusterAnalysis = async () => {
  const { data } = await api.get("/cluster-analysis");
  return data;
};

export default api;
