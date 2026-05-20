import axios from "axios";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 25000,
});

export async function analyzeFrame(imageBase64) {
  const response = await api.post("/analyze-frame", { image_base64: imageBase64 });
  return response.data;
}

export async function getHealth() {
  const response = await api.get("/health");
  return response.data;
}

export async function getSessionStats() {
  const response = await api.get("/session-stats");
  return response.data;
}

export async function resetSession() {
  const response = await api.post("/reset-session");
  return response.data;
}
