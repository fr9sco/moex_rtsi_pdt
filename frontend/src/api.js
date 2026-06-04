const API_URL = import.meta.env.VITE_API_URL || "";

async function getJson(path) {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) {
    throw new Error(`request failed: ${path}`);
  }
  return response.json();
}

export function getSummary() {
  return getJson("/api/summary");
}

export function getModelComparison(horizon, secid) {
  return getJson(`/api/model-comparison?horizon=${horizon}&secid=${secid}`);
}

export function getBacktest(horizon, secid) {
  return getJson(`/api/backtest?horizon=${horizon}&secid=${secid}`);
}

export function getBacktestSensitivity(horizon, secid) {
  return getJson(`/api/backtest-sensitivity?horizon=${horizon}&secid=${secid}`);
}

export function getPredictions(horizon, secid, model) {
  return getJson(
    `/api/predictions?horizon=${horizon}&secid=${secid}&model=${encodeURIComponent(model)}`
  );
}

export function getEquity(horizon, secid, model) {
  return getJson(
    `/api/equity?horizon=${horizon}&secid=${secid}&model=${encodeURIComponent(model)}`
  );
}

export function getFeatureImportance(source, horizon, secid) {
  return getJson(`/api/feature-importance?source=${source}&horizon=${horizon}&secid=${secid}`);
}

export function getPdtTree(horizon, secid) {
  return getJson(`/api/pdt-tree?horizon=${horizon}&secid=${secid}`);
}
