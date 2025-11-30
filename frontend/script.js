const BACKEND_URL = "http://127.0.0.1:8000";
const REFRESH_MS = 3000; // автообновление каждые 3 секунды

const metricSafeEl = document.getElementById("metric-safe");
const metricMediumEl = document.getElementById("metric-medium");
const metricDangerEl = document.getElementById("metric-danger");
const metricDangerSituationsEl = document.getElementById("metric-dangerous-situations");
const metricCleaningEl = document.getElementById("metric-cleaning");

const eventsTableBody = document.getElementById("events-table-body");
const dangerousTableBody = document.getElementById("dangerous-table-body");
const cleaningTableBody = document.getElementById("cleaning-table-body");
const refreshBtn = document.getElementById("refresh-btn");
const errorMessageEl = document.getElementById("error-message");
let isLoading = false;

async function fetchJson(endpoint) {
  const url = `${BACKEND_URL}${endpoint}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} при запросе ${endpoint}`);
  }
  return response.json();
}

function formatWindowRange(startSec, endSec) {
  const start = typeof startSec === "number" ? startSec : 0;
  const end = typeof endSec === "number" ? endSec : 0;
  return `${start.toFixed(0)}–${end.toFixed(0)}`;
}

function formatTrain(present, number) {
  if (!present) return "нет";
  return number ? `есть (${number})` : "есть";
}

function renderMetrics(currentState) {
  metricSafeEl.textContent = currentState.workers_in_safe ?? 0;
  metricMediumEl.textContent = currentState.workers_in_medium ?? 0;
  metricDangerEl.textContent = currentState.workers_in_danger ?? 0;
  metricDangerSituationsEl.textContent = currentState.dangerous_situations_count ?? 0;
  metricCleaningEl.textContent = currentState.cleaning_workers_count ?? 0;
}

function renderWindows(rows, targetEl) {
  targetEl.innerHTML = "";
  rows.forEach((w) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${formatWindowRange(w.window_start_sec, w.window_end_sec)}</td>
      <td>${w.workers_in_safe}</td>
      <td>${w.workers_in_medium}</td>
      <td>${w.workers_in_danger}</td>
      <td>${w.dangerous_situations_count}</td>
      <td>${w.cleaning_workers_count}</td>
      <td>${formatTrain(w.train_present, w.train_number)}</td>
    `;
    targetEl.appendChild(tr);
  });
}

function renderDangerous(rows, targetEl) {
  targetEl.innerHTML = "";
  rows.forEach((w) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${formatWindowRange(w.window_start_sec, w.window_end_sec)}</td>
      <td>${w.dangerous_situations_count}</td>
      <td>${formatTrain(w.train_present, w.train_number)}</td>
    `;
    targetEl.appendChild(tr);
  });
}

function renderCleaning(rows, targetEl) {
  targetEl.innerHTML = "";
  rows.forEach((w) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${formatWindowRange(w.window_start_sec, w.window_end_sec)}</td>
      <td>${w.cleaning_workers_count}</td>
      <td>${formatTrain(w.train_present, w.train_number)}</td>
    `;
    targetEl.appendChild(tr);
  });
}

async function loadData() {
  if (isLoading) return;
  isLoading = true;
  errorMessageEl.textContent = "";
  try {
    const [currentState, windows, dangerous] = await Promise.all([
      fetchJson("/api/metrics/current_state"),
      fetchJson("/api/windows"),
      fetchJson("/api/metrics/dangerous_situations"),
    ]);

    renderMetrics(currentState);
    renderWindows(windows, eventsTableBody);
    renderDangerous(dangerous, dangerousTableBody);
    const cleaning = windows.filter((w) => (w.cleaning_workers_count ?? 0) > 0);
    renderCleaning(cleaning, cleaningTableBody);
  } catch (err) {
    console.error(err);
    errorMessageEl.textContent =
      "Не удалось получить данные. Убедитесь, что backend работает на http://127.0.0.1:8000";
  } finally {
    isLoading = false;
  }
}

refreshBtn.addEventListener("click", loadData);

loadData();
setInterval(loadData, REFRESH_MS);
