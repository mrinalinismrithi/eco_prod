/* ═══════════════════════════════════════════════════════════
   ECOLENS — app.js
   Fixes:
     1. conversation_history uses {role, content} — matches agent.py
     2. Chaining: analysis_state persisted and sent on every request
     3. AI answers render HTML cleanly (no white-space:pre-wrap clash)
     4. Starts on dashboard, no hash needed
     5. Weather page renders a proper styled card
     6. Dynamic dashboard — works with any uploaded CSV, not just climate data
   ═══════════════════════════════════════════════════════════ */

"use strict";

// ── CONSTANTS ─────────────────────────────────────────────
const DEFAULT_API_BASE  = window.location.origin;
const TABLE_PAGE_SIZE   = 25;

const API = {
  health:    "/health",
  analytics: "/analytics",
  ask:       "/ask",
  weather:   "/weather",
  runEtl:    "/run-etl",
};

const TABLE_LABELS = {
  yearly_climate_data:     "Yearly Climate Data",
  country_warming_trends:  "Country Trends",
  regional_warming_trends: "Regional Trends",
  fastest_warming_regions: "Fastest Warming Regions",
  hottest_countries:       "Hottest Countries",
};

// ── STATE ─────────────────────────────────────────────────
const state = {
  analytics:       null,
  activePage:      "dashboard",
  activeTable:     "yearly_climate_data",
  tablePage:       1,
  conversation:    [],
  analysisState:   {},
  loadingAnalytics: false,
};

// ── HELPERS ───────────────────────────────────────────────
const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function esc(v) {
  return String(v ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtVal(v) {
  if (v === null || v === undefined || v === "") return "-";
  if (typeof v === "number") {
    return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3);
  }
  if (Array.isArray(v))  return v.map(fmtVal).join(", ");
  if (typeof v === "object") {
    return Object.entries(v).map(([k, x]) => `${prettyKey(k)}: ${fmtVal(x)}`).join("; ");
  }
  return String(v);
}

function prettyKey(k) {
  return String(k).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function getCol(row, candidates) {
  const k = candidates.find((c) => Object.prototype.hasOwnProperty.call(row, c));
  return k !== undefined ? row[k] : undefined;
}

function apiBase() {
  return ($("#api-base").value || DEFAULT_API_BASE).replace(/\/$/, "");
}

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), 3400);
}

function showNotice(msg, type = "info") {
  const el = $("#global-notice");
  el.textContent = msg;
  el.dataset.type = type;
  el.classList.toggle("hidden", !msg);
}

function setBusy(btn, label) {
  const orig = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = label;
  return () => { btn.disabled = false; btn.innerHTML = orig; };
}

// ── DYNAMIC COLUMN DETECTION ──────────────────────────────
function detectColumns(rows) {
  if (!rows || !rows.length) return { catCol: null, numCol: null, allCols: [], numCols: [] };

  const allCols = Object.keys(rows[0]);
  const numCols = allCols.filter((c) => typeof rows[0][c] === "number");
  const catCols = allCols.filter((c) => typeof rows[0][c] === "string");

  const catCol =
    allCols.find((c) => ["Country","country","Region","region","State","state"].includes(c))
    || catCols[0]
    || null;

  const numCol =
    allCols.find((c) => ["AvgTemperature","avgtemperature","cases","Cases","value","Value"].includes(c))
    || numCols[0]
    || null;

  return { catCol, numCol, allCols, numCols, catCols };
}

// ── HTTP ──────────────────────────────────────────────────
async function request(path, opts = {}) {
  const res = await fetch(`${apiBase()}${path}`, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    credentials: "include",
    ...opts,
  });
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = { message: text }; }
  if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
  return data;
}

// ── HEALTH ────────────────────────────────────────────────
async function checkHealth() {
  const dot = $("#status-dot");
  try {
    const h = await request(API.health);
    const ready  = h.data   === "ok";
    const aiOk   = h.openai === "ok";
    $("#api-status").textContent  = h.system_status || "System: Online";
    $("#data-status").textContent = `Data: ${h.data} · OpenAI: ${h.openai}`;
    dot.className = ready && aiOk ? "status-dot ok" : "status-dot";
    if (!ready) showNotice(`Data issue: ${h.data}`, "warning");
    return h;
  } catch (e) {
    $("#api-status").textContent  = "API unreachable";
    $("#data-status").textContent = e.message;
    dot.className = "status-dot err";
    showNotice(`Cannot reach API at ${apiBase()}. Start FastAPI or update the URL above.`, "error");
    throw e;
  }
}

// ── ANALYTICS ─────────────────────────────────────────────
async function loadAnalytics(force = false) {
  if (state.analytics && !force) return state.analytics;
  if (state.loadingAnalytics)    return state.analytics;
  state.loadingAnalytics = true;
  showNotice("Loading analytics…");
  try {
    state.analytics = await request(API.analytics);
    showNotice("");
    renderDashboard();
    renderAnalyticsTable();
    return state.analytics;
  } finally {
    state.loadingAnalytics = false;
  }
}

function uniqueCount(rows, col) {
  const s = new Set();
  rows.forEach((r) => {
    const v = Array.isArray(col) ? getCol(r, col) : r[col];
    if (v != null && v !== "") s.add(v);
  });
  return s.size || "-";
}

// ── DASHBOARD RENDERER ────────────────────────────────────
function renderDashboard() {
  const d       = state.analytics || {};
  const yearly  = d.yearly_climate_data      || [];
  const countries = d.country_warming_trends || [];
  const regions   = d.regional_warming_trends|| [];
  const fastest   = d.fastest_warming_regions|| [];
  const hottest   = d.hottest_countries      || [];

  // ── Metric 1: total rows
  $("#metric-rows").textContent = yearly.length.toLocaleString();

  // ── Metric 2: unique countries
  const countrySet = new Set();
  (countries.length ? countries : yearly).forEach((r) => {
    const v = r["Country"] || r["country"];
    if (v) countrySet.add(v);
  });
  $("#metric-countries").textContent = countrySet.size || "-";
  const labelCountries = $("#metric-countries").closest(".metric-card")?.querySelector(".metric-label");
  if (labelCountries) labelCountries.textContent = "Countries tracked";

  // ── Metric 3: unique regions
  const regionSet = new Set();
  (regions.length ? regions : fastest).forEach((r) => {
    const v = r["Region"] || r["region"];
    if (v) regionSet.add(v);
  });
  $("#metric-regions").textContent = regionSet.size || "-";
  const labelRegions = $("#metric-regions").closest(".metric-card")?.querySelector(".metric-label");
  if (labelRegions) labelRegions.textContent = "Regions tracked";

  // ── Metric 4: fastest warming region
  const fastestRow = fastest[0] || regions[0];
  $("#metric-fastest").textContent = fastestRow
    ? fmtVal(fastestRow["Region"] || fastestRow["region"] || fastestRow["Country"] || fastestRow["country"])
    : "-";

  renderTable("#hottest-preview",  hottest.slice(0, 8));
  renderTable("#fastest-preview",  fastest.slice(0, 8));
  renderRegionalBars(regions.length ? regions : fastest);
} 

// ── REGIONAL BAR CHART ────────────────────────────────────
function renderRegionalBars(rows) {
  const container = $("#regional-bars");
  if (!rows || !rows.length) {
    container.innerHTML = '<p class="empty-msg">Regional trend data unavailable.</p>';
    return;
  }

  const { catCol, numCol } = detectColumns(rows);

  if (!catCol || !numCol) {
    container.innerHTML = '<p class="empty-msg">Regional trend data unavailable.</p>';
    return;
  }

  const entries = rows
    .map((r) => ({
      label: fmtVal(r[catCol]),
      value: Number(r[numCol]),
    }))
    .filter((e) => e.label !== "-" && Number.isFinite(e.value))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 10);

  if (!entries.length) {
    container.innerHTML = '<p class="empty-msg">Regional trend data unavailable.</p>';
    return;
  }

  const max = Math.max(...entries.map((e) => Math.abs(e.value))) || 1;

  const title = container.closest(".dash-card")?.querySelector(".card-title");
  if (title) {
    title.innerHTML = `<i class="ti ti-chart-bar" style="color:#2563eb"></i> ${prettyKey(catCol)} Snapshot`;
  }

  container.innerHTML = entries.map((e) => `
    <div class="bar-row">
      <span title="${esc(e.label)}">${esc(e.label)}</span>
      <div class="bar-track" aria-hidden="true">
        <div class="bar-fill" style="width:${(Math.abs(e.value)/max)*100}%"></div>
      </div>
      <strong>${esc(fmtVal(e.value))}</strong>
    </div>`).join("");
}

// ── TABLE RENDERER ────────────────────────────────────────
function renderTable(selector, rows, opts = {}) {
  const table = $(selector);
  const visible = rows.slice(opts.start || 0, opts.end ?? rows.length);
  if (!visible.length) {
    table.innerHTML = "<tbody><tr><td style='padding:.75rem;color:#94a3b8'>No data available.</td></tr></tbody>";
    return;
  }
  const cols = Object.keys(visible[0]);
  table.innerHTML = `
    <thead><tr>${cols.map((c) => `<th>${esc(prettyKey(c))}</th>`).join("")}</tr></thead>
    <tbody>${visible.map((row) =>
      `<tr>${cols.map((c) => `<td>${esc(fmtVal(row[c]))}</td>`).join("")}</tr>`
    ).join("")}</tbody>`;
}

function currentRows() {
  const rows  = (state.analytics || {})[state.activeTable] || [];
  const query = $("#table-search").value.trim().toLowerCase();
  if (!query) return rows;
  return rows.filter((r) =>
    Object.values(r).some((v) => fmtVal(v).toLowerCase().includes(query))
  );
}

function renderAnalyticsTable() {
  const rows = currentRows();
  const totalPages = Math.max(1, Math.ceil(rows.length / TABLE_PAGE_SIZE));
  state.tablePage  = Math.min(Math.max(1, state.tablePage), totalPages);
  const start = (state.tablePage - 1) * TABLE_PAGE_SIZE;
  const end   = start + TABLE_PAGE_SIZE;
  $("#table-count").textContent = `${rows.length.toLocaleString()} rows · page ${state.tablePage}/${totalPages}`;
  $("#prev-page").disabled = state.tablePage <= 1;
  $("#next-page").disabled = state.tablePage >= totalPages;
  renderTable("#analytics-table", rows, { start, end });
}

function exportActiveTable() {
  const rows = currentRows();
  if (!rows.length) { toast("No rows to export."); return; }
  const cols = Object.keys(rows[0]);
  const csvEsc = (v) => { const t = fmtVal(v); return /[",\n]/.test(t) ? `"${t.replace(/"/g,'""')}"` : t; };
  const csv = [cols.join(","), ...rows.map((r) => cols.map((c) => csvEsc(r[c])).join(","))].join("\n");
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" })),
    download: `ecolens-${state.activeTable}.csv`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
  toast("CSV exported.");
}

// ── CHAT ──────────────────────────────────────────────────
function addMessage(role, content, isHtml = false) {
  const history = $("#chat-history");
  const empty   = $("#chat-empty");
  if (empty) empty.remove();

  const div = document.createElement("div");
  div.className = `msg msg-${role === "error" ? "error" : role === "thinking" ? "thinking" : role}`;

  if (isHtml) {
    div.style.whiteSpace = "normal";
    div.innerHTML = content;
  } else {
    div.textContent = content;
  }

  history.appendChild(div);
  history.scrollTop = history.scrollHeight;
  return div;
}

function renderChat() {
  const history = $("#chat-history");
  history.innerHTML = "";

  if (!state.conversation.length) {
    history.innerHTML = `
      <div class="chat-empty" id="chat-empty">
        <div class="chat-empty-icon"><i class="ti ti-message-bolt"></i></div>
        <p class="chat-empty-title">Ask EcoLens anything about your data</p>
        <p class="chat-empty-sub">Try the quick prompts above or type your own question.</p>
      </div>`;
    return;
  }

  state.conversation.forEach((m) => {
    addMessage(m.role, m.content, m.isHtml || false);
  });
}

function bindChips() {
  $$(".chip").forEach((b) =>
    b.addEventListener("click", () => {
      const q = b.dataset.question;
      $("#question-input").value = q;
      switchPage("assistant");
      setTimeout(() => $("#question-input").focus(), 50);
    })
  );
}

async function askAssistant(e) {
  e.preventDefault();
  const qInput   = $("#question-input");
  const question = qInput.value.trim();
  if (!question) return;

  const sendBtn = $("#send-btn");
  const restore = setBusy(sendBtn, '<i class="ti ti-loader-2"></i> <span>Thinking…</span>');

  addMessage("user", question);
  qInput.value = "";

  const thinkingEl = addMessage("thinking", "EcoLens is analysing…");

  const historyForBackend = state.conversation
    .filter((m) => ["user", "assistant"].includes(m.role))
    .map((m) => ({
      role:    m.role,
      content: m.plainText || m.content,
    }));

  try {
    const payload = {
      question,
      location:             $("#assistant-location").value.trim() || null,
      conversation_history: historyForBackend,
      analysis_state:       state.analysisState,
    };

    const answer = await request(API.ask, {
      method: "POST",
      body:   JSON.stringify(payload),
    });

    thinkingEl.remove();

    const htmlResponse  = answer.formatted_response;
    const plainResponse = answer.response || answer.answer || JSON.stringify(answer, null, 2);

    if (htmlResponse) {
      addMessage("assistant", htmlResponse, true);
      state.conversation.push({ role: "user",      content: question,      plainText: question,       isHtml: false });
      state.conversation.push({ role: "assistant", content: htmlResponse,  plainText: plainResponse,  isHtml: true  });
    } else {
      addMessage("assistant", plainResponse);
      state.conversation.push({ role: "user",      content: question,      plainText: question });
      state.conversation.push({ role: "assistant", content: plainResponse, plainText: plainResponse });
    }

    if (answer.analysis_state && typeof answer.analysis_state === "object") {
      state.analysisState = answer.analysis_state;
    }

  } catch (err) {
    thinkingEl.remove();
    addMessage("error", `Request failed: ${err.message}`);
  } finally {
    restore();
    $("#question-input").focus();
  }
}

// ── WEATHER ───────────────────────────────────────────────
function renderWeather(data) {
  const panel = $("#weather-result");

  if (data.success === false) {
    panel.innerHTML = `
      <div class="empty-state">
        <i class="ti ti-cloud-off empty-state-icon"></i>
        <p>${esc(data.response || data.message || "Weather request failed.")}</p>
      </div>`;
    return;
  }

  const loc     = data.location || {};
  const current = data.current  || {};
  const today   = data.today    || {};
  const place   = [loc.name, loc.admin1, loc.country].filter(Boolean).join(", ");

  const cells = [
    ["Feels like",  current.feels_like_c     != null ? `${current.feels_like_c} °C`   : "-"],
    ["Humidity",    current.humidity_percent  != null ? `${current.humidity_percent}%`  : "-"],
    ["Wind speed",  current.wind_speed_kmh   != null ? `${current.wind_speed_kmh} km/h`: "-"],
    ["Today min",   today.min_temperature_c  != null ? `${today.min_temperature_c} °C` : "-"],
    ["Today max",   today.max_temperature_c  != null ? `${today.max_temperature_c} °C` : "-"],
    ["Rainfall",    today.rain_sum_mm        != null ? `${today.rain_sum_mm} mm`        : "-"],
  ].map(([label, value]) => `
    <div class="weather-cell">
      <div class="weather-cell-label">${esc(label)}</div>
      <div class="weather-cell-value">${esc(value)}</div>
    </div>`).join("");

  panel.innerHTML = `
    <div class="weather-main">
      <p class="weather-loc-label">Current conditions</p>
      <p class="weather-loc">${esc(place || "Unknown location")}</p>
      <div class="weather-temp-row">
        <span class="weather-temp">${current.temperature_c != null ? current.temperature_c : "-"}°C</span>
      </div>
      <p class="weather-cond">${esc(current.condition || "-")}</p>
      <div class="weather-grid">${cells}</div>
    </div>`;
}

async function getWeather(e) {
  e.preventDefault();
  const btn     = $("#weather-form .btn-primary");
  const restore = setBusy(btn, '<i class="ti ti-loader-2"></i> Loading…');
  try {
    const loc = $("#weather-location").value.trim();
    const data = await request(API.weather, {
      method: "POST",
      body:   JSON.stringify({ location: loc }),
    });
    renderWeather(data);
  } catch (err) {
    $("#weather-result").innerHTML = `
      <div class="empty-state">
        <i class="ti ti-alert-triangle empty-state-icon"></i>
        <p>Weather request failed: ${esc(err.message)}</p>
      </div>`;
  } finally {
    restore();
  }
}

// ── ETL ───────────────────────────────────────────────────
async function runEtl() {
  const btn     = $("#run-etl");
  const restore = setBusy(btn, '<i class="ti ti-loader-2"></i> Running…');
  $("#etl-output").textContent = "Running ETL…";
  try {
    const data = await request(API.runEtl, { method: "POST" });
    if (data.success === false) throw new Error(data.error || data.message || "ETL failed.");
    $("#etl-output").textContent = JSON.stringify(data, null, 2);
    state.analytics = null;
    await loadAnalytics(true);
    toast("ETL completed — analytics refreshed.");
  } catch (err) {
    $("#etl-output").textContent = `Error: ${err.message}`;
    toast(`ETL failed: ${err.message}`);
  } finally {
    restore();
    checkHealth().catch(() => {});
  }
}

// ── UPLOAD ETL ────────────────────────────────────────────
async function runUploadEtl() {
  const fileInput = $("#csv-upload");
  const file = fileInput.files[0];
  const output = $("#etl-output");

  if (!file) {
    output.textContent = "Please select a CSV file first.";
    return;
  }

  const btn = $("#run-upload-etl");
  const restore = setBusy(btn, '<i class="ti ti-loader-2"></i> Processing…');
  output.textContent = "Uploading and processing CSV…";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${apiBase()}/api/etl/upload`, {
      method: "POST",
      credentials: "include",
      body: formData,
    });
    const data = await res.json();

    if (data.status === "success") {
      output.textContent = JSON.stringify(data.result, null, 2);
      state.analytics = null;
      await loadAnalytics(true);
      toast("CSV uploaded — dashboard updated with new data.");
      await updateDashboardLabels(); 
      await loadUploadHistory();
    } else {
      output.textContent = `Error: ${data.message}`;
      toast(`ETL failed: ${data.message}`);
    }
  } catch (err) {
    output.textContent = `Error: ${err.message}`;
    toast(`Upload failed: ${err.message}`);
  } finally {
    restore();
  }
}

// ── RESET TO DEFAULT ──────────────────────────────────────
async function resetToDefault() {
  const btn = $("#reset-to-default");
  const restore = setBusy(btn, '<i class="ti ti-loader-2"></i> Resetting…');
  const output = $("#etl-output");

  try {
    const res = await fetch(`${apiBase()}/api/etl/reset`, {
      method: "POST",
      credentials: "include",
    });
    const data = await res.json();
    output.textContent = data.message || "Reverted to default climate dataset.";
    state.analytics = null;
    await loadAnalytics(true);
    toast("Reverted to default climate data.");
    await updateDashboardLabels(); 
    await loadUploadHistory(); 
  } catch (err) {
    output.textContent = `Error: ${err.message}`;
    toast(`Reset failed: ${err.message}`);
  } finally {
    restore();
  }
} 
async function loadUploadHistory() {
  try {
    const res = await fetch(`${apiBase()}/api/etl/uploads`);
    const data = await res.json();
    const files = data.files || [];
    const activeFile = data.active;
    const section = document.getElementById("upload-history-section");
    const list = document.getElementById("upload-history-list");

    if (!section || !list) return;

    if (files.length === 0) {
      section.style.display = "none";
      return;
    }

    section.style.display = "block";
    list.innerHTML = files.map(f => `
      <div style="
        display:flex; align-items:center; justify-content:space-between;
        padding:8px 12px; border-radius:8px; cursor:pointer;
        background:${f === activeFile ? "#ecfdf5" : "var(--surface-2, #f9fafb)"};
        border:1px solid ${f === activeFile ? "#059669" : "var(--border)"};
      " onclick="switchToUpload('${f}')" title="Click to use this dataset">
        <div style="display:flex; align-items:center; gap:8px;">
          <i class="ti ti-file-text" style="color:${f === activeFile ? "#059669" : "#9ca3af"}; font-size:14px;"></i>
          <span style="font-size:13px; font-weight:600; color:var(--text-1);">${esc(f)}</span>
        </div>
        <div>
          ${f === activeFile
            ? `<span style="font-size:10px; font-weight:700; color:#059669; background:#d1fae5; padding:2px 8px; border-radius:99px;">ACTIVE</span>`
            : `<span style="font-size:10px; color:#9ca3af;">Click to activate</span>`
          }
        </div>
      </div>
    `).join("");

  } catch (err) {
    console.warn("Could not load upload history:", err);
  }
}  
async function switchToUpload(filename) {
  const output = $("#etl-output");
  output.textContent = `Switching to ${filename}...`;
  try {
    const res = await fetch(`${apiBase()}/api/etl/switch-upload?filename=${encodeURIComponent(filename)}`, {
      method: "POST",
      credentials: "include",
    });
    const data = await res.json();
    if (data.status === "success") {
      output.textContent = JSON.stringify(data.result, null, 2);
      state.analytics = null;
      await loadAnalytics(true);
      toast(`Switched to ${filename}`);
      await updateDashboardLabels();
      await loadUploadHistory();
    } else {
      output.textContent = `Error: ${data.message}`;
      toast(`Switch failed: ${data.message}`);
    }
  } catch (err) {
    output.textContent = `Error: ${err.message}`;
    toast(`Switch failed: ${err.message}`);
  }
} 

// ── UPDATE DASHBOARD LABELS ───────────────────────────────
async function updateDashboardLabels() {
  try {
    const res = await fetch(`${apiBase()}/api/etl/schema`);
    const schema = await res.json();
    const isUpload = schema.active_source === "upload";
    const yearRange = schema.year_range || [];
    const rowCount = schema.row_count || 0;
    const yearLabel = yearRange.length === 2 ? `${yearRange[0]}–${yearRange[1]}` : "";

    // ── Dashboard subtitle
    const pageSub = document.querySelector("#page-dashboard .page-sub");
    if (pageSub) {
      pageSub.textContent = isUpload
        ? `Uploaded climate dataset · ${yearLabel} · ${rowCount.toLocaleString()} records · GPT-4o-mini powered`
        : "Global surface temperature trends · historical CSV + live weather · GPT-4o-mini powered";
    }

    // ── Dashboard card titles — always climate terminology
    const cardTitle1 = document.querySelector("#page-dashboard .card-title:nth-of-type(1)");
    const cardTitle2 = document.querySelector("#page-dashboard .card-title:nth-of-type(2)");
    if (cardTitle1) cardTitle1.innerHTML = `<i class="ti ti-flame" style="color:#dc2626"></i> Hottest Countries`;
    if (cardTitle2) cardTitle2.innerHTML = `<i class="ti ti-trending-up" style="color:#f59e0b"></i> Fastest Warming Regions`;

    // ── Metric labels — always climate terminology
    const metricCountries = document.querySelector("#metric-countries")?.closest(".metric-card")?.querySelector(".metric-label");
    const metricRegions   = document.querySelector("#metric-regions")?.closest(".metric-card")?.querySelector(".metric-label");
    const metricRows      = document.querySelector("#metric-rows")?.closest(".metric-card")?.querySelector(".metric-label");
    if (metricCountries) metricCountries.textContent = "Countries tracked";
    if (metricRegions)   metricRegions.textContent   = "Regions tracked";
    if (metricRows)      metricRows.textContent      = "Climate records";

    // ── Analytics page title
    const analyticsTitle = document.getElementById("analytics-title");
    if (analyticsTitle) {
      analyticsTitle.textContent = isUpload
        ? `Climate Analytics · ${yearLabel}`
        : "Climate Analytics";
    }

    // ── Analytics tab labels — always climate terminology
    const tabs = document.querySelectorAll(".tab-bar .tab");
    if (tabs.length >= 5) {
      tabs[0].textContent = "Yearly Data";
      tabs[1].textContent = "Country Trends";
      tabs[2].textContent = "Regional Trends";
      tabs[3].textContent = "Fastest Regions";
      tabs[4].textContent = "Hottest Countries";
    }

    // ── Quick prompts — always climate terminology
    const chipsContainer = $("#quick-chips");
    if (chipsContainer) {
      if (isUpload) {
        chipsContainer.innerHTML = `
          <button class="chip" type="button" data-question="Which regions are warming the fastest?">
            <i class="ti ti-trending-up"></i> Fastest regions
          </button>
          <button class="chip" type="button" data-question="What are the top 10 hottest countries by average temperature?">
            <i class="ti ti-flame"></i> Hottest countries
          </button>
          <button class="chip" type="button" data-question="Summarize the temperature trends in this dataset.">
            <i class="ti ti-world"></i> Summarize trends
          </button>
          <button class="chip" type="button" data-question="Which country has the highest average temperature?">
            <i class="ti ti-chart-bar"></i> Highest temp
          </button>
          <button class="chip" type="button" data-question="What is the weather in London today?">
            <i class="ti ti-cloud"></i> London weather
          </button>`;
      } else {
        chipsContainer.innerHTML = `
          <button class="chip" type="button" data-question="Which regions are warming the fastest?">
            <i class="ti ti-trending-up"></i> Fastest regions
          </button>
          <button class="chip" type="button" data-question="What are the top 10 hottest countries by average temperature?">
            <i class="ti ti-flame"></i> Hottest countries
          </button>
          <button class="chip" type="button" data-question="Compare India and China temperature trends">
            <i class="ti ti-scale"></i> India vs China
          </button>
          <button class="chip" type="button" data-question="Summarize global warming trends in this dataset.">
            <i class="ti ti-world"></i> Global summary
          </button>
          <button class="chip" type="button" data-question="What is the weather in London today?">
            <i class="ti ti-cloud"></i> London weather
          </button>`;
      }
      bindChips();
    } 
    
    // ── Active dataset pill
    const pill = document.getElementById("active-dataset-pill");
    const pillName = document.getElementById("active-dataset-name");
    if (pill && pillName) {
      if (isUpload && schema.filename) {
        pillName.textContent = schema.filename;
        pill.style.display = "block";
        pill.onclick = async () => { await resetToDefault(); };
      } else {
        pill.style.display = "none";
      }
    }

  } catch (err) {
    console.warn("Could not update dashboard labels:", err);
  }
} 

// ── NAVIGATION ────────────────────────────────────────────
function switchPage(page, updateHash = true) {
  state.activePage = page;
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  $$(".page").forEach((s) => s.classList.toggle("active", s.id === `page-${page}`));
  if (updateHash) history.replaceState(null, "", `#${page}`);
  if (["dashboard", "analytics"].includes(page)) {
    loadAnalytics().catch((err) => toast(err.message));
  }
}

function switchTable(name) {
  state.activeTable = name;
  state.tablePage   = 1;
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.table === name));
  renderAnalyticsTable();
}

function jumpToTable(name) {
  switchTable(name);
  switchPage("analytics");
}

// ── EVENT BINDING ─────────────────────────────────────────
function bindEvents() {
  $$(".nav-item").forEach((b) =>
    b.addEventListener("click", () => switchPage(b.dataset.page))
  );

  $$(".tab").forEach((b) =>
    b.addEventListener("click", () => switchTable(b.dataset.table))
  );

  $$("[data-jump-table]").forEach((b) =>
    b.addEventListener("click", () => jumpToTable(b.dataset.jumpTable))
  );

  bindChips();

  $("#api-base").addEventListener("change", () => {
    state.analytics = null;
    checkHealth().catch(() => {});
    loadAnalytics(true).catch((err) => toast(err.message));
  });

  $("#table-search").addEventListener("input", () => { state.tablePage = 1; renderAnalyticsTable(); });
  $("#prev-page").addEventListener("click", () => { state.tablePage--; renderAnalyticsTable(); });
  $("#next-page").addEventListener("click", () => { state.tablePage++; renderAnalyticsTable(); });
  $("#export-table").addEventListener("click", exportActiveTable);

  $("#refresh-dashboard").addEventListener("click", () =>
    loadAnalytics(true).catch((err) => toast(err.message))
  );
  $("#refresh-analytics").addEventListener("click", () =>
    loadAnalytics(true).catch((err) => toast(err.message))
  );

  $("#chat-form").addEventListener("submit", askAssistant);

  $("#question-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("#chat-form").dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });

  $("#clear-chat").addEventListener("click", () => {
    state.conversation  = [];
    state.analysisState = {};
    renderChat();
    toast("Conversation cleared.");
  });

  $("#weather-form").addEventListener("submit", getWeather);
  $("#run-etl").addEventListener("click", runEtl);
  $("#run-upload-etl").addEventListener("click", runUploadEtl);
  $("#reset-to-default").addEventListener("click", resetToDefault);

  window.addEventListener("popstate", () => {
    const page = window.location.hash.replace("#", "") || "dashboard";
    if ($(`#page-${page}`)) switchPage(page, false);
  });
}

// ── AUTH ──────────────────────────────────────────────────
async function requireAuth() {
  try {
    const data = await request("/me");
    if (!data.authenticated) {
      window.location.href = "login.html";
      return null;
    }
    return data.username;
  } catch {
    window.location.href = "login.html";
    return null;
  }
}

async function logoutUser() {
  try { await request("/logout", { method: "POST" }); } catch {}
  window.location.href = "login.html";
}

// ── INIT ──────────────────────────────────────────────────
async function init() {
  const username = await requireAuth();
  if (!username) return;

  const userLabel = $("#current-username");
  if (userLabel) userLabel.textContent = username;
  const logoutBtn = $("#logout-btn");
  if (logoutBtn) logoutBtn.addEventListener("click", logoutUser);

  const apiInput = $("#api-base");
  if (apiInput) apiInput.value = window.location.origin;

  bindEvents();
  renderChat();

  const hash = window.location.hash.replace("#", "");
  const startPage = (hash && $(`#page-${hash}`)) ? hash : "dashboard";
  switchPage(startPage, false);
  updateDashboardLabels().catch(() => {}); 
  loadUploadHistory().catch(() => {}); 

  checkHealth()
    .then(() => loadAnalytics())
    .catch(() => {
      loadAnalytics().catch(() => {});
    });
}

init(); 