/**
 * frontend/static/app.js
 * ======================
 * Areca Nut Price Dashboard — Core Application Logic
 *
 * Responsibilities:
 *  - Fetch price/forecast data from FastAPI backend
 *  - Render animated price cards with trend indicators
 *  - Manage Chart.js history + forecast charts
 *  - Handle section navigation and filter interactions
 *  - Lazy-load Grafana iframe
 *  - Display HOLD / SELL recommendations
 */

'use strict';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const CONFIG = {
  apiBase:         window.API_BASE_URL || 'http://localhost:8000/api/v1',
  refreshInterval: 5 * 60 * 1000,    // Auto-refresh prices every 5 minutes
  priceLimit:      200,
  historyDays:     30,
  chartFontFamily: "'Inter', system-ui, sans-serif",
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  currentPrices:  [],
  historyData:    {},        // keyed by days (7, 30, 90)
  forecastData:   {},        // keyed by variety
  activeSection:  'prices',
  activeHistDays: 30,
  activeFcVariety: 'Chali',
  historyChart:   null,
  forecastChart:  null,
  debounceTimer:  null,
  refreshTimer:   null,
};

// ---------------------------------------------------------------------------
// Variety colour palette
// ---------------------------------------------------------------------------

const VARIETY_PALETTE = {
  Chali:  { accent: '#f5a623', bg: 'rgba(245,166,35,0.12)',  border: 'rgba(245,166,35,0.35)',  badge: 'rgba(245,166,35,0.15)',  badgeClr: '#f5a623' },
  Gotu:   { accent: '#2dce89', bg: 'rgba(45,206,137,0.10)',  border: 'rgba(45,206,137,0.3)',   badge: 'rgba(45,206,137,0.12)',  badgeClr: '#2dce89' },
  Kotte:  { accent: '#42a5f5', bg: 'rgba(66,165,245,0.10)',  border: 'rgba(66,165,245,0.3)',   badge: 'rgba(66,165,245,0.12)',  badgeClr: '#42a5f5' },
  Rashi:  { accent: '#a78bfa', bg: 'rgba(167,139,250,0.10)', border: 'rgba(167,139,250,0.3)',  badge: 'rgba(167,139,250,0.12)', badgeClr: '#a78bfa' },
  Saraku: { accent: '#26c6da', bg: 'rgba(38,198,218,0.10)',  border: 'rgba(38,198,218,0.3)',   badge: 'rgba(38,198,218,0.12)',  badgeClr: '#26c6da' },
  default:{ accent: '#8892a4', bg: 'rgba(136,146,164,0.08)', border: 'rgba(136,146,164,0.25)', badge: 'rgba(136,146,164,0.1)',  badgeClr: '#8892a4' },
};

function getVarietyPalette(variety) {
  return VARIETY_PALETTE[variety] || VARIETY_PALETTE.default;
}

// ---------------------------------------------------------------------------
// Chart.js Global Defaults
// ---------------------------------------------------------------------------

Chart.defaults.font.family       = CONFIG.chartFontFamily;
Chart.defaults.color              = '#8b949e';
Chart.defaults.borderColor        = 'rgba(255,255,255,0.06)';
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/**
 * Format a number as Indian currency (₹ with lakhs/crores notation).
 * @param {number} value  - Price value (in ₹ per quintal)
 * @param {boolean} short - If true, abbreviate (e.g. "₹42.5K")
 */
function formatINR(value, short = false) {
  if (value == null || isNaN(value)) return '—';
  const n = parseFloat(value);
  if (short && n >= 1000) return `₹${(n / 1000).toFixed(1)}K`;
  return new Intl.NumberFormat('en-IN', {
    style: 'currency', currency: 'INR',
    minimumFractionDigits: 0, maximumFractionDigits: 0,
  }).format(n);
}

function formatDate(dateStr) {
  if (!dateStr) return '—';
  return new Date(dateStr).toLocaleDateString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric', timeZone: 'Asia/Kolkata'
  });
}

function formatIST() {
  return new Date().toLocaleTimeString('en-IN', {
    hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Kolkata', hour12: false,
  }) + ' IST';
}

/** Animate a numeric value into an element (count-up effect). */
function animateNumber(el, targetValue, formatter = formatINR, duration = 700) {
  if (!el) return;
  const start     = parseFloat(el.dataset.raw || 0);
  const end       = parseFloat(targetValue);
  const startTime = performance.now();

  function step(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);   // ease-out cubic
    const current  = start + (end - start) * eased;
    el.textContent  = formatter(current);
    el.dataset.raw  = current;
    el.classList.add('number-animate');
    if (progress < 1) requestAnimationFrame(step);
    else el.dataset.raw = end;
  }
  requestAnimationFrame(step);
}

function showAlert(message, type = 'error') {
  const banner = document.getElementById('alert-banner');
  if (!banner) return;
  banner.textContent = message;
  banner.removeAttribute('hidden');
  banner.style.display = 'block';
  setTimeout(() => { banner.setAttribute('hidden', ''); banner.style.display = ''; }, 6000);
}

// ---------------------------------------------------------------------------
// API Client
// ---------------------------------------------------------------------------

async function apiFetch(path, params = {}) {
  const url = new URL(CONFIG.apiBase + path);
  Object.entries(params).forEach(([k, v]) => { if (v !== '' && v != null) url.searchParams.set(k, v); });

  const resp = await fetch(url.toString(), {
    headers: { 'Accept': 'application/json' },
    signal: AbortSignal.timeout(15000),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// ---------------------------------------------------------------------------
// Section Navigation
// ---------------------------------------------------------------------------

function showSection(name) {
  const sections = ['prices', 'forecast', 'grafana'];

  sections.forEach(s => {
    const el = document.getElementById(`section-${s}`);
    const btn = document.getElementById(`btn-${s}`);
    if (!el || !btn) return;

    if (s === name) {
      el.removeAttribute('hidden');
      btn.classList.add('nav-btn--active');
      btn.setAttribute('aria-current', 'page');
    } else {
      el.setAttribute('hidden', '');
      btn.classList.remove('nav-btn--active');
      btn.removeAttribute('aria-current');
    }
  });

  state.activeSection = name;

  // Lazy-load Grafana iframe on first view
  if (name === 'grafana') loadGrafanaIframe();
}

// ---------------------------------------------------------------------------
// Summary Stat Cards
// ---------------------------------------------------------------------------

function updateSummaryCard(id, priceEl, trendEl, price, trend, trendPct) {
  const priceElement = document.getElementById(priceEl);
  const trendElement = document.getElementById(trendEl);

  if (priceElement && price != null) {
    animateNumber(priceElement, price);
  }

  if (trendElement) {
    let label = '—';
    let cls = '';
    if (trend === 'rising')  { label = `▲ ${trendPct != null ? Math.abs(trendPct).toFixed(1) + '%' : ''} Rising`; cls = 'badge--rising'; }
    if (trend === 'falling') { label = `▼ ${trendPct != null ? Math.abs(trendPct).toFixed(1) + '%' : ''} Falling`; cls = 'badge--falling'; }
    if (trend === 'stable')  { label = `● Stable`; cls = 'badge--stable'; }

    trendElement.textContent = label;
    trendElement.className   = `stat-card__badge ${cls}`;
  }
}

function aggregateSummaryStats(prices) {
  const byVariety = {};
  for (const p of prices) {
    if (!byVariety[p.variety_name]) byVariety[p.variety_name] = { prices: [], trend_pcts: [] };
    byVariety[p.variety_name].prices.push(p.modal_price);
    if (p.trend_pct != null) byVariety[p.variety_name].trend_pcts.push(p.trend_pct);
  }

  for (const [variety, data] of Object.entries(byVariety)) {
    const avgPrice = data.prices.reduce((a, b) => a + b, 0) / data.prices.length;
    const avgTrendPct = data.trend_pcts.length
      ? data.trend_pcts.reduce((a, b) => a + b, 0) / data.trend_pcts.length
      : null;
    const trend = avgTrendPct == null ? null : avgTrendPct > 2 ? 'rising' : avgTrendPct < -2 ? 'falling' : 'stable';

    if (variety === 'Chali')  updateSummaryCard('stat-chali',  'chali-price',  'chali-trend',  avgPrice, trend, avgTrendPct);
    if (variety === 'Gotu')   updateSummaryCard('stat-gotu',   'gotu-price',   'gotu-trend',   avgPrice, trend, avgTrendPct);
    if (variety === 'Kotte')  updateSummaryCard('stat-kotte',  'kotte-price',  'kotte-trend',  avgPrice, trend, avgTrendPct);
  }

  const activeMarkets = new Set(prices.map(p => p.market_name)).size;
  const el = document.getElementById('active-markets');
  if (el) animateNumber(el, activeMarkets, v => Math.round(v), 500);
}

// ---------------------------------------------------------------------------
// Price Cards
// ---------------------------------------------------------------------------

function buildPriceCard(record) {
  const p = getVarietyPalette(record.variety_name);
  const trend = record.price_trend || 'stable';
  const trendPct = record.trend_pct;
  const rec = record.recommendation || 'HOLD';

  const trendIcon  = trend === 'rising' ? '▲' : trend === 'falling' ? '▼' : '●';
  const trendLabel = trend === 'rising' ? `${trendIcon} ${trendPct != null ? Math.abs(trendPct).toFixed(1) + '% ' : ''}Rising`
                   : trend === 'falling' ? `${trendIcon} ${trendPct != null ? Math.abs(trendPct).toFixed(1) + '% ' : ''}Falling`
                   : `${trendIcon} Stable`;
  const trendClass = `trend--${trend}`;
  const recClass   = rec === 'SELL' ? 'rec--sell' : 'rec--hold';
  const recIcon    = rec === 'SELL' ? '📉 SELL NOW' : '📈 HOLD';

  const arrivals = record.arrivals_tons != null
    ? `${parseFloat(record.arrivals_tons).toFixed(1)} tons arrived`
    : 'Arrivals: N/A';

  return `
<article
  class="price-card"
  role="listitem"
  style="--card-accent: ${p.accent};"
  aria-label="${record.variety_name} price at ${record.market_name}: ${formatINR(record.modal_price)}"
>
  <div class="price-card__accent-bar" aria-hidden="true"></div>
  <div class="price-card__header">
    <div>
      <div class="price-card__market">${escapeHtml(record.market_name)}</div>
      <div class="price-card__district">${escapeHtml(record.district || '')}</div>
    </div>
    <span
      class="price-card__variety-badge"
      style="--badge-bg: ${p.badge}; --badge-clr: ${p.badgeClr}; --badge-border: ${p.border};"
    >${escapeHtml(record.variety_name)}</span>
  </div>

  <div class="price-card__prices">
    <div class="price-cell">
      <div class="price-cell__label">Min</div>
      <div class="price-cell__value">${formatINR(record.min_price, true)}</div>
    </div>
    <div class="price-cell price-cell--modal">
      <div class="price-cell__label">Modal</div>
      <div class="price-cell__value">${formatINR(record.modal_price)}</div>
    </div>
    <div class="price-cell">
      <div class="price-cell__label">Max</div>
      <div class="price-cell__value">${formatINR(record.max_price, true)}</div>
    </div>
  </div>

  <div class="price-card__footer">
    <span class="price-card__arrivals" aria-label="${arrivals}">${arrivals}</span>
    <div style="display:flex; align-items:center; gap:8px;">
      <span class="price-card__trend ${trendClass}" aria-label="Price trend: ${trendLabel}">${trendLabel}</span>
      <span class="price-card__rec ${recClass}" aria-label="Recommendation: ${rec}">${recIcon}</span>
    </div>
  </div>
</article>`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Load Current Prices
// ---------------------------------------------------------------------------

async function loadCurrentPrices() {
  const grid    = document.getElementById('price-cards-grid');
  const noData  = document.getElementById('no-price-data');
  const variety = document.getElementById('filter-variety')?.value || '';
  const market  = document.getElementById('filter-market')?.value || '';
  const btn     = document.getElementById('btn-refresh-prices');

  if (btn) { btn.textContent = '↻ Loading…'; btn.disabled = true; }

  // Show skeletons
  grid.innerHTML = Array(6).fill('<div class="skeleton-card" aria-hidden="true"></div>').join('');
  noData.setAttribute('hidden', '');

  try {
    const resp = await apiFetch('/prices/current', {
      variety, market, limit: CONFIG.priceLimit
    });

    state.currentPrices = resp.data || [];

    if (!state.currentPrices.length) {
      grid.innerHTML = '';
      noData.removeAttribute('hidden');
    } else {
      grid.innerHTML = state.currentPrices.map(buildPriceCard).join('');
      aggregateSummaryStats(state.currentPrices);
    }

    // Update timestamp
    const tsEl = document.getElementById('last-updated');
    if (tsEl) { tsEl.textContent = formatIST(); tsEl.setAttribute('datetime', new Date().toISOString()); }

  } catch (err) {
    console.error('Price load error:', err);
    grid.innerHTML = '';
    showAlert(`Could not load prices: ${err.message}. Please check your connection.`);
  } finally {
    if (btn) { btn.textContent = '↻ Refresh'; btn.disabled = false; }
  }
}

function debounceLoadPrices() {
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(loadCurrentPrices, 400);
}

// ---------------------------------------------------------------------------
// History Chart
// ---------------------------------------------------------------------------

async function updateHistoryChart(days) {
  state.activeHistDays = days;

  // Update button styles
  ['7', '30', '90'].forEach(d => {
    const btn = document.getElementById(`hist-${d}d`);
    if (!btn) return;
    btn.classList.toggle('chart-btn--active', parseInt(d) === days);
    btn.setAttribute('aria-pressed', String(parseInt(d) === days));
  });

  if (!state.historyData[days]) {
    try {
      const resp = await apiFetch('/prices/history', { days, limit: 3000 });
      state.historyData[days] = resp.data || [];
    } catch (err) {
      console.error('History fetch error:', err);
      return;
    }
  }

  const data    = state.historyData[days];
  const canvas  = document.getElementById('history-chart');
  if (!canvas) return;

  // Group by variety and date
  const byVariety = {};
  for (const row of data) {
    if (!byVariety[row.variety_name]) byVariety[row.variety_name] = {};
    const d = row.record_date?.slice(0, 10);
    if (!byVariety[row.variety_name][d]) byVariety[row.variety_name][d] = [];
    byVariety[row.variety_name][d].push(row.modal_price);
  }

  // Build datasets
  const varieties = ['Chali', 'Gotu', 'Kotte'];
  const allDates  = [...new Set(data.map(r => r.record_date?.slice(0, 10)))].sort();

  const datasets = varieties.map(v => {
    const p = getVarietyPalette(v);
    const points = allDates.map(d => {
      const vals = byVariety[v]?.[d];
      return vals ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    });

    return {
      label: v,
      data:  allDates.map((d, i) => ({ x: d, y: points[i] })).filter(pt => pt.y != null),
      borderColor:           p.accent,
      backgroundColor:       p.bg,
      pointBackgroundColor:  p.accent,
      borderWidth:           2,
      pointRadius:           days <= 30 ? 3 : 0,
      pointHoverRadius:      6,
      tension:               0.4,
      fill:                  false,
      spanGaps:              true,
    };
  });

  if (state.historyChart) state.historyChart.destroy();

  state.historyChart = new Chart(canvas, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600, easing: 'easeInOutQuart' },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          backgroundColor: 'rgba(22,27,34,0.96)',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${formatINR(ctx.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          type: 'time',
          time: { unit: days <= 30 ? 'day' : 'week', displayFormats: { day: 'MMM d', week: 'MMM d' } },
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { maxTicksLimit: 10 },
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { callback: v => formatINR(v, true) },
          title: { display: true, text: '₹ / Quintal', font: { size: 11 } },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Forecast Section
// ---------------------------------------------------------------------------

async function loadForecastData() {
  try {
    const [fc7, fc30] = await Promise.all([
      apiFetch('/prices/forecast', { horizon: 7,  limit: 200 }),
      apiFetch('/prices/forecast', { horizon: 30, limit: 200 }),
    ]);

    // Aggregate: for each variety find nearest target date
    function extractLatestForecast(records, variety, horizon) {
      const filtered = (records || []).filter(r => r.variety_name === variety);
      if (!filtered.length) return null;
      // Take first future date
      return filtered.sort((a, b) => a.target_date.localeCompare(b.target_date))[0];
    }

    const chali7  = extractLatestForecast(fc7.data,  'Chali', 7);
    const chali30 = extractLatestForecast(fc30.data, 'Chali', 30);
    const gotu7   = extractLatestForecast(fc7.data,  'Gotu',  7);

    if (chali7) {
      setForecastCard('forecast-chali-7d', 'forecast-chali-7d-range', 'rec-chali-7d', chali7);
    }
    if (chali30) {
      setForecastCard('forecast-chali-30d', 'forecast-chali-30d-range', 'rec-chali-30d', chali30);
    }
    if (gotu7) {
      setForecastCard('forecast-gotu-7d', 'forecast-gotu-7d-range', 'rec-gotu-7d', gotu7);
    }

    // Store for chart
    state.forecastData = { '7': fc7.data || [], '30': fc30.data || [] };
    await updateForecastChart('Chali');

  } catch (err) {
    console.error('Forecast load error:', err);
    showAlert(`Could not load forecasts: ${err.message}`);
  }
}

function setForecastCard(priceId, rangeId, recId, data) {
  const priceEl = document.getElementById(priceId);
  const rangeEl = document.getElementById(rangeId);
  const recEl   = document.getElementById(recId);

  if (priceEl) animateNumber(priceEl, data.predicted_price);
  if (rangeEl) rangeEl.textContent = `90% CI: ${formatINR(data.confidence_lower)} – ${formatINR(data.confidence_upper)}`;

  if (recEl) {
    // Determine recommendation from forecast direction
    const current = state.currentPrices.find(p => p.variety_name === data.variety_name);
    const currentPrice = current ? current.modal_price : data.predicted_price;
    const direction = data.predicted_price > currentPrice * 1.02 ? 'HOLD'
                    : data.predicted_price < currentPrice * 0.98 ? 'SELL'
                    : 'HOLD';

    recEl.textContent = direction === 'HOLD' ? '📈 HOLD — Price likely rising' : '📉 SELL — Price may fall';
    recEl.className   = `forecast-card__recommendation ${direction === 'HOLD' ? 'rec-hold' : 'rec-sell'}`;
  }
}

async function updateForecastChart(variety) {
  state.activeFcVariety = variety;

  // Update button states
  ['Chali', 'Gotu', 'Kotte'].forEach(v => {
    const btn = document.getElementById(`fc-${v.toLowerCase()}`);
    if (!btn) return;
    btn.classList.toggle('chart-btn--active', v === variety);
    btn.setAttribute('aria-pressed', String(v === variety));
  });

  const canvas = document.getElementById('forecast-chart');
  if (!canvas) return;

  const p = getVarietyPalette(variety);

  // Actuals from history (last 60 days)
  let histData = state.historyData[60] || state.historyData[30] || [];
  if (!histData.length) {
    try {
      const resp = await apiFetch('/prices/history', { variety, days: 60, limit: 2000 });
      histData = resp.data || [];
      state.historyData[60] = histData;
    } catch { histData = []; }
  }

  const actualMap = {};
  for (const r of histData.filter(r => r.variety_name === variety)) {
    const d = r.record_date?.slice(0, 10);
    if (!actualMap[d]) actualMap[d] = [];
    actualMap[d].push(r.modal_price);
  }
  const actualData = Object.entries(actualMap)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([d, vals]) => ({ x: d, y: vals.reduce((a, b) => a + b, 0) / vals.length }));

  // Forecast from stored data
  const fcDataRaw7  = (state.forecastData['7']  || []).filter(r => r.variety_name === variety);
  const fcDataRaw30 = (state.forecastData['30'] || []).filter(r => r.variety_name === variety);
  const fcDataRaw   = [...fcDataRaw7, ...fcDataRaw30].sort((a, b) => a.target_date.localeCompare(b.target_date));

  const fcPredicted = fcDataRaw.map(r => ({ x: r.target_date?.slice(0, 10), y: r.predicted_price }));
  const fcLower     = fcDataRaw.map(r => ({ x: r.target_date?.slice(0, 10), y: r.confidence_lower }));
  const fcUpper     = fcDataRaw.map(r => ({ x: r.target_date?.slice(0, 10), y: r.confidence_upper }));

  if (state.forecastChart) state.forecastChart.destroy();

  state.forecastChart = new Chart(canvas, {
    type: 'line',
    data: {
      datasets: [
        {
          label: `${variety} — Actual`,
          data:  actualData,
          borderColor:       p.accent,
          backgroundColor:   p.bg,
          borderWidth:       2,
          pointRadius:       actualData.length < 60 ? 3 : 0,
          pointHoverRadius:  6,
          tension:           0.4,
          fill:              false,
          spanGaps:          true,
        },
        {
          label: `${variety} — Forecast`,
          data:  fcPredicted,
          borderColor:       p.accent,
          backgroundColor:   'transparent',
          borderWidth:       2,
          borderDash:        [6, 4],
          pointRadius:       3,
          pointBackgroundColor: p.accent,
          tension:           0.4,
          fill:              false,
        },
        {
          label: '90% CI Upper',
          data:  fcUpper,
          borderColor:     'transparent',
          backgroundColor: p.bg,
          borderWidth:     0,
          fill:            '+1',
          tension:         0.4,
          pointRadius:     0,
        },
        {
          label: '90% CI Lower',
          data:  fcLower,
          borderColor:     'transparent',
          backgroundColor: p.bg,
          borderWidth:     0,
          tension:         0.4,
          pointRadius:     0,
        },
      ],
    },
    options: {
      responsive:           true,
      maintainAspectRatio:  false,
      animation:            { duration: 600 },
      interaction:          { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: { filter: item => !item.text.includes('CI') },
        },
        tooltip: {
          backgroundColor: 'rgba(22,27,34,0.96)',
          borderColor:     'rgba(255,255,255,0.1)',
          borderWidth:     1,
          padding:         12,
          callbacks: {
            label: ctx => {
              if (ctx.dataset.label.includes('CI')) return null;
              return ` ${ctx.dataset.label}: ${formatINR(ctx.parsed.y)}`;
            },
          },
        },
        annotation: {},
      },
      scales: {
        x: {
          type: 'time',
          time: { unit: 'week', displayFormats: { week: 'MMM d' } },
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { maxTicksLimit: 12 },
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { callback: v => formatINR(v, true) },
          title: { display: true, text: '₹ / Quintal', font: { size: 11 } },
        },
      },
    },
  });

  // Update chart title
  const titleEl = canvas.closest('.chart-container')?.querySelector('.chart-container__title');
  if (titleEl) titleEl.textContent = `Forecast vs Historical — ${variety} (₹/Quintal)`;
}

// ---------------------------------------------------------------------------
// Grafana Iframe — Lazy Load
// ---------------------------------------------------------------------------

let grafanaLoaded = false;

function loadGrafanaIframe() {
  if (grafanaLoaded) return;
  const iframe  = document.getElementById('grafana-iframe');
  const loading = document.getElementById('grafana-loading');

  if (!iframe) return;

  // Swap data-src → src on first view
  const src = iframe.getAttribute('data-src');
  if (src) {
    iframe.addEventListener('load', () => {
      if (loading) loading.style.opacity = '0';
      setTimeout(() => { if (loading) loading.style.display = 'none'; }, 400);
    }, { once: true });

    iframe.src = src;
    grafanaLoaded = true;
  }
}

// ---------------------------------------------------------------------------
// Auto-refresh
// ---------------------------------------------------------------------------

function startAutoRefresh() {
  clearInterval(state.refreshTimer);
  state.refreshTimer = setInterval(() => {
    if (state.activeSection === 'prices') loadCurrentPrices();
  }, CONFIG.refreshInterval);
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

async function init() {
  // Load initial data
  await loadCurrentPrices();
  await updateHistoryChart(state.activeHistDays);

  // Start background refresh
  startAutoRefresh();

  // Preload forecast data so it's ready when user switches tab
  loadForecastData().catch(console.warn);
}

// Kick off when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

// Handle section switches with data loads
window.showSection = function (name) {
  // Update nav
  ['prices', 'forecast', 'grafana'].forEach(s => {
    const el  = document.getElementById(`section-${s}`);
    const btn = document.getElementById(`btn-${s}`);
    if (!el || !btn) return;
    if (s === name) {
      el.removeAttribute('hidden');
      btn.classList.add('nav-btn--active');
    } else {
      el.setAttribute('hidden', '');
      btn.classList.remove('nav-btn--active');
    }
  });

  state.activeSection = name;

  if (name === 'forecast') loadForecastData();
  if (name === 'grafana')  loadGrafanaIframe();
};

// Export for HTML onclick handlers
window.loadCurrentPrices   = loadCurrentPrices;
window.debounceLoadPrices  = debounceLoadPrices;
window.updateHistoryChart  = updateHistoryChart;
window.updateForecastChart = updateForecastChart;
