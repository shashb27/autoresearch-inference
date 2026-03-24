"""
leaderboard.py — Generate the community leaderboard HTML dashboard.

Reads all JSON files in leaderboard/runs/ and writes leaderboard/index.html —
a self-contained HTML page with charts and a sortable table.

Usage:
    uv run leaderboard.py                  # writes leaderboard/index.html
    uv run leaderboard.py --out path.html  # custom output path
    uv run leaderboard.py --dry-run        # print stats, don't write HTML
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_DIR, "leaderboard", "runs")
DEFAULT_OUT = os.path.join(PROJECT_DIR, "leaderboard", "index.html")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs(runs_dir: str) -> list[dict]:
    runs = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        if path.name == "__init__.py":
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            data["_file"] = path.name
            runs.append(data)
        except Exception as e:
            print(f"  WARNING: skipping {path.name}: {e}", file=sys.stderr)
    return runs


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def stats_summary(runs: list[dict]) -> dict:
    if not runs:
        return {}
    gpus = {r["hardware"]["gpu_name"] for r in runs}
    models = {r["model"]["id"] for r in runs}
    contributors = {r.get("contributor", "anonymous") for r in runs}
    best = max(runs, key=lambda r: r["results"]["best_tok_s"])
    most_gain = max(runs, key=lambda r: r["results"]["gain_pct"])
    return {
        "total_runs": len(runs),
        "total_gpus": len(gpus),
        "total_models": len(models),
        "total_contributors": len(contributors),
        "top_tok_s": best["results"]["best_tok_s"],
        "top_tok_s_gpu": best["hardware"]["gpu_name"],
        "top_tok_s_model": best["model"]["id"],
        "top_gain_pct": most_gain["results"]["gain_pct"],
        "top_gain_gpu": most_gain["hardware"]["gpu_name"],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


# ---------------------------------------------------------------------------
# JSON payload for charts
# ---------------------------------------------------------------------------

def build_chart_data(runs: list[dict]) -> dict:
    # Sort runs by best_tok_s descending for the bar chart
    sorted_runs = sorted(runs, key=lambda r: r["results"]["best_tok_s"], reverse=True)

    bar_labels = []
    bar_baseline = []
    bar_best = []
    bar_gain = []
    bar_colors = []

    family_palette = {
        "Ada Lovelace": "#10b981",
        "Hopper":       "#6366f1",
        "Ampere":       "#3b82f6",
        "Turing":       "#f59e0b",
        "Volta":        "#ef4444",
    }
    default_color = "#8b5cf6"

    for r in sorted_runs:
        gpu = r["hardware"]["gpu_name"].replace("NVIDIA ", "").replace("GeForce ", "")
        model_short = r["model"]["id"].split("/")[-1]
        label = f"{gpu}\n{model_short}"
        bar_labels.append(label)
        bar_baseline.append(round(r["results"]["baseline_tok_s"], 2))
        bar_best.append(round(r["results"]["best_tok_s"], 2))
        bar_gain.append(round(r["results"]["gain_pct"], 1))
        family = r["hardware"].get("gpu_family", "")
        bar_colors.append(family_palette.get(family, default_color))

    # Scatter: params_b vs best_tok_s
    scatter_points = []
    for r in runs:
        scatter_points.append({
            "x": r["model"]["params_b"],
            "y": round(r["results"]["best_tok_s"], 2),
            "gpu": r["hardware"]["gpu_name"].replace("NVIDIA ", "").replace("GeForce ", ""),
            "model": r["model"]["id"].split("/")[-1],
            "gain": round(r["results"]["gain_pct"], 1),
        })

    # Technique frequency
    tech_counts: dict[str, int] = {}
    for r in runs:
        for t in r.get("best_config", {}).get("techniques", []):
            if t != "baseline":
                tech_counts[t] = tech_counts.get(t, 0) + 1
    tech_sorted = sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)[:12]

    return {
        "bar": {
            "labels": bar_labels,
            "baseline": bar_baseline,
            "best": bar_best,
            "gain": bar_gain,
            "colors": bar_colors,
        },
        "scatter": scatter_points,
        "techniques": {
            "labels": [t for t, _ in tech_sorted],
            "counts": [c for _, c in tech_sorted],
        },
    }


# ---------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------

def build_table_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for r in runs:
        hw = r["hardware"]
        m = r["model"]
        res = r["results"]
        bc = r.get("best_config", {})
        rows.append({
            "run_id": r["run_id"],
            "contributor": r.get("contributor", "anonymous"),
            "gpu": hw["gpu_name"].replace("NVIDIA ", "").replace("GeForce ", ""),
            "gpu_family": hw.get("gpu_family", ""),
            "vram_gb": hw["vram_total_gb"],
            "model": m["id"].split("/")[-1],
            "params_b": m["params_b"],
            "baseline_tok_s": round(res["baseline_tok_s"], 2),
            "best_tok_s": round(res["best_tok_s"], 2),
            "gain_pct": round(res["gain_pct"], 1),
            "ttft_ms": round(res.get("best_ttft_ms", 0.0), 1),
            "vram_used_gb": round(res.get("best_vram_gb", 0.0), 1),
            "experiments": res["total_experiments"],
            "techniques": ", ".join(bc.get("techniques", [])),
            "submitted_at": r.get("submitted_at", "")[:10],
        })
    # Default sort: best_tok_s descending
    rows.sort(key=lambda x: x["best_tok_s"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>autoresearch-inference — Community Leaderboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d2e;
    --surface2: #252840;
    --border: #2e3150;
    --text: #e2e8f0;
    --text-muted: #8892a4;
    --accent: #6366f1;
    --green: #10b981;
    --yellow: #f59e0b;
    --red: #ef4444;
    --radius: 10px;
    --font: 'Inter', system-ui, -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 14px;
    line-height: 1.6;
    padding: 0 0 60px;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* Header */
  header {
    background: linear-gradient(135deg, #1a1d2e 0%, #12152a 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 24px;
  }
  header h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }
  header h1 span { color: var(--accent); }
  header p { color: var(--text-muted); margin-top: 4px; font-size: 13px; }
  .header-meta {
    margin-top: 10px;
    font-size: 12px;
    color: var(--text-muted);
  }

  /* Layout */
  .container { max-width: 1400px; margin: 0 auto; padding: 0 24px; }
  .section { margin-top: 32px; }
  .section-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 14px;
  }

  /* Stat cards */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 14px;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
  }
  .stat-card .value {
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }
  .stat-card .label {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .stat-card .sub {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Charts */
  .charts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }
  .chart-wide { grid-column: span 2; }
  .chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
  }
  .chart-card h3 {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .chart-wrapper { position: relative; }
  .chart-wrapper canvas { width: 100% !important; }

  @media (max-width: 768px) {
    .charts-grid { grid-template-columns: 1fr; }
    .chart-wide { grid-column: span 1; }
    header { padding: 20px; }
  }

  /* Table */
  .table-controls {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .table-controls input[type="search"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 13px;
    padding: 7px 12px;
    outline: none;
    width: 220px;
  }
  .table-controls input[type="search"]:focus {
    border-color: var(--accent);
  }
  .table-controls select {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 13px;
    padding: 7px 10px;
    outline: none;
    cursor: pointer;
  }
  .table-controls select:focus { border-color: var(--accent); }
  .table-controls label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .table-wrap {
    overflow-x: auto;
    border-radius: var(--radius);
    border: 1px solid var(--border);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  thead th {
    background: var(--surface2);
    color: var(--text-muted);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 10px 14px;
    text-align: left;
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
    border-bottom: 1px solid var(--border);
  }
  thead th:hover { color: var(--text); }
  thead th.sorted-asc::after  { content: " ↑"; color: var(--accent); }
  thead th.sorted-desc::after { content: " ↓"; color: var(--accent); }

  tbody tr {
    border-bottom: 1px solid var(--border);
    transition: background 0.1s;
  }
  tbody tr:last-child { border-bottom: none; }
  tbody tr:hover { background: var(--surface2); }
  td {
    padding: 10px 14px;
    white-space: nowrap;
    color: var(--text);
  }
  td.mono { font-family: var(--font-mono); font-size: 12px; }
  .rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    font-size: 11px;
    font-weight: 700;
    background: var(--surface2);
    color: var(--text-muted);
  }
  .rank-badge.gold   { background: #854d0e; color: #fde68a; }
  .rank-badge.silver { background: #374151; color: #d1d5db; }
  .rank-badge.bronze { background: #7c2d12; color: #fed7aa; }

  .gain-pill {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: var(--font-mono);
  }
  .gain-pill.high   { background: #064e3b; color: #6ee7b7; }
  .gain-pill.medium { background: #1c3a20; color: #86efac; }
  .gain-pill.low    { background: #1f2937; color: #9ca3af; }

  .tech-list {
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    max-width: 260px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .family-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 5px;
  }

  .no-results {
    text-align: center;
    padding: 40px;
    color: var(--text-muted);
    font-size: 13px;
  }

  /* Footer */
  footer {
    margin-top: 48px;
    padding: 20px 40px;
    border-top: 1px solid var(--border);
    text-align: center;
    font-size: 12px;
    color: var(--text-muted);
  }
</style>
</head>
<body>

<header>
  <div class="container">
    <h1>autoresearch-inference <span>Leaderboard</span></h1>
    <p>Community benchmark: maximize tok/s for open-weight LLMs on consumer and datacenter GPUs.</p>
    <div class="header-meta">
      Generated __GENERATED_AT__ &nbsp;·&nbsp;
      <a href="https://github.com/shashb27/autoresearch-inference">github.com/shashb27/autoresearch-inference</a>
    </div>
  </div>
</header>

<div class="container">

  <!-- Stats -->
  <div class="section">
    <div class="section-title">Overview</div>
    <div class="stats-grid" id="stats-grid">__STATS_CARDS__</div>
  </div>

  <!-- Charts -->
  <div class="section">
    <div class="section-title">Charts</div>
    <div class="charts-grid">

      <div class="chart-card chart-wide">
        <h3>Best tok/s vs Baseline — by GPU &amp; Model</h3>
        <div class="chart-wrapper" style="height:320px">
          <canvas id="barChart"></canvas>
        </div>
      </div>

      <div class="chart-card">
        <h3>tok/s vs Model Size (params)</h3>
        <div class="chart-wrapper" style="height:280px">
          <canvas id="scatterChart"></canvas>
        </div>
      </div>

      <div class="chart-card">
        <h3>Most-Used Optimization Techniques</h3>
        <div class="chart-wrapper" style="height:280px">
          <canvas id="techChart"></canvas>
        </div>
      </div>

    </div>
  </div>

  <!-- Table -->
  <div class="section">
    <div class="section-title">All Runs</div>
    <div class="table-controls">
      <input type="search" id="search-input" placeholder="Search GPU, model, contributor…">
      <label>GPU family:</label>
      <select id="family-filter">
        <option value="">All</option>
        <option>Ada Lovelace</option>
        <option>Hopper</option>
        <option>Ampere</option>
        <option>Turing</option>
        <option>Volta</option>
      </select>
    </div>
    <div class="table-wrap">
      <table id="runs-table">
        <thead>
          <tr>
            <th data-col="rank">#</th>
            <th data-col="contributor">Contributor</th>
            <th data-col="gpu">GPU</th>
            <th data-col="vram_gb">VRAM</th>
            <th data-col="model">Model</th>
            <th data-col="params_b">Params</th>
            <th data-col="baseline_tok_s">Baseline tok/s</th>
            <th data-col="best_tok_s" class="sorted-desc">Best tok/s</th>
            <th data-col="gain_pct">Gain</th>
            <th data-col="ttft_ms">TTFT ms</th>
            <th data-col="vram_used_gb">VRAM used</th>
            <th data-col="experiments">Expts</th>
            <th data-col="techniques">Techniques</th>
            <th data-col="submitted_at">Date</th>
          </tr>
        </thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>
  </div>

</div>

<footer>
  autoresearch-inference &nbsp;·&nbsp; Submit your run with
  <code>uv run submit_run.py</code>, then open a PR.
</footer>

<script>
// ---- Data ----
const ROWS = __TABLE_ROWS_JSON__;
const CHART = __CHART_DATA_JSON__;

// ---- Family colors ----
const FAMILY_COLORS = {
  "Ada Lovelace": "#10b981",
  "Hopper":       "#6366f1",
  "Ampere":       "#3b82f6",
  "Turing":       "#f59e0b",
  "Volta":        "#ef4444",
};
const familyColor = f => FAMILY_COLORS[f] || "#8b5cf6";

// ---- Chart.js defaults ----
Chart.defaults.color = "#8892a4";
Chart.defaults.borderColor = "#2e3150";
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 12;

// ---- Bar chart ----
(function() {
  const ctx = document.getElementById("barChart").getContext("2d");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: CHART.bar.labels,
      datasets: [
        {
          label: "Baseline tok/s",
          data: CHART.bar.baseline,
          backgroundColor: "rgba(99,102,241,0.35)",
          borderColor: "rgba(99,102,241,0.7)",
          borderWidth: 1,
          borderRadius: 3,
        },
        {
          label: "Best tok/s",
          data: CHART.bar.best,
          backgroundColor: CHART.bar.colors.map(c => c + "cc"),
          borderColor: CHART.bar.colors,
          borderWidth: 1,
          borderRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top", labels: { boxWidth: 12, padding: 16 } },
        tooltip: {
          callbacks: {
            afterBody: (items) => {
              const i = items[0].dataIndex;
              return [`Gain: +${CHART.bar.gain[i]}%`];
            },
          },
        },
      },
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 45, minRotation: 30, font: { size: 11 } } },
        y: { title: { display: true, text: "tok/s" }, grid: { color: "#2e3150" } },
      },
    },
  });
})();

// ---- Scatter chart ----
(function() {
  const ctx = document.getElementById("scatterChart").getContext("2d");
  // Group by GPU family for color
  const datasets = {};
  CHART.scatter.forEach(p => {
    const family = (function() {
      // Guess family from known GPU names — using the gpu field in scatter
      for (const [k, v] of Object.entries(FAMILY_COLORS)) {
        // We don't have family here directly; use the ROWS lookup
        const row = ROWS.find(r => r.model === p.model && r.gpu.includes(p.gpu.split(" ")[0]));
        if (row) return row.gpu_family;
      }
      return "Other";
    })();
    if (!datasets[family]) {
      datasets[family] = {
        label: family,
        data: [],
        backgroundColor: (familyColor(family) + "bb"),
        borderColor: familyColor(family),
        pointRadius: 7,
        pointHoverRadius: 9,
      };
    }
    datasets[family].data.push({ x: p.x, y: p.y, gpu: p.gpu, model: p.model, gain: p.gain });
  });

  new Chart(ctx, {
    type: "scatter",
    data: { datasets: Object.values(datasets) },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top", labels: { boxWidth: 10, padding: 12 } },
        tooltip: {
          callbacks: {
            label: (item) => {
              const d = item.raw;
              return [`${d.gpu} / ${d.model}`, `${d.x}B params → ${d.y} tok/s (+${d.gain}%)`];
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Model params (B)" },
          grid: { color: "#2e3150" },
          type: "logarithmic",
        },
        y: {
          title: { display: true, text: "Best tok/s" },
          grid: { color: "#2e3150" },
        },
      },
    },
  });
})();

// ---- Techniques chart ----
(function() {
  const ctx = document.getElementById("techChart").getContext("2d");
  const palette = ["#6366f1","#10b981","#3b82f6","#f59e0b","#ef4444","#8b5cf6",
                   "#06b6d4","#84cc16","#f97316","#ec4899","#14b8a6","#a855f7"];
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: CHART.techniques.labels,
      datasets: [{
        label: "Runs using technique",
        data: CHART.techniques.counts,
        backgroundColor: palette.slice(0, CHART.techniques.labels.length).map(c => c + "cc"),
        borderColor:     palette.slice(0, CHART.techniques.labels.length),
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: "#2e3150" }, ticks: { stepSize: 1 } },
        y: { grid: { display: false }, ticks: { font: { family: "'JetBrains Mono','Fira Code',monospace", size: 11 } } },
      },
    },
  });
})();

// ---- Table rendering ----
let sortCol = "best_tok_s";
let sortDir = -1; // -1 = desc
let filterText = "";
let filterFamily = "";

function gainClass(pct) {
  if (pct >= 50) return "high";
  if (pct >= 15) return "medium";
  return "low";
}

function rankBadge(i) {
  if (i === 0) return '<span class="rank-badge gold">1</span>';
  if (i === 1) return '<span class="rank-badge silver">2</span>';
  if (i === 2) return '<span class="rank-badge bronze">3</span>';
  return `<span class="rank-badge">${i+1}</span>`;
}

function familyDot(family) {
  return `<span class="family-dot" style="background:${familyColor(family)}"></span>`;
}

function renderTable() {
  const q = filterText.toLowerCase();
  let data = ROWS.filter(r => {
    if (filterFamily && r.gpu_family !== filterFamily) return false;
    if (q && !(
      r.gpu.toLowerCase().includes(q) ||
      r.model.toLowerCase().includes(q) ||
      r.contributor.toLowerCase().includes(q) ||
      r.techniques.toLowerCase().includes(q)
    )) return false;
    return true;
  });

  const numericCols = new Set(["vram_gb","params_b","baseline_tok_s","best_tok_s","gain_pct","ttft_ms","vram_used_gb","experiments"]);
  data.sort((a, b) => {
    let av = a[sortCol], bv = b[sortCol];
    if (numericCols.has(sortCol)) { av = Number(av); bv = Number(bv); }
    if (av < bv) return sortDir;
    if (av > bv) return -sortDir;
    return 0;
  });

  const tbody = document.getElementById("table-body");
  if (data.length === 0) {
    tbody.innerHTML = '<tr><td colspan="14" class="no-results">No matching runs.</td></tr>';
    return;
  }

  tbody.innerHTML = data.map((r, i) => `
    <tr>
      <td>${rankBadge(i)}</td>
      <td>${r.contributor}</td>
      <td>${familyDot(r.gpu_family)}${r.gpu}</td>
      <td class="mono">${r.vram_gb} GB</td>
      <td class="mono">${r.model}</td>
      <td class="mono">${r.params_b}B</td>
      <td class="mono">${r.baseline_tok_s.toFixed(2)}</td>
      <td class="mono" style="color:#e2e8f0;font-weight:600">${r.best_tok_s.toFixed(2)}</td>
      <td><span class="gain-pill ${gainClass(r.gain_pct)}">+${r.gain_pct}%</span></td>
      <td class="mono">${r.ttft_ms}</td>
      <td class="mono">${r.vram_used_gb} GB</td>
      <td class="mono">${r.experiments}</td>
      <td><div class="tech-list" title="${r.techniques}">${r.techniques}</div></td>
      <td class="mono">${r.submitted_at}</td>
    </tr>
  `).join("");
}

// Sort on header click
document.querySelectorAll("thead th[data-col]").forEach(th => {
  th.addEventListener("click", () => {
    const col = th.dataset.col;
    if (col === "rank") return;
    if (sortCol === col) {
      sortDir = -sortDir;
    } else {
      sortCol = col;
      sortDir = -1;
    }
    document.querySelectorAll("thead th").forEach(h => h.classList.remove("sorted-asc","sorted-desc"));
    th.classList.add(sortDir === -1 ? "sorted-desc" : "sorted-asc");
    renderTable();
  });
});

document.getElementById("search-input").addEventListener("input", e => {
  filterText = e.target.value;
  renderTable();
});
document.getElementById("family-filter").addEventListener("change", e => {
  filterFamily = e.target.value;
  renderTable();
});

renderTable();
</script>
</body>
</html>
"""


def build_stats_cards(stats: dict) -> str:
    def card(value, label, sub=""):
        sub_html = f'<div class="sub" title="{sub}">{sub}</div>' if sub else ""
        return f"""
    <div class="stat-card">
      <div class="value">{value}</div>
      <div class="label">{label}</div>
      {sub_html}
    </div>"""

    top_gpu_short = stats.get("top_tok_s_gpu", "").replace("NVIDIA ", "").replace("GeForce ", "")
    model_short = stats.get("top_tok_s_model", "").split("/")[-1]

    return (
        card(stats.get("total_runs", 0), "Total Runs") +
        card(stats.get("total_gpus", 0), "Unique GPUs") +
        card(stats.get("total_models", 0), "Models Tested") +
        card(stats.get("total_contributors", 0), "Contributors") +
        card(f"{stats.get('top_tok_s', 0):.1f}", "Top tok/s",
             f"{top_gpu_short} · {model_short}") +
        card(f"+{stats.get('top_gain_pct', 0):.1f}%", "Best Gain",
             stats.get("top_gain_gpu", "").replace("NVIDIA ", "").replace("GeForce ", ""))
    )


def render_html(runs: list[dict]) -> str:
    stats = stats_summary(runs)
    chart_data = build_chart_data(runs)
    table_rows = build_table_rows(runs)

    html = HTML_TEMPLATE
    html = html.replace("__GENERATED_AT__", stats.get("generated_at", ""))
    html = html.replace("__STATS_CARDS__", build_stats_cards(stats))
    html = html.replace("__TABLE_ROWS_JSON__", json.dumps(table_rows))
    html = html.replace("__CHART_DATA_JSON__", json.dumps(chart_data))
    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate the community leaderboard HTML dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run leaderboard.py
  uv run leaderboard.py --out docs/index.html
  uv run leaderboard.py --dry-run
        """,
    )
    parser.add_argument("--runs-dir", default=RUNS_DIR,
                        help=f"Directory containing run JSONs (default: {RUNS_DIR})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"Output HTML path (default: leaderboard/index.html)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats only, don't write HTML")
    args = parser.parse_args()

    print("=" * 55)
    print("  autoresearch-inference — Leaderboard Generator")
    print("=" * 55)

    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"  ERROR: No run JSONs found in {args.runs_dir}")
        print("  Submit a run first with: uv run submit_run.py")
        sys.exit(1)

    stats = stats_summary(runs)
    print(f"  Runs loaded    : {stats['total_runs']}")
    print(f"  GPUs           : {stats['total_gpus']}")
    print(f"  Models         : {stats['total_models']}")
    print(f"  Contributors   : {stats['total_contributors']}")
    print(f"  Top tok/s      : {stats['top_tok_s']:.2f}  ({stats['top_tok_s_gpu']} · {stats['top_tok_s_model'].split('/')[-1]})")
    print(f"  Best gain      : +{stats['top_gain_pct']:.1f}%  ({stats['top_gain_gpu']})")
    print()

    if args.dry_run:
        print("  DRY RUN — HTML not written.")
        return

    html = render_html(runs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(args.out) / 1024
    print(f"  Written to     : {os.path.relpath(args.out, PROJECT_DIR)}  ({size_kb:.1f} KB)")
    print()
    print("  Next steps:")
    print("    git add leaderboard/index.html")
    print("    git commit -m 'leaderboard: regenerate dashboard'")
    print("=" * 55)


if __name__ == "__main__":
    main()
