"""
analyze.py — Generate experiment run visualizations from results.tsv

Usage:
    uv run analyze.py                        # uses results.tsv, outputs to plots/
    uv run analyze.py --tsv results.tsv      # explicit input
    uv run analyze.py --out my_plots/        # explicit output dir
    uv run analyze.py --show                 # also open plots in browser
    uv run analyze.py --publish              # generate self-contained report.html

Generates:
    plots/tok_s_progression.png   — speed over time, best-so-far line
    plots/vram_vs_toks.png        — efficiency frontier (VRAM vs tok/s)
    plots/improvement_deltas.png  — % gain per kept experiment vs baseline
    plots/outcomes_donut.png      — keep / discard / crash ratio
    plots/tok_s_vs_ttft.png       — throughput vs latency dual-axis
    report.html                   — self-contained shareable report (--publish only)
"""

import argparse
import base64
import json
import os
import sys

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

STATUS_COLORS = {
    "keep":    "#2ea043",
    "discard": "#da3633",
    "crash":   "#6e7681",
}
BEST_LINE_COLOR  = "#58a6ff"
ACCENT_COLOR     = "#f0883e"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run at least one experiment first.")
        sys.exit(1)

    df = pd.read_csv(path, sep="\t")

    required = {"commit", "tok_s", "ttft_ms", "peak_vram_gb", "status", "description"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: results.tsv missing columns: {missing}")
        sys.exit(1)

    if df.empty:
        print("No experiments in results.tsv yet — nothing to plot.")
        sys.exit(0)

    df["experiment_num"] = range(1, len(df) + 1)
    df["color"] = df["status"].map(STATUS_COLORS).fillna("#8b949e")

    # Normalise status casing
    df["status"] = df["status"].str.strip().str.lower()
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig, out_dir: str, name: str, show: bool):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}")
    if show:
        fig.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. tok/s progression
# ---------------------------------------------------------------------------

def plot_tok_s_progression(df: pd.DataFrame, out_dir: str, show: bool):
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("Inference Speed Progression", fontweight="bold")

    # Best-so-far line
    best_so_far, current_best = [], 0.0
    for _, row in df.iterrows():
        if row["status"] == "keep" and row["tok_s"] > current_best:
            current_best = row["tok_s"]
        best_so_far.append(current_best)

    ax.plot(df["experiment_num"], best_so_far,
            color=BEST_LINE_COLOR, linewidth=2, label="Best tok/s so far", zorder=2)

    # All experiments (skip crash here — plotted separately at bottom with correct y)
    for status, grp in df.groupby("status"):
        if status == "crash":
            continue
        ax.scatter(grp["experiment_num"], grp["tok_s"],
                   c=STATUS_COLORS.get(status, "#8b949e"),
                   s=55, zorder=3, label=status)

    # Annotate significant jumps (>5% improvement over prev best)
    keeps = df[df["status"] == "keep"].reset_index(drop=True)
    prev_best = 0.0
    for _, row in keeps.iterrows():
        if prev_best > 0 and row["tok_s"] > prev_best * 1.05:
            gain_pct = (row["tok_s"] / keeps.iloc[0]["tok_s"] - 1) * 100
            ax.annotate(
                f"+{gain_pct:.0f}%\n{row['description'][:28]}",
                xy=(row["experiment_num"], row["tok_s"]),
                xytext=(6, 8), textcoords="offset points",
                fontsize=7, color="#c9d1d9",
                arrowprops=dict(arrowstyle="-", color="#444d56", lw=0.8),
            )
        if row["tok_s"] > prev_best:
            prev_best = row["tok_s"]

    # Exclude crash (tok_s == 0) from y-axis range so they don't compress the scale
    non_crash = df[df["tok_s"] > 0]["tok_s"]
    if not non_crash.empty:
        y_min = max(0, non_crash.min() * 0.97)
        y_max = non_crash.max() * 1.12
        ax.set_ylim(y_min, y_max)
        # Pin crashes to a narrow band at the very bottom with a distinct marker
        crashes = df[df["tok_s"] == 0]
        if not crashes.empty:
            crash_y = y_min + (y_max - y_min) * 0.015
            ax.scatter(crashes["experiment_num"], [crash_y] * len(crashes),
                       c=STATUS_COLORS["crash"], s=70, marker="X", zorder=4, label="crash",
                       edgecolors="#ffffff", linewidths=0.5)

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("tok/s")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir, "tok_s_progression.png", show)


# ---------------------------------------------------------------------------
# 2. VRAM vs tok/s efficiency frontier
# ---------------------------------------------------------------------------

def plot_vram_vs_toks(df: pd.DataFrame, out_dir: str, show: bool):
    valid = df[df["tok_s"] > 0].copy()
    if valid.empty:
        print("  Skipping vram_vs_toks.png — no successful runs yet.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("VRAM vs Speed — Efficiency Frontier", fontweight="bold")

    for status, grp in valid.groupby("status"):
        ax.scatter(grp["peak_vram_gb"], grp["tok_s"],
                   c=STATUS_COLORS.get(status, "#8b949e"),
                   alpha=0.75, s=60, label=status)

    # Number the kept points and show a legend below to avoid label overlap
    keeps = valid[valid["status"] == "keep"].reset_index(drop=True)
    for i, (_, row) in enumerate(keeps.iterrows()):
        ax.annotate(
            str(i + 1),
            xy=(row["peak_vram_gb"], row["tok_s"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=8, fontweight="bold", color="#f0e68c",
        )

    # Legend table below the chart
    legend_lines = [f"  {i+1}. {row['description'][:50]}"
                    for i, (_, row) in enumerate(keeps.iterrows())]
    legend_text = "\n".join(legend_lines)
    fig.text(0.01, -0.04 * max(1, len(keeps) // 3), legend_text,
             fontsize=7, color="#8b949e", va="top",
             transform=ax.transAxes)

    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_ylabel("tok/s")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.05 + 0.04 * len(keeps))
    _save(fig, out_dir, "vram_vs_toks.png", show)


# ---------------------------------------------------------------------------
# 3. Improvement deltas over baseline
# ---------------------------------------------------------------------------

def plot_improvement_deltas(df: pd.DataFrame, out_dir: str, show: bool):
    keeps = df[df["status"] == "keep"].copy()
    if len(keeps) < 2:
        print("  Skipping improvement_deltas.png — need ≥2 kept experiments.")
        return

    baseline = keeps.iloc[0]["tok_s"]
    keeps["pct_gain"] = (keeps["tok_s"] / baseline - 1) * 100

    fig, ax = plt.subplots(figsize=(max(10, len(keeps) * 1.1), 5))
    fig.suptitle("Cumulative Gains Over Baseline (Kept Experiments)", fontweight="bold")

    colors = [STATUS_COLORS["keep"] if g >= 0 else STATUS_COLORS["discard"]
              for g in keeps["pct_gain"]]
    ax.bar(range(len(keeps)), keeps["pct_gain"], color=colors, width=0.6)
    ax.set_xticks(range(len(keeps)))
    # Use full description but wrap long ones; rotation+alignment for readability
    ax.set_xticklabels(
        [d[:40] for d in keeps["description"]],
        rotation=42, ha="right", fontsize=7.5,
    )
    ax.axhline(0, color="#8b949e", linewidth=0.9, linestyle="--")

    # Value labels on bars
    for i, (_, row) in enumerate(keeps.iterrows()):
        va = "bottom" if row["pct_gain"] >= 0 else "top"
        offset = 0.5 if row["pct_gain"] >= 0 else -0.5
        ax.text(i, row["pct_gain"] + offset, f"{row['pct_gain']:+.1f}%",
                ha="center", va=va, fontsize=8, color="#c9d1d9")

    ax.set_ylabel("% gain over baseline")
    ax.grid(True, alpha=0.4, axis="y")
    fig.tight_layout()
    _save(fig, out_dir, "improvement_deltas.png", show)


# ---------------------------------------------------------------------------
# 4. Outcomes donut
# ---------------------------------------------------------------------------

def plot_experiment_outcomes(df: pd.DataFrame, out_dir: str, show: bool):
    counts = df["status"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(f"Experiment Outcomes  ({len(df)} total)", fontweight="bold")

    colors = [STATUS_COLORS.get(s, "#8b949e") for s in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        pctdistance=0.72,
        wedgeprops=dict(width=0.48, edgecolor="#0d1117", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color("#c9d1d9")

    fig.tight_layout()
    _save(fig, out_dir, "outcomes_donut.png", show)


# ---------------------------------------------------------------------------
# 5. tok/s vs TTFT dual-axis (latency/throughput tradeoff)
# ---------------------------------------------------------------------------

def plot_ttft_progression(df: pd.DataFrame, out_dir: str, show: bool):
    keeps = df[df["status"] == "keep"].copy()
    if len(keeps) < 2 or keeps["ttft_ms"].sum() == 0:
        print("  Skipping tok_s_vs_ttft.png — insufficient data.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Throughput vs Latency — Kept Experiments", fontweight="bold")

    # Use sequential indices so spacing is even (original exp numbers can be sparse)
    x = list(range(1, len(keeps) + 1))
    xlabels = [d[:22] for d in keeps["description"]]

    ax1.plot(x, keeps["tok_s"].values,
             "o-", color=STATUS_COLORS["keep"], linewidth=2, markersize=6, label="tok/s")
    ax1.fill_between(x, keeps["tok_s"].values, alpha=0.15, color=STATUS_COLORS["keep"])
    ax1.set_ylabel("Throughput (tok/s)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.4)

    ax2.plot(x, keeps["ttft_ms"].values,
             "s-", color=ACCENT_COLOR, linewidth=2, markersize=6, label="TTFT (ms)")
    ax2.fill_between(x, keeps["ttft_ms"].values, alpha=0.15, color=ACCENT_COLOR)
    ax2.set_ylabel("Time to First Token (ms)")
    ax2.set_xlabel("Kept experiment (sequential)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=7.5)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    _save(fig, out_dir, "tok_s_vs_ttft.png", show)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    keeps   = df[df["status"] == "keep"]
    crashes = df[df["status"] == "crash"]
    discards= df[df["status"] == "discard"]

    print("\n" + "=" * 55)
    print("  autoresearch-inference -- Run Summary")
    print("=" * 55)
    print(f"  Total experiments : {len(df)}")
    print(f"    [keep]    : {len(keeps)}")
    print(f"    [discard] : {len(discards)}")
    print(f"    [crash]   : {len(crashes)}")

    if not keeps.empty:
        baseline   = keeps.iloc[0]["tok_s"]
        best_row   = keeps.loc[keeps["tok_s"].idxmax()]
        best_tok_s = best_row["tok_s"]
        total_gain = (best_tok_s / baseline - 1) * 100 if baseline > 0 else 0

        print()
        print(f"  Baseline tok/s    : {baseline:.2f}")
        print(f"  Best tok/s        : {best_tok_s:.2f}  (+{total_gain:.1f}%)")
        print(f"  Best config       : {best_row['description']}")
        print(f"  Best VRAM         : {best_row['peak_vram_gb']:.1f} GB")

        if len(keeps) >= 2:
            gains = [(r["tok_s"] / keeps.iloc[0]["tok_s"] - 1) * 100
                     for _, r in keeps.iterrows()]
            print(f"\n  Gains per kept exp: {', '.join(f'{g:+.1f}%' for g in gains)}")

    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# HTML report (--publish)
# ---------------------------------------------------------------------------

def _img_to_b64(path: str) -> str:
    """Encode a PNG file as a base64 data URI."""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _read_file_safe(path: str, fallback: str = "") -> str:
    """Read a text file, returning fallback if it doesn't exist."""
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return fallback


def publish_report(df: pd.DataFrame, out_dir: str, tsv_path: str) -> str:
    """Generate a self-contained report.html with embedded plots and data.

    No external dependencies — uses stdlib base64 and string formatting only.
    All plots are embedded as base64 PNGs so the file is fully portable.

    Returns the path to the generated file.
    """
    project_dir = os.path.dirname(os.path.abspath(tsv_path))

    # Load context files
    config  = _read_file_safe(os.path.join(project_dir, "config.json"),  "{}")
    hw      = _read_file_safe(os.path.join(project_dir, "hardware.json"), "{}")
    learnings = _read_file_safe(os.path.join(project_dir, "LEARNINGS.md"), "*(no LEARNINGS.md found)*")

    # Parse for header stats
    try:
        cfg = json.loads(config)
        hw_cfg = json.loads(hw)
    except Exception:
        cfg, hw_cfg = {}, {}

    model_id    = cfg.get("model_id", "unknown")
    params_b    = cfg.get("model_params_b", "?")
    gpu_name    = hw_cfg.get("gpu_name", "unknown")
    vram_total  = hw_cfg.get("vram_total_gb", "?")

    keeps = df[df["status"] == "keep"]
    baseline  = keeps.iloc[0]["tok_s"] if not keeps.empty else 0
    best      = keeps["tok_s"].max() if not keeps.empty else 0
    total_gain = (best / baseline - 1) * 100 if baseline > 0 else 0
    best_desc  = keeps.loc[keeps["tok_s"].idxmax(), "description"] if not keeps.empty else "—"

    # Embed plots
    plot_names = [
        "tok_s_progression.png",
        "improvement_deltas.png",
        "vram_vs_toks.png",
        "tok_s_vs_ttft.png",
        "outcomes_donut.png",
    ]
    plot_titles = [
        "Speed Progression",
        "Gains Over Baseline",
        "VRAM vs Throughput",
        "Throughput vs Latency",
        "Experiment Outcomes",
    ]
    plot_html = ""
    for name, title in zip(plot_names, plot_titles):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            b64 = _img_to_b64(path)
            plot_html += (
                f'<div class="plot"><h3>{title}</h3>'
                f'<img src="{b64}" alt="{title}"></div>\n'
            )

    # Results table
    table_rows = ""
    for _, row in df.iterrows():
        color = {"keep": "#2ea043", "discard": "#da3633", "crash": "#6e7681"}.get(
            row["status"], "#8b949e"
        )
        table_rows += (
            f'<tr>'
            f'<td>{row["experiment_num"]}</td>'
            f'<td style="color:{color};font-weight:bold">{row["status"]}</td>'
            f'<td>{row["tok_s"]:.2f}</td>'
            f'<td>{row["ttft_ms"]:.1f}</td>'
            f'<td>{row["peak_vram_gb"]:.1f}</td>'
            f'<td class="desc">{row["description"]}</td>'
            f'<td class="commit">{str(row["commit"])[:7]}</td>'
            f'</tr>\n'
        )

    # Learnings as preformatted text (no markdown parser needed)
    learnings_html = f'<pre class="learnings">{learnings}</pre>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autoresearch-inference — {model_id}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0d1117; color: #c9d1d9; margin: 0; padding: 24px;
    line-height: 1.5;
  }}
  h1 {{ color: #f0f6fc; margin-bottom: 4px; }}
  h2 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 6px; margin-top: 32px; }}
  h3 {{ color: #8b949e; margin: 12px 0 6px; }}
  .meta {{ color: #8b949e; font-size: 0.9em; margin-bottom: 24px; }}
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin: 20px 0;
  }}
  .stat {{
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 14px; text-align: center;
  }}
  .stat .val {{ font-size: 1.8em; font-weight: bold; color: #58a6ff; }}
  .stat .lbl {{ font-size: 0.8em; color: #8b949e; margin-top: 4px; }}
  .best-desc {{
    background: #161b22; border-left: 3px solid #2ea043;
    padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-family: monospace;
  }}
  .plots {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
  .plot {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }}
  .plot img {{ width: 100%; border-radius: 4px; }}
  .plot:first-child {{ grid-column: 1 / -1; }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 0.88em;
    background: #161b22; border-radius: 6px; overflow: hidden;
  }}
  th {{
    background: #21262d; color: #8b949e; padding: 8px 10px;
    text-align: left; border-bottom: 1px solid #30363d;
  }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #21262d; }}
  td.desc {{ max-width: 320px; color: #c9d1d9; }}
  td.commit {{ font-family: monospace; color: #8b949e; font-size: 0.85em; }}
  tr:hover td {{ background: #1c2128; }}
  .context {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }}
  pre {{
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 14px; overflow-x: auto; font-size: 0.85em; color: #e6edf3;
  }}
  .learnings {{ white-space: pre-wrap; word-break: break-word; }}
  @media (max-width: 700px) {{
    .plots, .context {{ grid-template-columns: 1fr; }}
    .plot:first-child {{ grid-column: 1; }}
  }}
</style>
</head>
<body>

<h1>autoresearch-inference</h1>
<div class="meta">
  Model: <strong>{model_id}</strong> (~{params_b}B params) &nbsp;|&nbsp;
  GPU: <strong>{gpu_name}</strong> ({vram_total} GB VRAM) &nbsp;|&nbsp;
  {len(df)} experiments
</div>

<div class="stats">
  <div class="stat"><div class="val">{baseline:.1f}</div><div class="lbl">Baseline tok/s</div></div>
  <div class="stat"><div class="val">{best:.1f}</div><div class="lbl">Best tok/s</div></div>
  <div class="stat"><div class="val">+{total_gain:.1f}%</div><div class="lbl">Total gain</div></div>
  <div class="stat"><div class="val">{len(keeps)}</div><div class="lbl">Experiments kept</div></div>
  <div class="stat"><div class="val">{len(df) - len(keeps)}</div><div class="lbl">Discarded / crashed</div></div>
</div>

<div class="best-desc"><strong>Best config:</strong> {best_desc}</div>

<h2>Plots</h2>
<div class="plots">
{plot_html}
</div>

<h2>All Experiments</h2>
<table>
  <thead>
    <tr>
      <th>#</th><th>Status</th><th>tok/s</th>
      <th>TTFT (ms)</th><th>VRAM (GB)</th><th>Description</th><th>Commit</th>
    </tr>
  </thead>
  <tbody>
{table_rows}
  </tbody>
</table>

<h2>Hardware &amp; Config</h2>
<div class="context">
  <div><h3>config.json</h3><pre>{config}</pre></div>
  <div><h3>hardware.json</h3><pre>{hw}</pre></div>
</div>

<h2>LEARNINGS.md</h2>
{learnings_html}

</body>
</html>
"""

    out_path = os.path.join(project_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize autoresearch-inference experiment results"
    )
    parser.add_argument("--tsv",     default="results.tsv", help="Path to results.tsv")
    parser.add_argument("--out",     default="plots",       help="Output directory for plots")
    parser.add_argument("--show",    action="store_true",   help="Open plots after saving")
    parser.add_argument("--publish", action="store_true",
                        help="Generate self-contained report.html with embedded plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_results(args.tsv)
    print_summary(df)

    print("Generating plots...")
    plot_tok_s_progression(df,  args.out, args.show)
    plot_vram_vs_toks(df,       args.out, args.show)
    plot_improvement_deltas(df, args.out, args.show)
    plot_experiment_outcomes(df,args.out, args.show)
    plot_ttft_progression(df,   args.out, args.show)

    print(f"\nDone. All plots saved to {args.out}/")

    if args.publish:
        print("\nGenerating report.html...")
        report_path = publish_report(df, args.out, args.tsv)
        print(f"  Report: {os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
