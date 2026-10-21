"""
Benchmarking Reporting, CSV Export, and Visualization Module.

This module provides:
- CSV export for all empirical benchmark metrics.
- High-resolution PNG chart generation via Matplotlib.
- Native vector SVG chart generation (zero-dependency fallback).
- Standalone HTML performance dashboard.
- Clean ASCII terminal summary formatting.
"""

import csv
import os
from typing import Any, Dict, List, Optional, Sequence

from benchmark.runner import BenchmarkResult


def export_csv(results: Sequence[BenchmarkResult], output_path: str) -> None:
    """
    Export benchmark results to a structured CSV file.
    
    Args:
        results: Sequence of BenchmarkResult instances.
        output_path: Destination filepath for the CSV.
    """
    if not results:
        raise ValueError("Cannot export empty results list to CSV.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    records = [res.to_dict() for res in results]
    fieldnames = list(records[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def print_terminal_table(results: Sequence[BenchmarkResult], title: str = "BENCHMARK PERFORMANCE SUMMARY") -> None:
    """
    Print an aligned, human-readable ASCII summary table in the terminal.
    """
    if not results:
        print("No benchmark results to display.")
        return

    print("\n" + "=" * 98)
    print(f" {title.center(96)} ")
    print("=" * 98)
    header = (
        f"{'Experiment':<22} | {'N':>5} | {'M':>2} | {'efC':>3} | {'efS':>3} | "
        f"{'Recall@K':>8} | {'HNSW (ms)':>9} | {'BF (ms)':>8} | {'Speedup':>7} | {'Comps Reduc':>11}"
    )
    print(header)
    print("-" * 98)

    for r in results:
        line = (
            f"{r.experiment_name:<22} | {r.dataset_size:>5} | {r.M:>2} | {r.ef_construction:>3} | {r.ef_search:>3} | "
            f"{r.recall_at_k * 100:>7.1f}% | {r.hnsw_latencies.mean_ms:>9.3f} | {r.bf_latencies.mean_ms:>8.3f} | "
            f"{r.speedup_ratio:>6.1f}x | {r.distance_comp_reduction_ratio:>10.1f}x"
        )
        print(line)
    print("=" * 98 + "\n")


def generate_png_charts(
    scaling_results: Optional[Sequence[BenchmarkResult]] = None,
    ef_search_results: Optional[Sequence[BenchmarkResult]] = None,
    ef_const_results: Optional[Sequence[BenchmarkResult]] = None,
    output_dir: str = "results",
) -> List[str]:
    """
    Generate publication-quality PNG charts using Matplotlib.
    
    Returns:
        List of generated image file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    os.makedirs(output_dir, exist_ok=True)
    generated_files: List[str] = []

    # Chart 1: Latency Scaling vs Dataset Size N (HNSW vs Brute Force)
    if scaling_results and len(scaling_results) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
        
        n_vals = [r.dataset_size for r in scaling_results]
        hnsw_lats = [r.hnsw_latencies.mean_ms for r in scaling_results]
        bf_lats = [r.bf_latencies.mean_ms for r in scaling_results]
        hnsw_comps = [r.hnsw_avg_distance_comps_per_query for r in scaling_results]
        bf_comps = [r.bf_avg_distance_comps_per_query for r in scaling_results]

        # Subplot 1: Latency (ms)
        ax1.plot(n_vals, hnsw_lats, marker="o", color="#2563eb", linewidth=2.2, label="HNSW (O(log N))")
        ax1.plot(n_vals, bf_lats, marker="s", color="#dc2626", linewidth=2.2, linestyle="--", label="Exact Brute Force (O(N))")
        ax1.set_xlabel("Dataset Size (N)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Mean Query Latency (ms)", fontsize=11, fontweight="bold")
        ax1.set_title("Query Latency vs Dataset Size", fontsize=12, fontweight="bold", pad=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", frameon=True)

        # Subplot 2: Distance Computations per Query
        ax2.plot(n_vals, hnsw_comps, marker="o", color="#2563eb", linewidth=2.2, label="HNSW Distance Evaluations")
        ax2.plot(n_vals, bf_comps, marker="s", color="#dc2626", linewidth=2.2, linestyle="--", label="Brute Force Distance Evaluations (= N)")
        ax2.set_xlabel("Dataset Size (N)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Avg Distance Evaluations / Query", fontsize=11, fontweight="bold")
        ax2.set_title("Computational Work vs Dataset Size", fontsize=12, fontweight="bold", pad=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", frameon=True)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "chart_latency_vs_dataset_size.png")
        plt.savefig(chart_path)
        plt.close(fig)
        generated_files.append(chart_path)

    # Chart 2: Accuracy vs Speed Trade-off (Recall vs ef_search)
    if ef_search_results and len(ef_search_results) >= 2:
        fig, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=150)
        
        ef_vals = [r.ef_search for r in ef_search_results]
        recalls = [r.recall_at_k * 100 for r in ef_search_results]
        qps_vals = [r.hnsw_latencies.qps for r in ef_search_results]

        color1 = "#16a34a"
        ax1.set_xlabel("efSearch (Beam Search Width at Layer 0)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Recall@K (%)", color=color1, fontsize=11, fontweight="bold")
        line1 = ax1.plot(ef_vals, recalls, marker="o", color=color1, linewidth=2.5, label="Recall@K (%)")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(min(recalls) - 5, 102)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2 = ax1.twinx()
        color2 = "#9333ea"
        ax2.set_ylabel("Throughput (Queries Per Second)", color=color2, fontsize=11, fontweight="bold")
        line2 = ax2.plot(ef_vals, qps_vals, marker="^", color=color2, linewidth=2.0, linestyle="--", label="Throughput (QPS)")
        ax2.tick_params(axis="y", labelcolor=color2)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="lower right", frameon=True)
        ax1.set_title("Trade-off Curve: Recall@K vs Throughput across efSearch", fontsize=12, fontweight="bold", pad=12)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "chart_tradeoff_recall_vs_efsearch.png")
        plt.savefig(chart_path)
        plt.close(fig)
        generated_files.append(chart_path)

    # Chart 3: efConstruction vs Index Build Time & Recall
    if ef_const_results and len(ef_const_results) >= 2:
        fig, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=150)
        
        ef_c_vals = [r.ef_construction for r in ef_const_results]
        build_times = [r.hnsw_build_time_sec for r in ef_const_results]
        recalls = [r.recall_at_k * 100 for r in ef_const_results]

        color1 = "#ea580c"
        ax1.set_xlabel("efConstruction (Candidate List Size during Build)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Index Build Time (seconds)", color=color1, fontsize=11, fontweight="bold")
        line1 = ax1.plot(ef_c_vals, build_times, marker="s", color=color1, linewidth=2.2, label="Build Time (s)")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2 = ax1.twinx()
        color2 = "#0284c7"
        ax2.set_ylabel("Resulting Recall@K (%)", color=color2, fontsize=11, fontweight="bold")
        line2 = ax2.plot(ef_c_vals, recalls, marker="o", color=color2, linewidth=2.2, linestyle="--", label="Recall@K (%)")
        ax2.tick_params(axis="y", labelcolor=color2)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="lower right", frameon=True)
        ax1.set_title("Build Time & Index Quality vs efConstruction", fontsize=12, fontweight="bold", pad=12)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "chart_build_time_vs_efconstruction.png")
        plt.savefig(chart_path)
        plt.close(fig)
        generated_files.append(chart_path)

    return generated_files


def generate_svg_chart_scaling(scaling_results: Sequence[BenchmarkResult], output_path: str) -> None:
    """
    Generate a pure-SVG vector chart comparing HNSW vs Brute Force query latency.
    Zero external dependencies.
    """
    if not scaling_results or len(scaling_results) < 2:
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n_vals = [r.dataset_size for r in scaling_results]
    hnsw_lats = [r.hnsw_latencies.mean_ms for r in scaling_results]
    bf_lats = [r.bf_latencies.mean_ms for r in scaling_results]

    min_n, max_n = min(n_vals), max(n_vals)
    max_lat = max(max(hnsw_lats), max(bf_lats)) * 1.15
    if max_lat <= 0:
        max_lat = 1.0

    width, height = 750, 420
    pad_left, pad_right, pad_top, pad_bottom = 80, 50, 60, 60
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def map_x(n: float) -> float:
        return pad_left + ((n - min_n) / (max_n - min_n if max_n > min_n else 1)) * plot_w

    def map_y(lat: float) -> float:
        return (pad_top + plot_h) - (lat / max_lat) * plot_h

    hnsw_pts = " ".join(f"{map_x(n):.1f},{map_y(l):.1f}" for n, l in zip(n_vals, hnsw_lats))
    bf_pts = " ".join(f"{map_x(n):.1f},{map_y(l):.1f}" for n, l in zip(n_vals, bf_lats))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#ffffff; font-family:sans-serif;">',
        f'  <text x="{width/2}" y="32" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">Query Latency vs Dataset Size (N)</text>',
        f'  <text x="{width/2}" y="50" text-anchor="middle" font-size="12" fill="#64748b">Empirical Comparison: Sublinear HNSW vs Linear Brute-Force Scan</text>',
        # Grid lines
        f'  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_h}" stroke="#cbd5e1" stroke-width="1.5" />',
        f'  <line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{pad_left + plot_w}" y2="{pad_top + plot_h}" stroke="#cbd5e1" stroke-width="1.5" />',
    ]

    # Y-axis ticks
    for i in range(5):
        val = max_lat * (i / 4.0)
        y_pos = map_y(val)
        svg.append(f'  <line x1="{pad_left - 5}" y1="{y_pos:.1f}" x2="{pad_left + plot_w}" y2="{y_pos:.1f}" stroke="#f1f5f9" stroke-width="1" />')
        svg.append(f'  <text x="{pad_left - 10}" y="{y_pos + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{val:.2f}ms</text>')

    # X-axis ticks
    for n in n_vals:
        x_pos = map_x(n)
        svg.append(f'  <line x1="{x_pos:.1f}" y1="{pad_top + plot_h}" x2="{x_pos:.1f}" y2="{pad_top + plot_h + 5}" stroke="#94a3b8" />')
        svg.append(f'  <text x="{x_pos:.1f}" y="{pad_top + plot_h + 20}" text-anchor="middle" font-size="11" fill="#64748b">N={n}</text>')

    # Data lines
    svg.append(f'  <polyline fill="none" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5,5" points="{bf_pts}" />')
    svg.append(f'  <polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{hnsw_pts}" />')

    # Points
    for n, l in zip(n_vals, bf_lats):
        svg.append(f'  <circle cx="{map_x(n):.1f}" cy="{map_y(l):.1f}" r="4" fill="#dc2626" />')
    for n, l in zip(n_vals, hnsw_lats):
        svg.append(f'  <circle cx="{map_x(n):.1f}" cy="{map_y(l):.1f}" r="4" fill="#2563eb" />')

    # Legend
    leg_x = pad_left + 20
    leg_y = pad_top + 20
    svg.extend([
        f'  <rect x="{leg_x}" y="{leg_y}" width="220" height="60" rx="4" fill="#f8fafc" stroke="#e2e8f0" />',
        f'  <line x1="{leg_x + 15}" y1="{leg_y + 20}" x2="{leg_x + 45}" y2="{leg_y + 20}" stroke="#2563eb" stroke-width="2.5" />',
        f'  <circle cx="{leg_x + 30}" cy="{leg_y + 20}" r="3.5" fill="#2563eb" />',
        f'  <text x="{leg_x + 55}" y="{leg_y + 24}" font-size="12" fill="#1e293b">HNSW Graph O(log N)</text>',
        f'  <line x1="{leg_x + 15}" y1="{leg_y + 42}" x2="{leg_x + 45}" y2="{leg_y + 42}" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="4,4" />',
        f'  <circle cx="{leg_x + 30}" cy="{leg_y + 42}" r="3.5" fill="#dc2626" />',
        f'  <text x="{leg_x + 55}" y="{leg_y + 46}" font-size="12" fill="#1e293b">Brute Force O(N)</text>',
        '</svg>',
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def generate_html_report(
    results: Sequence[BenchmarkResult],
    system_info: Dict[str, Any],
    output_path: str,
    chart_filenames: Optional[List[str]] = None,
) -> None:
    """
    Generate a complete, self-contained HTML performance report.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    charts = chart_filenames or []

    rows_html = []
    for r in results:
        rows_html.append(
            f"<tr>"
            f"<td><strong>{r.experiment_name}</strong></td>"
            f"<td>{r.dataset_size}</td>"
            f"<td>{r.M}</td>"
            f"<td>{r.ef_construction}</td>"
            f"<td>{r.ef_search}</td>"
            f"<td class='highlight-recall'>{r.recall_at_k * 100:.1f}%</td>"
            f"<td>{r.hnsw_latencies.mean_ms:.3f} ms</td>"
            f"<td>{r.hnsw_latencies.p95_ms:.3f} ms</td>"
            f"<td>{r.hnsw_latencies.qps:.1f}</td>"
            f"<td>{r.bf_latencies.mean_ms:.3f} ms</td>"
            f"<td class='highlight-speedup'>{r.speedup_ratio:.1f}x</td>"
            f"<td>{r.distance_comp_reduction_ratio:.1f}x</td>"
            f"<td>{r.hnsw_peak_memory_mb:.2f} MB</td>"
            f"</tr>"
        )

    charts_html = []
    for c in charts:
        basename = os.path.basename(c)
        charts_html.append(
            f"<div class='chart-card'><h4>{basename.replace('.png', '').replace('chart_', '').replace('_', ' ').title()}</h4>"
            f"<img src='{basename}' alt='{basename}' /></div>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HNSW Search Engine Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 24px; background: #f8fafc; color: #1e293b; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 26px; color: #0f172a; margin-bottom: 8px; }}
  .subtitle {{ color: #64748b; margin-top: 0; margin-bottom: 24px; font-size: 15px; }}
  .card {{ background: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; padding: 20px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  .card h3 {{ margin-top: 0; color: #1e293b; font-size: 18px; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px; }}
  .sys-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
  .sys-item {{ background: #f1f5f9; padding: 10px 14px; border-radius: 6px; font-size: 13px; }}
  .sys-item strong {{ display: block; color: #475569; font-size: 11px; text-transform: uppercase; margin-bottom: 2px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
  th {{ background: #f8fafc; color: #475569; font-weight: 600; padding: 10px 12px; border-bottom: 2px solid #e2e8f0; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #f1f5f9; }}
  tr:hover {{ background: #f8fafc; }}
  .highlight-recall {{ color: #16a34a; font-weight: 600; }}
  .highlight-speedup {{ color: #2563eb; font-weight: 600; }}
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }}
  .chart-card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; text-align: center; }}
  .chart-card img {{ max-width: 100%; height: auto; border-radius: 4px; }}
  .chart-card h4 {{ margin: 0 0 10px 0; font-size: 14px; color: #334155; }}
  .note-box {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 4px; font-size: 13px; color: #1e40af; margin-top: 16px; line-height: 1.5; }}
</style>
</head>
<body>
<div class="container">
  <h1>HNSW Search Engine Benchmark Report</h1>
  <p class="subtitle">Empirical Performance, Accuracy (Recall@K), and Trade-off Analysis against Exact Baseline</p>

  <div class="card">
    <h3>Hardware & Software Test Environment</h3>
    <div class="sys-grid">
      <div class="sys-item"><strong>Operating System</strong>{system_info.get('os_name')} {system_info.get('os_release')} ({system_info.get('machine')})</div>
      <div class="sys-item"><strong>Processor</strong>{system_info.get('processor')}</div>
      <div class="sys-item"><strong>CPU Cores</strong>{system_info.get('cpu_count')}</div>
      <div class="sys-item"><strong>Total Memory</strong>{system_info.get('total_ram_gb')} GB RAM</div>
      <div class="sys-item"><strong>Python Version</strong>{system_info.get('python_version')}</div>
      <div class="sys-item"><strong>Benchmark Timestamp</strong>{system_info.get('timestamp')}</div>
    </div>
  </div>

  <div class="card">
    <h3>Empirical Results Table</h3>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Experiment</th>
            <th>N</th>
            <th>M</th>
            <th>efC</th>
            <th>efS</th>
            <th>Recall@K</th>
            <th>HNSW Mean</th>
            <th>HNSW p95</th>
            <th>HNSW QPS</th>
            <th>BF Mean</th>
            <th>Speedup</th>
            <th>Dist Reduction</th>
            <th>Peak RAM</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
    <div class="note-box">
      <strong>Algorithmic Analysis:</strong> HNSW trades search completeness for logarithmic query complexity. Because search paths are guided by local greedy graph routing, lower values of <code>efSearch</code> may miss global optima. Increasing <code>efSearch</code> smoothly improves Recall@K toward 100% while gracefully bounding distance computations compared to exhaustive O(N) brute force.
    </div>
  </div>

  <div class="card">
    <h3>Performance Visualizations</h3>
    <div class="charts-grid">
      {"".join(charts_html) if charts_html else "<p>Charts not rendered or Matplotlib unavailable.</p>"}
    </div>
  </div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
