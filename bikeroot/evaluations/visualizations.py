import os, pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List, Dict, Literal

"""
visualizations.py – helper utilities for loading `Evaluation` pickle files and
plotting/aggregating their time + quality metrics.

🔄 **Update (July 2025)**
------------------------------------------------------------
* Added utilities grouped *by the pre‑computed* ``dist_group`` column instead
  of manually binning by path length.
* New public helpers
    * ``compute_metric_by_group`` – aggregate *any* metric (eg. ``hypervolume``
      or ``count``) over the trials **within each ``dist_group`` & algorithm**.
    * ``plot_metric_by_group`` – quick bar/line visualisation of the aggregated
      table with optional log‑y scale (base *e* by default).

The rest of the original API is kept intact.
"""

# ---------------------------------------------------------------------
# ───────────────────────────  DATA LOADING  ───────────────────────────
# ---------------------------------------------------------------------

def load_evaluation_results(folder_path: str) -> pd.DataFrame:
    """Load every ``*.pkl``/``*.pickle`` in *folder_path* and explode trials into
    a tidy DataFrame (one row = one algorithm × trial)."""
    records: List[dict] = []

    for filename in os.listdir(folder_path):
        if not filename.endswith((".pickle", ".pkl")):
            continue
        fpath = os.path.join(folder_path, filename)
        try:
            eval_obj = pd.read_pickle(fpath)
        except Exception as e:  # pragma: no cover – robust against bad files
            print(f"Could not load {filename}: {e}")
            continue

        # Scenario‑level meta
        scenario_id = filename  # simple id – feel free to change
        graph_name = getattr(eval_obj, "graph_name", None)
        location   = getattr(eval_obj, "location",   None)
        src        = getattr(eval_obj, "src",        None)
        dst        = getattr(eval_obj, "dst",        None)
        dist_group = getattr(eval_obj, "dist_group", None)
        lent       = getattr(eval_obj, "lent",       None)

        criteria_list = getattr(eval_obj, "criteria", None) or []
        n_objectives  = len(criteria_list) if criteria_list else None
        try:
            criteria_names = [c.name if hasattr(c, "name") else str(c)
                              for c in criteria_list]
        except Exception:
            criteria_names = [str(c) for c in criteria_list]

        # --- explode trials ---
        for algo_name, result in getattr(eval_obj, "results", {}).items():
            for trial_idx, trial in enumerate(result.get("trials", [])):
                fail_info  = trial.get("fail")
                fail       = bool(fail_info)
                fail_reason = None if not fail else (
                    fail_info if isinstance(fail_info, str)
                    else fail_info.get("error", str(fail_info)) if isinstance(fail_info, dict)
                    else str(fail_info)
                )

                records.append({
                    "scenario": scenario_id,
                    "graph_name": graph_name,
                    "location": location,
                    "src": src,
                    "dst": dst,
                    "dist_group": dist_group,
                    "lent": lent,
                    "n_objectives": n_objectives,
                    "criteria": criteria_names,
                    "algorithm": algo_name,
                    "trial": trial_idx,
                    "time": trial.get("time"),
                    "hypervolume": trial.get("hypervolume"),
                    "count": trial.get("count"),
                    "fail": fail,
                    "fail_reason": fail_reason,
                })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------
# ─────────────────────────  COLOUR UTILITIES  ─────────────────────────
# ---------------------------------------------------------------------

def make_algorithm_color_dict(df: pd.DataFrame) -> Dict[str, tuple]:
    """Return *consistent* colour mapping for each algorithm."""
    algs = sorted(df["algorithm"].dropna().unique())
    return dict(zip(algs, sns.color_palette("tab10", len(algs))))

# ---------------------------------------------------------------------
# ────────────────────────  BASIC CLEANING HELP  ───────────────────────
# ---------------------------------------------------------------------

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove failed trials & treat "0 hypervolume" as missing (for means)."""
    df_ok = df[~df["fail"]].copy()
    df_ok.loc[df_ok["hypervolume"] == 0, "hypervolume"] = pd.NA
    return df_ok

# ---------------------------------------------------------------------
# ─────────────────── EXISTING time / LEN PLOTS ────────────────────
# ---------------------------------------------------------------------
# (unchanged – scroll past for new additions)

# ...[the rest of the original functions up to plot_ndcount_vs_dist_group stay
# exactly as they were]...

# ---------------------------------------------------------------------
# ───────────────────── NEW GROUP‑BY DISTANCE TOOLS ─────────────────────
# ---------------------------------------------------------------------

Agg = Literal["mean", "median", "max", "min", "sum"]

def compute_metric_by_group(
    df: pd.DataFrame,
    metric: str,
    *,
    agg: Agg = "mean",
    group_col: str = "dist_group",
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute *agg* of *metric* per ``(group_col, algorithm)`` after dropping
    failed trials and missing values.

    Example
    --------
    >>> tbl = compute_metric_by_group(df, "hypervolume", agg="mean")
    >>> tbl.head()
         dist_group algorithm   mean_hypervolume
    0         500m   namoa_tc        4.8e+10
    """
    df_ok = _clean_df(df)

    if metric not in df_ok.columns:
        raise KeyError(f"metric '{metric}' not present in DataFrame")

    out_col = f"{agg}_{metric}"

    tbl = (
        df_ok.groupby([group_col, "algorithm"])
              .agg(**{out_col: (metric, agg)})
              .reset_index()              # flat table for easy plotting
              .sort_values([group_col, out_col], ascending=[True, False])
    )

    if verbose:
        print(f"\n=== {metric} ({agg}) by {group_col} ===")
        print(tbl.to_string(index=False))

    return tbl


def plot_metric_by_group(
    df: pd.DataFrame,
    metric: str,
    *,
    agg: Agg = "mean",
    group_col: str = "dist_group",
    color_dict: Optional[Dict[str, tuple]] = None,
    log_y: bool | None = None,
    ax: Optional[plt.Axes] = None,
):
    """Bar/line chart of *metric* aggregated per ``dist_group`` & algorithm.

    Parameters
    ----------
    df : tidy DataFrame from ``load_evaluation_results``
    metric : column to aggregate ("hypervolume", "count", ...)
    agg : aggregation – default *mean*
    log_y : set ``True`` for log‑scale *y*.  If *None*, we default to
             ``True`` when *metric* == "hypervolume".
    """
    tbl = compute_metric_by_group(df, metric, agg=agg, group_col=group_col)
    agg_col = f"{agg}_{metric}"

    # Choose colour mapping
    if color_dict is None:
        color_dict = make_algorithm_color_dict(df)
    alg_order = list(color_dict.keys())

    # ▁▂▃▅ Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=tbl,
        x=group_col,
        y=agg_col,
        hue="algorithm",
        marker="o",
        palette=color_dict,
        hue_order=alg_order,
        ax=ax,
    )

    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylabel(f"{agg.capitalize()} {metric}")
    ax.set_title(f"{metric.title()} ({agg}) by {group_col}")

    # Decide log‑scale
    if log_y is None:
        log_y = metric.lower() == "hypervolume"
    if log_y:
        # Matplotlib ≥3.3 lets us pick base.  Use natural log so ticks are e^k.
        pos_vals = tbl[agg_col] > 0
        if pos_vals.any():
            ax.set_yscale("log", base=np.e)
        else:
            print("[plot_metric_by_group] All values <= 0 – cannot log‑scale.")

    ax.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return ax
# visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────  BASIC CLEANING HELP  ─────────────────────────

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with failed trials removed & obvious placeholder
    metrics marked as missing (for proper means/std)."""
    df_ok = df[~df["fail"]].copy()

    # Treat “0” placeholders as missing
    df_ok.loc[df_ok["hypervolume"] == 0, "hypervolume"] = pd.NA
    df_ok.loc[df_ok["time"] == 0,          "time"]       = pd.NA

    return df_ok


# ─────────────────  PER-LOCATION PERFORMANCE (NEW BEHAVIOUR) ────────────────

def plot_location_performance(
    df: pd.DataFrame,
    location: str,
    colors: dict | None = None,
    *,
    agg: str = "mean",          # mean / median
    show_std: bool = True,
    verbose: bool = False,
):
    """
    For *location*, draw **three separate figures**:

        • Hypervolume vs Route Distance  
        • Runtime (s) vs Route Distance  
        • # Non-dominated Solutions vs Route Distance

    Failed trials are stripped out, and 0-placeholder metrics ignored.
    """
    # ── filter + clean ───────────────────────────────────────────────────
    df_ok  = _clean_df(df)
    df_loc = df_ok[df_ok["location"] == location]
    if df_loc.empty:
        print(f"[plot_location_performance] No data for location “{location}”.")
        return

    if verbose:
        print(f"{len(df_loc)} rows for {location} after cleaning.")

    metrics = ["hypervolume", "time", "count"]
    ylabels = ["Hypervolume", "Runtime (s)", "# Non-dominated Solutions"]

    for metric, ylabel in zip(metrics, ylabels):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for algo, g in df_loc.groupby("algorithm"):
            g = g.sort_values("dist_group")
            series = g.groupby("dist_group")[metric]

            y = getattr(series, agg)()
            x = y.index.values              # dist_group

            col = (
                colors[algo] if colors and algo in colors
                else None
            )

            ax.plot(x, y.values, marker="o", label=algo, color=col)

            if show_std and metric in {"hypervolume", "time"}:
                std = series.std()
                ax.fill_between(x, y - std, y + std, alpha=0.20, color=col)

        ax.set_xlabel("Route Distance (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Path Length — {location}")
        ax.grid(True)
        ax.legend(fontsize="small")
        plt.tight_layout()
        plt.show()
        fname = f"figs/{location.replace(', ', '_')}_{metric}.png"   # or ".png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")    


def compute_metrics_by_group(df, groupby_cols=['algorithm'], metrics=['hypervolume', 'time', 'count']):
    """
    Compute aggregate metrics (mean, std, count) grouped by given columns.
    Returns a DataFrame with aggregated statistics.
    
    Parameters:
    - df: pandas DataFrame with performance data.
    - groupby_cols: list of column names to group by.
    - metrics: list of metric column names to aggregate.
    """
    # Group by specified columns and compute statistics
    agg_funcs = {}
    for metric in metrics:
        agg_funcs[metric] = ['mean', 'std', 'count']
    agg_df = df.groupby(groupby_cols).agg(agg_funcs)
    # Flatten column MultiIndex
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()
    return agg_df

# Alias for convenience if called "computer_metric_by_group"
computer_metric_by_group = compute_metrics_by_group


def plot_overall_bar_chart(df, metric, colors=None):
    """
    Plot a bar chart comparing mean of a given metric across algorithms.
    
    Parameters:
    - df: pandas DataFrame containing 'algorithm' and the specified metric columns.
    - metric: the metric column name (e.g., 'hypervolume', 'time', 'count').
    - colors: dict mapping algorithm names to colors.
    """
    # Compute mean metric by algorithm
    mean_vals = df.groupby('algorithm')[metric].mean()
    std_vals = df.groupby('algorithm')[metric].std()
    # Sort by mean descending for clarity
    mean_vals = mean_vals.sort_values(ascending=False)
    std_vals = std_vals[mean_vals.index]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors_list = []
    for algo in mean_vals.index:
        if colors and algo in colors:
            colors_list.append(colors[algo])
        else:
            colors_list.append('C0')
    ax.bar(mean_vals.index, mean_vals.values, yerr=std_vals.values,
           color=colors_list, alpha=0.7, capsize=5)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(metric.replace('_', ' ').capitalize())
    ax.set_title(f"Average {metric.replace('_', ' ').capitalize()} by Algorithm")
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y')
    plt.tight_layout()
    plt.show()


def plot_hypervolume_vs_time(df, colors=None):
    """
    Scatter plot of average hypervolume vs average time for each algorithm.
    
    Parameters:
    - df: pandas DataFrame containing columns ['algorithm', 'hypervolume', 'time'].
    - colors: dict mapping algorithm names to colors.
    """
    # Compute mean hypervolume and mean time for each algorithm
    summary = df.groupby('algorithm').agg({'hypervolume': 'mean', 'time': 'mean'}).reset_index()
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, row in summary.iterrows():
        algo = row['algorithm']
        hv = row['hypervolume']
        rt = row['time']
        color = colors[algo] if colors and algo in colors else 'C0'
        ax.scatter(rt, hv, s=50, color=color)
        ax.text(rt, hv, algo, fontsize=8, ha='right', va='bottom')
    ax.set_xlabel("Average time (s)")
    ax.set_ylabel("Average Hypervolume")
    ax.set_title("Algorithm Performance: Hypervolume vs time")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_combined_performance_vs_distance(df, metric, colors=None):
    """
    Plot a line chart of a metric vs path length aggregated across all locations.
    
    Parameters:
    - df: pandas DataFrame containing 'dist_group', 'algorithm', and metric column.
    - metric: metric to plot ('hypervolume' or 'time').
    - colors: dict mapping algorithm names to colors.
    """
    # Compute mean metric by algorithm and dist_group across all locations
    df_group = df.groupby(['algorithm', 'dist_group'])[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6,4))
    for algo, group in df_group.groupby('algorithm'):
        group = group.sort_values('dist_group')
        x = group['dist_group']
        y = group[metric]
        color = colors[algo] if colors and algo in colors else None
        ax.plot(x, y, marker='o', label=algo, color=color)
    ax.set_xlabel("Route Distance (m)")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Mean {metric.capitalize()} vs Path Length (All Locations)")
    ax.legend(fontsize='small')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def compare_short_vs_long(df, short_distance, long_distance, metrics=['hypervolume', 'time']):
    """
    Compare algorithm performance on short and long routes.
    Computes average metric at short and long distances and the percentage change.
    
    Parameters:
    - df: pandas DataFrame containing 'algorithm', 'dist_group', and metric columns.
    - short_distance: the dist_group value considered as short route.
    - long_distance: the dist_group value considered as long route.
    - metrics: list of metrics to compare (e.g. ['hypervolume', 'time']).
    
    Returns:
    - pandas DataFrame summarizing performance at short vs long routes for each algorithm.
    """
    # Filter for short and long distances
    df_short = df[df['dist_group'] == short_distance]
    df_long = df[df['dist_group'] == long_distance]
    records = []
    for algo, group_s in df_short.groupby('algorithm'):
        group_l = df_long[df_long['algorithm'] == algo]
        if group_l.empty:
            continue
        record = {'algorithm': algo}
        for metric in metrics:
            mean_s = group_s[metric].mean()
            mean_l = group_l[metric].mean()
            pct_change = (mean_l - mean_s) / mean_s * 100 if mean_s != 0 else np.nan
            record[f'{metric}_short'] = mean_s
            record[f'{metric}_long'] = mean_l
            record[f'{metric}_pct_change'] = pct_change
        records.append(record)
    return pd.DataFrame(records)
# ---------------------------------------------------------------------
# ─────────────  PER-LOCATION SUMMARY  →  LaTeX  TABLE  ───────────────
# ---------------------------------------------------------------------

def _compute_location_metrics(
    df: pd.DataFrame,
    metrics: list[str] = ["hypervolume", "time", "count"],
    agg: str = "mean",
) -> pd.DataFrame:
    """Return <location × metric> table (means, medians, … as chosen)."""
    df_ok = _clean_df(df)
    agg_funcs = {m: agg for m in metrics}
    tbl = (
        df_ok.groupby("location")
             .agg(agg_funcs)
             .reset_index()
             .rename(columns={m: f"{agg}_{m}" for m in metrics})
    )
    return tbl


def save_location_metrics_tex(
    df: pd.DataFrame,
    filepath: str = "location_metrics.tex",
    metrics: list[str] | None = None,
    agg: str = "mean",
    *,
    caption: str = "Mean performance metrics per location "
                   "(failed trials removed)",
    label: str = "tab:location_metrics",
    float_fmt: str = "{:,.2f}",
) -> str:
    """
    Compute the <location × metric> table and write it to *filepath*
    as a stand-alone LaTeX table environment.

    Returns the LaTeX string (so notebooks can `print` it too).
    """
    if metrics is None:
        metrics = ["hypervolume", "time", "count"]

    tbl = _compute_location_metrics(df, metrics=metrics, agg=agg)

    # ── build LaTeX code ------------------------------------------------
    col_spec = "|l|" + "|".join("r" for _ in metrics) + "|"
    header   = ["\\textbf{Location}"] + [
        f"\\textbf{{{m.capitalize()}}}" for m in metrics
    ]

    lines = [
        "\\begin{table}[h!]",
        "    \\centering",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "        \\hline",
        "        " + " & ".join(header) + " \\\\",
        "        \\hline",
    ]

    for _, row in tbl.iterrows():
        vals = [row["location"]]
        for m in metrics:
            col = f"{agg}_{m}"
            val = row[col]
            vals.append(float_fmt.format(val) if pd.notna(val) else "--")
        lines.append("        " + " & ".join(vals) + " \\\\")
    lines.append("        \\hline")
    lines.append("    \\end{tabular}")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append("\\end{table}")

    tex_str = "\n".join(lines)

    # write to disk
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(tex_str)

    return tex_str


from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────────────  LaTeX helpers (add once)  ─────────────────────
import re, numpy as np, pandas as pd
_LATEX_ESC = {
    '&':  r'\&',  '%':  r'\%',  '$':  r'\$',  '#':  r'\#',
    '_':  r'\_',  '{':  r'\{',  '}':  r'\}',  '~':  r'\textasciitilde{}',
    '^':  r'\textasciicircum{}', '\\': r'\textbackslash{}',
}
def _tex(s: str) -> str:
    """Escape every TeX-special character in *s*."""
    return re.sub(r'[&%$#_\{}~^\\]', lambda m: _LATEX_ESC[m.group()], s)


# ───────────  RAW-DATA → longtable  (dist_group-sorted)  ────────────
from pathlib import Path

def save_raw_tables_by_location_tex(
    df: pd.DataFrame,
    out_dir: str = "figs/raw_tables",
    *,
    metrics: list[str] | None = None,
    include_trial_idx: bool = True,
    float_fmt: str = "{:,.4g}",   # for time / count
    exp_fmt:   str = "{:,.2f}",   # precision for ln(HV)
    vbars: bool = True,
):
    metrics = metrics or ["hypervolume", "time", "count"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df_ok = _clean_df(df)                      # your existing cleaner

    # column order for the table
    col_order = ["algorithm", "dist_group"]
    if include_trial_idx:
        col_order.append("trial")
    col_order += metrics

    files = []
    for loc in sorted(df_ok["location"].dropna().unique()):
        sub = df_ok[df_ok["location"] == loc][col_order]
        if sub.empty:
            continue

        # sort rows so the reader scans shortest → longest distance
        sort_by = ["dist_group", "algorithm"] + (["trial"] if include_trial_idx else [])
        sub = sub.sort_values(sort_by, kind="stable")

        slug = loc.replace(", ", "_").replace(" ", "_")
        path = Path(out_dir) / f"{slug}_raw.tex"

        # column specification: |l|r|r|…|
        spec = (
            "|l|" + "|".join("r" for _ in col_order[1:]) + "|"
            if vbars else
            "l"   +        "".join("r" for _ in col_order[1:])
        )

        header = ["\\textbf{" + _tex(c.title().replace('_', ' ')) + "}"
                  for c in col_order]

        lines = []
        lines.append(f"\\begin{{longtable}}{{{spec}}}")
        lines.append("    \\caption{Raw trial metrics for "
                     f"{_tex(loc)} ($\\mathrm{{e}}^x$ for hypervolume)\\\\")
        lines.append("    \\hline")
        lines.append("    " + " & ".join(header) + " \\\\")
        lines.append("    \\hline\\endfirsthead")
        lines.append("    \\hline")
        lines.append("    " + " & ".join(header) + " \\\\")
        lines.append("    \\hline\\endhead")
        lines.append("    \\hline\\endfoot")
        lines.append("    \\hline\\endlastfoot")

        # rows
        for _, row in sub.iterrows():
            cells = []
            for col in col_order:
                val = row[col]
                if pd.isna(val):
                    cells.append("--")
                elif col == "algorithm":
                    cells.append(r"\texttt{" + _tex(str(val)) + "}")
                elif col == "hypervolume":
                    cells.append(f"$\\mathrm{{e}}^{{{exp_fmt.format(np.log(val))}}}$")
                elif isinstance(val, (int, np.integer)):
                    cells.append(f"{val:,}")
                elif isinstance(val, (float, np.floating)):
                    cells.append(float_fmt.format(val))
                else:
                    cells.append(_tex(str(val)))
            lines.append("    " + " & ".join(cells) + " \\\\")

        lines.append("\\end{longtable}")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files.append(str(path))

    return files
# ---------------------------------------------------------------------
# ──────────  ALL-LOCATIONS MEDIAN TABLE  +  FAILURE REPORT  ──────────
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# ─────  ONE [h!] TABLE PER DIST_GROUP  •  BEST VALUES IN BOLD  ───────
# ---------------------------------------------------------------------
from pathlib import Path
import pandas as pd, numpy as np

def save_median_tables_per_dist_group_tex(
    df: pd.DataFrame,
    out_dir: str = "figs/tables",
    metrics: list[str] | None = None,
    *,
    float_fmt: str = "{:,.3g}",
):
    """
    For each `dist_group` write an [h!] LaTeX `table` whose rows are
    algorithms and columns are the median of *metrics* across all
    locations.  Best values (min runtime, max hypervolume/max count)
    are \\textbf{}-ed.

    Returns a list of generated filenames.
    """
    if metrics is None:
        metrics = ["hypervolume", "time", "count"]

    # ensure canonical case/order
    metrics = [m.lower() for m in metrics]

    df_ok = _clean_df(df)

    med = (
        df_ok.groupby(["dist_group", "algorithm"])
             .agg({m: "median" for m in metrics})
             .reset_index()
             .rename(columns={m: f"median_{m}" for m in metrics})
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # helper for bolding
    def _fmt(val, best, bold_if_best):
        if pd.isna(val):
            return "--"
        s = float_fmt.format(val)
        return r"\textbf{" + s + "}" if bold_if_best else s

    for dist, sub in med.groupby("dist_group"):
        sub = sub.sort_values("algorithm", kind="stable")

        # determine best values for this distance group
        best_vals = {}
        if "time" in metrics:
            best_vals["time"] = sub["median_time"].min(skipna=True)
        if "hypervolume" in metrics:
            best_vals["hypervolume"] = sub["median_hypervolume"].max(skipna=True)
        if "count" in metrics:
            best_vals["count"] = sub["median_count"].max(skipna=True)

        file_path = out_dir / f"median_{dist}.tex"
        paths.append(str(file_path))

        col_spec = "|l|" + "|".join("r" for _ in metrics) + "|"
        header   = ["\\textbf{Algorithm}"] + [
            f"\\textbf{{Median {m.capitalize()}}}" for m in metrics
        ]

        lines = [
            "\\begin{table}[h!]",
            "    \\centering",
            f"    \\begin{{tabular}}{{{col_spec}}}",
            "        \\hline",
            "        " + " & ".join(header) + " \\\\",
            "        \\hline",
        ]

        for _, row in sub.iterrows():
            algo_tex = r"\texttt{" + _tex(row["algorithm"]) + "}"
            cells = [algo_tex]

            for m in metrics:
                col = f"median_{m}"
                val = row[col]
                best = best_vals[m]
                bold = np.isclose(val, best, rtol=1e-9, atol=0, equal_nan=False)
                cells.append(_fmt(val, best, bold))
            lines.append("        " + " & ".join(cells) + r" \\")
        lines += [
            "        \\hline",
            "    \\end{tabular}",
            f"    \\caption{{Median performance for \\textbf{{{dist}}}\\,m "
            "haversine routes (aggregated over the four city graphs)}}",
            f"    \\label{{tab:median_{dist}}}",
            "\\end{table}",
        ]

        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return paths

def print_failure_summary(
    df: pd.DataFrame,
    *,
    max_ids: int = 5,
) -> None:
    """
    Print a quick overview of failed trials per algorithm, including
    up to *max_ids* scenario IDs for context.
    """
    fails = df[df["fail"]]
    if fails.empty:
        print("✅ No failed trials recorded.")
        return

    print("\n=== Failure summary ===")
    for algo, g in fails.groupby("algorithm"):
        scen_ids = g["scenario"].unique()[:max_ids]
        id_list  = ", ".join(map(str, scen_ids))
        more = "" if len(scen_ids) == len(g["scenario"].unique()) else " …"
        print(f"• {algo:20s}  |  {len(g):3d} failed trials  |  IDs: {id_list}{more}")


def print_failed_trials_by_file(
    df: pd.DataFrame,
    *,
    max_algos: int | None = None,   # None = show all failing algos
) -> None:
    """
    Print each *scenario* (pickle filename) that contains failed trials,
    followed by the algorithms that failed inside it.

    Parameters
    ----------
    df : output of `load_evaluation_results`
    max_algos : truncate the algorithm list to this many entries for brevity
    """
    fails = df[df["fail"]]
    if fails.empty:
        print("✅  No failed trials recorded.")
        return

    print("\n=== Failed trials by pickle file ===")
    for scen, g in fails.groupby("scenario"):
        algos = sorted(g["algorithm"].unique())
        if max_algos is not None:
            more = " …" if len(algos) > max_algos else ""
            algos = algos[:max_algos]
        else:
            more = ""

        print(f"• {scen}  ({len(g)} failed rows)")
        print("    ↳  " + ", ".join(algos) + more)