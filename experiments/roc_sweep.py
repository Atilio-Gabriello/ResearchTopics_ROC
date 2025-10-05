import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for headless/scripted runs
import matplotlib.pyplot as plt
import pysubdisc


def _build_curve(roc_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build a simple ROC curve by:
    - grouping by FPR and taking the max TPR per unique FPR (dominant operating point),
    - adding anchors (0,0) and (1,1),
    - sorting by FPR, and
    - computing trapezoidal AUC.
    Returns: (xs, ys, auc)
    """
    df = roc_df[['FPR', 'TPR']].copy()
    if df.empty:
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        return xs, ys, 0.5
    # Keep the best TPR for each FPR bucket
    df = df.groupby('FPR', as_index=False)['TPR'].max()
    # Add anchors
    df = pd.concat([pd.DataFrame({'FPR': [0.0], 'TPR': [0.0]}), df, pd.DataFrame({'FPR': [1.0], 'TPR': [1.0]})], ignore_index=True)
    df = df.drop_duplicates(subset=['FPR']).sort_values('FPR')
    xs = df['FPR'].to_numpy(dtype=float)
    ys = df['TPR'].to_numpy(dtype=float)
    # Trapezoidal AUC
    auc = float(np.trapezoid(ys, xs))
    return xs, ys, auc


def _plot_roc(roc_df: pd.DataFrame, title: str, out_path: Path) -> float:
    xs, ys, auc = _build_curve(roc_df)
    plt.figure(figsize=(8, 6))
    # Scatter of all subgroup operating points
    if not roc_df.empty:
        plt.scatter(roc_df['FPR'], roc_df['TPR'], s=18, alpha=0.6, label='Subgroups')
    # Simple ROC curve
    plt.plot(xs, ys, '-', color='tab:blue', lw=2, label=f'Curve (AUC={auc:.3f})')
    # Diagonal baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return auc


def _upper_roc_hull(points: list[tuple[float, float]]):
    """Andrew's monotone chain but return only the upper hull (ROC convex hull)."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    # Build upper hull: iterate in increasing x but enforce clockwise turns
    upper = []
    for p in pts:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) >= 0:
            upper.pop()
        upper.append(p)
    # Ensure anchors present
    if (0.0, 0.0) not in upper:
        upper = [(0.0, 0.0)] + upper
    if (1.0, 1.0) not in upper:
        upper = upper + [(1.0, 1.0)]
    # Sort by x for integration
    upper = sorted(upper, key=lambda t: t[0])
    # Deduplicate any x ties keeping max y (upper frontier)
    x_to_y = {}
    for x, y in upper:
        x_to_y[x] = max(y, x_to_y.get(x, y))
    return [(x, x_to_y[x]) for x in sorted(x_to_y.keys())]


def _roc_hull_auc(roc_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    pts = [(float(x), float(y)) for x, y in roc_df[['FPR', 'TPR']].to_numpy()] if not roc_df.empty else []
    hull_pts = _upper_roc_hull(pts + [(0.0, 0.0), (1.0, 1.0)])
    xs = np.array([p[0] for p in hull_pts], dtype=float)
    ys = np.array([p[1] for p in hull_pts], dtype=float)
    auc = float(np.trapezoid(ys, xs))
    return xs, ys, auc


def run_one(data: pd.DataFrame, alpha: float, depth: int, width: int,
            min_cov: int | None, out_dir: Path,
            target_col: str = 'target', positive_value: str = 'gr50K',
            nr_threads: int | None = None,
            postproc: bool | None = None,
            postproc_count: int | None = None,
            numeric_strategy: str | None = None,
            nr_bins: int | None = None,
            strategy: str = 'ROC_BEAM') -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = pysubdisc.singleNominalTarget(data, target_col, positive_value)
    sd.searchStrategy = strategy
    sd.searchDepth = depth
    sd.searchStrategyWidth = width
    if min_cov is not None:
        sd.minimumCoverage = int(min_cov)

    # Try to set ROC params if supported by the JAR; safe no-ops if not
    try:
        sd.alpha = float(alpha)
    except Exception:
        pass
    # Reproducibility and numeric options
    if nr_threads is not None:
        sd.nrThreads = int(nr_threads)
    if postproc is not None:
        sd.postProcessingDoAutoRun = bool(postproc)
    if postproc_count is not None:
        sd.postProcessingCount = int(postproc_count)
    if numeric_strategy is not None:
        sd.numericStrategy = str(numeric_strategy)
    if nr_bins is not None:
        sd.nrBins = int(nr_bins)

    sd.run()
    df = sd.asDataFrame()
    df.to_csv(out_dir / f'subgroups_alpha_{alpha}.csv', index=False)

    # Compute ROC points
    y_true = (data[target_col] == positive_value)
    P = int(y_true.sum())
    N = int(len(y_true) - P)

    roc = []
    for i in range(len(df)):
        y_pred = sd.getSubgroupMembers(i).astype(bool)
        TP = int((y_true & y_pred).sum())
        FP = int(((~y_true) & y_pred).sum())
        TPR = TP / P if P else 0.0
        FPR = FP / N if N else 0.0
        roc.append({
            'index': i,
            'FPR': float(FPR),
            'TPR': float(TPR),
            'Coverage': int(df.loc[i, 'Coverage']),
            'Quality': float(df.loc[i, 'Quality']),
            'Conditions': df.loc[i, 'Conditions'],
        })

    roc_df = pd.DataFrame(roc)
    roc_csv = out_dir / f'roc_points_alpha_{alpha}.csv'
    roc_df.to_csv(roc_csv, index=False)

    # Plot scatter + curve and save PNG
    png_path = out_dir / f'roc_alpha_{alpha}.png'
    auc_env = _plot_roc(roc_df, title=f'ROC (alpha={alpha}, depth={depth}, width={width}, strat={strategy})', out_path=png_path)
    # Convex-hull curve and AUC (optional additional file)
    xs_h, ys_h, auc_h = _roc_hull_auc(roc_df)
    plt.figure(figsize=(8, 6))
    plt.plot(xs_h, ys_h, '-', lw=2, label=f'Hull (AUC={auc_h:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC convex hull (alpha={alpha}, depth={depth}, width={width}, strat={strategy})')
    plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hull_png = out_dir / f'roc_hull_alpha_{alpha}.png'
    plt.savefig(hull_png)
    plt.close()
    print(f"Saved {roc_csv}, {png_path} (env AUC={auc_env:.3f}), and {hull_png} (hull AUC={auc_h:.3f})")

    # Compute table-2-like metrics
    def tpr_at_fpr(thresh: float) -> float:
        if roc_df.empty:
            return 0.0
        subset = roc_df[roc_df['FPR'] <= thresh]
        return float(subset['TPR'].max()) if not subset.empty else 0.0

    summary = {
        'alpha': alpha,
        'strategy': strategy,
        'num_subgroups': int(len(df)),
        'best_quality': float(df['Quality'].max()) if not df.empty else float('nan'),
        'mean_quality': float(df['Quality'].mean()) if not df.empty else float('nan'),
        'median_quality': float(df['Quality'].median()) if not df.empty else float('nan'),
        'mean_coverage': float(df['Coverage'].mean()) if not df.empty else float('nan'),
        'median_coverage': float(df['Coverage'].median()) if not df.empty else float('nan'),
        'AUC_env': float(auc_env),
        'AUC_hull': float(auc_h),
        'max_TPR': float(roc_df['TPR'].max()) if not roc_df.empty else 0.0,
        'min_FPR': float(roc_df['FPR'].min()) if not roc_df.empty else 0.0,
        'TPR@FPR<=0.05': tpr_at_fpr(0.05),
        'TPR@FPR<=0.10': tpr_at_fpr(0.10),
        'TPR@FPR<=0.20': tpr_at_fpr(0.20),
    }
    return summary


def main():
    p = argparse.ArgumentParser(description='ROC_BEAM sweep and export')
    p.add_argument('--data', type=str, default='./tests/adult.txt', help='Path to CSV')
    p.add_argument('--target-col', type=str, default='target', help='Name of target column')
    p.add_argument('--positive', type=str, default='gr50K', help='Positive class value')
    p.add_argument('--alphas', type=float, nargs='+', default=[0.5], help='Alpha values to run')
    p.add_argument('--depth', type=int, default=2, help='Search depth')
    p.add_argument('--width', type=int, default=100, help='Beam width')
    p.add_argument('--min-coverage', type=int, default=None, help='Minimum coverage (absolute)')
    p.add_argument('--out', type=str, default='./runs/roc', help='Output directory')
    p.add_argument('--strategies', type=str, nargs='*', default=['ROC_BEAM', 'WIDE_BEAM', 'BEAM', 'BEST_FIRST'],
                   help='Strategies to compare: e.g., ROC_BEAM, BEAM, BEST_FIRST, BREADTH_FIRST, COVER_BASED_BEAM_SELECTION, or WIDE_BEAM (alias for BEAM with larger width).')
    p.add_argument('--wide-width', type=int, default=500, help='Beam width to use when strategy is WIDE_BEAM')
    # Reproducibility and numeric handling params
    p.add_argument('--nr-threads', type=int, default=None, help='Number of threads (set 1 for determinism)')
    p.add_argument('--no-postproc', action='store_true', help='Disable post-processing')
    p.add_argument('--postproc-count', type=int, default=None, help='Post-processing count (if enabled)')
    p.add_argument('--numeric-strategy', type=str, default=None, help="Numeric strategy, e.g. 'NUMERIC_BINS' or 'NUMERIC_BEST'")
    p.add_argument('--nr-bins', type=int, default=None, help='Number of bins for NUMERIC_BINS strategy')
    args = p.parse_args()

    data = pd.read_csv(args.data)
    out_dir = Path(args.out)

    # Strategy runs + collect summaries
    summaries = []
    comparison_rows = []
    for strat in args.strategies:
        strat_upper = strat.upper()
        # strategy_name is what the engine receives; label_name is how we label/save outputs
        if strat_upper == 'WIDE_BEAM':
            strategy_name = 'BEAM'
            label_name = 'WIDE_BEAM'
            width_to_use = args.wide_width
        else:
            strategy_name = strat_upper
            label_name = strat_upper
            width_to_use = args.width

        if strategy_name == 'ROC_BEAM':
            for a in args.alphas:
                s = run_one(
                    data=data,
                    alpha=a,
                    depth=args.depth,
                    width=width_to_use,
                    min_cov=args.min_coverage,
                    out_dir=out_dir / label_name / f'alpha_{a}',
                    target_col=args.target_col,
                    positive_value=args.positive,
                    nr_threads=args.nr_threads,
                    postproc=(False if args.no_postproc else None),
                    postproc_count=args.postproc_count,
                    numeric_strategy=args.numeric_strategy,
                    nr_bins=args.nr_bins,
                    strategy=strategy_name,
                )
                s['width_used'] = width_to_use
                # Label the strategy as requested (e.g., WIDE_BEAM)
                s['strategy'] = label_name
                summaries.append(s)
                comparison_rows.append(s)
        else:
            # Non-ROC strategies: run once (alpha provided but not used for search)
            a = args.alphas[0] if args.alphas else 0.5
            s = run_one(
                data=data,
                alpha=a,
                depth=args.depth,
                width=width_to_use,
                min_cov=args.min_coverage,
                out_dir=out_dir / label_name,
                target_col=args.target_col,
                positive_value=args.positive,
                nr_threads=args.nr_threads,
                postproc=(False if args.no_postproc else None),
                postproc_count=args.postproc_count,
                numeric_strategy=args.numeric_strategy,
                nr_bins=args.nr_bins,
                strategy=strategy_name,
            )
            s['width_used'] = width_to_use
            s['strategy'] = label_name
            comparison_rows.append(s)

    # Combined overlay of ROC curves across alphas (ROC_BEAM only if available)
    plt.figure(figsize=(8, 6))
    for a in args.alphas:
        # Prefer new per-strategy path
        roc_csv = out_dir / 'ROC_BEAM' / f'alpha_{a}' / f'roc_points_alpha_{a}.csv'
        if not roc_csv.exists():
            # Backward compatibility
            roc_csv = out_dir / f'alpha_{a}' / f'roc_points_alpha_{a}.csv'
            if not roc_csv.exists():
                continue
        roc_df = pd.read_csv(roc_csv)
        xs, ys, auc = _build_curve(roc_df)
        plt.plot(xs, ys, lw=2, label=f'alpha={a} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curves overlay (depth={args.depth}, width={args.width})')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    overlay_png = out_dir / 'roc_overlay.png'
    plt.savefig(overlay_png)
    plt.close()
    print(f"Saved overlay plot {overlay_png}")

    # Write a table2-like summary CSV for ROC_BEAM alphas and a strategy comparison table
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_csv = out_dir / 'table2_like_summary.csv'
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved summary table {summary_csv}")
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        # Friendly column order if present
        preferred = ['strategy', 'alpha', 'width_used', 'num_subgroups', 'best_quality', 'mean_quality', 'median_quality', 'mean_coverage', 'median_coverage', 'AUC_env', 'AUC_hull', 'TPR@FPR<=0.05', 'TPR@FPR<=0.10', 'TPR@FPR<=0.20', 'max_TPR', 'min_FPR']
        cols = [c for c in preferred if c in comparison_df.columns] + [c for c in comparison_df.columns if c not in preferred]
        comparison_df = comparison_df[cols]
        comparison_csv = out_dir / 'strategy_comparison.csv'
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"Saved strategy comparison table {comparison_csv}")


if __name__ == '__main__':
    main()
