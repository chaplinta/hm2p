#!/usr/bin/env python3
"""Penk+ vs Penk- cell-type programme: animal-aware statistics and LR classification.

Runs the cell-type comparison on the ex vivo patching metrics table
(``results/patching/analysis/metrics.csv``) at three levels:

1. Cell-level descriptive statistics and Mann-Whitney U — pseudoreplicated
   across the 6 animals, kept for reference only.
2. Animal-aware primary tests — within-animal cluster permutation on group
   median differences, and a paired animal-level Wilcoxon signed-rank test
   (both cell types are patched in every animal).
3. Supplementary linear mixed model (skipped when statsmodels is unavailable).

It then tests hypothesis H1 — Penk+ cells are the low-rheobase (LR) neuron type
of granular RSC L2/3 — with threshold-based classification, an animal-stratified
enrichment permutation, a threshold-free PCA/mixture axis, and a comparison of
group medians with published LR/RS reference ranges.

Statistical framework: docs/stats-strategy.md.

References
----------
Brennan et al. 2020. "Hyperexcitable Neurons Enable Precise and Persistent
Information Encoding in the Superficial Retrosplenial Cortex." Cell Reports
30:1598-1612. doi:10.1016/j.celrep.2019.12.093

Yousuf et al. 2020. "Modular Organization of Murine Retrosplenial Cortex."
Frontiers in Neural Circuits 14:576504. doi:10.3389/fncir.2020.576504

Brennan et al. 2021. "Thalamus and claustrum control parallel layer 1 circuits
in retrosplenial cortex." eLife 10:e62207. doi:10.7554/eLife.62207

Usage
-----
    python scripts/run_patching_celltype.py --dry-run
    python scripts/run_patching_celltype.py --n-perms 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.patching.lr_classify import (  # noqa: E402
    BRENNAN_2020_REFERENCE,
    LR_CRITERIA,
    classify_lr_rs,
    compare_to_reference,
    data_driven_lr_axis,
    enrichment_test,
)
from hm2p.patching.statistics import (  # noqa: E402
    animal_level_comparison,
    cluster_permutation_comparison,
    compute_summary_stats,
    mann_whitney_comparison,
    ordered_groups,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("patching_celltype")

DEFAULT_INPUT = Path("results/patching/analysis/metrics.csv")
DEFAULT_OUTPUT = Path("results/patching/celltype_programme")

GROUP_COL = "cell_type"
ANIMAL_COL = "animal_id"
REQUIRED_COLS = (GROUP_COL, ANIMAL_COL)

EPHYS_PREFIXES = ("ephys_passive_", "ephys_active_")
MORPH_PREFIXES = ("morph_api_", "morph_bas_")

#: Ephys metrics forming the candidate excitability axis (Brennan et al. 2020).
LR_AXIS_COLS = (
    "ephys_passive_rin",
    "ephys_passive_rhreo",
    "ephys_passive_maxsp",
    "ephys_passive_tau",
    "ephys_active_halfWidth",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def metric_columns(df: pd.DataFrame, prefixes: Sequence[str]) -> list[str]:
    """Numeric columns of *df* whose name starts with one of *prefixes*.

    Parameters
    ----------
    df : DataFrame
        Metrics table.
    prefixes : sequence of str
        Column-name prefixes to select.

    Returns
    -------
    list of str
        Matching numeric columns, in table order.
    """
    return [
        c
        for c in df.columns
        if c.startswith(tuple(prefixes)) and pd.api.types.is_numeric_dtype(df[c])
    ]


def missing_columns(df: pd.DataFrame, required: Sequence[str]) -> list[str]:
    """Return the entries of *required* that are absent from *df*.

    Parameters
    ----------
    df : DataFrame
        Table to check.
    required : sequence of str
        Column names that must be present.

    Returns
    -------
    list of str
        Missing column names (empty when all are present).
    """
    return [c for c in required if c not in df.columns]


def validate_input(df: pd.DataFrame) -> dict[str, Any]:
    """Describe the loaded table and check that the required columns exist.

    Parameters
    ----------
    df : DataFrame
        Loaded metrics table.

    Returns
    -------
    dict
        ``n_cells``, ``n_animals``, ``groups``, ``group_counts``,
        ``ephys_cols``, ``morph_cols``, ``lr_axis_cols`` and ``missing``.

    Raises
    ------
    ValueError
        If a required identifier column is missing.
    """
    missing = missing_columns(df, REQUIRED_COLS)
    if missing:
        raise ValueError(f"input table is missing required columns: {missing}")

    ephys_cols = metric_columns(df, EPHYS_PREFIXES)
    morph_cols = metric_columns(df, MORPH_PREFIXES)
    groups = [str(g) for g in ordered_groups(df[GROUP_COL])]
    return {
        "n_cells": int(len(df)),
        "n_animals": int(df[ANIMAL_COL].nunique()),
        "groups": groups,
        "group_counts": {g: int((df[GROUP_COL] == g).sum()) for g in groups},
        "ephys_cols": ephys_cols,
        "morph_cols": morph_cols,
        "lr_axis_cols": [c for c in LR_AXIS_COLS if c in df.columns],
        "missing": missing,
    }


def run_mixed_model(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame | None:
    """Run the supplementary linear mixed model, or return None if unavailable.

    Parameters
    ----------
    df : DataFrame
        Cell-level metrics table.
    metric_cols : sequence of str
        Metrics to fit.

    Returns
    -------
    DataFrame or None
        Output of ``mixed_model_comparison``, or None when statsmodels is not
        installed.
    """
    try:
        from hm2p.patching.statistics import mixed_model_comparison
    except ImportError:  # pragma: no cover - defensive, module is always present
        log.warning("mixed_model_comparison unavailable; skipping LMM")
        return None
    try:
        return mixed_model_comparison(df, list(metric_cols), GROUP_COL, ANIMAL_COL)
    except ImportError:
        log.warning("statsmodels not installed; skipping supplementary mixed model")
        return None


def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars/arrays inside *obj* into JSON-serialisable types.

    Parameters
    ----------
    obj : object
        Value, list or dict, possibly containing numpy types.

    Returns
    -------
    object
        Structure containing only builtin types.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def format_report(results: dict[str, Any]) -> str:
    """Render the markdown summary of a completed run.

    Parameters
    ----------
    results : dict
        Output of :func:`run_analysis`.

    Returns
    -------
    str
        Markdown text.
    """
    info = results["info"]
    enrich = results["enrichment"]
    axis = results["lr_axis"]
    g1, g2 = enrich["groups"]

    def top_rows(df: pd.DataFrame, p_col: str, n: int = 10) -> list[str]:
        if df.empty or p_col not in df.columns:
            return ["- (no results)"]
        ranked = df.dropna(subset=[p_col]).sort_values(p_col).head(n)
        if ranked.empty:
            return ["- (no p-values could be computed)"]
        lines = []
        for _, row in ranked.iterrows():
            diff = row.get("observed_diff", row.get("median_diff", float("nan")))
            lines.append(
                f"- `{row['metric']}`: diff={diff:.4g}, p={row[p_col]:.4g}, "
                f"p_FDR={row.get('p_fdr', float('nan')):.4g}"
            )
        return lines

    lines = [
        "# Penk+ vs Penk- cell-type programme (ex vivo patching)",
        "",
        f"Cells: {info['n_cells']} in {info['n_animals']} animals "
        f"({', '.join(f'{g}={n}' for g, n in info['group_counts'].items())}).",
        f"Metrics: {len(info['ephys_cols'])} ephys, {len(info['morph_cols'])} morphology.",
        "",
        "Differences are reported as "
        f"`{g1} - {g2}`. Cell-level Mann-Whitney is descriptive only "
        "(cells are nested within animals); the primary tests are the "
        "within-animal cluster permutation and the paired animal-level "
        "Wilcoxon signed-rank test. See docs/stats-strategy.md.",
        "",
        "## Cluster permutation (primary, within-animal label shuffle)",
        *top_rows(results["cluster_permutation"], "p_perm"),
        "",
        "## Animal-level paired Wilcoxon (primary, one difference per animal)",
        *top_rows(results["animal_level"], "p_value"),
        "",
        "## Cell-level Mann-Whitney (descriptive, pseudoreplicated)",
        *top_rows(results["mann_whitney"], "p_value"),
        "",
        "## H1: are Penk+ cells the low-rheobase (LR) type?",
        f"- LR criteria: {LR_CRITERIA.source}",
        f"- LR fraction: {g1}={enrich[f'frac_lr_{g1}']:.3f} "
        f"({enrich[f'n_lr_{g1}']}/{enrich[f'n_{g1}']}), "
        f"{g2}={enrich[f'frac_lr_{g2}']:.3f} "
        f"({enrich[f'n_lr_{g2}']}/{enrich[f'n_{g2}']})",
        f"- Difference={enrich['observed_diff']:.3f}, "
        f"within-animal permutation p={enrich['p_perm']:.4g} "
        f"({enrich['n_perms']} permutations); "
        f"Fisher exact p={enrich['fisher_p_descriptive']:.4g} (descriptive, ignores animal)",
        f"- Threshold-free axis: PC1 explains "
        f"{axis['explained_variance_ratio'][0]:.2f} of ephys variance over "
        f"{axis['n_cells']} cells; PC1 medians "
        + ", ".join(f"{g}={v:.3f}" for g, v in axis["pc1_median_by_group"].items())
        + f"; mixture BIC(1)-BIC(2)={axis['bic_delta']:.2f} "
        f"(bimodal={axis['bimodal']})",
        "",
        "## Reference comparison",
        f"- {BRENNAN_2020_REFERENCE.source}",
    ]
    ref = results["reference"]
    for _, row in ref.iterrows():
        lines.append(
            f"- `{row['metric']}` {row['group']}: median={row['median']:.4g} → {row['verdict']}"
        )
    lines += [
        "",
        "## Caveats",
        "- 6 animals: the animal-level Wilcoxon cannot reach p < 0.05 with fewer "
        "than 6 non-zero differences; sign counts carry most of the information.",
        "- LR thresholds and reference ranges are approximate and must be checked "
        "against Brennan et al. 2020 and against this dataset's recording "
        "conditions before publication.",
        "- The mixture BIC bimodality check is descriptive, not a test.",
    ]
    if results.get("mixed_model") is None:
        lines.append("- Supplementary mixed model skipped (statsmodels not installed).")
    return "\n".join(lines)


def run_analysis(
    df: pd.DataFrame,
    n_perms: int = 5000,
    seed: int = 0,
    min_criteria_met: int = 3,
    within_animal: bool = True,
) -> dict[str, Any]:
    """Run every cell-type comparison and the LR classification.

    Parameters
    ----------
    df : DataFrame
        Cell-level metrics table.
    n_perms : int
        Permutations for the cluster permutation and enrichment tests.
    seed : int
        Seed for the permutation random generator.
    min_criteria_met : int
        Criteria required for an LR or RS call.
    within_animal : bool
        Shuffle group labels within each animal (appropriate when both cell
        types are recorded in every mouse). When cell types segregate by
        animal, use ``False`` to shuffle labels at the animal level instead;
        the enrichment permutation is then reported as descriptive only.

    Returns
    -------
    dict
        ``info``, ``summary``, ``mann_whitney``, ``cluster_permutation``,
        ``animal_level``, ``mixed_model`` (DataFrame or None), ``classified``,
        ``enrichment``, ``lr_axis`` and ``reference``.
    """
    info = validate_input(df)
    info["within_animal"] = within_animal
    info["cells_per_animal_by_type"] = (
        df.groupby([ANIMAL_COL, GROUP_COL]).size().unstack(fill_value=0).to_dict("index")
    )
    metric_cols = info["ephys_cols"] + info["morph_cols"]
    rng = np.random.default_rng(seed)

    log.info("summary + cell-level Mann-Whitney over %d metrics", len(metric_cols))
    summary = compute_summary_stats(df, metric_cols, group_col=GROUP_COL)
    mw = mann_whitney_comparison(df, metric_cols, group_col=GROUP_COL)

    mode = "within animal" if within_animal else "between animals"
    log.info("cluster permutation (%d permutations, %s)", n_perms, mode)
    cluster = cluster_permutation_comparison(
        df,
        metric_cols,
        group_col=GROUP_COL,
        animal_col=ANIMAL_COL,
        n_perms=n_perms,
        rng=rng,
        within_animal=within_animal,
    )
    animal = animal_level_comparison(df, metric_cols, group_col=GROUP_COL, animal_col=ANIMAL_COL)
    mixed = run_mixed_model(df, metric_cols)

    log.info("LR classification and enrichment")
    classified = classify_lr_rs(df, criteria=LR_CRITERIA, min_criteria_met=min_criteria_met)
    enrichment = enrichment_test(
        classified,
        group_col=GROUP_COL,
        animal_col=ANIMAL_COL,
        n_perms=n_perms,
        rng=rng,
    )
    axis = data_driven_lr_axis(df, info["lr_axis_cols"], group_col=GROUP_COL)
    reference = compare_to_reference(df, metric_cols=None, group_col=GROUP_COL)

    return {
        "info": info,
        "summary": summary,
        "mann_whitney": mw,
        "cluster_permutation": cluster,
        "animal_level": animal,
        "mixed_model": mixed,
        "classified": classified,
        "enrichment": enrichment,
        "lr_axis": axis,
        "reference": reference,
    }


def write_outputs(results: dict[str, Any], output_dir: Path) -> list[Path]:
    """Write every result table, the JSON side-outputs and the markdown report.

    Parameters
    ----------
    results : dict
        Output of :func:`run_analysis`.
    output_dir : Path
        Directory to create and write into.

    Returns
    -------
    list of Path
        Paths written, in write order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    tables = {
        "summary_stats.csv": results["summary"],
        "mann_whitney_cell_level.csv": results["mann_whitney"],
        "cluster_permutation.csv": results["cluster_permutation"],
        "animal_level.csv": results["animal_level"],
        "lr_classification.csv": results["classified"],
        "reference_comparison.csv": results["reference"],
    }
    if results.get("mixed_model") is not None:
        tables["mixed_model_supplementary.csv"] = results["mixed_model"]

    for name, table in tables.items():
        path = output_dir / name
        table.to_csv(path, index=False)
        written.append(path)

    enrichment_path = output_dir / "lr_enrichment.json"
    enrichment_path.write_text(json.dumps(_json_safe(results["enrichment"]), indent=2))
    written.append(enrichment_path)

    axis = dict(results["lr_axis"])
    axis.pop("pc1_scores", None)
    axis.pop("pc2_scores", None)
    axis.pop("groups", None)
    axis.pop("pc1_by_group", None)
    axis_path = output_dir / "lr_axis.json"
    axis_path.write_text(json.dumps(_json_safe(axis), indent=2))
    written.append(axis_path)

    report_path = output_dir / "report.md"
    report_path.write_text(format_report(results))
    written.append(report_path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``--input``, ``--output``, ``--n-perms``, ``--seed``,
        ``--min-criteria`` and ``--dry-run``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="metrics CSV to load")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output directory")
    parser.add_argument(
        "--n-perms", type=int, default=5000, help="permutations per test (default 5000)"
    )
    parser.add_argument("--seed", type=int, default=0, help="permutation seed (default 0)")
    parser.add_argument(
        "--min-criteria",
        type=int,
        default=3,
        help="LR criteria required for an LR/RS call (default 3)",
    )
    parser.add_argument(
        "--between-animal",
        dest="within_animal",
        action="store_false",
        help="shuffle labels at the animal level (cell types segregate by mouse)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the input and print what would run, writing nothing",
    )
    return parser


def describe_dry_run(info: dict[str, Any], args: argparse.Namespace) -> str:
    """Render the dry-run description.

    Parameters
    ----------
    info : dict
        Output of :func:`validate_input`.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    str
        Human-readable plan.
    """
    lines = [
        f"input: {args.input}",
        f"cells: {info['n_cells']} in {info['n_animals']} animals",
        "groups: " + ", ".join(f"{g}={n}" for g, n in info["group_counts"].items()),
        f"metrics: {len(info['ephys_cols'])} ephys + {len(info['morph_cols'])} morphology",
        f"LR axis metrics: {', '.join(info['lr_axis_cols']) or '(none present)'}",
        "would run: compute_summary_stats, mann_whitney_comparison (descriptive), "
        "cluster_permutation_comparison, animal_level_comparison, "
        "mixed_model_comparison (if statsmodels), classify_lr_rs, enrichment_test, "
        "data_driven_lr_axis, compare_to_reference",
        f"permutations: {args.n_perms}; seed: {args.seed}; min LR criteria: {args.min_criteria}",
        f"would write into: {args.output}",
        "dry run: nothing written",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (0 on success, 1 on a validation failure).
    """
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        log.error("input table not found: %s", args.input)
        return 1
    df = pd.read_csv(args.input)

    try:
        info = validate_input(df)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    if args.dry_run:
        print(describe_dry_run(info, args))
        return 0

    results = run_analysis(
        df,
        n_perms=args.n_perms,
        seed=args.seed,
        min_criteria_met=args.min_criteria,
        within_animal=args.within_animal,
    )
    written = write_outputs(results, args.output)
    for path in written:
        log.info("wrote %s", path)
    print(format_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
