"""Hypothesis Test Report — results from docs/hypotheses.md.

Runs or loads precomputed hypothesis tests comparing Penk+ vs Penk⁻CamKII+
RSP neurons using non-parametric animal-level and cluster permutation tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.data import DERIVATIVES_BUCKET, download_s3_bytes

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "hypotheses"


@st.cache_data(ttl=600)
def _load_results() -> tuple[pd.DataFrame | None, str | None]:
    """Load hypothesis results CSV and markdown report."""
    # Try local first (faster)
    csv_path = RESULTS_DIR / "hypothesis_results.csv"
    md_path = RESULTS_DIR / "hypothesis_report.md"

    results_df = None
    report_md = None

    if csv_path.exists():
        results_df = pd.read_csv(csv_path)
    if md_path.exists():
        report_md = md_path.read_text()

    return results_df, report_md


@st.cache_data(ttl=600)
def _load_confounds() -> pd.DataFrame | None:
    path = RESULTS_DIR / "confound_checks.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def _sig_color(p: float) -> str:
    if np.isnan(p):
        return "gray"
    if p < 0.01:
        return "#2ca02c"  # green
    if p < 0.05:
        return "#ff7f0e"  # orange
    return "#d62728"  # red


def _format_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Hypothesis Test Report")
st.caption(
    "Non-parametric tests from docs/hypotheses.md. "
    "Animal-level Mann-Whitney U + cluster permutation (primary). "
    "See docs/stats-strategy.md for methodology."
)

# --- Run tests button ---
col_btn, col_signal, col_perms = st.columns([2, 2, 2])
with col_signal:
    signal_choice = st.selectbox("Signal", ["dff", "events"], index=0, key="hyp_signal")
with col_perms:
    n_perms_choice = st.selectbox("Permutations", [500, 1000, 5000, 10000], index=0, key="hyp_perms")
with col_btn:
    run_clicked = st.button("Run hypothesis tests", type="primary", key="run_hyp")

if run_clicked:
    with st.spinner(f"Running tests ({signal_choice}, {n_perms_choice} perms)..."):
        import subprocess
        cmd = [
            sys.executable, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "test_hypotheses.py"),
            "--signal", signal_choice,
            "--n-perms", str(n_perms_choice),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            st.success("Tests complete. Refreshing...")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(f"Tests failed:\n```\n{result.stderr[-1000:]}\n```")

results_df, report_md = _load_results()

if results_df is None:
    st.info("No results yet. Click **Run hypothesis tests** above.")
    st.stop()

# --- Summary metrics ---
n_tests = len(results_df[results_df["test"] != "descriptive"])
n_sig = len(results_df[(results_df["p_value"] < 0.05) & (results_df["test"] != "descriptive")])
hypotheses = results_df["hypothesis"].unique()

c1, c2, c3 = st.columns(3)
c1.metric("Hypotheses tested", len(hypotheses))
c2.metric("Statistical tests", n_tests)
c3.metric("Significant (p < 0.05)", n_sig)

# --- Tabs ---
tab_overview, tab_detail, tab_confounds, tab_report = st.tabs([
    "Overview", "Detailed Results", "Confound Checks", "Full Report",
])

# --- Overview: forest plot ---
with tab_overview:
    st.header("Results Overview")

    # Build summary: one row per hypothesis, best p-value
    hyp_summary = []
    for hid in hypotheses:
        hyp_rows = results_df[results_df["hypothesis"] == hid]
        hname = hyp_rows.iloc[0].get("hypothesis_name", hid)

        # Get p-values from different tests
        animal_p = hyp_rows.loc[hyp_rows["test"] == "animal_summary", "p_value"]
        perm_p = hyp_rows.loc[hyp_rows["test"] == "cluster_perm", "p_value"]
        wilcox_p = hyp_rows.loc[hyp_rows["test"].isin(
            ["wilcoxon", "wilcoxon_interaction", "wilcoxon_onesample"]
        ), "p_value"]

        row = {"hypothesis": hid, "name": hname}
        row["animal_p"] = float(animal_p.iloc[0]) if len(animal_p) > 0 else np.nan
        row["perm_p"] = float(perm_p.iloc[0]) if len(perm_p) > 0 else np.nan
        row["wilcoxon_p"] = float(wilcox_p.iloc[0]) if len(wilcox_p) > 0 else np.nan

        # Best p across tests
        ps = [row["animal_p"], row["perm_p"], row["wilcoxon_p"]]
        valid_ps = [p for p in ps if not np.isnan(p)]
        row["best_p"] = min(valid_ps) if valid_ps else np.nan

        # Determine verdict
        if np.isnan(row["best_p"]):
            row["verdict"] = "No data"
        elif not np.isnan(row["animal_p"]) and not np.isnan(row["perm_p"]):
            # Between-group: both must agree
            if row["animal_p"] < 0.05 and row["perm_p"] < 0.05:
                row["verdict"] = "Supported"
            elif row["animal_p"] < 0.05 or row["perm_p"] < 0.05:
                row["verdict"] = "Inconsistent"
            else:
                row["verdict"] = "Not supported"
        elif not np.isnan(row["wilcoxon_p"]):
            row["verdict"] = "Supported" if row["wilcoxon_p"] < 0.05 else "Not supported"
        else:
            row["verdict"] = "Descriptive"

        hyp_summary.append(row)

    summary_df = pd.DataFrame(hyp_summary)

    # Verdict colours
    verdict_colors = {
        "Supported": "#2ca02c",
        "Not supported": "#d62728",
        "Inconsistent": "#ff7f0e",
        "Descriptive": "#7f7f7f",
        "No data": "#cccccc",
    }

    # Display as styled table
    for _, row in summary_df.iterrows():
        verdict = row["verdict"]
        color = verdict_colors.get(verdict, "#999")
        icon = {"Supported": ":white_check_mark:", "Not supported": ":x:",
                "Inconsistent": ":warning:", "Descriptive": ":bar_chart:",
                "No data": ":grey_question:"}.get(verdict, "")

        cols = st.columns([1, 4, 2, 2, 2, 2])
        cols[0].markdown(f"**{row['hypothesis']}**")
        cols[1].markdown(row["name"])
        cols[2].markdown(f"Animal: {_format_p(row['animal_p'])}")
        cols[3].markdown(f"Perm: {_format_p(row['perm_p'])}")
        cols[4].markdown(f"Wilcoxon: {_format_p(row['wilcoxon_p'])}")
        cols[5].markdown(f"{icon} {verdict}")

    # Legend
    st.caption(
        "**Supported** = both animal-level and permutation tests p < 0.05. "
        "**Inconsistent** = one test significant, the other not (interpret with caution). "
        "**Not supported** = both tests p > 0.05."
    )

# --- Detailed results ---
with tab_detail:
    st.header("All Test Results")

    # Filter
    test_filter = st.multiselect(
        "Filter by test type",
        options=results_df["test"].unique().tolist(),
        default=results_df["test"].unique().tolist(),
    )
    filtered = results_df[results_df["test"].isin(test_filter)].copy()

    # Highlight significant
    filtered["significant"] = filtered["p_value"].apply(
        lambda p: "Yes" if p < 0.05 else ("" if np.isnan(p) else "No")
    )

    display_cols = [c for c in filtered.columns if c not in ("signal",)]
    st.dataframe(
        filtered[display_cols].sort_values("p_value"),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download results CSV",
        results_df.to_csv(index=False).encode(),
        file_name="hypothesis_results.csv",
        mime="text/csv",
    )

# --- Confound checks ---
with tab_confounds:
    st.header("Signal Quality Confound Checks")
    st.caption(
        "Spearman correlation between significant metrics and signal quality "
        "confounds (SNR, peak dF/F, bleaching). |ρ| > 0.3 is flagged."
    )

    confounds_df = _load_confounds()
    if confounds_df is not None and len(confounds_df) > 0:
        flagged = confounds_df[confounds_df.get("flagged", False) == True]  # noqa: E712
        if len(flagged) > 0:
            st.warning(f"**{len(flagged)} confound(s) flagged** (|ρ| > 0.3)")
            st.dataframe(flagged, use_container_width=True, hide_index=True)
        else:
            st.success("No confounds flagged (all |ρ| < 0.3)")

        st.subheader("All confound checks")
        st.dataframe(confounds_df, use_container_width=True, hide_index=True)
    else:
        st.info("No confound checks computed (no results were near significance).")

# --- Full markdown report ---
with tab_report:
    st.header("Full Report")
    if report_md:
        st.markdown(report_md)
    else:
        st.info("No markdown report found.")

    with st.expander("Methods & References"):
        st.markdown("""
**Animal-level Mann-Whitney U:** Collapse to one mean per animal, then
unpaired Mann-Whitney U test. Conservative — each observation is independent.

**Cluster permutation test:** Shuffle celltype labels at the animal level
(10,000 permutations). Respects nesting structure without distributional
assumptions. Minimum achievable p ≈ 0.0005 with 16 animals.

**Wilcoxon signed-rank:** Paired within-cell comparison (e.g. light vs dark).
Non-parametric alternative to paired t-test.

**Confound control:** Spearman correlation between significant metrics and
signal quality variables (SNR, peak dF/F₀, bleaching slope). Results with
|ρ| > 0.3 are flagged as potentially confounded.

**References:**
- Mann & Whitney 1947. "On a test of whether one of two random variables
  is stochastically larger than the other." *Ann Math Stat*.
- Wilcoxon 1945. "Individual comparisons by ranking methods."
  *Biometrics Bulletin* 1(6):80-83.
""")
