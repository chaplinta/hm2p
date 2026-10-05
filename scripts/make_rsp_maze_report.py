#!/usr/bin/env python3
# ruff: noqa: E501  (report prose and table rows)
"""Data for the RSP maze-exploration findings report (all cells pooled).

Collects, from existing result folders:

- dead-end recency (``scripts/run_dead_end_recency.py``): summaries for every
  signal/window/control variant, per-animal dark and light z, and the
  per-entry population response binned by recency within each session;
- behaviour around dead-end entries: approach speed vs recency by condition;
- trip destinations vs random-walk expectations (``mazemem`` behaviour table);
- pooled population decoding (``popdec``) and the follow-up decoders that did
  not survive their controls (``mazemem``).

Writes ``docs/figures/rsp_maze/data/report.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
OUT = REPO / "docs" / "figures" / "rsp_maze" / "data" / "report.json"
MAIN = "spikes_w1.5"
VARIANTS = {
    "spikes_w1.5": "CASCADE spikes, 1.5 s window",
    "spikes_w1": "CASCADE spikes, 1.0 s window",
    "spikes_w2.5": "CASCADE spikes, 2.5 s window",
    "dff_rise_w1.5": "dF/F rises (onset-weighted, no spike inference), 1.5 s",
    "dff_w1.5": "dF/F, 1.5 s window",
    "dff_w3": "dF/F, 3.0 s window",
    "spikes_w1.5_ctrl": "CASCADE spikes + time since any dead end + visit count",
    "dff_rise_w1.5_ctrl": "dF/F rises + time since any dead end + visit count",
    "spikes_w1.5_min5": "CASCADE spikes, re-entries within 5 s dropped",
    "dff_rise_w1.5_min5": "dF/F rises, re-entries within 5 s dropped",
}


def _wilcoxon(x: pd.Series | np.ndarray) -> float:
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v) & (v != 0)]
    return float(stats.wilcoxon(v).pvalue) if v.size >= 4 else float("nan")


def recency_variants(root: Path) -> list[dict[str, Any]]:
    """Summary rows (dark, light, dark_prev_dark, dark_minus_light) for each variant."""
    rows = []
    for key, label in VARIANTS.items():
        p = root / "dead_end_recency" / key / "summary.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p).set_index("contrast")
        row: dict[str, Any] = {"key": key, "label": label}
        for c in ("dark", "light", "dark_prev_dark", "dark_minus_light"):
            if c in s.index:
                row[c] = {
                    "median_z": float(s.loc[c, "median_z"]),
                    "animal_p": float(s.loc[c, "animal_p"]),
                    "session_frac_pos": float(s.loc[c, "session_frac_pos"]),
                    "n_animals": int(s.loc[c, "n_animals"]),
                }
        rows.append(row)
    return rows


def per_animal(cells: pd.DataFrame) -> list[dict[str, Any]]:
    """Median z per animal for dark and light, and their paired difference."""
    w = cells.pivot_table(
        index=["exp_id", "roi", "animal_id"], columns="cond", values="z"
    ).reset_index()
    out = []
    for a, g in w.groupby("animal_id"):
        out.append(
            {
                "animal_id": str(a),
                "dark": float(g["dark"].median()) if "dark" in g else float("nan"),
                "light": float(g["light"].median()) if "light" in g else float("nan"),
                "n_cells": int(len(g)),
            }
        )
    return out


def binned_population(entries: pd.DataFrame, n_bins: int = 3) -> dict[str, Any]:
    """Population entry response by recency tertile (within session and condition).

    Raw (not covariate-adjusted) per-entry population responses, averaged per
    session and bin; summary across sessions as median and interquartile range.
    """
    e = entries.copy()
    e["cond"] = np.where(
        e.light_frac > 0.9, "light", np.where(e.light_frac < 0.1, "dark", "mixed")
    )
    e = e[e.cond != "mixed"]
    out: dict[str, Any] = {}
    for cond, g in e.groupby("cond"):
        per_session, rec = [], []
        for _, sdf in g.groupby("exp_id"):
            if len(sdf) < 3 * n_bins:
                continue
            b = pd.qcut(sdf.recency_s.rank(method="first"), n_bins, labels=False)
            per_session.append(
                sdf.groupby(b).pop_response.mean().reindex(range(n_bins)).to_numpy()
            )
            rec.append(sdf.groupby(b).recency_s.median().reindex(range(n_bins)).to_numpy())
        if not per_session:
            continue
        m, r = np.array(per_session), np.array(rec)
        out[cond] = {
            "n_sessions": int(m.shape[0]),
            "median": np.nanmedian(m, axis=0).tolist(),
            "q25": np.nanpercentile(m, 25, axis=0).tolist(),
            "q75": np.nanpercentile(m, 75, axis=0).tolist(),
            "long_minus_short_p": _wilcoxon(m[:, -1] - m[:, 0]),
            "median_recency_s": np.nanmedian(r, axis=0).tolist(),
        }
    return out


def approach_behaviour(entries: pd.DataFrame) -> dict[str, Any]:
    """Spearman of recency with approach speed, per session and condition; animal-level test."""
    e = entries.copy()
    e["cond"] = np.where(
        e.light_frac > 0.9, "light", np.where(e.light_frac < 0.1, "dark", "mixed")
    )
    out = {}
    for cond in ("light", "dark"):
        rows = []
        for (_ex, an), g in e[e.cond == cond].groupby(["exp_id", "animal_id"]):
            if len(g) >= 10:
                rows.append(
                    {
                        "animal_id": an,
                        "rho": stats.spearmanr(g.recency_s, g.speed_pre).statistic,
                        "median_recency_s": float(g.recency_s.median()),
                    }
                )
        d = pd.DataFrame(rows)
        am = d.groupby("animal_id").rho.median()
        out[cond] = {
            "median_rho": float(am.median()),
            "animal_p": _wilcoxon(am),
            "n_animals": int(am.size),
            "median_recency_s": float(d.median_recency_s.median()),
        }
    return out


def trip_behaviour(root: Path) -> list[dict[str, Any]]:
    """Fraction of trips ending at a dead end not among the 3 most recent, vs walk expectations."""
    p = root / "celltype_programme_mazemem_ec2" / "spikes" / "mazemem" / "behaviour.csv"
    if not p.exists():
        return []
    b = pd.read_csv(p)
    b = b[(b.label == "novel") & b.condition.isin(["light", "dark"])]
    rows = []
    for (cond, nm), g in b.groupby(["condition", "null_model"]):
        am = g.groupby("animal_id")[["observed_frac", "expected_frac", "excess"]].median()
        rows.append(
            {
                "condition": cond,
                "null_model": nm,
                "observed": float(am.observed_frac.median()),
                "expected": float(am.expected_frac.median()),
                "animal_p": _wilcoxon(am.excess),
                "n_animals": int(len(am)),
            }
        )
    return rows


def decoding(root: Path) -> dict[str, Any]:
    """Headline pooled population decoders and the mazemem follow-ups that did not hold."""
    out: dict[str, Any] = {}
    p = root / "celltype_programme_popdec_ec2" / "spikes" / "popdec" / "summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        s = s[(s.group == "all") & s.headline]
        out["popdec"] = s[
            [
                "decoder",
                "metric",
                "feature_set",
                "n_sessions",
                "median_observed",
                "median_null_mean",
                "animal_wilcoxon_p",
            ]
        ].to_dict(orient="records")
    p = root / "celltype_programme_mazemem_ec2" / "spikes" / "mazemem" / "summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        s = s[s.group == "all"]
        plan = s[
            (s.analysis == "planning")
            & (s.target == "novel")
            & s.variant.isin(["pre", "pre_noreturn"])
            & s.condition.isin(["light", "dark"])
        ]
        out["planning"] = plan[
            [
                "variant",
                "condition",
                "feature_set",
                "n_sessions",
                "median_observed",
                "median_null_mean",
                "animal_wilcoxon_p",
            ]
        ].to_dict(orient="records")
        dist = s[
            (s.analysis == "distance")
            & (s.target == "remaining")
            & (s.metric == "partial_spearman")
            & s.variant.isin(["place_controlled", "place_direction_controlled"])
            & s.condition.isin(["dark", "all"])
            & (s.feature_set != "behaviour")
        ]
        out["distance"] = dist[
            [
                "variant",
                "condition",
                "feature_set",
                "n_sessions",
                "median_observed",
                "median_null_mean",
                "animal_wilcoxon_p",
            ]
        ].to_dict(orient="records")
        spl = s[(s.analysis == "splitter") & (s.condition == "all") & (s.feature_set == "neural")]
        out["splitter"] = spl[
            ["target", "n_sessions", "median_observed", "median_null_mean", "animal_wilcoxon_p"]
        ].to_dict(orient="records")
    for key in ("popdec", "planning", "distance", "splitter"):
        for r in out.get(key, []):
            if "animal_wilcoxon_p" in r:
                r["animal_p"] = r.pop("animal_wilcoxon_p")
    return out


def inout_tables(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-session in/out decoding (pooled frames and light/dark only) and its summary."""
    frames = []
    p = root / "celltype_programme_popdec_ec2" / "spikes" / "popdec" / "sessions.csv"
    if p.exists():
        a = pd.read_csv(p)
        frames.append(a.assign(condition="all"))
    q = root / "popdec_light_dark" / "sessions_spikes.csv"
    if q.exists():
        frames.append(pd.read_csv(q))
    if not frames:
        return [], []
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.decoder == "inout") & (d.metric == "balanced_accuracy")].dropna(subset=["observed"])
    d["animal_id"] = d["animal_id"].astype(str)
    sessions = d[
        ["condition", "feature_set", "exp_id", "animal_id", "observed", "null_mean", "excess"]
    ]
    summary = []
    for (cond, feat), g in d.groupby(["condition", "feature_set"]):
        am = g.groupby("animal_id").excess.median()
        summary.append(
            {
                "condition": cond,
                "feature_set": feat,
                "n_sessions": int(len(g)),
                "n_animals": int(am.size),
                "median_observed": float(g.observed.median()),
                "median_null_mean": float(g.null_mean.median()),
                "animal_p": _wilcoxon(am),
            }
        )
    order = {"all": 0, "light": 1, "dark": 2}
    summary.sort(key=lambda r: (order.get(r["condition"], 9), r["feature_set"]))
    return sessions.to_dict(orient="records"), summary


def _fmtp(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.0e}" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def narrative(d: dict[str, Any]) -> dict[str, Any]:
    """Text for the report, with numbers taken from the assembled data."""
    io = {(r["condition"], r["feature_set"]): r for r in d.get("inout_summary", [])}
    a, ah, ab = (
        io.get(("all", "neural"), {}),
        io.get(("all", "neural_hd_removed"), {}),
        io.get(("all", "behaviour"), {}),
    )
    lt, dk = io.get(("light", "neural_hd_removed"), {}), io.get(("dark", "neural_hd_removed"), {})
    var = {v["key"]: v for v in d.get("variants", [])}
    main_dl = var.get("spikes_w1.5", {}).get("dark_minus_light", {})
    min5_dl = var.get("spikes_w1.5_min5", {}).get("dark_minus_light", {})
    trips = {(r["condition"], r["null_model"]): r for r in d.get("trips", [])}
    tu_l, tu_d = trips.get(("light", "uniform"), {}), trips.get(("dark", "uniform"), {})
    tn_l = trips.get(("light", "nonbacktracking"), {})
    ap = d.get("approach", {})
    dec = d.get("decoding", {})
    plan = {(r["variant"], r["condition"], r["feature_set"]): r for r in dec.get("planning", [])}
    pl_n = plan.get(("pre", "light", "neural"), {})
    pl_s = plan.get(("pre", "light", "behaviour_plus_syllable"), {})
    pl_nr = plan.get(("pre_noreturn", "light", "neural"), {})
    dist = {(r["variant"], r["condition"], r["feature_set"]): r for r in dec.get("distance", [])}
    di_p = dist.get(("place_controlled", "dark", "neural_hd_removed"), {})
    di_pd = dist.get(("place_direction_controlled", "all", "neural_hd_removed"), {})
    spl = dec.get("splitter", [])
    n_spl = max([r["n_sessions"] for r in spl], default=0)
    g = lambda r, k: r.get(k, float("nan"))  # noqa: E731
    headline = (
        "Pooling all cells, RSP population activity tells whether the mouse is running into or out of a "
        "dead end, beyond head direction, speed and turning. None of the memory or planning signals we "
        "tested survived its controls, and no population signal differed between light and dark."
    )
    summary = [
        [
            "Holds",
            "within",
            f"Inbound vs outbound runs decoded at {g(a, 'median_observed'):.2f} (null {g(a, 'median_null_mean'):.2f}; "
            f"animal-level p {_fmtp(g(a, 'animal_p'))}); {g(ah, 'median_observed'):.2f} with head direction removed "
            f"(p {_fmtp(g(ah, 'animal_p'))}); behaviour alone {g(ab, 'median_observed'):.2f} (p {_fmtp(g(ab, 'animal_p'))}).",
        ],
        [
            "Light / dark",
            "null",
            f"Split by condition the effect is underpowered: head-direction-removed decoding {g(lt, 'median_observed'):.2f} "
            f"in light ({lt.get('n_sessions', 0)} sessions, p {_fmtp(g(lt, 'animal_p'))}) and {g(dk, 'median_observed'):.2f} "
            f"in dark ({dk.get('n_sessions', 0)} sessions, p {_fmtp(g(dk, 'animal_p'))}); no light-dark difference in any decoder.",
        ],
        [
            "Behaviour",
            "lean",
            f"Trips reached dead ends not among the last three visited at the random-walk rate "
            f"({g(tu_l, 'observed'):.2f} light, {g(tu_d, 'observed'):.2f} dark vs {g(tu_l, 'expected'):.2f} expected), well "
            f"below a walker that never reverses ({g(tn_l, 'expected'):.2f}). Exploration is not memory-guided at the trip level.",
        ],
        [
            "Did not hold",
            "null",
            "Dead-end recency (time since last visit), planning of the next trip, distance to go, and trip origin or "
            "destination: each explained by immediate re-entries, posture, travel direction or too few samples.",
        ],
    ]
    direction_notes = [
        "Runs were matched on duration and mean speed between the two classes, and decoding was cross-validated over "
        "contiguous time blocks, so neither locomotion nor slow drift explains the result. Removing head direction "
        "from each cell does not reduce it, so it is not compass heading: two runs with the same lab-frame heading can "
        "be inbound in one corridor and outbound in another.",
        "Single cells show the same structure weakly and with mixed signs: about 15 % of cells are individually "
        "selective for runs into vs out of dead ends, some preferring each direction, which averages to zero.",
    ]
    behaviour_points = [
        f"<strong>Trips are not memory-guided.</strong> The fraction of trips ending at a dead end not visited among the "
        f"last three was {g(tu_l, 'observed'):.2f} in the light and {g(tu_d, 'observed'):.2f} in the dark, matching a "
        f"uniform random walk ({g(tu_l, 'expected'):.2f}; p {_fmtp(g(tu_l, 'animal_p'))}) and far below a walker that never "
        f"reverses ({g(tn_l, 'expected'):.2f}): mice often turn back the way they came.",
        f"<strong>Approach speed follows recency in both conditions.</strong> Mice run faster into dead ends they have not "
        f"visited for longer (median per-session ρ {g(ap.get('light', {}), 'median_rho'):.2f} in light, "
        f"{g(ap.get('dark', {}), 'median_rho'):.2f} in dark; p {_fmtp(g(ap.get('light', {}), 'animal_p'))} and "
        f"{_fmtp(g(ap.get('dark', {}), 'animal_p'))}). Longer gaps also follow longer trips, so this may reflect trip length "
        "rather than memory.",
        "<strong>Earlier result for context.</strong> Over a whole session, coverage of the maze beyond a random walk is "
        "higher in the light than the dark (directed exploration lost in darkness), even though single trips look random "
        "in both; the directedness accumulates over many choices rather than showing in any one trip.",
    ]
    leads = [
        {
            "name": "Dead-end recency",
            "question": "Does the arrival response depend on how long since this dead end was last visited?",
            "first": f"Dark-specific: dark minus light z {g(main_dl, 'median_z'):+.2f}, animal-level p {_fmtp(g(main_dl, 'animal_p'))}; "
            "robust to window, dead-end identity, behaviour and light-switch covariates, and to onset-weighted dF/F.",
            "control": f"Dropping re-entries within 5 s of leaving (about 30 % of entries) removed it "
            f"(dark minus light z {g(min5_dl, 'median_z'):+.2f}, p {_fmtp(g(min5_dl, 'animal_p'))}). It reflected "
            "immediate step-out-and-back re-entries, not memory over tens of seconds.",
            "verdict": "Did not hold",
            "held": False,
        },
        {
            "name": "Planning the next trip",
            "question": "Does activity before leaving a dead end predict whether the next trip goes somewhere new?",
            "first": f"Light: {g(pl_n, 'median_observed'):.2f} vs null {g(pl_n, 'median_null_mean'):.2f} (p {_fmtp(g(pl_n, 'animal_p'))}).",
            "control": f"Posture (syllables) and behaviour in the same second predict it as well ({g(pl_s, 'median_observed'):.2f}, "
            f"p {_fmtp(g(pl_s, 'animal_p'))}); excluding bounce-back trips leaves no neural signal "
            f"(p {_fmtp(g(pl_nr, 'animal_p'))}); a fresh light/dark subsample showed it in the dark too.",
            "verdict": "Did not hold",
            "held": False,
        },
        {
            "name": "Distance to go",
            "question": "At the same place, does activity track how many steps remain to the end of the trip?",
            "first": f"Dark, place-controlled, head direction removed: partial ρ {g(di_p, 'median_observed'):.2f} "
            f"(p {_fmtp(g(di_p, 'animal_p'))}).",
            "control": f"Controlling place and direction of travel together removes it (ρ {g(di_pd, 'median_observed'):.2f}, "
            f"p {_fmtp(g(di_pd, 'animal_p'))}): it was the inbound/outbound signal.",
            "verdict": "Did not hold",
            "held": False,
        },
        {
            "name": "Trip origin and destination",
            "question": "At the same place and direction, does activity depend on where the trip started or where it will end?",
            "first": f"Too few repeated trips per place and direction: at most {n_spl} sessions had enough samples.",
            "control": "Not testable with these session lengths.",
            "verdict": "Not testable",
            "held": False,
        },
        {
            "name": "Junction exit",
            "question": "Does activity before leaving a junction predict which arm is taken?",
            "first": "Behaviour (head direction and turning) predicts the exit; the population does not beyond it.",
            "control": "Head-direction-removed decoding at chance.",
            "verdict": "Did not hold",
            "held": False,
        },
    ]
    return {
        "headline": headline,
        "summary": summary,
        "direction_notes": direction_notes,
        "behaviour_points": behaviour_points,
        "leads": leads,
    }


def build(root: Path = RES) -> dict[str, Any]:
    """All report data."""
    out: dict[str, Any] = {"variants": recency_variants(root)}
    base = root / "dead_end_recency" / MAIN
    if (base / "cells.csv").exists():
        cells = pd.read_csv(base / "cells.csv")
        out["per_animal"] = per_animal(cells)
        out["n_cells"] = int(cells.drop_duplicates(["exp_id", "roi"]).shape[0])
        out["n_sessions"] = int(cells.exp_id.nunique())
        out["n_animals"] = int(cells.animal_id.nunique())
    if (base / "entries.csv").exists():
        ent = pd.read_csv(base / "entries.csv")
        out["n_entries"] = int(len(ent))
        if "pop_response" in ent.columns:
            out["binned"] = binned_population(ent)
        out["approach"] = approach_behaviour(ent)
    out["trips"] = trip_behaviour(root)
    out["decoding"] = decoding(root)
    out["inout_sessions"], out["inout_summary"] = inout_tables(root)
    out.update(narrative(out))
    return out


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def main() -> None:  # pragma: no cover - file I/O entry point
    data = _clean(build())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT}")


if __name__ == "__main__":  # pragma: no cover
    main()
