"""Junction-choice decision rules: does the navigation controller switch in dark?

Behaviour-only. At each junction the mouse chooses a departure arm. Two candidate
decision rules predict that choice:

- **Egocentric / body-based**: continue the left/right alternation habit, or
  (failing a defined last turn) keep going forward. No map, no vision needed.
- **Allocentric / world-based**: head toward the least-recently-visited arm
  (novelty / recency). Needs memory of where the animal has been.

The key analysis is the *conflict trial*: junctions where the two rules predict
different arms. On those, which rule does the animal follow, and does the
fraction following the allocentric rule change between light and dark? No model
fitting; a direct paired proportion.

This module makes **no assumption about RSP function** — it is behaviour only.
Turn geometry uses ``hm2p.maze.analysis.classify_turn``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from hm2p.maze.analysis import classify_turn
from hm2p.maze.topology import RoseMaze


def neighbours(maze: RoseMaze, cell_idx: int) -> list[int]:
    """Indices (into maze.cell_list) of the accessible neighbours of a cell."""
    cell = maze.cell_list[cell_idx]
    return [maze.cell_to_idx[nb] for nb in maze.adj.get(cell, [])]


def _turn_label(maze: RoseMaze, prev_idx: int, junction_idx: int, cand_idx: int) -> str:
    return classify_turn(
        maze.cell_list[prev_idx], maze.cell_list[junction_idx], maze.cell_list[cand_idx]
    )


def egocentric_choice(
    maze: RoseMaze,
    prev_idx: int,
    junction_idx: int,
    candidates: list[int],
    last_turn: str | None,
) -> int | None:
    """Arm the body-based rule predicts: continue L/R alternation, else go forward.

    Returns the predicted candidate index, or None if undefined or ambiguous
    (no candidate with the required turn label, or more than one).
    """
    labels = {c: _turn_label(maze, prev_idx, junction_idx, c) for c in candidates}

    def _unique(label: str) -> int | None:
        hits = [c for c, lab in labels.items() if lab == label]
        return hits[0] if len(hits) == 1 else None

    if last_turn in ("left", "right"):
        want = "right" if last_turn == "left" else "left"
        pred = _unique(want)
        if pred is not None:
            return pred
    # Fallback: momentum / keep going forward.
    return _unique("forward")


def allocentric_choice(
    candidates: list[int],
    recency: dict[int, float],
) -> int | None:
    """Arm the world-based rule predicts: the least-recently-visited target.

    ``recency[c]`` = steps since the target cell of candidate ``c`` was last
    visited (math.inf if never). Returns the candidate with the strictly largest
    recency, or None if there is no unique maximum (e.g. several never-visited
    arms — genuinely undecidable by recency).
    """
    if not candidates:
        return None
    vals = [recency.get(c, math.inf) for c in candidates]
    mx = max(vals)
    winners = [c for c, v in zip(candidates, vals) if v == mx]
    return winners[0] if len(winners) == 1 else None


def allocentric_frontier_choice(
    maze: RoseMaze,
    candidates: list[int],
    step: int,
    last_visit_step: dict[int, int],
    frontier_window: int,
) -> int | None:
    """Arm heading toward unexplored territory (non-myopic, region-level).

    A cell is a "frontier" if it has not been visited within the last
    ``frontier_window`` cell-visits (or never). For each candidate arm, the
    distance to the nearest frontier cell (graph shortest path from the
    candidate's target cell) is computed; the rule predicts the candidate that
    minimises this distance — i.e. the arm that most directly heads toward
    somewhere the animal has not been recently. A candidate that is itself a
    frontier has distance 0.

    Returns the candidate with the strictly smallest distance-to-frontier, or
    None if there is no frontier (everything visited recently) or no unique
    minimum.
    """
    if not candidates:
        return None
    # Frontier = never visited, or not visited within the last frontier_window steps.
    frontier = []
    for i in range(maze.n_cells):
        last = last_visit_step.get(i)
        if last is None or (step - last) >= frontier_window:
            frontier.append(i)
    if not frontier:
        return None
    fidx = np.asarray(frontier, dtype=int)
    dists = [int(maze.dist[c, fidx].min()) for c in candidates]
    mn = min(dists)
    winners = [c for c, d in zip(candidates, dists) if d == mn]
    return winners[0] if len(winners) == 1 else None


def extract_choice_events(
    visit_cells: np.ndarray,
    visit_frames: np.ndarray,
    maze: RoseMaze,
    light_on: np.ndarray,
    allo_rule: str = "myopic",
    frontier_window: int = 10,
) -> list[dict[str, Any]]:
    """Build per-junction choice events from a distinct-cell visit sequence.

    Parameters
    ----------
    visit_cells : (n_visits,) int
        Sequence of distinct consecutive maze-cell indices (>=0).
    visit_frames : (n_visits,) int
        Frame index where each visit begins (to read the light condition).
    maze : RoseMaze
    light_on : (n_frames,) bool
    allo_rule : {"myopic", "frontier"}
        "myopic" = least-recently-visited neighbour arm; "frontier" = arm whose
        target is graph-closest to unexplored territory (see
        :func:`allocentric_frontier_choice`).
    frontier_window : int
        Visits-since-last-visit beyond which a cell counts as frontier (frontier
        rule only).

    Returns
    -------
    list[dict] with keys: junction, prev, chosen, candidates, condition
    ('light'/'dark'), last_turn, ego_pred, allo_pred, conflict (bool),
    followed ('ego'/'allo'/'both'/'neither'). Recency is whole-sequence
    (memory does not reset across epochs).
    """
    vc = np.asarray(visit_cells, dtype=int)
    vf = np.asarray(visit_frames, dtype=int)
    light = np.asarray(light_on, dtype=bool)
    n = vc.size
    events: list[dict[str, Any]] = []
    if n < 3:
        return events

    junction_set = {maze.cell_to_idx[j] for j in maze.junctions}
    last_visit_step: dict[int, int] = {}
    last_turn: str | None = None

    for k in range(n):
        cur = int(vc[k])
        # Decision happens at an interior junction with a real prev and next.
        if 0 < k < n - 1 and cur in junction_set and vc[k - 1] >= 0 and vc[k + 1] >= 0:
            prev_idx = int(vc[k - 1])
            chosen = int(vc[k + 1])
            cands = neighbours(maze, cur)
            recency = {
                c: (k - last_visit_step[c]) if c in last_visit_step else math.inf
                for c in cands
            }
            ego = egocentric_choice(maze, prev_idx, cur, cands, last_turn)
            if allo_rule == "frontier":
                allo = allocentric_frontier_choice(
                    maze, cands, k, last_visit_step, frontier_window
                )
            else:
                allo = allocentric_choice(cands, recency)
            conflict = ego is not None and allo is not None and ego != allo
            followed = "neither"
            hit_ego = ego is not None and chosen == ego
            hit_allo = allo is not None and chosen == allo
            if hit_ego and hit_allo:
                followed = "both"
            elif hit_ego:
                followed = "ego"
            elif hit_allo:
                followed = "allo"
            frame = int(vf[k]) if vf[k] < light.size else light.size - 1
            events.append({
                "junction": cur,
                "prev": prev_idx,
                "chosen": chosen,
                "candidates": cands,
                "condition": "light" if light[frame] else "dark",
                "last_turn": last_turn,
                "ego_pred": ego,
                "allo_pred": allo,
                "conflict": conflict,
                "followed": followed,
            })
            # Update the alternation memory with the realised turn.
            turn = _turn_label(maze, prev_idx, cur, chosen)
            if turn in ("left", "right"):
                last_turn = turn
        # Visiting cur updates its recency for future decisions.
        last_visit_step[cur] = k

    return events


def conflict_follow_rate(events: list[dict[str, Any]], condition: str) -> tuple[float, int]:
    """Fraction of conflict trials (in ``condition``) where the allocentric rule
    was followed. Returns (rate, n_conflict). NaN rate if no conflict trials.

    Conflict trials where the animal followed neither rule (e.g. backtracked)
    count in the denominator: the question is how often the world-based option
    is taken when it is on offer and disagrees with the habit.
    """
    conf = [e for e in events if e["conflict"] and e["condition"] == condition]
    if not conf:
        return float("nan"), 0
    n_allo = sum(1 for e in conf if e["followed"] == "allo")
    return n_allo / len(conf), len(conf)


def rule_accuracies(events: list[dict[str, Any]], condition: str) -> dict[str, float]:
    """How often each rule alone predicts the actual choice, in ``condition``.

    Over events where that rule makes a prediction. Returns dict with
    ``ego_acc``, ``allo_acc``, ``n_ego``, ``n_allo``.
    """
    ev = [e for e in events if e["condition"] == condition]
    ego_def = [e for e in ev if e["ego_pred"] is not None]
    allo_def = [e for e in ev if e["allo_pred"] is not None]
    ego_acc = (
        sum(1 for e in ego_def if e["chosen"] == e["ego_pred"]) / len(ego_def)
        if ego_def else float("nan")
    )
    allo_acc = (
        sum(1 for e in allo_def if e["chosen"] == e["allo_pred"]) / len(allo_def)
        if allo_def else float("nan")
    )
    return {"ego_acc": ego_acc, "allo_acc": allo_acc,
            "n_ego": len(ego_def), "n_allo": len(allo_def)}
