"""Tests for junction-choice decision rules. Synthetic sequences on the real maze."""

from __future__ import annotations

import math

import numpy as np

from hm2p.maze.choice_models import (
    allocentric_choice,
    allocentric_frontier_choice,
    conflict_follow_rate,
    egocentric_choice,
    extract_choice_events,
    neighbours,
    rule_accuracies,
)
from hm2p.maze.topology import build_rose_maze

MAZE = build_rose_maze()
# Junction (5,4)=idx 20; neighbours 22=(6,4), 15=(4,4), 19=(5,3).
# Approaching from 22: 15 is "forward", 19 is "left", 22 is "back".
J, FROM22, FWD, LEFT = 20, 22, 15, 19
CANDS = [22, 15, 19]


# ---------------------------------------------------------------------------
# neighbours
# ---------------------------------------------------------------------------


def test_neighbours_match_graph():
    assert set(neighbours(MAZE, J)) == {22, 15, 19}


# ---------------------------------------------------------------------------
# egocentric_choice (alternation, then forward fallback)
# ---------------------------------------------------------------------------


def test_ego_alternation_after_right_picks_left():
    assert egocentric_choice(MAZE, FROM22, J, CANDS, "right") == LEFT


def test_ego_alternation_after_left_falls_back_forward():
    # no "right" arm from this approach -> momentum/forward
    assert egocentric_choice(MAZE, FROM22, J, CANDS, "left") == FWD


def test_ego_no_last_turn_is_forward():
    assert egocentric_choice(MAZE, FROM22, J, CANDS, None) == FWD


# ---------------------------------------------------------------------------
# allocentric_choice (least recently visited)
# ---------------------------------------------------------------------------


def test_allo_picks_unique_unvisited():
    assert allocentric_choice(CANDS, {22: 0, 15: 5, 19: math.inf}) == LEFT


def test_allo_ambiguous_two_unvisited_is_none():
    assert allocentric_choice(CANDS, {22: 0, 15: math.inf, 19: math.inf}) is None


def test_allo_picks_largest_finite_recency():
    assert allocentric_choice(CANDS, {22: 0, 15: 5, 19: 3}) == FWD


def test_allo_empty_is_none():
    assert allocentric_choice([], {}) is None


# ---------------------------------------------------------------------------
# allocentric_frontier_choice (non-myopic, distance-to-unexplored)
# ---------------------------------------------------------------------------


def test_frontier_picks_arm_toward_unexplored():
    # All cells visited recently EXCEPT a far one; the arm graph-closest to it wins.
    # Mark every cell as just-visited at step 100, window 10 -> none stale...
    last = {i: 100 for i in range(MAZE.n_cells)}
    # ...then make one distant cell stale by removing it (never visited).
    far = MAZE.cell_to_idx[(1, 4)]  # a dead-end far from junction (5,4)
    del last[far]
    # step-100 = 5 < window 10, so visited cells are NOT stale; only `far` is frontier.
    pred = allocentric_frontier_choice(MAZE, CANDS, step=105, last_visit_step=last,
                                       frontier_window=10)
    # the predicted arm should be the candidate with smallest graph distance to `far`
    dists = {c: int(MAZE.dist[c, far]) for c in CANDS}
    assert pred == min(dists, key=dists.get)


def test_frontier_none_when_all_recent():
    last = {i: 100 for i in range(MAZE.n_cells)}
    pred = allocentric_frontier_choice(MAZE, CANDS, step=101, last_visit_step=last,
                                       frontier_window=10)
    assert pred is None


def test_frontier_candidate_itself_stale_wins():
    # Candidate 19 never visited -> it is itself a frontier (distance 0) -> wins.
    last = {i: 100 for i in range(MAZE.n_cells) if i != 19}
    pred = allocentric_frontier_choice(MAZE, CANDS, step=105, last_visit_step=last,
                                       frontier_window=10)
    assert pred == 19


def test_frontier_empty_candidates_none():
    assert allocentric_frontier_choice(MAZE, [], 10, {0: 1}, 5) is None


# ---------------------------------------------------------------------------
# extract_choice_events
# ---------------------------------------------------------------------------


def test_extract_conflict_followed_ego():
    # Visit forward arm (15) first so it is non-novel; approach J from 22 with no
    # prior turn -> ego=forward=15; 19 never visited -> allo=19. Choosing 15
    # follows the habit (ego), not novelty (allo).
    seq = np.array([15, 22, 20, 15])
    frames = np.array([0, 1, 2, 3])
    light = np.array([True, True, True, True])
    ev = extract_choice_events(seq, frames, MAZE, light)
    assert len(ev) == 1
    e = ev[0]
    assert e["ego_pred"] == FWD and e["allo_pred"] == LEFT
    assert e["conflict"] is True
    assert e["followed"] == "ego"
    assert e["condition"] == "light"


def test_extract_choice_follows_allo():
    # Same setup but choose the novel arm (19) -> follows allocentric.
    seq = np.array([15, 22, 20, 19])
    ev = extract_choice_events(seq, np.array([0, 1, 2, 3]), MAZE, np.array([True] * 4))
    assert ev[0]["followed"] == "allo"
    assert ev[0]["conflict"] is True


def test_extract_condition_from_light():
    seq = np.array([15, 22, 20, 19])
    # junction decision frame is visit_frames[2] = 2 -> dark
    light = np.array([True, True, False, False])
    ev = extract_choice_events(seq, np.array([0, 1, 2, 3]), MAZE, light)
    assert ev[0]["condition"] == "dark"


def test_extract_too_short_returns_empty():
    assert extract_choice_events(np.array([20, 22]), np.array([0, 1]), MAZE, np.array([True, True])) == []


def test_extract_keys_and_followed_domain():
    rng = np.random.default_rng(0)
    # random walk over accessible cells (consecutive picks are arbitrary, but the
    # function tolerates it) to exercise the structural contract
    cells = [MAZE.cell_to_idx[c] for c in MAZE.cell_list]
    seq = rng.choice(cells, size=60)
    ev = extract_choice_events(seq, np.arange(60), MAZE, np.ones(60, bool))
    for e in ev:
        assert set(e) >= {"junction", "prev", "chosen", "candidates", "condition",
                          "ego_pred", "allo_pred", "conflict", "followed"}
        assert e["followed"] in {"ego", "allo", "both", "neither"}
        if e["conflict"]:
            assert e["ego_pred"] is not None and e["allo_pred"] is not None
            assert e["ego_pred"] != e["allo_pred"]


# ---------------------------------------------------------------------------
# conflict_follow_rate / rule_accuracies
# ---------------------------------------------------------------------------


def _ev(condition, conflict, followed):
    return {"condition": condition, "conflict": conflict, "followed": followed,
            "ego_pred": 1, "allo_pred": 2, "chosen": 1}


def test_conflict_follow_rate_counts():
    events = [
        _ev("light", True, "allo"),
        _ev("light", True, "ego"),
        _ev("light", True, "allo"),
        _ev("light", False, "ego"),   # not a conflict -> excluded
        _ev("dark", True, "ego"),
    ]
    rate_l, n_l = conflict_follow_rate(events, "light")
    assert n_l == 3 and math.isclose(rate_l, 2 / 3)
    rate_d, n_d = conflict_follow_rate(events, "dark")
    assert n_d == 1 and rate_d == 0.0


def test_conflict_follow_rate_no_conflict_is_nan():
    rate, n = conflict_follow_rate([_ev("light", False, "ego")], "light")
    assert n == 0 and math.isnan(rate)


def test_rule_accuracies_basic():
    events = [
        {"condition": "light", "ego_pred": 1, "allo_pred": 2, "chosen": 1},  # ego hit
        {"condition": "light", "ego_pred": 3, "allo_pred": 2, "chosen": 2},  # allo hit
        {"condition": "light", "ego_pred": None, "allo_pred": 2, "chosen": 9},
    ]
    acc = rule_accuracies(events, "light")
    assert acc["n_ego"] == 2 and math.isclose(acc["ego_acc"], 0.5)
    assert acc["n_allo"] == 3 and math.isclose(acc["allo_acc"], 1 / 3)
