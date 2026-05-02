"""Reusable Plotly components for the sync diagnostics page.

Each function returns a ``plotly.graph_objects.Figure`` so the caller
can pass it directly to ``st.plotly_chart``. All accept an optional
``time_window`` so the deep-dive panel can lock zoom across the
pulse-train raster, cumulative-count line, and ISI histograms.

See ``docs/sync-pipeline-design.md`` §1.6 / §4.4.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Page-wide palette for sync_status — must match the legend in the page header.
COLOR_OK: str = "#16a34a"
COLOR_WARN: str = "#d97706"
COLOR_FAIL: str = "#dc2626"


def pulse_train_raster(
    channels: dict[str, np.ndarray],
    *,
    light_on: np.ndarray | None = None,
    light_off: np.ndarray | None = None,
    time_window: tuple[float, float] | None = None,
) -> go.Figure:
    """Multi-row pulse-train raster.

    One row per channel, one tick per pulse. Light-on segments are
    shaded yellow; light-off intervals are shaded grey lightly.

    Parameters
    ----------
    channels:
        Mapping of channel name (e.g. ``"camera"``, ``"line clock"``,
        ``"imaging"``) to 1D pulse-time arrays in seconds.
    light_on, light_off:
        Light edge-time arrays (seconds). ``light_off`` follows
        ``light_on`` in pairs; trailing edges are left open.
    time_window:
        Optional ``(t0, t1)`` window in seconds; pulses outside the
        window are dropped from the trace.
    """
    fig = go.Figure()
    names = list(channels.keys())
    for i, name in enumerate(names):
        times = np.asarray(channels[name], dtype=np.float64)
        if time_window is not None:
            mask = (times >= time_window[0]) & (times <= time_window[1])
            times = times[mask]
        fig.add_trace(
            go.Scattergl(
                x=times,
                y=np.full(times.size, i),
                mode="markers",
                marker=dict(symbol="line-ns", line=dict(width=1), size=10),
                name=name,
                hovertemplate=f"{name}<br>t=%{{x:.4f}} s<extra></extra>",
            )
        )
    # Light shading.
    if light_on is not None and light_off is not None:
        on = np.asarray(light_on, dtype=np.float64)
        off = np.asarray(light_off, dtype=np.float64)
        for s, e in _light_intervals(on, off):
            if time_window is not None and (e < time_window[0] or s > time_window[1]):
                continue
            fig.add_vrect(x0=s, x1=e, fillcolor="#fde68a", opacity=0.25, line_width=0)
    fig.update_layout(
        height=200 + 30 * len(names),
        margin=dict(l=60, r=20, t=20, b=40),
        xaxis_title="Time (s)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(names))),
            ticktext=names,
            range=[-0.5, len(names) - 0.5],
        ),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def cumulative_pulses(
    channels: dict[str, np.ndarray],
    *,
    time_window: tuple[float, float] | None = None,
) -> go.Figure:
    """Cumulative pulse count curve — slope deviation reveals drift."""
    fig = go.Figure()
    for name, times in channels.items():
        t = np.asarray(times, dtype=np.float64)
        if time_window is not None:
            t = t[(t >= time_window[0]) & (t <= time_window[1])]
        if t.size == 0:
            continue
        cumulative = np.arange(1, t.size + 1)
        fig.add_trace(
            go.Scatter(
                x=t,
                y=cumulative,
                mode="lines",
                name=name,
                line=dict(width=1.5),
            )
        )
    fig.update_layout(
        height=300,
        margin=dict(l=60, r=20, t=20, b=40),
        xaxis_title="Time (s)",
        yaxis_title="Cumulative pulse count",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def isi_histogram(
    times: np.ndarray,
    fps_nominal: float,
    *,
    log_y: bool = True,
    n_bins: int = 80,
) -> go.Figure:
    """ISI histogram with a vertical line at the nominal interval.

    Non-parametric: median + MAD are robust to the heavy-tailed
    artefacts (single dropped/duplicate pulses) that are most likely to
    appear here.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.size < 2:
        return go.Figure().update_layout(
            annotations=[dict(text="not enough pulses", x=0.5, y=0.5, showarrow=False)],
            height=240,
        )
    isis_ms = np.diff(times) * 1000.0
    fig = go.Figure(
        go.Histogram(
            x=isis_ms,
            nbinsx=n_bins,
            marker=dict(color="#3b82f6"),
        )
    )
    if fps_nominal > 0:
        nominal_ms = 1000.0 / fps_nominal
        fig.add_vline(
            x=nominal_ms,
            line_dash="dash",
            line_color="#16a34a",
            annotation_text=f"{nominal_ms:.2f} ms",
        )
    fig.update_layout(
        height=240,
        margin=dict(l=50, r=20, t=20, b=40),
        xaxis_title="ISI (ms)",
        yaxis_title="count",
        yaxis_type="log" if log_y else "linear",
        showlegend=False,
    )
    return fig


def light_cycle_strip(
    light_on: np.ndarray,
    light_off: np.ndarray,
    t_max: float,
) -> go.Figure:
    """Single horizontal strip of light-on/off shading."""
    fig = go.Figure()
    on = np.asarray(light_on, dtype=np.float64)
    off = np.asarray(light_off, dtype=np.float64)
    # Background strip (entire duration) in light-off grey.
    fig.add_vrect(x0=0, x1=t_max, fillcolor="#e5e7eb", opacity=0.4, line_width=0)
    for s, e in _light_intervals(on, off):
        if e > t_max:
            e = t_max
        if s < 0:
            s = 0.0
        fig.add_vrect(x0=s, x1=e, fillcolor="#fde68a", opacity=0.7, line_width=0)
    fig.update_layout(
        height=80,
        margin=dict(l=60, r=20, t=10, b=30),
        xaxis_title="Time (s)",
        yaxis=dict(visible=False, range=[0, 1]),
        showlegend=False,
    )
    return fig


def _light_intervals(
    light_on: np.ndarray,
    light_off: np.ndarray,
) -> list[tuple[float, float]]:
    """Pair light-on and light-off times into intervals.

    A simple sweep: every ON edge is paired with the next OFF edge
    that follows it. Unpaired trailing edges are dropped.
    """
    intervals: list[tuple[float, float]] = []
    on = np.sort(np.asarray(light_on, dtype=np.float64))
    off = np.sort(np.asarray(light_off, dtype=np.float64))
    j = 0
    for s in on:
        while j < off.size and off[j] <= s:
            j += 1
        if j >= off.size:
            break
        intervals.append((float(s), float(off[j])))
        j += 1
    return intervals
