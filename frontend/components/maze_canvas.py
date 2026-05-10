"""HTML5 Canvas component for maze animation.

Builds an HTML+JS+CSS template that renders mouse trajectory, skeleton,
head direction arrow, and light/dark state on a ``<canvas>`` element.
All playback controls (play/pause, speed, scrubber) live in JS and run
at 60 fps without Streamlit reruns.

Rendered via ``st.html()`` with ``unsafe_allow_javascript=True``, or
falls back to the deprecated ``st.components.v1.html()`` on older
Streamlit versions.
"""

from __future__ import annotations

import json
import uuid
from typing import Any


def _js_animation_code(uid: str) -> str:
    """Return the JavaScript source for the canvas animation loop.

    Parameters
    ----------
    uid : str
        Unique suffix appended to all DOM element IDs to avoid collisions
        when Streamlit re-renders the component.
    """
    return rf"""
(function() {{
    // ── Data from Python ────────────────────────────────────────────
    const D = window.__MAZE_DATA_{uid}__;
    const N = D.n_frames;
    if (N === 0) return;

    const canvas = document.getElementById('maze-canvas-{uid}');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // ── Sizing ──────────────────────────────────────────────────────
    const PAD = 40;
    const MAZE_X_MIN = -0.5, MAZE_X_MAX = 7.5;
    const MAZE_Y_MIN = -0.5, MAZE_Y_MAX = 5.5;
    const MAZE_W = MAZE_X_MAX - MAZE_X_MIN;  // 8
    const MAZE_H = MAZE_Y_MAX - MAZE_Y_MIN;  // 6

    // Maintain aspect ratio inside the canvas
    const availW = canvas.width - 2 * PAD;
    const availH = canvas.height - 2 * PAD;
    const scale = Math.min(availW / MAZE_W, availH / MAZE_H);
    const offX = PAD + (availW - MAZE_W * scale) / 2;
    const offY = PAD + (availH - MAZE_H * scale) / 2;

    function toCanvasX(mx) {{ return offX + (mx - MAZE_X_MIN) * scale; }}
    // Canvas y is inverted (0 at top), maze y increases upward
    function toCanvasY(my) {{ return offY + (MAZE_Y_MAX - my) * scale; }}

    // ── Controls ────────────────────────────────────────────────────
    const btnPlay = document.getElementById('btn-play-{uid}');
    const selSpeed = document.getElementById('sel-speed-{uid}');
    const scrubber = document.getElementById('scrubber-{uid}');
    const lblTime = document.getElementById('lbl-time-{uid}');
    const lblFrame = document.getElementById('lbl-frame-{uid}');
    const lblHD = document.getElementById('lbl-hd-{uid}');
    const lblSpeed = document.getElementById('lbl-speed-{uid}');
    const lblLight = document.getElementById('lbl-light-{uid}');

    scrubber.max = N - 1;
    scrubber.value = 0;

    let playing = false;
    let frameIdx = 0;
    let lastTimestamp = null;
    let animId = null;

    // Effective playback dt between data frames (seconds)
    const dataDt = N > 1
        ? (D.frame_times[N - 1] - D.frame_times[0]) / (N - 1)
        : 0.1;

    btnPlay.addEventListener('click', function() {{
        playing = !playing;
        btnPlay.textContent = playing ? 'Pause' : 'Play';
        lastTimestamp = null;
    }});

    scrubber.addEventListener('input', function() {{
        frameIdx = parseInt(this.value, 10);
        drawFrame(frameIdx);
    }});

    // ── Pre-compute bodypart index map ──────────────────────────────
    const bpIndex = {{}};
    for (let b = 0; b < D.bp_names.length; b++) {{
        bpIndex[D.bp_names[b]] = b;
    }}

    // Find head_midpoint and nose_tip by name (with legacy aliases)
    function bpIdx(names) {{
        for (const n of names) {{
            if (bpIndex[n] !== undefined) return n;
        }}
        return null;
    }}
    const headBp = bpIdx(['head_midpoint', 'implant_base_rear']);
    const noseBp = bpIdx(['nose_tip', 'nose']);

    // ── Trail buffer ────────────────────────────────────────────────
    const trailFrames = Math.max(1, Math.round(D.trail_seconds / dataDt));

    // ── Drawing functions ───────────────────────────────────────────

    function drawMazeWalls(isLight) {{
        ctx.beginPath();
        const wx = D.maze_walls_x;
        const wy = D.maze_walls_y;
        ctx.moveTo(toCanvasX(wx[0]), toCanvasY(wy[0]));
        for (let i = 1; i < wx.length; i++) {{
            ctx.lineTo(toCanvasX(wx[i]), toCanvasY(wy[i]));
        }}
        ctx.closePath();
        ctx.strokeStyle = isLight ? '#000000' : 'rgba(200,200,200,0.8)';
        ctx.lineWidth = 2;
        ctx.stroke();
    }}

    function drawTrail(fi) {{
        const start = Math.max(0, fi - trailFrames);
        const count = fi - start + 1;
        if (count < 1 || !headBp) return;

        const xArr = D.bp_x[headBp];
        const yArr = D.bp_y[headBp];
        const isLight = D.light_on[fi];

        for (let j = start; j <= fi; j++) {{
            const px = xArr[j];
            const py = yArr[j];
            if (px === null || py === null) continue;
            const t = (j - start) / Math.max(count - 1, 1);
            const alpha = 0.1 + 0.7 * t;
            ctx.beginPath();
            ctx.arc(toCanvasX(px), toCanvasY(py), 2.5, 0, 2 * Math.PI);
            ctx.fillStyle = isLight
                ? 'rgba(255, 109, 56, ' + alpha + ')'
                : 'rgba(140, 140, 140, ' + alpha + ')';
            ctx.fill();
        }}
    }}

    function drawSkeleton(fi) {{
        const isLight = D.light_on[fi];

        // Bone connections
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = isLight
            ? 'rgba(120, 120, 120, 0.7)'
            : 'rgba(180, 180, 180, 0.5)';

        for (const pair of D.skeleton) {{
            const bp1 = pair[0], bp2 = pair[1];
            if (!(bp1 in D.bp_x) || !(bp2 in D.bp_x)) continue;
            const x1 = D.bp_x[bp1][fi], y1 = D.bp_y[bp1][fi];
            const x2 = D.bp_x[bp2][fi], y2 = D.bp_y[bp2][fi];
            if (x1 === null || y1 === null || x2 === null || y2 === null) continue;

            ctx.beginPath();
            ctx.moveTo(toCanvasX(x1), toCanvasY(y1));
            ctx.lineTo(toCanvasX(x2), toCanvasY(y2));
            ctx.stroke();
        }}

        // Bodypart dots
        for (const bpName of D.bp_names) {{
            const bx = D.bp_x[bpName][fi];
            const by = D.bp_y[bpName][fi];
            if (bx === null || by === null) continue;
            const cx = toCanvasX(bx);
            const cy = toCanvasY(by);
            ctx.beginPath();
            ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
            ctx.fillStyle = D.bp_colors[bpName] || '#888888';
            ctx.fill();
            ctx.strokeStyle = isLight ? '#000' : '#fff';
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }}
    }}

    function drawPositionDot(fi) {{
        if (!headBp) return;
        const hx = D.bp_x[headBp][fi];
        const hy = D.bp_y[headBp][fi];
        if (hx === null || hy === null) return;
        const isLight = D.light_on[fi];
        const dotSize = D.show_skeleton ? 4 : 8;

        ctx.beginPath();
        ctx.arc(toCanvasX(hx), toCanvasY(hy), dotSize, 0, 2 * Math.PI);
        ctx.fillStyle = isLight ? '#FF6D38' : '#8C8C8C';
        ctx.fill();
        ctx.strokeStyle = isLight ? '#000' : '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
    }}

    function drawArrow(fi) {{
        if (!headBp || !noseBp) return;
        const hx = D.bp_x[headBp][fi];
        const hy = D.bp_y[headBp][fi];
        const nx = D.bp_x[noseBp][fi];
        const ny = D.bp_y[noseBp][fi];
        if (hx === null || hy === null || nx === null || ny === null) return;

        const dx = nx - hx;
        const dy = ny - hy;
        const norm = Math.sqrt(dx * dx + dy * dy);
        if (norm < 1e-6) return;

        const ux = dx / norm;
        const uy = dy / norm;
        const arrowLen = D.arrow_length;

        const endMazeX = hx + ux * arrowLen;
        const endMazeY = hy + uy * arrowLen;

        const cx1 = toCanvasX(hx);
        const cy1 = toCanvasY(hy);
        const cx2 = toCanvasX(endMazeX);
        const cy2 = toCanvasY(endMazeY);

        const isLight = D.light_on[fi];
        const arrowColor = isLight ? '#7F00FF' : '#A080D0';

        // Arrow shaft
        ctx.beginPath();
        ctx.moveTo(cx1, cy1);
        ctx.lineTo(cx2, cy2);
        ctx.strokeStyle = arrowColor;
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Arrowhead (two lines from tip)
        const headLen = 8;
        const headAngle = Math.PI / 6;
        const angle = Math.atan2(cy2 - cy1, cx2 - cx1);
        ctx.beginPath();
        ctx.moveTo(cx2, cy2);
        ctx.lineTo(
            cx2 - headLen * Math.cos(angle - headAngle),
            cy2 - headLen * Math.sin(angle - headAngle)
        );
        ctx.moveTo(cx2, cy2);
        ctx.lineTo(
            cx2 - headLen * Math.cos(angle + headAngle),
            cy2 - headLen * Math.sin(angle + headAngle)
        );
        ctx.strokeStyle = arrowColor;
        ctx.lineWidth = 2;
        ctx.stroke();
    }}

    function drawLegend() {{
        const legendX = canvas.width - 130;
        let legendY = 15;
        ctx.font = 'bold 11px sans-serif';
        ctx.fillStyle = '#333';
        ctx.fillText('Bodyparts', legendX, legendY);
        legendY += 5;

        ctx.font = '10px sans-serif';
        for (const bpName of D.bp_names) {{
            legendY += 15;
            ctx.beginPath();
            ctx.arc(legendX + 6, legendY - 3, 4, 0, 2 * Math.PI);
            ctx.fillStyle = D.bp_colors[bpName] || '#888';
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 0.5;
            ctx.stroke();
            ctx.fillStyle = '#333';
            ctx.fillText(bpName, legendX + 15, legendY);
        }}
    }}

    function drawFrame(fi) {{
        fi = Math.max(0, Math.min(N - 1, fi));
        frameIdx = fi;
        const isLight = D.light_on[fi];

        // Clear
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Background
        ctx.fillStyle = isLight ? '#fafafa' : '#2a2a2a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Maze interior fill
        ctx.beginPath();
        const wx = D.maze_walls_x;
        const wy = D.maze_walls_y;
        ctx.moveTo(toCanvasX(wx[0]), toCanvasY(wy[0]));
        for (let i = 1; i < wx.length; i++) {{
            ctx.lineTo(toCanvasX(wx[i]), toCanvasY(wy[i]));
        }}
        ctx.closePath();
        ctx.fillStyle = isLight ? '#ffffff' : '#3a3a3a';
        ctx.fill();

        drawMazeWalls(isLight);
        drawTrail(fi);

        if (D.show_skeleton) drawSkeleton(fi);
        if (D.show_position) {{
            drawPositionDot(fi);
            drawArrow(fi);
        }}
        if (D.show_skeleton) drawLegend();

        // Update readout text
        scrubber.value = fi;
        const t0 = D.frame_times[0];
        const t = D.frame_times[fi] - t0;
        lblTime.textContent = (t / 60).toFixed(1) + ' min';
        lblFrame.textContent = fi + ' / ' + (N - 1);

        const hdVal = D.hd_deg[fi];
        if (hdVal !== null && !isNaN(hdVal)) {{
            lblHD.textContent = (((hdVal % 360) + 360) % 360).toFixed(0) + '\u00b0';
        }} else {{
            lblHD.textContent = '--';
        }}

        const spdVal = D.speed[fi];
        if (spdVal !== null && !isNaN(spdVal)) {{
            lblSpeed.textContent = spdVal.toFixed(1) + ' cm/s';
        }} else {{
            lblSpeed.textContent = '--';
        }}

        lblLight.textContent = isLight ? 'Light ON' : 'Light OFF';
        lblLight.style.color = isLight ? '#b87333' : '#888';
    }}

    // ── Animation loop ──────────────────────────────────────────────
    function tick(timestamp) {{
        if (playing) {{
            if (lastTimestamp === null) lastTimestamp = timestamp;
            const elapsed = (timestamp - lastTimestamp) / 1000;
            lastTimestamp = timestamp;
            const playbackRate = parseFloat(selSpeed.value);
            const dataElapsed = elapsed * playbackRate;
            const framesToAdvance = dataElapsed / dataDt;
            frameIdx += framesToAdvance;
            if (frameIdx >= N) frameIdx = 0;
        }}
        drawFrame(Math.floor(frameIdx));
        animId = requestAnimationFrame(tick);
    }}

    // Draw initial frame and start loop
    drawFrame(0);
    animId = requestAnimationFrame(tick);
}})();
"""


def _css_styles(uid: str) -> str:
    """Return scoped CSS for the canvas component layout.

    Parameters
    ----------
    uid : str
        Unique suffix for CSS scoping.
    """
    return f"""
    <style>
        .maze-container-{uid} {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 920px;
        }}
        .maze-container-{uid} .maze-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 6px;
            flex-wrap: wrap;
        }}
        .maze-container-{uid} .maze-controls button {{
            padding: 5px 18px;
            font-size: 13px;
            font-weight: 600;
            border: 1px solid #ccc;
            border-radius: 4px;
            background: #f0f0f0;
            cursor: pointer;
            min-width: 65px;
        }}
        .maze-container-{uid} .maze-controls button:hover {{
            background: #e0e0e0;
        }}
        .maze-container-{uid} .maze-controls select {{
            padding: 4px 8px;
            font-size: 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background: #fff;
        }}
        .maze-container-{uid} .maze-scrubber {{
            width: 100%;
            margin: 4px 0;
        }}
        .maze-container-{uid} .maze-readouts {{
            display: flex;
            gap: 16px;
            font-size: 12px;
            color: #555;
            margin-top: 2px;
            flex-wrap: wrap;
        }}
        .maze-container-{uid} .maze-readouts span {{
            white-space: nowrap;
        }}
        .maze-container-{uid} .maze-readouts .label {{
            font-weight: 600;
            color: #333;
        }}
        #maze-canvas-{uid} {{
            border: 1px solid #ddd;
            border-radius: 4px;
            display: block;
        }}
    </style>
    """


def build_maze_canvas_html(payload: dict[str, Any]) -> str:
    """Build the complete HTML string for the maze canvas component.

    Parameters
    ----------
    payload : dict
        Serializable dict with the animation data. Expected keys:
        ``n_frames``, ``bp_names``, ``skeleton``, ``bp_colors``,
        ``maze_walls_x``, ``maze_walls_y``, ``bp_x``, ``bp_y``,
        ``hd_deg``, ``speed``, ``light_on``, ``frame_times``,
        ``arrow_length``, ``trail_seconds``, ``show_position``,
        ``show_skeleton``.

    Returns
    -------
    str
        Self-contained HTML fragment (CSS + canvas + JS) ready for
        ``st.html()`` or ``st.components.v1.html()``.
    """
    # Unique ID suffix for DOM elements — avoids collisions on rerender
    uid = uuid.uuid4().hex[:8]
    data_json = json.dumps(payload, separators=(",", ":"))

    # Canvas dimensions: 880px wide preserves the 8:6 maze aspect ratio
    # with padding, fitting inside a standard Streamlit column.
    canvas_width = 880
    canvas_height = 680

    html = f"""\
{_css_styles(uid)}
<div class="maze-container-{uid}">
    <div class="maze-controls">
        <button id="btn-play-{uid}">Play</button>
        <label style="font-size:12px;">Speed:
            <select id="sel-speed-{uid}">
                <option value="0.25">0.25x</option>
                <option value="0.5">0.5x</option>
                <option value="1" selected>1x</option>
                <option value="2">2x</option>
                <option value="4">4x</option>
            </select>
        </label>
    </div>
    <input type="range" id="scrubber-{uid}" class="maze-scrubber"
           min="0" max="0" value="0" step="1">
    <canvas id="maze-canvas-{uid}" width="{canvas_width}" height="{canvas_height}"></canvas>
    <div class="maze-readouts">
        <span><span class="label">Time:</span> <span id="lbl-time-{uid}">0.0 min</span></span>
        <span><span class="label">Frame:</span> <span id="lbl-frame-{uid}">0 / 0</span></span>
        <span><span class="label">HD:</span> <span id="lbl-hd-{uid}">--</span></span>
        <span><span class="label">Speed:</span> <span id="lbl-speed-{uid}">--</span></span>
        <span><span class="label">Light:</span> <span id="lbl-light-{uid}">--</span></span>
    </div>
</div>
<script>
window.__MAZE_DATA_{uid}__ = {data_json};
{_js_animation_code(uid)}
</script>
"""
    return html


def render_maze_canvas(payload: dict[str, Any], height: int = 780) -> None:
    """Render the maze canvas component in Streamlit.

    Uses ``st.html`` (Streamlit >= 1.44) with ``unsafe_allow_javascript=True``
    for the animation loop. Falls back to the deprecated
    ``st.components.v1.html`` on older versions.

    Parameters
    ----------
    payload : dict
        Animation data payload (see ``build_maze_canvas_html``).
    height : int
        Height of the iframe in pixels. Used only by the
        ``st.components.v1.html`` fallback; ``st.html`` sizes to content.
    """
    import streamlit as st

    html = build_maze_canvas_html(payload)

    if hasattr(st, "html"):
        # st.html (>= 1.44) — current API, not iframed
        st.html(html, unsafe_allow_javascript=True)
    else:
        # Fallback for older Streamlit versions
        import streamlit.components.v1 as components

        components.html(html, height=height, scrolling=False)
