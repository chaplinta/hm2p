<script>
/* Shared helpers for the hm2p data-quality report pages. */
const QC = (() => {
  const PAGES = [
    ["qc-tracking.html", "Tracking"],
    ["qc-movement.html", "Movement metrics"],
    ["qc-syllables.html", "Syllables"],
    ["qc-rois.html", "Cell classification"],
    ["qc-spikes.html", "Spike extraction"],
  ];
  const CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };

  function css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
  function isNum(v) { return v !== null && v !== undefined && !Number.isNaN(v) && Number.isFinite(Number(v)); }
  function fmt(v, d = 2) { return isNum(v) ? Number(v).toFixed(d) : "—"; }
  function pct(v, d = 1) { return isNum(v) ? (100 * v).toFixed(d) + " %" : "—"; }
  function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
  function median(a) {
    const v = a.filter(isNum).map(Number).sort((x, y) => x - y);
    if (!v.length) return null;
    const m = Math.floor(v.length / 2);
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
  }

  /* int16 base64 traces written by hm2p.qc.common.encode_i16 */
  function b64bytes(b64) {
    const s = atob(b64), u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return u.buffer;
  }
  function decode(enc) {
    if (!enc) return [];
    const q = new Int16Array(b64bytes(enc.b64));
    const out = new Array(q.length);
    for (let i = 0; i < q.length; i++) out[i] = q[i] === -32768 ? null : enc.offset + enc.scale * q[i];
    return out;
  }
  function decodeTyped(b64, Type) { return new Type(b64bytes(b64)); }

  function layout(extra) {
    const muted = css("--muted"), rule = css("--rule");
    const ax = { gridcolor: rule, zerolinecolor: rule, linecolor: rule, color: muted, automargin: true, title: { font: { size: 12 } } };
    const base = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: css("--sans") || "sans-serif", size: 12, color: css("--ink") },
      margin: { l: 56, r: 14, t: 10, b: 44 }, showlegend: false,
      hoverlabel: { bgcolor: css("--surface"), bordercolor: rule, font: { color: css("--ink") } },
      legend: { orientation: "h", y: 1.12, x: 0, font: { size: 12 } },
      xaxis: Object.assign({}, ax), yaxis: Object.assign({}, ax),
    };
    const out = Object.assign(base, extra || {});
    for (const k of Object.keys(out)) if (/^[xy]axis\d*$/.test(k)) out[k] = Object.assign({}, ax, out[k]);
    return out;
  }
  function axTitle(t) { return { text: t, font: { size: 12 } }; }

  /* {lo, hi, counts} -> centres and (optionally normalised) heights */
  function histXY(h, norm = true) {
    if (!h || !h.counts) return { x: [], y: [] };
    const n = h.counts.length, w = (h.hi - h.lo) / n;
    const tot = norm ? Math.max(1, h.counts.reduce((a, b) => a + b, 0)) : 1;
    return { x: h.counts.map((_, i) => h.lo + (i + 0.5) * w), y: h.counts.map(c => c / tot), w };
  }
  function histLine(h, name, color, extra) {
    const { x, y } = histXY(h);
    return Object.assign({ type: "scatter", mode: "lines", line: { shape: "hvh", width: 2, color }, x, y, name,
      hovertemplate: `${esc(name)}: %{x:.3g} → %{y:.2%}<extra></extra>` }, extra || {});
  }
  function histBar(h, name, color, extra) {
    const { x, y, w } = histXY(h);
    return Object.assign({ type: "bar", x, y, width: w * 0.92, name, marker: { color, line: { width: 0 } },
      hovertemplate: `%{x:.3g} → %{y:.2%}<extra>${esc(name)}</extra>` }, extra || {});
  }

  /* Shaded rectangles for dark epochs given a 0..1 light fraction per x sample */
  function darkShapes(xs, light, yref = "paper") {
    const shapes = [];
    if (!light || !light.length) return shapes;
    let start = null;
    const half = xs.length > 1 ? (xs[1] - xs[0]) / 2 : 0;
    for (let i = 0; i <= light.length; i++) {
      const dark = i < light.length && light[i] !== null && light[i] < 0.5;
      if (dark && start === null) start = i;
      if (!dark && start !== null) {
        shapes.push({ type: "rect", xref: "x", yref, x0: xs[start] - half, x1: xs[i - 1] + half, y0: 0, y1: 1,
          fillcolor: css("--dark-c"), opacity: 0.12, line: { width: 0 }, layer: "below" });
        start = null;
      }
    }
    return shapes;
  }

  function nav(current) {
    const el = document.getElementById("qc-nav");
    if (!el) return;
    el.innerHTML = PAGES.map(([href, label]) =>
      `<a href="${href}"${href === current ? ' aria-current="page"' : ""}>${label}</a>`).join("");
  }

  function sessionLabel(s) {
    const tags = [s.celltype === "penk" ? "Penk+" : s.celltype === "nonpenk" ? "Penk⁻CamKII+" : (s.celltype || "?")];
    if (s.exclude) tags.push("excluded");
    return `${s.exp_id} · ${tags.join(" · ")}`;
  }
  function chips(s) {
    let h = `<span class="chip">${s.celltype === "penk" ? "Penk+" : s.celltype === "nonpenk" ? "Penk⁻CamKII+" : esc(s.celltype || "?")}</span>`;
    if (s.exclude) h += ' <span class="chip excl">excluded</span>';
    if (s.primary) h += ' <span class="chip prim">primary</span>';
    return h;
  }

  /* Shared selection state across tables, scatter plots and selects */
  const listeners = [];
  const state = { session: null };
  function onSession(fn) { listeners.push(fn); }
  function setSession(id) {
    state.session = id;
    document.querySelectorAll("select.session-select").forEach(s => { if (s.value !== id) s.value = id; });
    document.querySelectorAll("tr.pick[data-session]").forEach(tr => tr.setAttribute("aria-selected", tr.dataset.session === id ? "true" : "false"));
    listeners.forEach(fn => { try { fn(id); } catch (e) { console.error(e); } });
    try { localStorage.setItem("qc-session", id); } catch (e) { /* storage unavailable */ }
  }
  function initialSession(sessions) {
    let id = null;
    try { id = localStorage.getItem("qc-session"); } catch (e) { /* storage unavailable */ }
    const ok = sessions.filter(s => s.summary);
    if (id && ok.some(s => s.exp_id === id)) return id;
    return ok.length ? ok[0].exp_id : null;
  }
  function sessionSelect(el, sessions) {
    el.classList.add("session-select");
    el.innerHTML = sessions.map(s => `<option value="${esc(s.exp_id)}"${s.summary ? "" : " disabled"}>${esc(sessionLabel(s))}${s.summary ? "" : " (no data)"}</option>`).join("");
    el.addEventListener("change", () => setSession(el.value));
  }

  /* Sortable session table. columns: {key, label, get(s), fmt(v), flag(v) -> 'warn'|'bad'|null, title} */
  function sessionTable(el, sessions, columns) {
    let sortKey = null, dir = 1;
    function value(s, c) { return c.get ? c.get(s) : (s.overview ? s.overview[c.key] : null); }
    function render() {
      let rows = sessions.slice();
      if (sortKey !== null) {
        const c = columns[sortKey];
        rows.sort((a, b) => {
          const va = value(a, c), vb = value(b, c);
          if (!isNum(va) && !isNum(vb)) return String(a.exp_id).localeCompare(b.exp_id);
          if (!isNum(va)) return 1;
          if (!isNum(vb)) return -1;
          return dir * (va - vb);
        });
      }
      const head = `<tr><th class="sortable" data-i="-1">Session</th>${columns.map((c, i) =>
        `<th class="sortable" data-i="${i}" title="${esc(c.title || "")}"${sortKey === i ? ` aria-sort="${dir > 0 ? "ascending" : "descending"}"` : ""}>${c.label}</th>`).join("")}</tr>`;
      const body = rows.map(s => {
        if (!s.summary) return `<tr><td>${esc(s.exp_id)} ${chips(s)}</td><td colspan="${columns.length}" class="na">${esc(s.error || "no data")}</td></tr>`;
        const cells = columns.map(c => {
          const v = value(s, c);
          const f = c.flag && isNum(v) ? c.flag(Number(v)) : null;
          const txt = c.fmt ? c.fmt(v, s) : fmt(v);
          return `<td class="num${f ? " flag-" + f : ""}${!isNum(v) && !c.fmt ? " na" : ""}">${txt}</td>`;
        }).join("");
        return `<tr class="pick" tabindex="0" data-session="${esc(s.exp_id)}" aria-selected="${s.exp_id === state.session}"><td>${esc(s.exp_id)} ${chips(s)}</td>${cells}</tr>`;
      }).join("");
      el.innerHTML = `<table class="compact"><thead>${head}</thead><tbody>${body}</tbody></table>`;
      el.querySelectorAll("th.sortable").forEach(th => th.addEventListener("click", () => {
        const i = Number(th.dataset.i);
        if (i < 0) { sortKey = null; dir = 1; } else if (sortKey === i) dir = -dir; else { sortKey = i; dir = 1; }
        render();
      }));
      el.querySelectorAll("tr.pick").forEach(tr => {
        tr.addEventListener("click", () => setSession(tr.dataset.session));
        tr.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setSession(tr.dataset.session); } });
      });
    }
    render();
    return render;
  }

  /* One dot per session for a chosen overview metric, coloured by cell type */
  function stripPlot(div, sessions, col) {
    const groups = [["penk", "Penk+", "--penk"], ["nonpenk", "Penk⁻CamKII+", "--nonpenk"]];
    const traces = groups.map(([g, name, c]) => {
      const S = sessions.filter(s => s.summary && s.celltype === g);
      const v = S.map(s => (col.get ? col.get(s) : s.overview[col.key]));
      return { type: "scatter", mode: "markers", name, x: v, y: S.map(() => name), customdata: S.map(s => s.exp_id),
        marker: { size: 10, color: css(c), line: { width: 2, color: css("--surface") }, symbol: S.map(s => s.exclude ? "x" : "circle") },
        hovertemplate: `%{customdata}<br>${esc(col.label.replace(/<[^>]+>/g, ""))}: %{x:.4g}<extra>${name}</extra>` };
    });
    const lay = layout({ margin: { l: 110, r: 14, t: 6, b: 40 }, showlegend: false });
    lay.xaxis.title = axTitle(col.label.replace(/<[^>]+>/g, ""));
    Plotly.react(div, traces, lay, CONFIG);
    div.removeAllListeners && div.removeAllListeners("plotly_click");
    div.on("plotly_click", ev => { const p = ev.points && ev.points[0]; if (p && p.customdata) setSession(p.customdata); });
  }

  function metricStrip(selectEl, div, sessions, columns) {
    const usable = columns.filter(c => !c.noStrip);
    selectEl.innerHTML = usable.map((c, i) => `<option value="${i}">${c.label.replace(/<[^>]+>/g, "")}</option>`).join("");
    const draw = () => stripPlot(div, sessions, usable[Number(selectEl.value)]);
    selectEl.addEventListener("change", draw);
    draw();
    return draw;
  }

  function kv(el, pairs) {
    el.innerHTML = pairs.filter(p => p).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v === null || v === undefined || v === "" ? '<span class="na">—</span>' : esc(v)}</dd>`).join("");
  }
  /* Show a message in place of a chart. The div stays a Plotly plot (empty
     traces + centred annotation) so pending Plotly redraws never hit a purged div. */
  function empty(div, msg) {
    div.dataset.empty = msg;
    const lay = layout({ xaxis: { visible: false }, yaxis: { visible: false }, margin: { l: 10, r: 10, t: 10, b: 10 },
      annotations: [{ text: esc(msg), xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false, font: { size: 14, color: css("--muted") } }] });
    return Plotly.react(div, [], lay, CONFIG);
  }
  function ensurePlot(div) { delete div.dataset.empty; }
  function plot(id, traces, lay) {
    const div = typeof id === "string" ? document.getElementById(id) : id;
    ensurePlot(div);
    return Plotly.react(div, traces, lay, CONFIG);
  }

  function header(data) {
    const g = document.getElementById("generated");
    if (g) g.textContent = data.generated || "?";
    const c = document.getElementById("champion");
    if (c) c.textContent = (data.champion && data.champion.champion_id) || "none declared";
    const n = document.getElementById("n-ok");
    if (n) {
      const ok = data.sessions.filter(s => s.summary).length;
      n.textContent = `${ok} of ${data.sessions.length}`;
    }
  }

  /* Redraw on theme change so plot colours follow the page tokens */
  function onTheme(fn) {
    const mq = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
    if (mq && mq.addEventListener) mq.addEventListener("change", fn);
    new MutationObserver(fn).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }

  return { CONFIG, css, isNum, fmt, pct, esc, median, decode, decodeTyped, layout, axTitle, histXY, histLine, histBar,
    darkShapes, nav, sessionLabel, chips, state, onSession, setSession, initialSession, sessionSelect, sessionTable,
    stripPlot, metricStrip, kv, empty, ensurePlot, plot, header, onTheme };
})();
</script>
