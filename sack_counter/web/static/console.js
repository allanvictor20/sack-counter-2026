/* web/static/console.js — Console UI interactions & live polling.
 *
 * Every screen that needs client-side behaviour calls one method on the
 * ``Console`` object from a tiny inline <script> at the foot of its
 * template.  Each method is a no-op when the elements it expects are
 * absent, so a missing block never throws on a screen that doesn't use
 * it.  ``Console.historyScreen`` drives the All / Sacks in / Sacks out
 * segmented filter on the history page; ``Console.diagnosticsScreen``
 * drives the Run check button on the detection-check page.
 */

const Console = (function () {
  let _pollTimer = null;

  /* ── helpers ──────────────────────────────────────────────── */

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body || {}),
    });
    let payload = null;
    try { payload = await res.json(); } catch (_) { /* empty body */ }
    if (!res.ok) {
      const msg = (payload && (payload.detail || payload.error))
        || `Request failed (${res.status})`;
      throw new Error(msg);
    }
    return payload || {};
  }

  function el(id) { return document.getElementById(id); }

  function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); }

  /* ── snapshot → live sidebar / live page elements ────────── */

  function updateSidebar(snap) {
    if (!snap) return;

    const dot   = el("status-dot");
    const label = el("status-label");
    const meta  = el("status-meta");

    const LABELS = {
      running: "Counting", starting: "Starting",
      finished: "Finished", error:    "Stopped", idle: "Idle",
    };

    if (dot)   dot.classList.toggle("on", !!snap.running);
    if (label) label.textContent = LABELS[snap.status] || "Idle";
    if (meta) {
      const frame = (snap.frame || 0).toLocaleString();
      meta.innerHTML =
        `Frame ${frame}<br>${snap.fps || 0} fps · ${snap.elapsed || "0:00"}`;
    }
  }

  function updateSnapshot(snap) {
    if (!snap) return;
    updateSidebar(snap);

    const countEl  = el("live-count");
    const modeLbl  = el("mode-label");
    const statusEl = el("session-status");

    if (statusEl && snap.status) {
      statusEl.textContent = String(snap.status).toUpperCase();
    }

    const totals = snap.totals || {};
    const isExit = snap.mode === "exit";
    const sacks  = isExit
      ? (totals.sacks ?? snap.sacks_exited ?? snap.count ?? 0)
      : (totals.sacks ?? snap.sacks_entered ?? snap.count ?? 0);

    if (countEl) countEl.textContent = sacks;
    if (modeLbl) {
      modeLbl.textContent = isExit
        ? "Counting Exits (Landing Zone)"
        : "Counting Entries (Doorway)";
    }
  }

  /* Live page — poll /api/session and render the workers / events
   * panels that the Live template wires up. */
  function pollSession(opts) {
    opts = opts || {};
    const workersEl = opts.workers;
    const eventsEl  = opts.events;
    const warnEl    = opts.warning;
    const modeTag   = opts.modeTag;
    const sourceEl  = opts.source;
    const stopBtn   = opts.stopBtn;

    function render(snap) {
      updateSidebar(snap);
      if (sourceEl) sourceEl.textContent = snap.source || "No session running";
      if (modeTag) {
        modeTag.textContent = snap.mode === "exit"
          ? "Counting sacks out" : "Counting sacks in";
      }
      if (stopBtn) stopBtn.disabled = !snap.running;

      if (warnEl) {
        if (snap.warning) {
          warnEl.innerHTML =
            '<div class="callout"><h6>Heads up</h6><p>' +
            snap.warning + "</p></div>";
        } else {
          warnEl.innerHTML = "";
        }
      }

      const workers = snap.workers || [];
      if (workersEl) {
        clear(workersEl);
        if (!workers.length) {
          workersEl.innerHTML =
            '<div class="screen-sub">No carriers confirmed yet.</div>';
        } else {
          const max = Math.max(1, ...workers.map((w) => w.delivered || 0));
          workers.forEach((w) => {
            const row = document.createElement("div");
            row.className = "worker-row";
            const pct = Math.round(((w.delivered || 0) / max) * 100);
            row.innerHTML =
              '<div class="worker-line"><span>Carrier ' + w.id + "</span>" +
              '<span class="mono">' + (w.delivered || 0) + " sacks · " +
              (w.boxes || 0) + " boxes</span></div>" +
              '<div class="meter"><span style="width:' + pct + '%"></span></div>';
            workersEl.appendChild(row);
          });
        }
      }

      const events = snap.events || [];
      if (eventsEl) {
        clear(eventsEl);
        if (!events.length) {
          eventsEl.innerHTML =
            '<div class="event"><div class="what screen-sub">Nothing yet.</div></div>';
        } else {
          events.forEach((e) => {
            const row = document.createElement("div");
            row.className = "event";
            const tone = e.tone || "neutral";
            row.innerHTML =
              '<div class="when mono">' + (e.time || "") + "</div>" +
              '<div class="what">' + (e.text || "") +
              (e.tag ? ' <span class="tag tag-' + tone + '">' + e.tag + "</span>" : "") +
              "</div>";
            eventsEl.appendChild(row);
          });
        }
      }

      if (snap.status === "error" && snap.error) {
        console.error("Session error:", snap.error);
      }
    }

    if (_pollTimer) clearInterval(_pollTimer);
    _pollTimer = setInterval(async () => {
      try {
        const res = await fetch("/api/session");
        if (res.ok) render(await res.json());
      } catch (err) {
        console.error("Failed to poll session status:", err);
      }
    }, 1000);
  }

  /* Legacy alias kept for templates that call startPolling directly. */
  function startPolling() {
    if (_pollTimer) return;
    _pollTimer = setInterval(async () => {
      try {
        const res = await fetch("/api/session");
        if (res.ok) updateSnapshot(await res.json());
      } catch (err) {
        console.error("Failed to poll session status:", err);
      }
    }, 1000);
  }

  /* Shared POST helper used by the Stop button on the live page. */
  async function post(url, body) { return postJSON(url, body); }

  /* ── Calibrate (stub — full canvas logic lives in calibrate.html) ─ */
  function calibrateScreen(config) {
    console.log("Calibrate initialized with config:", config);
  }

  /* ── History page: segmented filter + free-text search ───── */

  function historyScreen() {
    const table  = el("history-table");
    const search = el("history-search");
    if (!table) return;

    const radios = Array.from(document.querySelectorAll('input[name="hf"]'));
    const labels = Array.from(document.querySelectorAll("label.seg-opt"));
    const rows   = Array.from(table.querySelectorAll("tbody tr"));
    if (!radios.length || !rows.length) return;

    function applyFilters() {
      const q    = (search && search.value || "").trim().toLowerCase();
      const mode = (radios.find((r) => r.checked) || {}).value || "all";

      rows.forEach((tr) => {
        const rowMode = tr.dataset.mode || "";
        const rowSrc  = (tr.dataset.source || "").toLowerCase();

        const modeOk =
          mode === "all" ||
          (mode === "Sacks in"  && rowMode === "enter") ||
          (mode === "Sacks out" && rowMode === "exit");

        const qOk = !q || rowSrc.indexOf(q) !== -1;
        tr.style.display = (modeOk && qOk) ? "" : "none";
      });
    }

    if (search) search.addEventListener("input", applyFilters);
    radios.forEach((r) => r.addEventListener("change", applyFilters));

    /* The radios sit inside <label class="seg-opt"> with pointer-events:none
     * on the input itself, so a click on the label still toggles the input
     * but the change event can race the visual highlight on some browsers.
     * Re-applying on click is a cheap belt-and-braces measure. */
    labels.forEach((lab) =>
      lab.addEventListener("click", () =>
        setTimeout(applyFilters, 0)));

    applyFilters();
  }

  /* ── Diagnostics page: mode toggle + cutoff + Run check ──── */

  function diagnosticsScreen(initial) {
    const runBtn   = el("diag-run");
    const srcInput = el("diag-source");
    const everyInp = el("diag-every");
    const cutoffInp = el("diag-cutoff");
    const statusEl = el("diag-status");
    const resultEl = el("diag-result");

    /* Mode segmented control: one radio per option. */
    const modeRadios = Array.from(
      document.querySelectorAll('input[name="diag-mode"]'));
    const modeLabels = Array.from(
      document.querySelectorAll("#diag-mode .seg-opt"));

    function mode() {
      const r = modeRadios.find((x) => x.checked);
      return r ? r.value : "enter";
    }

    function setCutoffDisabled() {
      /* Cutoff is only meaningful for the sack detector — both modes
       * use it, so it's always enabled.  Kept as a hook in case the
       * model picker ever grows more options. */
      if (cutoffInp) cutoffInp.disabled = false;
    }

    function escapeHTML(s) {
      return String(s == null ? "" : s)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function renderResult(r) {
      if (!resultEl) return;
      if (!r || !r.status || r.status === "idle") {
        return;  /* leave the empty-state copy the template rendered */
      }
      if (r.status === "running") {
        resultEl.innerHTML =
          '<div class="empty">Checking <span class="mono">' +
          escapeHTML(r.source || "") + "</span>… " +
          (r.frames_checked || 0) + " frames sampled so far.</div>";
        return;
      }
      if (r.status === "error") {
        resultEl.innerHTML =
          '<div class="empty err">Check failed: ' +
          escapeHTML(r.error || "unknown error") + "</div>";
        return;
      }
      /* finished */
      const parts = [
        '<div class="diag-grid">',
        '<div class="diag-panel">',
        '<h6>Frames checked</h6>',
        '<div class="value accent" style="font-family:var(--font-heading);' +
        'font-weight:800;font-size:44px">' +
        (r.frames_checked || 0) + "</div>",
        '<div class="screen-sub">Sampled one in every ' +
        (r.every || 30) + " frames.</div>",
        "</div>",
        '<div class="diag-panel">',
        "<h6>Detections</h6>",
        '<div class="value" style="font-family:var(--font-heading);' +
        'font-weight:800;font-size:44px">' +
        (r.detections || 0) + "</div>",
        '<div class="screen-sub">Sacks the model could see at or above ' +
        "the cutoff. If this is much lower than the count on your report, " +
        "the model is missing sacks; if much higher, the counting rules " +
        "are filtering them out.</div>",
        "</div>",
        "</div>",
      ];
      if (r.model_used) {
        parts.push(
          '<div class="note-box" style="margin:16px 26px">Model used: ' +
          '<span class="mono">' + escapeHTML(r.model_used) + "</span></div>");
      }
      resultEl.innerHTML = parts.join("");
    }

    function setStatus(msg, isError) {
      if (!statusEl) return;
      statusEl.textContent = msg || "";
      statusEl.classList.toggle("err", !!isError);
    }

    let pollTimer = null;
    function stopPoll() {
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    }
    function startPoll() {
      stopPoll();
      pollTimer = setInterval(async () => {
        try {
          const res = await fetch("/api/diagnostics");
          if (!res.ok) return;
          const r = await res.json();
          renderResult(r);
          if (r.status === "running") {
            setStatus("Checking " + (r.source || "") + "…");
          } else {
            stopPoll();
            if (r.status === "error") {
              setStatus("Check failed: " + (r.error || ""), true);
            } else {
              setStatus("Check finished.");
            }
            if (runBtn) runBtn.disabled = false;
          }
        } catch (err) {
          console.error("Diagnostics poll failed:", err);
        }
      }, 1000);
    }

    async function runCheck() {
      if (!srcInput) return;
      const source = (srcInput.value || "").trim();
      if (!source) {
        setStatus("Choose a video to check first.", true);
        return;
      }
      const every  = parseInt(everyInp && everyInp.value, 10) || 30;
      const cutoff = parseFloat(cutoffInp && cutoffInp.value);
      const body = { source: source, every: every, mode: mode() };
      if (!isNaN(cutoff)) body.cutoff = cutoff;

      if (runBtn) runBtn.disabled = true;
      setStatus("Starting check…");
      try {
        await postJSON("/api/diagnostics/start", body);
        startPoll();
      } catch (err) {
        setStatus(err.message || "Could not start check.", true);
        if (runBtn) runBtn.disabled = false;
      }
    }

    if (runBtn)   runBtn.addEventListener("click", runCheck);
    modeRadios.forEach((r) => r.addEventListener("change", setCutoffDisabled));
    modeLabels.forEach((lab) =>
      lab.addEventListener("click", () => setTimeout(setCutoffDisabled, 0)));

    /* Re-render whatever the server already produced (initial page load). */
    if (initial && initial.status) renderResult(initial);
  }

  /* ── public surface ──────────────────────────────────────── */

  return {
    startPolling:      startPolling,
    pollSession:       pollSession,
    post:              post,
    updateSnapshot:    updateSnapshot,
    calibrateScreen:   calibrateScreen,
    historyScreen:     historyScreen,
    diagnosticsScreen: diagnosticsScreen,
  };
})();

document.addEventListener("DOMContentLoaded", () => {
  /* Live sidebar status updates on every page, cheaply.  Individual
   * screens wire their own behaviour by calling the matching method
   * from a per-template <script>. */
  Console.startPolling();
});
