/* web/static/console.js — Console UI interactions, live polling, and
   per-screen wiring for the Sack Counter console.

   Every screen extends base.html, which calls Console.bindSidebar() on
   load — that owns the shared status footer in the sidebar.  Each
   screen that needs its own behaviour calls its matching Console.<xScreen>()
   function from a {% block scripts %} tag.

   Conventions
   -----------
   - DOM ids referenced here are declared in the matching template; if
     you rename one, update both ends.
   - All HTTP goes through Console.post / Console.get so a single fetch
     wrapper can attach the JSON headers and surface errors.
   - Polling is one setInterval at 1s, shared by the sidebar and the
     live screen, so we never have two intervals racing. */

const Console = (function () {
  let pollInterval = null;
  let liveBindings = null;        // set by pollSession() on the live screen
  const subscribers = [];          // sidebar + live both react to a snapshot

  /* ── HTTP helpers ────────────────────────────────────────────── */

  async function get(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }

  async function post(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`;
      try {
        const data = await res.json();
        if (data && data.detail) msg = String(data.detail);
      } catch (_) { /* not JSON */ }
      throw new Error(msg);
    }
    return res.json();
  }

  /* ── Snapshot fan-out ────────────────────────────────────────── */

  function subscribe(fn) {
    subscribers.push(fn);
  }

  function emit(snap) {
    for (const fn of subscribers) {
      try { fn(snap); } catch (err) { console.error("subscriber failed:", err); }
    }
  }

  function startPolling() {
    if (pollInterval) return;
    pollInterval = setInterval(async () => {
      try {
        const snap = await get("/api/session");
        emit(snap);
      } catch (err) {
        // Network blips happen — don't spam the console, just keep polling.
      }
    }, 1000);
  }

  /* ── Sidebar status footer (every page) ──────────────────────── */

  const STATUS_LABELS = {
    running: "Counting", starting: "Starting",
    finished: "Finished", error: "Stopped", idle: "Idle",
  };

  function bindSidebar() {
    subscribe(updateSidebar);
    startPolling();
  }

  function updateSidebar(snap) {
    const dot = document.getElementById("status-dot");
    const label = document.getElementById("status-label");
    const meta = document.getElementById("status-meta");
    if (!snap) return;

    if (dot)   dot.classList.toggle("on", !!snap.running);
    if (label) label.textContent = STATUS_LABELS[snap.status] || "Idle";
    if (meta) {
      const frame = (snap.frame || 0).toLocaleString();
      meta.innerHTML =
        `Frame ${frame}<br>${snap.fps || 0} fps · ${snap.elapsed || "0:00"}`;
    }
  }

  /* ── Live screen ─────────────────────────────────────────────── */

  function pollSession(opts) {
    liveBindings = opts || {};
    subscribe(renderLive);
    startPolling();
  }

  function renderLive(snap) {
    if (!snap || !liveBindings) return;
    const b = liveBindings;

    setText(b.source, snap.source || "No session running");

    if (b.modeTag) {
      const isExit = snap.mode === "exit";
      b.modeTag.textContent = isExit ? "Counting sacks out" : "Counting sacks in";
    }

    if (b.stopBtn) b.stopBtn.disabled = !snap.running;

    // Count tiles.
    const t = snap.totals || {};
    setText(document.getElementById("live-count-sacks"),  t.sacks   ?? 0);
    setText(document.getElementById("live-count-boxes"),  t.boxes   ?? 0);
    setText(document.getElementById("live-count-workers"),t.workers ?? 0);
    setText(document.getElementById("live-count-review"), t.review  ?? 0);
    const sacksLabel = document.getElementById("live-count-sacks-label");
    if (sacksLabel && t.label) sacksLabel.textContent = t.label;

    // Warning slot (sacks-out discrepancy warnings, etc.)
    if (b.warning) {
      if (snap.warning) {
        b.warning.innerHTML =
          `<div class="callout"><h6>Please review</h6><p>${escapeHtml(snap.warning)}</p></div>`;
      } else {
        b.warning.innerHTML = "";
      }
    }

    renderWorkers(b.workers, snap.workers || []);
    renderEvents(b.events, snap.events || []);
  }

  function renderWorkers(container, workers) {
    if (!container) return;
    if (!workers.length) {
      container.innerHTML = `<div class="screen-sub">No carriers confirmed yet.</div>`;
      return;
    }
    const max = Math.max(1, ...workers.map((w) => w.delivered));
    container.innerHTML = workers.map((w) => `
      <div class="worker-row">
        <div class="worker-line">
          <span class="mono">Worker ${escapeHtml(String(w.id))}</span>
          <span class="mono">${w.delivered} sack${w.delivered === 1 ? "" : "s"}</span>
        </div>
        <div class="meter"><span style="width:${(w.delivered / max) * 100}%"></span></div>
        <div class="screen-sub mono" style="font-size:11px;margin-top:2px">
          ${w.boxes || 0} box${(w.boxes || 0) === 1 ? "" : "es"} · carrying ${w.carrying || 0}
        </div>
      </div>`).join("");
  }

  function renderEvents(container, events) {
    if (!container) return;
    if (!events.length) {
      container.innerHTML =
        `<div class="event"><div class="what screen-sub">Nothing yet.</div></div>`;
      return;
    }
    container.innerHTML = events.map((e) => `
      <div class="event">
        <span class="when mono">${escapeHtml(e.time || "")}</span>
        <span class="what">${escapeHtml(e.text || "")}
          ${e.tag && e.tone ? `<span class="tag tag-${e.tone}" style="margin-left:6px">${escapeHtml(e.tag)}</span>` : ""}
        </span>
      </div>`).join("");
  }

  /* ── File picker ─────────────────────────────────────────────── */

  /* Browsers hide the real path of files chosen via <input type=file>,
     so the only way to get the video to the backend is to upload the
     bytes and persist them next to the app. attachFilePicker wires the
     Browse button → hidden file input → /api/upload → fill the source
     input with the returned local filename. Works on any screen whose
     source field needs a video path. */
  function attachFilePicker(opts) {
    const input   = document.getElementById(opts.inputId);
    const browse  = document.getElementById(opts.browseId);
    const fileEl  = document.getElementById(opts.fileId);
    const errBox  = opts.errId
      ? document.getElementById(opts.errId)
      : null;
    if (!input || !browse || !fileEl) return;

    browse.addEventListener("click", () => fileEl.click());

    fileEl.addEventListener("change", async () => {
      if (!fileEl.files || !fileEl.files.length) return;
      const f = fileEl.files[0];
      const original = browse.textContent;
      browse.disabled = true;
      browse.textContent = "Uploading…";
      try {
        const buf = await f.arrayBuffer();
        const res = await fetch(
          `/api/upload?filename=${encodeURIComponent(f.name)}`,
          {
            method: "POST",
            body: buf,
            headers: { "Content-Type": "application/octet-stream" },
          },
        );
        if (!res.ok) {
          let msg = `${res.status} ${res.statusText}`;
          try {
            const d = await res.json();
            if (d && d.detail) msg = String(d.detail);
          } catch (_) { /* not JSON */ }
          throw new Error(msg);
        }
        const data = await res.json();
        input.value = data.path;
        // Let any listeners on the source input react (e.g. refreshSourceInfo).
        input.dispatchEvent(new Event("change"));
      } catch (err) {
        if (errBox) errBox.textContent = `Upload failed: ${err.message}`;
        else alert(`Upload failed: ${err.message}`);
      } finally {
        browse.disabled = false;
        browse.textContent = original;
        // Allow re-picking the same file if the user cancels and tries again.
        fileEl.value = "";
      }
    });
  }

  /* ── Setup screen ────────────────────────────────────────────── */

  function setupScreen() {
    const sourceInput = document.getElementById("source");
    const sourceInfo  = document.getElementById("source-info");
    const webcamBtn   = document.getElementById("use-webcam");
    const startBtn    = document.getElementById("start-btn");
    const errBox      = document.getElementById("setup-error");
    const modeCards   = document.querySelectorAll(".mode-card");
    const sliders     = ["conf_sack", "conf_person", "conf_box"];

    // Wire the Browse button → upload → fill #source
    attachFilePicker({
      inputId:  "source",
      browseId: "browse-btn",
      fileId:   "source-file",
      errId:    "setup-error",
    });

    let mode = document.querySelector(".mode-card.on")?.dataset.mode || "enter";

    modeCards.forEach((card) => {
      card.addEventListener("click", () => {
        modeCards.forEach((c) => c.classList.remove("on"));
        card.classList.add("on");
        mode = card.dataset.mode;
      });
    });

    sliders.forEach((key) => {
      const el = document.getElementById(key);
      const val = document.getElementById(`${key}-val`);
      if (el && val) {
        el.addEventListener("input", () => { val.textContent = parseFloat(el.value).toFixed(2); });
      }
    });

    if (webcamBtn) {
      webcamBtn.addEventListener("click", () => {
        if (sourceInput) {
          sourceInput.value = "webcam";
          refreshSourceInfo("webcam");
        }
      });
    }

    if (sourceInput) {
      sourceInput.addEventListener("change", () => refreshSourceInfo(sourceInput.value));
    }

    async function refreshSourceInfo(source) {
      if (!source || !sourceInfo) return;
      if (source === "webcam") {
        sourceInfo.textContent = "Using the webcam";
        return;
      }
      try {
        const info = await get(`/api/source-info?source=${encodeURIComponent(source)}`);
        sourceInfo.textContent =
          `${info.width}×${info.height} · ${info.frames.toLocaleString()} frames · ${info.fps} fps`;
      } catch (err) {
        sourceInfo.textContent = "Cannot read that file — check the path.";
      }
    }

    if (startBtn) {
      startBtn.addEventListener("click", async () => {
        if (errBox) errBox.textContent = "";
        startBtn.disabled = true;
        const body = {
          source: sourceInput?.value?.trim() || "",
          mode,
          conf_sack:   parseFloat(document.getElementById("conf_sack")?.value),
          conf_person: parseFloat(document.getElementById("conf_person")?.value),
          conf_box:    parseFloat(document.getElementById("conf_box")?.value),
          save_output:       document.getElementById("save_output")?.checked || false,
          use_door_crossing: document.getElementById("use_door_crossing")?.checked || false,
        };
        try {
          await post("/api/session/start", body);
          window.location = "/";
        } catch (err) {
          startBtn.disabled = false;
          if (errBox) errBox.textContent = err.message;
        }
      });
    }
  }

  /* ── Calibrate screen ────────────────────────────────────────── */

  function calibrateScreen(config) {
    const ctx = config || {};
    const stage   = document.getElementById("cal-stage");
    const img     = document.getElementById("cal-frame");
    const svg     = document.getElementById("cal-svg");
    const poly    = document.getElementById("cal-poly");
    const marks   = document.getElementById("cal-marks");
    const room    = document.getElementById("cal-room");
    const polyL   = document.getElementById("cal-poly-landing");
    const marksL  = document.getElementById("cal-marks-landing");
    const hint    = document.getElementById("cal-hint");
    const errBox  = document.getElementById("cal-error");
    const count   = document.getElementById("cal-count");

    const loadBtn = document.getElementById("load-frame");
    const undoBtn = document.getElementById("undo-btn");
    const clearBtn= document.getElementById("clear-btn");
    const finishBtn = document.getElementById("finish-btn");
    const saveBtn = document.getElementById("save-btn");

    const srcInput= document.getElementById("cal-source");
    const rdoDoor = document.getElementById("cal-target-door");
    const rdoLand = document.getElementById("cal-target-landing");
    const doorSteps = document.getElementById("door-steps");
    const landSteps = document.getElementById("landing-steps");

    // Wire the Browse button → upload → fill #cal-source
    attachFilePicker({
      inputId:  "cal-source",
      browseId: "cal-browse-btn",
      fileId:   "cal-source-file",
      errId:    "cal-error",
    });

    // State — door points, room point, landing points, current target.
    let doorPts   = Array.isArray(ctx.points) ? ctx.points.slice() : [];
    let roomPt    = ctx.roomPoint || null;
    let landPts   = Array.isArray(ctx.landingPoints) ? ctx.landingPoints.slice() : [];
    let target    = ctx.target === "landing" ? "landing" : "door";
    let landingClosed = landPts.length >= 3;

    function syncTarget() {
      const isLanding = target === "landing";
      if (doorSteps) doorSteps.style.display = isLanding ? "none" : "";
      if (landSteps) landSteps.style.display = isLanding ? "" : "none";
      if (rdoDoor) rdoDoor.checked = !isLanding;
      if (rdoLand) rdoLand.checked = isLanding;
      // Dim the door polygon when the landing zone is the active target.
      if (poly)    poly.setAttribute("opacity",    isLanding ? "0.35" : "1");
      if (marks)   marks.setAttribute("opacity",   isLanding ? "0.35" : "1");
      if (polyL)   polyL.setAttribute("opacity",   isLanding ? "1" : "0.35");
      if (marksL)  marksL.setAttribute("opacity",  isLanding ? "1" : "0.35");
      if (finishBtn) finishBtn.style.display =
        (isLanding && !landingClosed && landPts.length >= 3) ? "" : "none";
      redraw();
    }

    function activePts() { return target === "landing" ? landPts : doorPts; }

    function currentHint() {
      if (target === "landing") {
        if (landingClosed) return "Landing zone ready. Press Save and continue.";
        const n = landPts.length;
        if (n < 3) return `Click ${3 - n} more point${3 - n === 1 ? "" : "s"} to outline the landing area.`;
        return `Click Finish shape, or keep adding points (up to 8).`;
      }
      const n = doorPts.length;
      if (n < 4) return `Click ${4 - n} more corner${4 - n === 1 ? "" : "s"} of the doorway.`;
      if (!roomPt) return "Click one spot inside the room.";
      return "Press Save and continue.";
    }

    function redraw() {
      if (hint) hint.textContent = currentHint();
      // Door polyline.
      if (poly) {
        const all = doorPts.length >= 4 ? doorPts.concat([doorPts[0]]) : doorPts;
        poly.setAttribute("points", all.map((p) => `${p[0]},${p[1]}`).join(" "));
      }
      if (marks) {
        marks.innerHTML = doorPts.map((p, i) =>
          svgMark(p, "#ec3013", String(i + 1))
        ).join("");
      }
      if (room) {
        room.innerHTML = roomPt
          ? `<circle cx="${roomPt[0]}" cy="${roomPt[1]}" r="14" fill="#ec3013" />
             <circle cx="${roomPt[0]}" cy="${roomPt[1]}" r="6" fill="#f3f2f2" />`
          : "";
      }
      // Landing polyline.
      if (polyL) {
        const all = landingClosed && landPts.length >= 3
          ? landPts.concat([landPts[0]]) : landPts;
        polyL.setAttribute("points", all.map((p) => `${p[0]},${p[1]}`).join(" "));
      }
      if (marksL) {
        marksL.innerHTML = landPts.map((p, i) =>
          svgMark(p, "#e15b47", String(i + 1))
        ).join("");
      }
      if (count) {
        const n = target === "landing" ? landPts.length : doorPts.length;
        const saved = target === "landing"
          ? (landPts.length >= 3 ? "Saved to config.yaml" : "Not saved yet")
          : (doorPts.length === 4 && roomPt ? "Saved to config.yaml" : "Not saved yet");
        count.innerHTML = `${n} point(s) marked<br>${saved}`;
      }
    }

    function svgMark(p, color, label) {
      return `<circle cx="${p[0]}" cy="${p[1]}" r="12" fill="${color}" />
              <text x="${p[0]}" y="${p[1] + 4}" text-anchor="middle"
                    fill="#f3f2f2" font-size="13" font-family="monospace">${label}</text>`;
    }

    function svgCoords(evt) {
      const rect = svg.getBoundingClientRect();
      const x = (evt.clientX - rect.left) / rect.width  * 1600;
      const y = (evt.clientY - rect.top)  / rect.height * 900;
      return [Math.round(x), Math.round(y)];
    }

    function onStageClick(evt) {
      if (errBox) errBox.textContent = "";
      const [x, y] = svgCoords(evt);

      if (target === "landing") {
        if (landingClosed) return;
        if (landPts.length >= 8) {
          if (errBox) errBox.textContent = "Eight points is the most the landing zone can have.";
          return;
        }
        landPts.push([x, y]);
        if (landPts.length >= 3) {
          if (finishBtn) finishBtn.style.display = "";
        }
        redraw();
        return;
      }

      // Door target.
      if (doorPts.length < 4) {
        doorPts.push([x, y]);
        redraw();
        return;
      }
      if (!roomPt) {
        roomPt = [x, y];
        redraw();
      }
    }

    function onUndo() {
      if (errBox) errBox.textContent = "";
      if (target === "landing") {
        if (landingClosed) {
          landingClosed = false;
        } else if (landPts.length) {
          landPts.pop();
        }
      } else {
        if (roomPt) roomPt = null;
        else if (doorPts.length) doorPts.pop();
      }
      redraw();
    }

    function onClear() {
      if (target === "landing") {
        landPts = [];
        landingClosed = false;
      } else {
        doorPts = [];
        roomPt  = null;
      }
      redraw();
    }

    function onFinish() {
      if (target !== "landing") return;
      if (landPts.length < 3) return;
      landingClosed = true;
      if (finishBtn) finishBtn.style.display = "none";
      redraw();
    }

    async function onLoadFrame() {
      if (errBox) errBox.textContent = "";
      const source = srcInput?.value?.trim();
      if (!source) {
        if (errBox) errBox.textContent = "Pick a video file or the webcam first.";
        return;
      }
      const url = `/api/frame.jpg?source=${encodeURIComponent(source)}`;
      img.src = url;
      img.onload = () => redraw();
      img.onerror = () => {
        if (errBox) errBox.textContent = "Could not read a frame from that source.";
      };
    }

    async function onSave() {
      if (errBox) errBox.textContent = "";
      try {
        if (target === "landing") {
          if (landPts.length < 3) throw new Error("Mark at least three points around the landing area.");
          await post("/api/calibration", {
            target: "landing", landing_points: landPts,
          });
        } else {
          if (doorPts.length !== 4) throw new Error("Mark exactly four corners of the doorway.");
          if (!roomPt) throw new Error("Click one spot inside the room as well.");
          await post("/api/calibration", {
            target: "door", points: doorPts, room_point: roomPt,
          });
        }
        window.location = "/setup";
      } catch (err) {
        if (errBox) errBox.textContent = err.message;
      }
    }

    // Wire events.
    if (rdoDoor) rdoDoor.addEventListener("change", () => { target = "door"; syncTarget(); });
    if (rdoLand) rdoLand.addEventListener("change", () => { target = "landing"; syncTarget(); });
    if (stage)   stage.addEventListener("click", onStageClick);
    if (undoBtn) undoBtn.addEventListener("click", onUndo);
    if (clearBtn)clearBtn.addEventListener("click", onClear);
    if (finishBtn) finishBtn.addEventListener("click", onFinish);
    if (saveBtn) saveBtn.addEventListener("click", onSave);
    if (loadBtn) loadBtn.addEventListener("click", onLoadFrame);

    syncTarget();
  }

  /* ── History screen ──────────────────────────────────────────── */

  function historyScreen() {
    const table  = document.getElementById("history-table");
    const search = document.getElementById("history-search");
    // The radios live inside <label class="seg-opt"> wrappers.  We
    // bind to BOTH the radios (for the change event) and the labels
    // (for click), because some browsers don't reliably fire
    // `change` on a radio that has `pointer-events: none` and is
    // activated via a label click.  Listening on the label too means
    // the filter updates no matter which element actually received
    // the click.
    const radios = Array.from(document.querySelectorAll('input[name="hf"]'));
    const labels = Array.from(document.querySelectorAll("label.seg-opt"));
    if (!table || !radios.length) return;
    const rows   = Array.from(table.querySelectorAll("tbody tr"));

    function applyFilters() {
      const q = (search?.value || "").trim().toLowerCase();
      const mode = radios.find((r) => r.checked)?.value || "all";
      rows.forEach((tr) => {
        const rowMode = tr.dataset.mode || "";
        const rowSrc  = tr.dataset.source || "";
        const modeOk  = mode === "all" ||
          (mode === "Sacks in"  && rowMode === "enter") ||
          (mode === "Sacks out" && rowMode === "exit");
        const qOk = !q || rowSrc.includes(q);
        tr.style.display = (modeOk && qOk) ? "" : "none";
      });
    }

    if (search) search.addEventListener("input", applyFilters);
    // `change` is the canonical event for radios — fires when the
    // checked state actually flips.
    radios.forEach((r) => r.addEventListener("change", applyFilters));
    // `click` on the label is a fallback for browsers that don't
    // bubble a `change` event when the radio is activated via a
    // label click with `pointer-events: none` on the input.  The
    // microtask delay lets the browser flip `checked` before we
    // re-read it.
    labels.forEach((lab) =>
      lab.addEventListener("click", () =>
        Promise.resolve().then(applyFilters)));
  }

  /* ── Diagnostics screen ──────────────────────────────────────── */

  function diagnosticsScreen(initial) {
    const runBtn   = document.getElementById("diag-run");
    const srcInput = document.getElementById("diag-source");
    const everyInput = document.getElementById("diag-every");
    const statusEl = document.getElementById("diag-status");
    const resultEl = document.getElementById("diag-result");

    // Wire the Browse button → upload → fill #diag-source
    attachFilePicker({
      inputId:  "diag-source",
      browseId: "diag-browse-btn",
      fileId:   "diag-source-file",
    });

    let pollTimer = null;

    function render(r) {
      if (!r || !r.status || r.status === "idle") {
        if (resultEl) {
          resultEl.innerHTML =
            `<div class="empty">Nothing checked yet. Point this at the video
             that produced the counts you doubt and press
             <strong>Run check</strong> — it samples every 30th frame, so a
             long video takes seconds rather than minutes.</div>`;
        }
        return;
      }
      if (r.status === "error") {
        if (resultEl) resultEl.innerHTML =
          `<div class="err">${escapeHtml(r.error || "Diagnostics failed.")}</div>`;
        return;
      }

      // Running or finished — both render the same layout; the chart
      // just grows as samples arrive.
      const samples = Array.isArray(r.samples) ? r.samples : [];
      const frames  = r.frames_checked || 0;
      const total   = r.detections || 0;
      const every   = r.every || 30;
      const model   = r.model_used || "";
      const mode    = r.mode || "enter";
      const cutoff  = r.cutoff;
      const running = r.status === "running";

      // Top-row KPI tiles.
      const kpi = [
        `<div class="diag-kpi">
           <div class="label">Frames checked</div>
           <div class="value mono">${frames.toLocaleString()}</div>
           <div class="conf-note">Sampled every ${every} frames
             ${r.source ? `from <span class="mono">${escapeHtml(r.source)}</span>` : ""}.</div>
         </div>`,
        `<div class="diag-kpi">
           <div class="label">Detections</div>
           <div class="value mono">${total.toLocaleString()}</div>
           <div class="conf-note">Above cutoff ${cutoff != null ? Number(cutoff).toFixed(2) : "—"}.</div>
         </div>`,
        `<div class="diag-kpi">
           <div class="label">Mode</div>
           <div class="value mono">${escapeHtml(mode)}</div>
           <div class="conf-note">${mode === "exit" ? "Floor-sack model" : "Sack-carrying model"}.</div>
         </div>`,
        `<div class="diag-kpi diag-kpi-wide">
           <div class="label">Model</div>
           <div class="value mono small">${escapeHtml(model || "—")}</div>
           <div class="conf-note">${model ? "Loaded for this check." : "No model file — counts will be zero."}</div>
         </div>`,
      ].join("");

      // Line chart of detections per sample over time.
      const chart = renderDiagChart(samples, { running });

      const parts = [
        `<div class="diag-kpis">${kpi}</div>`,
        chart,
      ];
      if (resultEl) resultEl.innerHTML = parts.join("");
    }

    /* ── Diagnostics line chart ──────────────────────────────── */
    /* Renders an inline SVG of per-sample detection counts over
       time.  Mirrors the trend chart on the report page but uses a
       polyline + area fill instead of bars, because the user wants
       to *see the shape* of detection activity across the video. */

    function renderDiagChart(samples, opts) {
      opts = opts || {};
      const running = !!opts.running;
      const W = 880, H = 280;
      const M = { top: 18, right: 28, bottom: 38, left: 48 };
      const iw = W - M.left - M.right;
      const ih = H - M.top - M.bottom;

      // Empty state — no samples yet.
      if (!samples || samples.length === 0) {
        const note = running
          ? "Sampling frames — chart will appear as the check runs."
          : "No samples recorded.";
        return `<div class="diag-chart empty">
          <div class="diag-chart-empty">${escapeHtml(note)}</div>
        </div>`;
      }

      // Pack the values we need for scaling.
      const counts = samples.map(s => Number(s.count) || 0);
      const n = counts.length;
      const realMax = counts.length ? Math.max(...counts) : 0;
      const peakIdx = realMax > 0 ? counts.indexOf(realMax) : -1;
      const mean = counts.reduce((a, b) => a + b, 0) / n;
      // Y axis always reaches at least 4 so a flat-zero line still
      // shows a baseline + a couple of gridlines.
      const niceMax = Math.max(4, niceMaxNum(realMax));

      // X scale: index 0..(n-1) → 0..iw.  We use index rather than
      // frame number so uneven sample spacing (e.g. truncated last
      // chunk) doesn't squish the right side of the chart.
      const xAt = (i) => (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
      const yAt = (v) => ih - (v / niceMax) * ih;

      // Build polyline points.
      const pts = counts.map((v, i) => {
        const x = M.left + xAt(i);
        const y = M.top + yAt(v);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      const line = pts.join(" ");

      // Area fill (line + down to baseline + back).
      const baselineY = M.top + ih;
      const areaPts = [
        `${M.left + xAt(0)},${baselineY}`,
        ...pts,
        `${M.left + xAt(n - 1)},${baselineY}`,
      ].join(" ");

      // Gridlines: 4 horizontal, value labels on the Y axis.
      const gridLines = [];
      const yTicks = 4;
      for (let g = 0; g <= yTicks; g++) {
        const v = (niceMax / yTicks) * g;
        const y = M.top + yAt(v);
        gridLines.push(
          `<line class="diag-grid-line" x1="${M.left}" y1="${y.toFixed(1)}"
                 x2="${M.left + iw}" y2="${y.toFixed(1)}" />`
        );
        gridLines.push(
          `<text class="diag-axis-label" x="${M.left - 8}" y="${(y + 3).toFixed(1)}"
                 text-anchor="end">${Math.round(v)}</text>`
        );
      }

      // X-axis ticks: ~6 evenly spaced, labelled with sample #.
      const xTicks = Math.min(6, n);
      const xTickEls = [];
      for (let g = 0; g < xTicks; g++) {
        const i = xTicks === 1 ? 0 : Math.round((g / (xTicks - 1)) * (n - 1));
        const x = M.left + xAt(i);
        xTickEls.push(
          `<line class="diag-tick" x1="${x.toFixed(1)}" y1="${baselineY}"
                 x2="${x.toFixed(1)}" y2="${baselineY + 4}" />`
        );
        xTickEls.push(
          `<text class="diag-axis-label" x="${x.toFixed(1)}"
                 y="${baselineY + 18}" text-anchor="middle">${i + 1}</text>`
        );
      }

      // Mean line.
      const meanY = M.top + yAt(mean);
      const meanEl =
        `<line class="diag-mean-line" x1="${M.left}" y1="${meanY.toFixed(1)}"
               x2="${M.left + iw}" y2="${meanY.toFixed(1)}" />
         <text class="diag-mean-label" x="${M.left + iw - 4}"
               y="${(meanY - 6).toFixed(1)}" text-anchor="end">
           avg ${mean.toFixed(1)}</text>`;

      // Peak marker (only when there's a real peak to mark).
      const peakEl = (realMax > 0 && peakIdx >= 0)
        ? `<circle class="diag-peak" cx="${(M.left + xAt(peakIdx)).toFixed(1)}"
                   cy="${(M.top + yAt(realMax)).toFixed(1)}" r="4" />
           <text class="diag-peak-label"
                 x="${(M.left + xAt(peakIdx)).toFixed(1)}"
                 y="${(M.top + yAt(realMax) - 10).toFixed(1)}"
                 text-anchor="middle">${realMax}</text>`
        : "";

      // Sample dots.
      const dots = counts.map((v, i) => {
        const x = M.left + xAt(i);
        const y = M.top + yAt(v);
        return `<circle class="diag-point" cx="${x.toFixed(1)}"
                        cy="${y.toFixed(1)}" r="2.5" />`;
      }).join("");

      // Axis titles.
      const axisTitles =
        `<text class="diag-axis-title" x="${M.left + iw / 2}" y="${H - 4}"
               text-anchor="middle">sample #</text>
         <text class="diag-axis-title" transform="translate(13,${M.top + ih / 2}) rotate(-90)"
               text-anchor="middle">detections</text>`;

      // Legend.
      const legend =
        `<div class="diag-legend">
           <span class="diag-leg"><span class="sw diag-leg-line"></span>per-sample</span>
           <span class="diag-leg"><span class="sw diag-leg-mean"></span>average</span>
           <span class="diag-leg"><span class="sw diag-leg-peak"></span>peak</span>
         </div>`;

      const headTitle = running
        ? `Detections per sample <span class="diag-live">· live</span>`
        : `Detections per sample`;

      return `<div class="diag-chart">
        <div class="diag-chart-head">
          <div class="diag-chart-title">${headTitle}</div>
          <div class="diag-chart-meta">${n} sample${n === 1 ? "" : "s"}
            · peak ${realMax} · avg ${mean.toFixed(1)}</div>
        </div>
        <svg class="diag-line-svg" viewBox="0 0 ${W} ${H}"
             preserveAspectRatio="xMidYMid meet" role="img"
             aria-label="Detections per sample line chart">
          ${gridLines.join("")}
          ${xTickEls.join("")}
          <polygon class="diag-area" points="${areaPts}" />
          <line class="diag-baseline" x1="${M.left}" y1="${baselineY}"
                x2="${M.left + iw}" y2="${baselineY}" />
          ${meanEl}
          <polyline class="diag-line" points="${line}"
                    fill="none" stroke-linejoin="round" stroke-linecap="round" />
          ${dots}
          ${peakEl}
          ${axisTitles}
        </svg>
        ${legend}
      </div>`;
    }

    function niceMaxNum(v) {
      if (v <= 0) return 4;
      const pow = Math.pow(10, Math.floor(Math.log10(v)));
      const n = v / pow;
      let nice;
      if      (n <= 1) nice = 1;
      else if (n <= 2) nice = 2;
      else if (n <= 2.5) nice = 2.5;
      else if (n <= 5) nice = 5;
      else             nice = 10;
      const out = nice * pow;
      // Make sure the max value sits inside the chart, not on the top edge.
      return out < v ? out * 2 : out;
    }

    function setStatus(text) {
      if (statusEl) statusEl.textContent = text || "";
    }

    async function refresh() {
      try {
        const r = await get("/api/diagnostics");
        setStatus(r.status === "running" ? "Running…" : "");
        render(r);
        if (r.status === "running") {
          if (!pollTimer) pollTimer = setInterval(refresh, 1500);
        } else {
          if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        }
      } catch (err) {
        setStatus(`Error: ${err.message}`);
      }
    }

    if (runBtn) {
      runBtn.addEventListener("click", async () => {
        const source = srcInput?.value?.trim();
        if (!source) { setStatus("Pick a video file first."); return; }
        const every = parseInt(everyInput?.value || "30", 10) || 30;
        setStatus("Starting…");
        try {
          await post("/api/diagnostics/start", { source, every });
          refresh();
        } catch (err) {
          setStatus(err.message);
        }
      });
    }

    render(initial);
    // If a check is mid-flight when the page loads, keep polling.
    if (initial && initial.status === "running") refresh();
  }

  /* ── Utils ───────────────────────────────────────────────────── */

  function setText(el, value) {
    if (el) el.textContent = value;
  }

  function escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  /* ── Public surface ──────────────────────────────────────────── */
  return {
    bindSidebar,
    pollSession,
    setupScreen,
    calibrateScreen,
    historyScreen,
    diagnosticsScreen,
    get,
    post,
  };
})();

document.addEventListener("DOMContentLoaded", () => {
  Console.bindSidebar();
});
