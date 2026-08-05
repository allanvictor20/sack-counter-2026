/* console.js — behaviour for the Sack Counter console.
 *
 * The screens are server-rendered; this file handles the parts that
 * cannot be: polling a running session, marking the doorway by clicking,
 * and running a detection check. No framework, no build step — the
 * markup it produces uses the same design-system classes as the
 * templates.
 */
const Console = (() => {

  const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  async function post(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    let data = {};
    try { data = await res.json(); } catch { /* empty body */ }
    if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
    return data;
  }

  async function get(url) {
    const res = await fetch(url);
    let data = {};
    try { data = await res.json(); } catch { /* empty body */ }
    if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
    return data;
  }

  /* ── Shared: the sidebar status block ───────────────────── */
  function paintStatus(s) {
    const dot = document.getElementById('status-dot');
    const label = document.getElementById('status-label');
    const meta = document.getElementById('status-meta');
    if (dot) dot.classList.toggle('on', !!s.running);
    if (label) {
      label.textContent = { running: 'Counting', starting: 'Starting',
        finished: 'Finished', error: 'Stopped', idle: 'Idle' }[s.status] || 'Idle';
    }
    if (meta) {
      meta.innerHTML = `Frame ${Number(s.frame).toLocaleString()}<br>`
        + `${s.fps} fps · ${s.elapsed}`;
    }
  }

  /* ── Live ───────────────────────────────────────────────── */
  function pollSession(el) {
    let stopped = false;

    function paintWorkers(workers) {
      if (!el.workers) return;
      if (!workers.length) {
        el.workers.innerHTML =
          '<div class="screen-sub">No carriers confirmed yet.</div>';
        return;
      }
      const top = Math.max(...workers.map((w) => w.delivered), 1);
      el.workers.innerHTML = workers.map((w) => `
        <div class="worker-row">
          <div class="worker-line">
            <span class="mono">Worker ${esc(w.id)}${w.carrying ? ' · carrying ' + esc(w.carrying) : ''}</span>
            <span class="mono" style="color:var(--color-neutral-700)">${esc(w.delivered)} sacks · ${esc(w.boxes)} ${w.boxes === 1 ? 'box' : 'boxes'}</span>
          </div>
          <div class="meter"><span style="width:${Math.round(w.delivered / top * 100)}%"></span></div>
        </div>`).join('');
    }

    function paintEvents(events) {
      if (!el.events) return;
      if (!events.length) {
        el.events.innerHTML =
          '<div class="event"><div class="what screen-sub">Nothing yet.</div></div>';
        return;
      }
      el.events.innerHTML = events.map((e) => `
        <div class="event">
          <span class="when mono">${esc(e.time)}</span>
          <div style="min-width:0">
            <div class="what">${esc(e.text)}</div>
            <div style="margin-top:5px">
              <span class="tag tag-${esc(e.tone)}" style="font-size:10px;padding:2px 7px">${esc(e.tag)}</span>
            </div>
          </div>
        </div>`).join('');
    }

    async function tick() {
      if (stopped) return;
      try {
        const s = await get('/api/session');
        paintStatus(s);
        paintWorkers(s.workers || []);
        paintEvents(s.events || []);
        if (el.modeTag) {
          el.modeTag.textContent = s.mode === 'exit'
            ? 'Counting sacks out' : 'Counting sacks in';
        }
        if (el.source && s.source) {
          el.source.textContent = `${s.source} · ${s.fps} fps · frame ${Number(s.frame).toLocaleString()}`;
        }
        if (el.stopBtn) el.stopBtn.disabled = !s.running;
        if (el.warning) {
          const msg = s.error || s.warning;
          if (msg) {
            el.warning.innerHTML =
              `<div class="callout"><h6>Check this</h6><p>${esc(msg)}</p></div>`;
          } else if (el.warning.dataset.static !== '1') {
            el.warning.innerHTML = '';
          }
        }
      } catch (err) {
        /* The server may be restarting; keep polling. */
      }
      setTimeout(tick, 1000);
    }

    // A static warning rendered server-side (e.g. "not calibrated") should
    // survive the first poll rather than being wiped by it.
    if (el.warning && el.warning.children.length) el.warning.dataset.static = '1';
    tick();
    window.addEventListener('beforeunload', () => { stopped = true; });
  }

  /* ── Setup ──────────────────────────────────────────────── */
  function setupScreen() {
    let mode = 'enter';

    document.querySelectorAll('.mode-card').forEach((card) => {
      card.addEventListener('click', () => {
        document.querySelectorAll('.mode-card').forEach((c) => c.classList.remove('on'));
        card.classList.add('on');
        mode = card.dataset.mode;
      });
    });

    ['conf_sack', 'conf_person', 'conf_box'].forEach((key) => {
      const input = document.getElementById(key);
      const out = document.getElementById(`${key}-val`);
      if (!input) return;
      input.addEventListener('input', () => {
        out.textContent = Number(input.value).toFixed(2);
      });
    });

    const source = document.getElementById('source');
    const info = document.getElementById('source-info');

    document.getElementById('use-webcam')?.addEventListener('click', () => {
      source.value = 'webcam';
      source.dispatchEvent(new Event('change'));
    });

    source?.addEventListener('change', async () => {
      if (!source.value.trim()) { info.textContent = ''; return; }
      info.textContent = 'Checking…';
      try {
        const d = await get(`/api/source-info?source=${encodeURIComponent(source.value.trim())}`);
        info.textContent = `${d.width}×${d.height} · ${d.fps} fps`
          + (d.frames > 0 ? ` · ${d.frames.toLocaleString()} frames` : '');
      } catch (err) {
        info.innerHTML = `<span class="err">${esc(err.message)}</span>`;
      }
    });

    const err = document.getElementById('setup-error');
    document.getElementById('start-btn')?.addEventListener('click', async (ev) => {
      const btn = ev.currentTarget;
      err.textContent = '';
      btn.disabled = true;
      btn.textContent = 'Starting…';
      try {
        await post('/api/session/start', {
          source: source.value.trim(),
          mode,
          conf_sack: document.getElementById('conf_sack')?.value,
          conf_person: document.getElementById('conf_person')?.value,
          conf_box: document.getElementById('conf_box')?.value,
          save_output: document.getElementById('save_output')?.checked,
          use_door_crossing: document.getElementById('use_door_crossing')?.checked,
        });
        window.location = '/';
      } catch (e) {
        err.textContent = e.message;
        btn.disabled = false;
        btn.textContent = 'Start counting';
      }
    });
  }

  /* ── Calibrate ──────────────────────────────────────────── */
  function calibrateScreen(initial) {
    const stage = document.getElementById('cal-stage');
    const img = document.getElementById('cal-frame');
    const svg = document.getElementById('cal-svg');
    const poly = document.getElementById('cal-poly');
    const marks = document.getElementById('cal-marks');
    const roomG = document.getElementById('cal-room');
    const hint = document.getElementById('cal-hint');
    const count = document.getElementById('cal-count');
    const err = document.getElementById('cal-error');

    // Points are stored in the source video's pixel space, which is what
    // the pipeline expects; the SVG viewBox is rewritten to match once a
    // frame is loaded, so clicks map 1:1 regardless of display size.
    let W = 1600, H = 900;
    let points = (initial.points || []).map((p) => ({ x: p[0], y: p[1] }));
    let room = initial.roomPoint ? { x: initial.roomPoint[0], y: initial.roomPoint[1] } : null;

    const HINTS = [
      'Click the first corner of the doorway.',
      'Click the next corner, going around the doorway.',
      'Two more corners to go.',
      'One more corner and the doorway is outlined.',
      'Doorway outlined. Now click any spot inside the room, so the system knows which way is in.',
      'Done — press Save and you will not need to do this again for this camera.',
    ];

    function draw() {
      poly.setAttribute('points', points.map((p) => `${Math.round(p.x)},${Math.round(p.y)}`).join(' '));
      const size = Math.max(W, H) / 62;
      marks.innerHTML = points.map((p, i) => `
        <rect x="${p.x - size / 2}" y="${p.y - size / 2}" width="${size}" height="${size}" fill="#ec3013"></rect>
        <text x="${p.x}" y="${p.y + size * 0.34}" fill="#f3f2f2" font-family="Archivo,sans-serif"
              font-weight="800" font-size="${size * 0.7}" text-anchor="middle">${i + 1}</text>`).join('');
      roomG.innerHTML = room ? `
        <rect x="${room.x - size / 2}" y="${room.y - size / 2}" width="${size}" height="${size}" fill="#f3f2f2"></rect>
        <text x="${room.x + size}" y="${room.y + size * 0.3}" fill="#f3f2f2"
              font-family="Archivo,sans-serif" font-weight="800" font-size="${size * 0.6}">INSIDE</text>` : '';

      const done = room ? 5 : points.length;
      hint.textContent = HINTS[Math.min(done, HINTS.length - 1)];
      count.innerHTML = `${points.length} point(s) marked<br>Saved to config.yaml`;
      const active = points.length < 4 ? 1 : (!room ? 2 : 3);
      [1, 2, 3].forEach((n) => {
        document.getElementById(`step-${n}`).classList.toggle('on', n === active);
      });
    }

    async function loadFrame() {
      const src = document.getElementById('cal-source').value.trim();
      if (!src) { err.textContent = 'Type the video file to take a frame from.'; return; }
      err.textContent = '';
      try {
        const d = await get(`/api/source-info?source=${encodeURIComponent(src)}`);
        W = d.width || 1600; H = d.height || 900;
        svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
        img.src = `/api/frame.jpg?source=${encodeURIComponent(src)}&at=0&t=${Date.now()}`;
        draw();
      } catch (e) {
        err.textContent = e.message;
      }
    }

    stage.addEventListener('click', (ev) => {
      if (!img.src) { err.textContent = 'Load a frame first.'; return; }
      const r = stage.getBoundingClientRect();
      const x = Math.round((ev.clientX - r.left) / r.width * W);
      const y = Math.round((ev.clientY - r.top) / r.height * H);
      if (points.length < 4) points.push({ x, y });
      else if (!room) room = { x, y };
      draw();
    });

    document.getElementById('load-frame').addEventListener('click', loadFrame);
    document.getElementById('undo-btn').addEventListener('click', () => {
      if (room) room = null; else points.pop();
      draw();
    });
    document.getElementById('clear-btn').addEventListener('click', () => {
      points = []; room = null; draw();
    });
    document.getElementById('save-btn').addEventListener('click', async () => {
      err.textContent = '';
      try {
        await post('/api/calibration', {
          points: points.map((p) => [p.x, p.y]),
          room_point: room ? [room.x, room.y] : null,
        });
        window.location = '/setup';
      } catch (e) {
        err.textContent = e.message;
      }
    });

    draw();
    if (document.getElementById('cal-source').value.trim()) loadFrame();
  }

  /* ── History ────────────────────────────────────────────── */
  function historyScreen() {
    const search = document.getElementById('history-search');
    const rows = () => Array.from(document.querySelectorAll('#history-table tbody tr'));

    function apply() {
      const term = (search?.value || '').trim().toLowerCase();
      const mode = document.querySelector('input[name="hf"]:checked')?.value || 'all';
      rows().forEach((tr) => {
        const okMode = mode === 'all' || tr.dataset.mode === mode;
        const okTerm = !term || tr.dataset.source.includes(term);
        tr.style.display = okMode && okTerm ? '' : 'none';
      });
    }

    search?.addEventListener('input', apply);
    document.querySelectorAll('input[name="hf"]').forEach((r) =>
      r.addEventListener('change', apply));
  }

  /* ── Detection check ────────────────────────────────────── */
  function diagnosticsScreen(initial) {
    const out = document.getElementById('diag-result');
    const status = document.getElementById('diag-status');

    function paint(r) {
      if (!r) return;
      out.innerHTML = `
        <div class="diag-grid">
          <section class="diag-panel">
            <h6 style="margin:0 0 4px">How sure the model was, sack by sack</h6>
            <div class="screen-sub" style="margin-bottom:16px">
              Bars to the left of the cut-off (${r.cutoff.toFixed(2)}) are thrown away.
              ${r.kept.toLocaleString()} of ${r.detections.toLocaleString()} detections survive it.
            </div>
            <div class="hist">
              ${r.histogram.map((h) => `
                <div class="col">
                  <div class="mono" style="font-size:10px;text-align:center;color:var(--color-neutral-700);margin-bottom:3px">${h.v}</div>
                  <div class="bar ${h.kept ? 'kept' : ''}" style="height:${h.h}px"></div>
                </div>`).join('')}
            </div>
            <div class="chart-labels">
              ${r.histogram.map((h) => `<span class="mono">${h.label}</span>`).join('')}
            </div>
            <div class="note-box" style="margin-top:16px">
              Scores cluster around ${r.stats.median}. If most sacks score below your
              cut-off, lower it on the Setup screen — or, for sacks lying on the
              floor, use the separate floor-sack model, otherwise almost nothing
              is detected.
            </div>
          </section>

          <section class="diag-panel">
            <h6 style="margin:0 0 4px">Sacks seen per checked frame</h6>
            <div class="screen-sub" style="margin-bottom:16px">
              Long flat stretches at zero usually mean the camera view is blocked,
              not that the room is empty. ${r.zero_frames} of ${r.checked_frames}
              checked frames saw nothing.
            </div>
            <div class="spark">
              ${r.sparkline.map((s) => `<span class="${s.zero ? 'zero' : ''}" style="height:${s.h}px"></span>`).join('')}
            </div>
            <div class="mono" style="display:flex;justify-content:space-between;font-size:10px;color:var(--color-neutral-600);margin-top:5px">
              <span>frame 0</span><span>frame ${r.total_frames.toLocaleString()}</span>
            </div>
            <div class="hr"></div>
            <h6 style="margin:0 0 10px">Sample frames</h6>
            <div class="samples">
              ${r.samples.map((s) => `
                <figure><div><img src="${s.data_uri}" alt="${esc(s.caption)}"></div>
                <figcaption class="mono">${esc(s.caption)}</figcaption></figure>`).join('')}
            </div>
          </section>
        </div>`;
    }

    async function poll() {
      const s = await get('/api/diagnostics');
      if (s.status === 'running') {
        status.textContent = 'Running the check — sampling frames…';
        setTimeout(poll, 1200);
        return;
      }
      if (s.status === 'error') {
        status.innerHTML = `<span class="err">${esc(s.error)}</span>`;
        return;
      }
      if (s.result) {
        status.textContent = `${s.result.checked_frames} frames checked · `
          + `${s.result.resolution} · ${s.result.detections.toLocaleString()} detections`;
        paint(s.result);
      }
    }

    document.getElementById('diag-run').addEventListener('click', async () => {
      status.textContent = 'Starting…';
      try {
        await post('/api/diagnostics/start', {
          source: document.getElementById('diag-source').value.trim(),
          every: document.getElementById('diag-every').value,
        });
        poll();
      } catch (e) {
        status.innerHTML = `<span class="err">${esc(e.message)}</span>`;
      }
    });

    if (initial) { paint(initial); }
  }

  return { post, get, paintStatus, pollSession, setupScreen,
           calibrateScreen, historyScreen, diagnosticsScreen };
})();
