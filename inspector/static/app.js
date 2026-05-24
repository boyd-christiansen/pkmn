/* ============================================================
   pkmn SFT inspector — frontend
   No framework, no build step. Vanilla JS.
   Three pane layout for Browse + a Compare view.
   ============================================================ */

const state = {
  files: { current: [], legacy: [] },
  currentFile: null,         // path string
  currentRowIdx: null,       // int
  rowDetail: null,           // last fetched detail payload
  pinned: { A: null, B: null },  // each: { file, idx, payload }
};

// ------------------------------------------------------------
// API helpers
// ------------------------------------------------------------

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    const err = await r.text();
    throw new Error(`${r.status}: ${err.slice(0, 200)}`);
  }
  return r.json();
}

function setStatus(msg, kind = "info") {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.style.color = kind === "error" ? "var(--danger)" : "";
}

// ------------------------------------------------------------
// Initial load: file index
// ------------------------------------------------------------

async function loadFiles() {
  setStatus("loading file index…");
  try {
    state.files = await api("/api/files");
    renderFileList();
    setStatus(
      `${state.files.current.length} current + ${state.files.legacy.length} legacy files`,
    );
  } catch (e) {
    setStatus("file index failed: " + e.message, "error");
  }
}

function renderFileList() {
  const renderInto = (ulId, files) => {
    const ul = document.getElementById(ulId);
    ul.innerHTML = "";
    if (!files.length) {
      ul.innerHTML = `<li class="placeholder" style="cursor:default">none</li>`;
      return;
    }
    for (const f of files) {
      const li = document.createElement("li");
      if (f.bucket === "legacy") li.classList.add("legacy");
      li.dataset.path = f.path;
      li.innerHTML = `
        <span>
          <span class="file-kind">${f.kind === "sft" ? "SFT" : f.kind === "parsed_match" ? "MATCH" : "?"}</span>
          ${escapeHtml(f.path.replace(/^legacy\//, ""))}
        </span>
        <span class="file-rows">${f.rows}</span>
      `;
      li.addEventListener("click", () => selectFile(f.path));
      ul.appendChild(li);
    }
  };
  renderInto("files-current", state.files.current);
  renderInto("files-legacy", state.files.legacy);
}

// ------------------------------------------------------------
// File → row list
// ------------------------------------------------------------

async function selectFile(path) {
  document.querySelectorAll("#file-sidebar li").forEach(li => {
    li.classList.toggle("active", li.dataset.path === path);
  });
  state.currentFile = path;
  state.currentRowIdx = null;

  document.getElementById("rows-header").innerHTML = `<code>${escapeHtml(path)}</code>`;
  document.getElementById("row-list").innerHTML = `<li class="placeholder">loading…</li>`;
  document.getElementById("detail-content").innerHTML = "";
  document.getElementById("detail-header").innerHTML = `<span class="placeholder">pick a row to see its details →</span>`;

  try {
    const data = await api(`/api/file/${encodeURIComponent(path)}/rows`);
    renderRowList(data.rows);
  } catch (e) {
    document.getElementById("row-list").innerHTML =
      `<li class="placeholder">load failed: ${escapeHtml(e.message)}</li>`;
  }
}

function renderRowList(rows) {
  const ul = document.getElementById("row-list");
  ul.innerHTML = "";
  if (!rows.length) {
    ul.innerHTML = `<li class="placeholder">empty file</li>`;
    return;
  }

  // For SFT files, group rows by (match_id, game_index) under a divider
  // for visual hierarchy. Parsed-match files: one row per match, no
  // grouping.
  let lastGameKey = null;
  rows.forEach((r, i) => {
    if (r.kind === "sft") {
      const gk = `${r.match_id}__${r.game_index}`;
      if (gk !== lastGameKey) {
        lastGameKey = gk;
        const divider = document.createElement("li");
        divider.className = "game-divider";
        divider.textContent = `${r.match_id}  · game ${r.game_index}`;
        ul.appendChild(divider);
      }
      const li = document.createElement("li");
      li.dataset.idx = String(r.idx);
      li.innerHTML = `
        <span class="row-id">turn ${r.turn}</span>
        <span class="row-summary">${escapeHtml(r.format_id || "")}</span>
      `;
      li.addEventListener("click", () => selectRow(r.idx));
      ul.appendChild(li);
    } else if (r.kind === "parsed_match") {
      const li = document.createElement("li");
      li.dataset.idx = String(r.idx);
      li.innerHTML = `
        <span class="row-id">${escapeHtml(r.match_id)}</span>
        <span class="row-summary">
          ${r.format} · ${r.game_count} game${r.game_count === 1 ? "" : "s"} ·
          ${r.turn_count} turns ·
          ${escapeHtml((r.players || []).join(" vs "))}
        </span>
      `;
      li.addEventListener("click", () => selectRow(r.idx));
      ul.appendChild(li);
    } else {
      const li = document.createElement("li");
      li.innerHTML = `<span class="placeholder">unknown row #${r.idx}</span>`;
      ul.appendChild(li);
    }
  });
}

// ------------------------------------------------------------
// Row → detail view
// ------------------------------------------------------------

async function selectRow(idx) {
  state.currentRowIdx = idx;
  document.querySelectorAll("#row-list li").forEach(li => {
    li.classList.toggle("active", li.dataset.idx === String(idx));
  });
  document.getElementById("detail-content").innerHTML =
    `<div class="placeholder">loading row…</div>`;

  try {
    const detail = await api(
      `/api/file/${encodeURIComponent(state.currentFile)}/row/${idx}`,
    );
    state.rowDetail = detail;
    renderRowDetail(detail, document.getElementById("detail-content"),
                    document.getElementById("detail-header"));
  } catch (e) {
    document.getElementById("detail-content").innerHTML =
      `<div class="placeholder">load failed: ${escapeHtml(e.message)}</div>`;
  }
}

function renderRowDetail(d, contentEl, headerEl) {
  if (d.kind === "sft") return renderSftRow(d, contentEl, headerEl);
  if (d.kind === "parsed_match") return renderParsedMatchRow(d, contentEl, headerEl);
  contentEl.innerHTML = `<pre>${escapeHtml(JSON.stringify(d.raw, null, 2))}</pre>`;
}

// ------------------------------------------------------------
// SFT row renderer — the main detail view
// ------------------------------------------------------------

function renderSftRow(d, contentEl, headerEl) {
  // ---- HEADER BAR ----
  const isPinnedA = state.pinned.A && state.pinned.A.file === d.path && state.pinned.A.idx === d.idx;
  const isPinnedB = state.pinned.B && state.pinned.B.file === d.path && state.pinned.B.idx === d.idx;

  // Dry-run detection: the master_pipeline --dry-run path emits a stub
  // submit_decision tool call whose ack is `{"status":"decision_committed_dry_run"}`.
  // Surface this with a prominent badge so the user knows at a glance
  // they're looking at a preview row (full prompts, placeholder action)
  // and not a real teacher-LLM-synthesized one.
  const isDryRun = (d.tool_loop || []).some(e =>
    e.type === "submit" &&
    e.ack && typeof e.ack === "object" &&
    e.ack.status === "decision_committed_dry_run"
  );

  headerEl.innerHTML = `
    <div class="detail-header-bar">
      <strong>${escapeHtml(d.match_id)}</strong>
      <span class="kv-pair"><span class="k">game</span><span class="v">${d.game_index}</span></span>
      <span class="kv-pair"><span class="k">turn</span><span class="v">${d.turn}</span></span>
      <span class="kv-pair"><span class="k">format</span><span class="v">${escapeHtml(d.format_id || "")}</span></span>
      <span class="schema-badge ${d.user.parsed.schema}">${d.user.parsed.schema}</span>
      ${isDryRun ? `<span class="dry-run-badge" title="Preview row — full prompts but no real LLM synthesis. The action card surfaces the human's actual play from the source replay.">DRY RUN</span>` : ""}
      <button class="pin-btn ${isPinnedA ? "pinned" : ""}" data-slot="A">
        ${isPinnedA ? "📌 A pinned" : "📌 pin to slot A"}
      </button>
      <button class="pin-btn ${isPinnedB ? "pinned" : ""}" data-slot="B">
        ${isPinnedB ? "📌 B pinned" : "📌 pin to slot B"}
      </button>
    </div>
  `;
  headerEl.querySelectorAll(".pin-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const slot = btn.dataset.slot;
      state.pinned[slot] = { file: d.path, idx: d.idx, payload: d };
      // Re-render to update buttons.
      selectRow(d.idx);
      renderCompareSlot(slot);
      setStatus(`pinned to slot ${slot}: ${d.match_id} g${d.game_index} t${d.turn}`);
    });
  });

  // ---- CARDS ----
  const cards = [];

  // INPUTS card (source data summary)
  if (d.source) {
    cards.push(card("Inputs (parsed source)", renderSourceCard(d.source)));
  } else {
    cards.push(card("Inputs (parsed source)",
      `<div class="placeholder">source parsed-match not in pipeline/parsed_data/ — can't show snapshot/events for this row</div>`));
  }

  // SYSTEM card
  cards.push(card("System prompt", renderSystemPrompt(d.system.parsed, d.system.raw),
              { collapsed: true }));

  // USER card
  cards.push(card("User prompt", renderUserPrompt(d.user.parsed, d.user.raw)));

  // TOOL LOOP card
  cards.push(card("Tool loop", renderToolLoop(d.tool_loop)));

  // RAW MESSAGES card (collapsed)
  cards.push(card("Raw messages JSON",
              `<pre>${escapeHtml(JSON.stringify(d.raw_messages, null, 2))}</pre>`,
              { collapsed: true }));

  contentEl.innerHTML = cards.join("");
  attachCardCollapse(contentEl);
}

function renderSourceCard(src) {
  const snap = src.snapshot || {};
  const events = snap.events || [];
  const p1 = snap.p1 || {};
  const p2 = snap.p2 || {};
  const fld = snap.field || {};

  let s = `
    <div class="kv-pair">
      <span class="k">file</span><span class="v">${escapeHtml(src.file)}</span>
      &nbsp;<span class="k">match</span><span class="v">${escapeHtml(src.match_id)}</span>
      &nbsp;<span class="k">game</span><span class="v">${src.game_index}</span>
      &nbsp;<span class="k">turn</span><span class="v">${src.turn}</span>
      &nbsp;<span class="k">format</span><span class="v">${escapeHtml(src.match_format || "")}</span>
      &nbsp;<span class="k">protocol-winner</span><span class="v">${src.post_winner || "?"}</span>
    </div>
  `;
  s += `<details><summary>Snapshot field/p1/p2 summary</summary><div style="margin-top:8px;">`;
  s += `<div class="kv-pair"><span class="k">field</span><span class="v">${escapeHtml(JSON.stringify(fld))}</span></div>`;
  s += `<div class="kv-pair"><span class="k">p1.active</span><span class="v">${escapeHtml(JSON.stringify((p1.active || []).map(a => a.species)))}</span></div>`;
  s += `<div class="kv-pair"><span class="k">p1.bench</span><span class="v">${escapeHtml(JSON.stringify((p1.bench || []).map(b => b.species + (b.fainted ? " (F)" : ""))))}</span></div>`;
  s += `<div class="kv-pair"><span class="k">p2.active</span><span class="v">${escapeHtml(JSON.stringify((p2.active || []).map(a => a.species)))}</span></div>`;
  s += `<div class="kv-pair"><span class="k">p2.bench</span><span class="v">${escapeHtml(JSON.stringify((p2.bench || []).map(b => b.species + (b.fainted ? " (F)" : ""))))}</span></div>`;
  s += `<div class="kv-pair"><span class="k">p1.seenSpecies</span><span class="v">${escapeHtml(JSON.stringify(p1.seenSpecies || []))}</span></div>`;
  s += `<div class="kv-pair"><span class="k">p2.seenSpecies</span><span class="v">${escapeHtml(JSON.stringify(p2.seenSpecies || []))}</span></div>`;
  s += `</div></details>`;

  s += `<details style="margin-top:8px"><summary>events (${events.length})</summary>`;
  s += `<pre>${escapeHtml(JSON.stringify(events, null, 2))}</pre></details>`;

  s += `<details style="margin-top:8px"><summary>full snapshot JSON</summary>`;
  s += `<pre>${escapeHtml(JSON.stringify(snap, null, 2))}</pre></details>`;

  return s;
}

function renderSystemPrompt(parsed, raw) {
  let s = "";
  if (parsed.prelude) {
    s += `<details><summary>Prelude (team blocks + intro)</summary><pre>${escapeHtml(parsed.prelude)}</pre></details>`;
  }
  if (parsed.rules.length) {
    s += `<div style="margin-top:8px">`;
    for (const r of parsed.rules) {
      s += `<div class="rule">
              <span class="rule-num">${r.number}.</span>
              ${r.title ? `<span class="rule-title">${escapeHtml(r.title)}</span>` : ""}
              <pre style="margin-top:4px">${escapeHtml(stripRulePrefix(r.body))}</pre>
            </div>`;
    }
    s += `</div>`;
  }
  if (!parsed.rules.length && !parsed.prelude) {
    s += `<pre>${escapeHtml(raw || "")}</pre>`;
  }
  return s;
}

function stripRulePrefix(body) {
  // remove "1. **Foo Rule**:" or "1. The Foo Rule:" header from the rendered body
  return body.replace(/^\d+\.\s+(?:\*\*[^*]+\*\*[:\.\-\s]+|The\s+[A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*\s+Rule:\s+)/, "");
}

function renderUserPrompt(parsed, raw) {
  const h = parsed.header || {};
  let s = "";

  // Header block (board state)
  s += `<div class="section-block">
          <div class="section-title"><span>HEADER (turn ${h.turn ?? "?"})</span></div>
          ${h.field_str ? `<div class="kv-pair"><span class="k">field</span><span class="v">${escapeHtml(h.field_str)}</span></div>` : ""}
          ${h.p1_active ? `<div style="margin-top:4px"><div class="kv-pair"><span class="k">YOUR (P1) ACTIVE</span></div><pre>${escapeHtml(h.p1_active)}</pre></div>` : ""}
          ${h.p1_bench ? `<div class="kv-pair"><span class="k">YOUR (P1) BENCH</span><span class="v">${escapeHtml(h.p1_bench)}</span></div>` : ""}
          ${h.p2_active ? `<div style="margin-top:4px"><div class="kv-pair"><span class="k">OPP (P2) ACTIVE</span></div><pre>${escapeHtml(h.p2_active)}</pre></div>` : ""}
          ${h.p2_bench ? `<div class="kv-pair"><span class="k">OPP (P2) BENCH</span><span class="v">${escapeHtml(h.p2_bench)}</span></div>` : ""}
        </div>`;

  // Each named section, in document order
  for (const sec of parsed.sections) {
    s += `<div class="section-block scroll-target" data-section="${sec.name}">
            <div class="section-title">
              <span>${escapeHtml(sec.title)}</span>
            </div>
            <pre>${escapeHtml(sec.body)}</pre>
          </div>`;
  }

  // Missing sections
  if (parsed.missing_sections && parsed.missing_sections.length) {
    for (const m of parsed.missing_sections) {
      s += `<div class="section-block missing">
              <div class="section-title missing-section"><span>${escapeHtml(m)} — not present in this row's schema</span></div>
            </div>`;
    }
  }

  s += `<details style="margin-top:12px"><summary>raw user prompt string</summary><pre>${escapeHtml(raw || "")}</pre></details>`;
  return s;
}

function renderToolLoop(loop) {
  if (!loop.length) {
    return `<div class="placeholder">no tool calls — likely a dry-run row or pre-tool-architecture sample</div>`;
  }
  let s = "";
  for (const e of loop) {
    if (e.type === "submit") {
      s += `<div class="tool-iter submit">
              <div class="tool-iter-header">
                <span class="iter-num">iter ${e.iteration}</span>
                <span class="iter-name">submit_decision (final)</span>
              </div>
              <div class="tool-iter-body">
                ${e.thought ? `<div class="thought">${escapeHtml(e.thought)}</div>` : ""}
                ${renderActionGrid(e.action)}
                ${e.ack ? `<details style="margin-top:8px"><summary>tool ack</summary><pre>${escapeHtml(JSON.stringify(e.ack, null, 2))}</pre></details>` : ""}
              </div>
            </div>`;
    } else if (e.type === "calc" || e.type === "tool") {
      s += `<div class="tool-iter">
              <div class="tool-iter-header">
                <span class="iter-num">iter ${e.iteration}</span>
                <span class="iter-name">${escapeHtml(e.name || "")}</span>
              </div>
              <div class="tool-iter-body">
                <div class="args">
                  <h4>Arguments</h4>
                  <pre>${escapeHtml(JSON.stringify(e.args, null, 2))}</pre>
                </div>
                <div class="response">
                  <h4>Tool response</h4>
                  <pre>${e.response == null ? '<span class="placeholder">(no response — orphan call)</span>' : escapeHtml(JSON.stringify(e.response, null, 2))}</pre>
                </div>
              </div>
            </div>`;
    } else if (e.type === "text") {
      s += `<div class="tool-iter">
              <div class="tool-iter-header">
                <span class="iter-num">iter ${e.iteration}</span>
                <span class="iter-name">assistant text</span>
              </div>
              <div class="tool-iter-body" style="display:block">
                <pre>${escapeHtml(e.content || "")}</pre>
              </div>
            </div>`;
    } else if (e.type === "tool_orphan") {
      s += `<div class="tool-iter">
              <div class="tool-iter-header">
                <span class="iter-num">iter ${e.iteration}</span>
                <span class="iter-name placeholder">orphan tool response (no matching tool_call_id)</span>
              </div>
              <div class="tool-iter-body" style="display:block">
                <pre>${escapeHtml(e.raw_response || "")}</pre>
              </div>
            </div>`;
    }
  }
  return s;
}

function renderActionGrid(action) {
  if (!action || typeof action !== "object") {
    return `<div class="placeholder">no action object</div>`;
  }
  const slots = ["slot_1", "slot_2"];
  let s = `<div class="action-grid">`;
  for (const k of slots) {
    const a = action[k] || {};
    s += `<div class="action-card">
            <div class="slot-label">${k}</div>
            <span class="action-type">${escapeHtml(a.action_type || "?")}</span>
            <span class="action-detail">
              ${a.move ? `move=${escapeHtml(a.move)}` : ""}
              ${a.target ? ` target=${escapeHtml(a.target)}` : ""}
              ${a.tera ? ` <strong style="color:var(--good)">+TERA</strong>` : ""}
              ${a.switch_to ? `switch_to=${escapeHtml(a.switch_to)}` : ""}
            </span>
          </div>`;
  }
  s += `</div>`;
  return s;
}

// ------------------------------------------------------------
// Parsed-match row renderer (for bo1.jsonl / bo3.jsonl)
// ------------------------------------------------------------

function renderParsedMatchRow(d, contentEl, headerEl) {
  headerEl.innerHTML = `
    <div class="detail-header-bar">
      <strong>${escapeHtml(d.match_id)}</strong>
      <span class="kv-pair"><span class="k">format</span><span class="v">${escapeHtml(d.format)}</span></span>
      <span class="kv-pair"><span class="k">players</span><span class="v">${escapeHtml((d.players || []).join(" vs "))}</span></span>
    </div>
  `;

  let s = "";
  for (const game of d.games) {
    const snaps = game.snapshots || [];
    s += `<div class="card">
            <div class="card-header">
              Game ${snaps[0]?.turn ? "" : ""}${escapeHtml(game.replay_id || "")}
              <span class="meta">${snaps.length} snapshots · winner: ${game.winner || "?"}</span>
            </div>
            <div class="card-body">`;
    if (game.teamSheets) {
      s += `<details><summary>teamSheets</summary><pre>${escapeHtml(JSON.stringify(game.teamSheets, null, 2))}</pre></details>`;
    }
    for (const snap of snaps) {
      const events = snap.events || [];
      s += `<details style="margin-top:6px"><summary>turn ${snap.turn} (${events.length} events)</summary>
              <pre>${escapeHtml(JSON.stringify(snap, null, 2))}</pre>
            </details>`;
    }
    s += `</div></div>`;
  }
  contentEl.innerHTML = s;
  attachCardCollapse(contentEl);
}

// ------------------------------------------------------------
// Cards: collapsible
// ------------------------------------------------------------

function card(title, bodyHtml, opts = {}) {
  const collapsedClass = opts.collapsed ? " collapsed" : "";
  return `<div class="card${collapsedClass}">
            <div class="card-header">${escapeHtml(title)}</div>
            <div class="card-body">${bodyHtml}</div>
          </div>`;
}

function attachCardCollapse(root) {
  root.querySelectorAll(".card-header").forEach(h => {
    h.addEventListener("click", () => h.parentElement.classList.toggle("collapsed"));
  });
}

// ------------------------------------------------------------
// Compare tab
// ------------------------------------------------------------

async function renderCompareSlot(slot) {
  const pin = state.pinned[slot];
  const el = document.getElementById(`compare-slot-${slot}`);
  if (!pin) {
    el.innerHTML = `<div class="placeholder">slot ${slot} — empty</div>`;
    return;
  }
  el.innerHTML = `<div class="placeholder">loading…</div>`;
  try {
    const detail = await api(`/api/file/${encodeURIComponent(pin.file)}/row/${pin.idx}`);
    pin.payload = detail;
    const headerEl = document.createElement("div");
    headerEl.className = "pane-header";
    const contentEl = document.createElement("div");
    contentEl.className = "detail-content";
    el.innerHTML = "";
    el.appendChild(headerEl);
    el.appendChild(contentEl);
    renderRowDetail(detail, contentEl, headerEl);
  } catch (e) {
    el.innerHTML = `<div class="placeholder">load failed: ${escapeHtml(e.message)}</div>`;
  }
}

document.getElementById("clear-pinned").addEventListener("click", () => {
  state.pinned.A = null;
  state.pinned.B = null;
  renderCompareSlot("A");
  renderCompareSlot("B");
  if (state.currentRowIdx != null) selectRow(state.currentRowIdx);
});

// ------------------------------------------------------------
// Tab switching
// ------------------------------------------------------------

document.querySelectorAll(".tab").forEach(t => {
  t.addEventListener("click", () => {
    const which = t.dataset.tab;
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === t));
    document.querySelectorAll(".tab-pane").forEach(p => {
      p.classList.toggle("active", p.id === `tab-${which}`);
    });
    if (which === "compare") {
      renderCompareSlot("A");
      renderCompareSlot("B");
    }
  });
});

// Sidebar collapsible for legacy section
document.querySelectorAll(".collapsible").forEach(h => {
  h.addEventListener("click", () => {
    h.classList.toggle("collapsed");
    const target = document.getElementById(h.dataset.target);
    if (target) target.classList.toggle("hidden");
  });
});

// ------------------------------------------------------------
// Utilities
// ------------------------------------------------------------

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

// ------------------------------------------------------------
// Boot
// ------------------------------------------------------------

loadFiles();
