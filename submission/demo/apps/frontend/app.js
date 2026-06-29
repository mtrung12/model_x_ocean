"use strict";

// ---- Trait display order (must match backend TRAIT_ORDER) -----------------
const VERDICT_ORDER = [
  "Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism",
];

// ---- DOM handles ----------------------------------------------------------
const $ = (id) => document.getElementById(id);
const essaySelect = $("essay-select");
const essayText = $("essay-text");
const essayLabels = $("essay-labels");
const essayPreview = $("essay-preview");
const essayPreviewBody = $("essay-preview-body");
const essayPreviewToggle = $("essay-preview-toggle");
const charCount = $("char-count");
const gtBadge = $("gt-badge");
const runBtn = $("run-btn");
const statusLine = $("status-line");
const timelineFill = $("timeline-fill");
const loaderMessage = $("loader-message");

let essays = [];
const STAGES = ["profile", "retrieval", "predict", "verdict"];

function setLoadingMessage(message) {
  if (loaderMessage) loaderMessage.textContent = message;
}

function hideAppLoader() {
  document.body.classList.remove("is-initializing");
}

// ---------------------------------------------------------------------------
// Init: load essays into the dropdown
// ---------------------------------------------------------------------------
async function init() {
  setLoadingMessage("Loading labelled essays and preparing the default example.");
  try {
    const res = await fetch("/essays");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    essays = await res.json();
    for (const e of essays) {
      const opt = document.createElement("option");
      opt.value = String(e.id);
      const def = e.is_default ? " ★" : "";
      opt.textContent = `#${e.id}${def}  —  ${e.preview}`;
      essaySelect.appendChild(opt);
    }
    const def = essays.find((e) => e.is_default);
    if (def) { essaySelect.value = String(def.id); onEssayPicked(); }
    setLoadingMessage("Ready.");
  } catch (err) {
    setLoadingMessage("Essay list unavailable. Free-text mode is ready.");
    statusLine.textContent = "Could not load test essays — free-text still works.";
  } finally {
    window.setTimeout(hideAppLoader, 350);
  }
  updateRunState();
}

const TRAIT_FULL = {
  O: "Openness", C: "Conscientiousness", E: "Extraversion",
  A: "Agreeableness", N: "Neuroticism",
};

function setEssayPreview(text) {
  if (text) {
    essayPreviewBody.textContent = text;
    essayPreview.classList.remove("hidden");
    essayPreviewBody.classList.add("hidden");
    essayPreviewToggle.textContent = "Show";
    essayPreviewToggle.setAttribute("aria-expanded", "false");
  } else {
    essayPreview.classList.add("hidden");
    essayPreviewBody.classList.add("hidden");
  }
}

essayPreviewToggle.addEventListener("click", () => {
  const expanded = essayPreviewToggle.getAttribute("aria-expanded") === "true";
  essayPreviewBody.classList.toggle("hidden", expanded);
  essayPreviewToggle.textContent = expanded ? "Show" : "Hide";
  essayPreviewToggle.setAttribute("aria-expanded", String(!expanded));
});

function onEssayPicked() {
  if (essaySelect.value !== "") {
    essayText.value = "";
    const e = essays.find((x) => x.id === Number(essaySelect.value));
    gtBadge.textContent = e ? `ground truth · ${e.label_summary}` : "";
    gtBadge.classList.toggle("hidden", !e);

    if (e && e.label_summary) {
      const chips = e.label_summary.trim().split(/\s+/).map((part) => {
        const [letter, val] = part.split(":");
        const name = TRAIT_FULL[letter] || letter;
        return `<span class="essay-label-chip label-${esc(val)}">`
          + `<span class="elc-name">${esc(name)}</span>`
          + `<span class="elc-val">${esc(val)}</span>`
          + `</span>`;
      }).join("");
      essayLabels.innerHTML = `<span class="elc-heading">Ground truth labels</span>${chips}`;
      essayLabels.classList.remove("hidden");
    } else {
      essayLabels.classList.add("hidden");
    }

    setEssayPreview(e ? e.full_text : null);
  } else {
    essayLabels.classList.add("hidden");
    setEssayPreview(null);
  }
  updateCharCount();
  updateRunState();
}

function updateCharCount() { charCount.textContent = `${essayText.value.length} chars`; }

function updateRunState() {
  const hasEssay = essaySelect.value !== "";
  const hasText = essayText.value.trim().length >= 200;
  runBtn.disabled = !(hasEssay || hasText);
}

essaySelect.addEventListener("change", onEssayPicked);
essayText.addEventListener("input", () => {
  if (essayText.value.trim().length > 0) {
    essaySelect.value = "";
    gtBadge.classList.add("hidden");
    essayLabels.classList.add("hidden");
  }
  updateCharCount();
  updateRunState();
});

// ---------------------------------------------------------------------------
// Timeline + stage-state helpers
// ---------------------------------------------------------------------------
function setTimeline(stage, state) {
  // state: "active" | "done"
  document.querySelectorAll(`.tl-step[data-step="${stage}"]`).forEach((el) => {
    el.classList.remove("active");
    if (state === "active") el.classList.add("active");
    if (state === "done") { el.classList.add("done"); }
  });
  const doneCount = document.querySelectorAll(".tl-step.done").length;
  const activeCount = document.querySelectorAll(".tl-step.active").length;
  const progress = (doneCount + (activeCount ? 0.5 : 0)) / STAGES.length;
  timelineFill.style.width = `${Math.min(100, progress * 100)}%`;
}

function setStageState(stage, txt, cls) {
  document.querySelectorAll(`.stage-state[data-stage="${stage}"]`).forEach((el) => {
    el.textContent = txt;
    el.className = `stage-state ${cls || ""}`;
  });
  const card = $(`stage-${stage}`);
  if (card) card.classList.toggle("active", cls === "running");
}

function setVerdictEssayPreview(text) {
  const wrap = $("verdict-essay-preview");
  const body = $("verdict-preview-body");
  const toggle = $("verdict-preview-toggle");
  if (text) {
    body.textContent = text;
    wrap.classList.remove("hidden");
    body.classList.add("hidden");
    toggle.textContent = "Show";
    toggle.setAttribute("aria-expanded", "false");
    toggle.onclick = () => {
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      body.classList.toggle("hidden", expanded);
      toggle.textContent = expanded ? "Show" : "Hide";
      toggle.setAttribute("aria-expanded", String(!expanded));
    };
  } else {
    wrap.classList.add("hidden");
  }
}

function resetUI() {
  for (const id of ["profile-body", "retrieval-body", "predict-body", "verdict-body"]) {
    $(id).innerHTML = "";
  }
  $("profile-note").classList.add("hidden");
  $("verdict-essay-preview").classList.add("hidden");
  for (const s of STAGES) {
    setStageState(s, "", "");
    document.querySelectorAll(`.tl-step[data-step="${s}"]`).forEach((el) =>
      el.classList.remove("active", "done"));
  }
  timelineFill.style.width = "0%";
}

// ---------------------------------------------------------------------------
// Render: Stage 1 — Profile
// ---------------------------------------------------------------------------
const SIG = {
  high: ["sig-high", "bar-high"], mod: ["sig-mod", "bar-mod"],
  low: ["sig-low", "bar-low"], none: ["sig-none", "bar-none"],
  "n/e": ["sig-ne", "bar-ne"],
};

function renderProfile(profile) {
  if (profile.note) {
    const n = $("profile-note");
    n.textContent = "⚠ " + profile.note;
    n.classList.remove("hidden");
  }
  const body = $("profile-body");
  body.innerHTML = "";

  const grid = document.createElement("div");
  grid.className = "facet-grid reveal";
  for (const trait of Object.keys(profile.by_trait || {})) {
    const facets = profile.by_trait[trait];
    const group = document.createElement("div");
    group.className = "facet-group";
    group.innerHTML = `<h3>${esc(trait)}</h3>`;
    for (const f of facets) {
      const [sigCls, barCls] = SIG[f.signal] || ["sig-none", "bar-none"];
      const row = document.createElement("div");
      row.className = "facet-row";
      row.innerHTML =
        `<div class="facet-code">${esc(f.code)}</div>` +
        `<div class="facet-main">` +
          `<div class="facet-head">` +
            `<span class="facet-name">${esc(f.name)}</span>` +
            `<span class="facet-bar"><i class="${barCls}"></i></span>` +
            `<span class="sig-tag ${sigCls}">${esc(f.signal || "—")}</span>` +
          `</div>` +
          (f.evidence ? `<div class="facet-evidence">${esc(f.evidence)}</div>` : "") +
        `</div>`;
      group.appendChild(row);
    }
    grid.appendChild(group);
  }
  body.appendChild(grid);

  const ling = (profile.linguistic || []).filter((l) => l.value);
  if (ling.length) {
    const card = document.createElement("div");
    card.className = "ling-card reveal";
    card.innerHTML = "<h3>Linguistic fingerprint</h3>";
    const g = document.createElement("div");
    g.className = "ling-grid";
    for (const l of ling) {
      const item = document.createElement("div");
      item.className = "ling-item";
      item.innerHTML = `<b>${esc(l.key)}</b> · ${esc(l.value)}`;
      g.appendChild(item);
    }
    card.appendChild(g);
    body.appendChild(card);
  }
}

// ---------------------------------------------------------------------------
// Render: Stage 2 — Retrieval (one block per trait)
// ---------------------------------------------------------------------------
function ensureBlock(parentId, trait, prefix, pendingTxt) {
  let block = document.getElementById(`${prefix}-${trait}`);
  if (!block) {
    block = document.createElement("div");
    block.id = `${prefix}-${trait}`;
    block.className = "trait-block reveal";
    block.innerHTML = `<h3>${esc(trait)} <span class="pending">${pendingTxt}</span></h3>` +
      `<div class="${prefix}-list"></div>`;
    $(parentId).appendChild(block);
  }
  return block;
}

function renderRetrieval(trait, retrieved) {
  const block = ensureBlock("retrieval-body", trait, "retr", "retrieving");
  block.querySelector("h3").innerHTML = esc(trait);
  const list = block.querySelector(".retr-list");
  list.innerHTML = "";
  const fmt = (x) => (x == null ? "—" : Number(x).toFixed(3));
  retrieved.forEach((r, i) => {
    const s = r.scores || {};
    const card = document.createElement("div");
    card.className = "retr-card";
    card.innerHTML =
      `<div class="retr-head">` +
        `<span class="rank">${i + 1}</span>` +
        `<span class="uid">${esc(String(r.user_id))}</span>` +
        `<span class="label label-${esc(r.label)}">${esc(r.label)}</span>` +
        `<span class="scores">` +
          `<span class="score-chip">fused <b>${fmt(s.fused)}</b></span>` +
          `<span class="score-chip">dense ${fmt(s.dense)}</span>` +
          `<span class="score-chip">facet ${fmt(s.facet)}</span>` +
        `</span>` +
      `</div>` +
      `<div class="retr-snippet">${esc(r.snippet || "")}</div>`;
    list.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Render: Stage 3 — Reasoned prediction
// ---------------------------------------------------------------------------
function renderPrediction(trait, tr) {
  const block = ensureBlock("predict-body", trait, "pred", "predicting");
  const lbl = tr.label || "?";
  block.querySelector("h3").innerHTML =
    `${esc(trait)} <span class="label label-${esc(lbl)}">${esc(lbl)}</span>`;
  let list = block.querySelector(".pred-list");
  list.innerHTML = "";
  const card = document.createElement("div");
  card.className = "pred-card";
  const sections = [
    ["Evidence", tr.evidence],
    ["Facet check", tr.facet_check],
    ["Example alignment", tr.example_alignment],
    ["Verdict", tr.verdict],
  ];
  let html = "";
  for (const [title, val] of sections) {
    if (!val) continue;
    html += `<div class="pred-section"><div class="pred-title">${title}</div>` +
      `<div class="pred-text">${esc(val)}</div></div>`;
  }
  card.innerHTML = html || `<div class="pred-text">No structured sections parsed.</div>`;
  list.appendChild(card);
}

// ---------------------------------------------------------------------------
// Render: Stage 4 — Verdict
// ---------------------------------------------------------------------------
function renderVerdict(result) {
  const body = $("verdict-body");
  const gt = result.ground_truth;
  const traits = result.traits || {};
  let correct = 0, total = 0;

  let thead = "<thead><tr><th>Trait</th><th>Predicted</th>";
  if (gt) thead += "<th>True</th><th>Match</th>";
  thead += "</tr></thead>";

  let rows = "";
  for (const trait of VERDICT_ORDER) {
    const pred = (traits[trait] || {}).label || "?";
    rows += `<tr><td>${trait}</td>` +
      `<td><span class="label label-${esc(pred)}">${esc(pred)}</span></td>`;
    if (gt) {
      const tv = gt[trait] || "?";
      const ok = tv !== "?" && pred === tv;
      if (tv !== "?") { total++; if (ok) correct++; }
      rows += `<td><span class="label label-${esc(tv)}">${esc(tv)}</span></td>` +
        `<td class="mark ${ok ? "ok" : "bad"}">${ok ? "✓" : "✗"}</td>`;
    }
    rows += "</tr>";
  }

  const table = `<table class="verdict-table">${thead}<tbody>${rows}</tbody></table>`;

  let gauge = "";
  if (gt && total > 0) {
    const pct = Math.round((100 * correct) / total);
    gauge =
      `<div class="gauge">` +
        `<div class="ring" style="--pct:${pct}">` +
          `<div class="ring-inner"><div class="ring-pct">${pct}%</div>` +
          `<div class="ring-sub">${correct}/${total} traits</div></div>` +
        `</div>` +
        `<div class="gauge-cap">accuracy on this essay</div>` +
      `</div>`;
  }

  body.innerHTML = `<div class="verdict-wrap reveal"><div>${table}</div>${gauge}</div>`;
  setVerdictEssayPreview(result.input_text || null);
}

// ---------------------------------------------------------------------------
// Run: stream NDJSON events
// ---------------------------------------------------------------------------
async function run() {
  resetUI();
  runBtn.disabled = true;
  runBtn.classList.add("busy");
  statusLine.textContent = "Starting…";

  const payload = { top_k: 3, model_name: "gpt-4o-mini" };
  if (essaySelect.value !== "") payload.essay_id = Number(essaySelect.value);
  else payload.text = essayText.value.trim();

  let resp;
  try {
    resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    finishRun("Network error: " + err);
    return;
  }
  if (!resp.ok) {
    finishRun(`Error ${resp.status}: ${await resp.text()}`);
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line.startsWith("data: ")) handleEvent(JSON.parse(line.slice(6)));
    }
  }
  runBtn.disabled = false;
  runBtn.classList.remove("busy");
}

function finishRun(msg) {
  statusLine.textContent = msg;
  runBtn.disabled = false;
  runBtn.classList.remove("busy");
}

function handleEvent(ev) {
  switch (ev.stage) {
    case "profile":
      if (ev.status === "running") {
        setStageState("profile", "profiling…", "running");
        setTimeline("profile", "active");
        statusLine.textContent = "Profiling the text (30 facets)…";
      } else {
        setStageState("profile", "done", "done");
        setTimeline("profile", "done");
        renderProfile(ev.profile);
      }
      break;

    case "retrieval":
      if (ev.status === "running") {
        setStageState("retrieval", `${ev.trait} ${ev.index}/${ev.total}`, "running");
        setTimeline("retrieval", "active");
        statusLine.textContent = `Retrieving similar essays · ${ev.trait} (${ev.index}/${ev.total})…`;
        ensureBlock("retrieval-body", ev.trait, "retr", "retrieving");
      } else {
        renderRetrieval(ev.trait, ev.retrieved);
      }
      break;

    case "predict":
      if (ev.status === "running") {
        setStageState("retrieval", "done", "done");
        setTimeline("retrieval", "done");
        setStageState("predict", `${ev.trait} ${ev.index}/${ev.total}`, "running");
        setTimeline("predict", "active");
        statusLine.textContent = `Reasoned prediction · ${ev.trait} (${ev.index}/${ev.total})…`;
        ensureBlock("predict-body", ev.trait, "pred", "predicting");
      } else {
        renderPrediction(ev.trait, ev.trait_result);
      }
      break;

    case "done":
      setStageState("predict", "done", "done");
      setTimeline("predict", "done");
      setStageState("verdict", "done", "done");
      setTimeline("verdict", "done");
      renderVerdict(ev.result);
      statusLine.textContent = "✓ Pipeline complete.";
      break;

    case "error":
      statusLine.textContent = "Pipeline error: " + ev.message;
      break;
  }
}

runBtn.addEventListener("click", run);

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

init();
