/* ESCOX 3.0 — one-page front-end (no external dependencies) */
(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  let lastResult = null;
  let lastText = "";

  const EXAMPLES = {
    job: `Environmental Engineer – Renewable Energy (m/f/d)

SolarGrid Hellas is an independent power producer that develops, builds and operates utility-scale solar photovoltaic parks and battery storage facilities across Southern Europe.

We are seeking an environmental engineer with experience in renewable energy systems, environmental impact assessments, and energy efficiency.

Your responsibilities:
• Prepare environmental impact assessments and permitting documentation for new photovoltaic plants
• Monitor compliance with environmental legislation and biodiversity protection measures on site
• Carry out energy audits and propose energy efficiency improvements for our operations
• Design waste management and recycling plans for decommissioned solar panels

Requirements: degree in environmental or energy engineering; knowledge of EU environmental legislation; strong reporting skills.`,
    cv: `Maria Papadopoulou — Sustainability Specialist

Profile: Agronomist with eight years of experience helping food producers reduce their environmental footprint.

Experience
2019–today  Sustainability officer, Olympus Dairy Cooperative (production of milk, yoghurt and cheese)
 - Implemented a food waste monitoring system and trained staff to reduce food waste in production lines
 - Led the transition to renewable energy sources for the cooling plants
 - Prepared the cooperative's annual sustainability report and carbon footprint calculation
2015–2019  Field agronomist, AgroNova
 - Advised farmers on sustainable tillage techniques and organic farming
 - Promoted water conservation and efficient irrigation

Skills: life cycle assessment, circular economy, environmental legislation, stakeholder engagement.`,
    course: `MSc Course: Circular Economy and Waste Management (10 ECTS)

This course introduces students to circular economy principles and their application in industry and municipalities. Students learn to design waste collection and recycling schemes, assess the environmental impact of products using life cycle assessment, and develop strategies for waste prevention and resource efficiency.

Learning outcomes: on completion students will be able to apply circular economy business models, analyse municipal waste streams, evaluate recycling technologies, and advise organisations on environmental legislation regarding waste.`,
  };

  // ------------------------------------------------------------------ input handling
  const textarea = $("#text");
  const updateCount = () => { $("#char-count").textContent = `${textarea.value.length.toLocaleString()} characters`; };
  textarea.addEventListener("input", updateCount);
  $$("[data-example]").forEach((b) => b.addEventListener("click", () => {
    textarea.value = EXAMPLES[b.dataset.example]; $("#file-info").textContent = ""; updateCount(); textarea.focus();
  }));
  $("#clear-btn").addEventListener("click", () => { textarea.value = ""; $("#file-info").textContent = ""; updateCount(); });

  async function uploadFile(file) {
    const info = $("#file-info");
    info.textContent = `Reading ${file.name} …`;
    const fd = new FormData(); fd.append("file", file);
    try {
      const r = await fetch("/api/extract-text", { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "Could not read the file.");
      textarea.value = data.text; updateCount();
      const i = data.info; info.textContent = `✓ ${i.filename} · ${i.type}${i.pages ? ` · ${i.pages} page(s)` : ""} · ${i.chars.toLocaleString()} characters extracted`;
      info.style.color = "var(--ok)";
    } catch (e) {
      info.textContent = `⚠ ${e.message}`; info.style.color = "var(--coral-ink)";
    }
  }
  $("#file").addEventListener("change", (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); e.target.value = ""; });
  $(".upload-btn").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file").click(); } });
  const dz = $("#dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadFile(f); });

  // modules
  $$(".module").forEach((m) => m.addEventListener("click", () => {
    m.setAttribute("aria-pressed", m.getAttribute("aria-pressed") === "true" ? "false" : "true");
    $("#form-error").textContent = "";
  }));
  if (window.ESCOX.defaultMode) $("#mode").value = window.ESCOX.defaultMode;
  if (!window.ESCOX.llmAvailable) $("#mode").value = "retrieval";

  // ------------------------------------------------------------------ analyse
  let stepTimer = null;
  function startLoading() {
    const steps = $$(".loading__steps span"); let i = 0;
    steps.forEach((s) => s.className = "");
    steps[0].className = "active";
    $("#loading").hidden = false;
    $("#loading-note").textContent = $("#mode").value === "retrieval"
      ? "Semantic retrieval only — this takes a few seconds."
      : "A local LLM on CPU can take up to a minute or two per document.";
    stepTimer = setInterval(() => {
      if (i < steps.length - 1) { steps[i].className = "done"; i++; steps[i].className = "active"; }
    }, $("#mode").value === "retrieval" ? 600 : 4000);
  }
  function stopLoading() { clearInterval(stepTimer); $("#loading").hidden = true; }

  async function analyse() {
    const text = textarea.value.trim();
    const sel = Object.fromEntries($$(".module").map((m) => [m.dataset.module, m.getAttribute("aria-pressed") === "true"]));
    const err = $("#form-error"); err.textContent = "";
    if (!text) { err.textContent = "Please paste some text or upload a document first."; textarea.focus(); return; }
    if (!Object.values(sel).some(Boolean)) { err.textContent = "Select at least one extraction module."; return; }
    const btn = $("#analyse-btn"); btn.disabled = true;
    $("#results").hidden = true; startLoading();
    try {
      const r = await fetch("/api/analyse", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, extract_green_skills: sel.green_skills, extract_green_occupations: sel.green_occupations,
          extract_sectors: sel.sectors, mode: $("#mode").value, label_language: $("#lang").value }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || `Server error ${r.status}`);
      lastResult = data; lastText = text.replace(/\r\n/g, "\n").trim();
      render(data);
    } catch (e) {
      err.textContent = `⚠ ${e.message}`;
    } finally {
      stopLoading(); btn.disabled = false;
    }
  }
  $("#analyse-btn").addEventListener("click", analyse);
  textarea.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyse(); });

  // ------------------------------------------------------------------ rendering
  const ICON = {
    green_skills: '<svg viewBox="0 0 24 24"><path d="M5 19c0-8 5-13 14-14 0 9-5 14-13 14H5Zm0 0 7-7"/></svg>',
    green_occupations: '<svg viewBox="0 0 24 24"><path d="M4 8h16v11H4zM9 8V5h6v3M4 13h16"/></svg>',
    sectors: '<svg viewBox="0 0 24 24"><path d="M3 20V10l5 3V10l5 3V6l8 4v10H3Zm4-3h2m4 0h2"/></svg>',
  };
  const TITLE = { green_skills: "Green Skills", green_occupations: "Green Occupations", sectors: "Economic Sectors" };
  const CLS = { green_skills: "skills", green_occupations: "occ", sectors: "sec" };
  const chev = '<svg class="item__chev" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg>';

  const supportBadge = (it) => {
    if (it.support === "explicit") return '<span class="badge badge--explicit" title="Stated in the text">explicit</span>';
    if (it.support === "implicit") return '<span class="badge badge--implicit" title="Implied by the context">inferred</span>';
    return '<span class="badge badge--sim" title="Semantic similarity only (no LLM validation)">similarity match</span>';
  };
  const simBadge = (it) => it.similarity == null ? "" : `<span class="badge badge--sim" title="Cosine similarity of the best matching text fragment">sim ${it.similarity.toFixed(2)}</span>`;
  const confDot = (it) => `<span class="conf conf--${esc(it.confidence)}" title="Confidence: ${esc(it.confidence)}"></span>`;
  const quote = (it) => it.evidence ? `<blockquote class="quote">“${esc(it.evidence)}”${it.evidence_verified === false ? ' <span class="badge badge--undet">quote not found verbatim</span>' : ""}</blockquote>` : "";
  const simbar = (it) => it.similarity == null ? "—" : `<span class="simbar"><span style="width:${Math.max(0, Math.min(1, it.similarity)) * 100}%"></span></span>${it.similarity.toFixed(3)} (rank ${it.retrieval_rank})`;
  const decision = (it) => ({ accepted: "Accepted by LLM", not_applied: "Not applied (retrieval only)", generated: "Generated by LLM, matched to taxonomy" }[it.llm_decision] || esc(it.llm_decision));

  function greenBadge(gs) {
    if (!gs) return "";
    if (gs.value === "green") return `<span class="badge badge--green" title="${esc(gs.basis)}">● Green occupation</span>`;
    if (gs.value === "not_green") return `<span class="badge badge--notgreen" title="${esc(gs.basis)}">Not green</span>`;
    return `<span class="badge badge--undet" title="${esc(gs.basis)}">Green status undetermined</span>`;
  }

  function itemSkill(it) {
    const broader = (it.broader || []).map((b) => esc(b.label || "(unresolved)")).join(", ");
    return `<details class="item"><summary>${confDot(it)}<div class="item__main"><div class="item__label">${esc(it.label)}</div>
      <div class="item__badges">${supportBadge(it)}${simBadge(it)}<span class="badge">${esc(it.skill_type)}</span></div></div>${chev}</summary>
      <div class="item__detail">${quote(it)}
      <dl class="kv"><dt>Definition</dt><dd>${esc(it.description)}</dd>
      <dt>Green classification</dt><dd>${esc(it.green_basis)}</dd>
      <dt>Reuse level</dt><dd>${esc(it.reuse_level)}</dd>
      ${broader ? `<dt>Broader concept</dt><dd>${broader}</dd>` : ""}
      <dt>Similarity</dt><dd>${simbar(it)}</dd><dt>LLM validation</dt><dd>${decision(it)}</dd>
      <dt>ESCO identifier</dt><dd><code>${esc(it.esco_uri)}</code></dd></dl></div></details>`;
  }

  function itemOcc(it) {
    const crumbs = (it.isco_hierarchy || []).map((g) => `<li><b>${esc(g.code)}</b>${esc(g.label || "unresolved")}</li>`).join("");
    return `<details class="item"><summary>${confDot(it)}<div class="item__main"><div class="item__label">${esc(it.label)}</div>
      <div class="item__badges">${it.isco_code ? `<span class="badge badge--code" title="ISCO-08 unit group">ISCO-08 ${esc(it.isco_code)}</span>` : '<span class="badge badge--undet">ISCO not mapped</span>'}
      ${greenBadge(it.green_status)}${supportBadge(it)}${simBadge(it)}</div></div>${chev}</summary>
      <div class="item__detail">${quote(it)}
      <dl class="kv"><dt>ISCO-08</dt><dd><ul class="crumbs">${crumbs}</ul></dd>
      <dt>ISCO basis</dt><dd>${esc(it.isco_basis)}</dd>
      <dt>Green status</dt><dd>${esc(it.green_status?.basis)}</dd>
      <dt>Definition</dt><dd>${esc(it.description)}</dd>
      <dt>Similarity</dt><dd>${simbar(it)}</dd><dt>LLM validation</dt><dd>${decision(it)}</dd>
      <dt>ESCO occupation</dt><dd>code ${esc(it.esco_code)} · <code>${esc(it.esco_uri)}</code></dd>
      <dt>Linked NACE</dt><dd>${(it.nace_codes || []).map(esc).join(", ") || "—"}</dd></dl></div></details>`;
  }

  function itemSec(it) {
    const crumbs = (it.hierarchy || []).map((h) => `<li title="${esc(h.level_name)}"><b>${esc(h.code)}</b>${h.title ? esc(h.title) : '<i class="muted">title not in supplied data</i>'}</li>`).join("");
    const src = (it.candidate_source || []).map((s) => s === "occupation_crosswalk" ? "ESCO–NACE crosswalk of the identified occupation" : s === "semantic" ? "semantic similarity" : s).join(" + ");
    return `<details class="item"><summary>${confDot(it)}<div class="item__main"><div class="item__label">${esc(it.label)}</div>
      <div class="item__badges"><span class="badge badge--code" title="NACE Rev. 2.1">NACE ${esc(it.nace_code)}</span><span class="badge">${esc(it.level_name)}</span>${supportBadge(it)}${simBadge(it)}</div></div>${chev}</summary>
      <div class="item__detail">${quote(it)}
      <dl class="kv"><dt>NACE hierarchy</dt><dd><ul class="crumbs">${crumbs}</ul></dd>
      <dt>Candidate source</dt><dd>${esc(src)}</dd>
      <dt>Similarity</dt><dd>${simbar(it)}</dd><dt>LLM validation</dt><dd>${decision(it)}</dd>
      ${it.label !== it.label_en ? `<dt>English title</dt><dd>${esc(it.label_en)}</dd>` : ""}
      <dt>Basis</dt><dd>${esc(it.nace_basis)}</dd></dl></div></details>`;
  }

  function rejectedList(list) {
    if (!list || !list.length) return "";
    const li = list.slice(0, 12).map((r) => `<li>${esc(r.label || r.id || "?")}${r.similarity != null ? ` <span class="muted">(sim ${Number(r.similarity).toFixed(2)})</span>` : ""} — ${esc(r.reason)}</li>`).join("");
    return `<details class="rejected"><summary>Uncertain / rejected candidates (${list.length})</summary><ul>${li}</ul></details>`;
  }

  function card(module, data) {
    const items = data[module] || [];
    const status = data.module_status[module];
    const fn = { green_skills: itemSkill, green_occupations: itemOcc, sectors: itemSec }[module];
    const body = items.length ? items.map(fn).join("") : `<p class="empty">${esc(status.message)}</p>`;
    return `<article class="rcard rcard--${CLS[module]}"><div class="rcard__head">${ICON[module]}<h3>${TITLE[module]}</h3><span class="rcard__count">${items.length}</span></div>
      <div class="rcard__body">${body}${rejectedList((data.rejected_candidates || {})[module])}</div></article>`;
  }

  function highlight(text, data) {
    const spans = [];
    const add = (arr, cls, labelKey) => (arr || []).forEach((it) => {
      if (it.evidence_span && it.evidence_verified) spans.push({ s: it.evidence_span[0], e: it.evidence_span[1], cls, title: it[labelKey] });
    });
    add(data.green_skills, "ev-skills", "label"); add(data.green_occupations, "ev-occ", "label"); add(data.sectors, "ev-sec", "label");
    if (!spans.length) return null;
    spans.sort((a, b) => a.s - b.s || b.e - a.e);
    let out = "", pos = 0;
    for (const sp of spans) {
      if (sp.s < pos) continue; // skip overlaps
      out += esc(text.slice(pos, sp.s)) + `<mark class="${sp.cls}" title="${esc(sp.title)}">${esc(text.slice(sp.s, sp.e))}</mark>`;
      pos = sp.e;
    }
    return out + esc(text.slice(pos));
  }

  function render(data) {
    const md = data.metadata;
    const modules = md.modules_requested;
    $("#result-grid").innerHTML = modules.map((m) => card(m, data)).join("");
    const llmCalls = (md.llm_calls || []).filter((c) => c.invoked);
    const chips = [
      `<span class="meta-chip">${{ hybrid: "Hybrid: retrieval + LLM", retrieval: "Retrieval only", llm_only: "LLM only" }[md.mode]}</span>`,
      md.llm_invoked ? `<span class="meta-chip meta-chip--llm">LLM invoked ×${llmCalls.length} · ${esc(md.model || "")}</span>` : `<span class="meta-chip">LLM not invoked</span>`,
      `<span class="meta-chip">${md.processing_time.toFixed(1)} s</span>`,
    ];
    $("#results-meta").innerHTML = chips.join("");
    $("#warnings").innerHTML = (md.warnings || []).map((w) => `<div class="warn-box">⚠ ${esc(w)}</div>`).join("");

    const hl = highlight(lastText, data);
    $("#evidence-card").hidden = !hl;
    if (hl) $("#evidence-text").innerHTML = hl;

    // technical details
    const rows = (md.llm_calls || []).map((c) => `<tr><td>${esc(c.module)}</td><td>${c.invoked ? "yes" : "no — " + esc(c.reason || "")}</td>
      <td>${c.n_candidates ?? "—"}</td><td>${c.latency_s ?? "—"}</td><td>${c.prompt_tokens ?? "—"} / ${c.completion_tokens ?? "—"}</td>
      <td>${c.grammar_constrained ? "yes" : "no"}</td><td>${c.valid_json ? "yes" : (c.invoked ? "no" : "—")}</td>
      <td>${c.validation ? `${c.validation.accepted ?? 0} accepted · ${(c.validation.rejected || []).length || c.validation.invalid_labels || 0} rejected` : "—"}</td></tr>`).join("");
    const llm = md.models.llm || {};
    $("#tech-body").innerHTML = `
      <h4>LLM calls</h4>
      <table><thead><tr><th>Module</th><th>Invoked</th><th>Candidates</th><th>Latency (s)</th><th>Tokens in/out</th><th>Grammar-constrained</th><th>Valid JSON</th><th>Validation</th></tr></thead><tbody>${rows || '<tr><td colspan="8">No LLM calls (retrieval-only mode).</td></tr>'}</tbody></table>
      <h4>Models</h4>
      <dl class="kv"><dt>LLM</dt><dd>${esc(llm.model_name || llm.reason || "—")} ${llm.model_source ? `· ${esc(llm.model_source)}` : ""} ${llm.runtime ? `· ${esc(llm.runtime)}` : ""}</dd>
      ${llm.model_sha256 ? `<dt>Model SHA-256</dt><dd><code>${esc(llm.model_sha256)}</code></dd>` : ""}
      <dt>Prompt version</dt><dd>${esc(md.prompt_version)}</dd>
      <dt>Embeddings</dt><dd>${esc(md.models.embedding.name)} (${esc(md.models.embedding.device)})</dd>
      <dt>Taxonomies</dt><dd>${esc(Object.entries(md.taxonomy || {}).map(([k, v]) => `${k}: ${v}`).join(" · "))}</dd>
      <dt>Green occupation reference</dt><dd>${esc(md.green_occupation_reference)}</dd>
      <dt>Text</dt><dd>${md.text.chars} characters · SHA-256 <code>${esc(md.text.sha256.slice(0, 16))}…</code></dd></dl>
      <h4>Full JSON result</h4><pre>${esc(JSON.stringify(data, null, 2))}</pre>`;

    $("#results").hidden = false;
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ------------------------------------------------------------------ export
  function download(name, blob) {
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }
  $("#export-json").addEventListener("click", () => {
    if (lastResult) download("escox3_results.json", new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" }));
  });
  $("#export-csv").addEventListener("click", async () => {
    if (!lastResult) return;
    const r = await fetch("/api/export/csv", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(lastResult) });
    download("escox3_results.csv", await r.blob());
  });

  // ------------------------------------------------------------------ data gaps
  $("#gaps-btn").addEventListener("click", async () => {
    const box = $("#gaps");
    if (!box.hidden) { box.hidden = true; return; }
    const r = await fetch("/api/taxonomy"); const d = await r.json();
    const counts = Object.entries(d.counts || {}).map(([k, v]) => `${k.replace(/_/g, " ")}: <b>${v}</b>`).join(" · ");
    box.innerHTML = `<p>${counts}</p>` + (d.gaps || []).map((g) => `<div class="gap gap--${esc(g.severity)}"><b>${esc(g.id.replace(/_/g, " "))}</b> (${esc(g.severity)})<br>${esc(g.detail)}</div>`).join("");
    box.hidden = false;
  });

  updateCount();
})();
