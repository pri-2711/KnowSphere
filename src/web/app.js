// Knowledge Map frontend: plain JavaScript, no build step.
"use strict";

const NOT_FOUND = "Not found in the uploaded documents.";
const PALETTE = ["#9cc9b4", "#f2b8a2", "#a9c4e8", "#f3d48b", "#c8b6e2", "#b9d98f", "#f0a8b9",
                 "#8fd0d4", "#e3c0a0", "#b8c2cc", "#d9a7d4", "#a3d9a5"];

const $ = (selector) => document.querySelector(selector);
let cy = null;
let documentsCache = [];
let pollTimer = null;

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (e) { /* not JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

// A quote rendered verbatim, with matched keywords wrapped in <mark>.
function highlighted(quote, ranges) {
  let html = "";
  let cursor = 0;
  for (const [start, end] of ranges || []) {
    if (start < cursor) continue;
    html += escapeHtml(quote.slice(cursor, start)) + "<mark>" + escapeHtml(quote.slice(start, end)) + "</mark>";
    cursor = end;
  }
  return html + escapeHtml(quote.slice(cursor));
}

function citationButton(citation, unitId, quote) {
  const q = quote ? ` data-quote="${escapeHtml(quote)}"` : "";
  return `<button type="button" class="citation" data-unit="${escapeHtml(unitId)}"${q} data-label="${escapeHtml(citation)}">${escapeHtml(citation)}</button>`;
}

// ------------------------------------------------------------------ navigation
function showScreen(name) {
  const screens = ["documents", "map", "ask"];
  if (!screens.includes(name)) name = "documents";
  for (const screen of screens) $(`#screen-${screen}`).hidden = screen !== name;
  document.querySelectorAll(".nav-link").forEach((link) => link.classList.toggle("active", link.dataset.screen === name));
  if (name === "map") ensureGraph();
}
window.addEventListener("hashchange", () => showScreen(location.hash.slice(1)));

async function loadStats() {
  try {
    const stats = await api("/api/stats");
    $("#stats").innerHTML = `${stats.documents} documents<br>${stats.pages} pages<br>${stats.images} images<br>` +
      `${stats.concepts} concepts<br>${stats.relationships} relationships`;
  } catch (e) { $("#stats").textContent = ""; }
}

// ------------------------------------------------------------------- documents
function metadataInput(doc, field, cls = "") {
  const value = doc[field] || "";
  const source = doc.user_edited.includes(field) ? "edited by you" : (doc.detected_from[field] ? `found: ${doc.detected_from[field]}` : "not found");
  return `<input class="${cls}" data-field="${field}" value="${escapeHtml(value)}" placeholder="Unknown"><span class="hint">${escapeHtml(source)}</span>`;
}

function renderDocuments(docs) {
  documentsCache = docs;
  const rows = docs.map((doc) => {
    const busy = doc.status === "queued" || doc.status === "running";
    const progress = busy || doc.status === "processed" ? `<div class="progress"><span style="width:${Math.round((doc.progress || 0) * 100)}%"></span></div>` : "";
    const stage = busy ? `<span class="hint">${escapeHtml(doc.stage || "")}</span>` : "";
    const error = doc.error ? `<span class="hint">${escapeHtml(doc.error)}</span>` : "";
    return `<tr data-id="${escapeHtml(doc.id)}">
      <td><span class="doc-name">${escapeHtml(doc.filename)}</span><span class="hint">${doc.images} images, ${doc.concepts} concepts</span></td>
      <td>${doc.pages}</td>
      <td><span class="badge ${escapeHtml(doc.status)}">${escapeHtml(doc.status)}</span>${progress}${stage}${error}</td>
      <td>${metadataInput(doc, "title")}</td>
      <td>${metadataInput(doc, "author")}</td>
      <td>${metadataInput(doc, "year", "year")}</td>
      <td><button type="button" class="small save-meta" ${busy || !doc.pages ? "disabled" : ""}>Save</button></td>
    </tr>`;
  });
  $("#doc-rows").innerHTML = rows.join("") || `<tr><td colspan="7" class="muted">No documents yet.</td></tr>`;
  fillDocumentFilter(docs);
}

async function loadDocuments() {
  try {
    const docs = await api("/api/documents");
    renderDocuments(docs);
    const busy = docs.some((doc) => doc.status === "queued" || doc.status === "running");
    clearTimeout(pollTimer);
    if (busy) pollTimer = setTimeout(loadDocuments, 1500);
    else loadStats();
  } catch (e) {
    $("#doc-rows").innerHTML = `<tr><td colspan="7" class="muted">Could not load documents: ${escapeHtml(e.message)}</td></tr>`;
  }
}

function notify(message, isError = false) {
  const box = $("#upload-message");
  box.textContent = message;
  box.classList.toggle("error", isError);
  box.hidden = false;
}

async function uploadFiles(files) {
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith(".pdf")) { notify(`${file.name} is not a PDF.`, true); continue; }
    const form = new FormData();
    form.append("file", file);
    try {
      await api("/api/documents/upload", { method: "POST", body: form });
      notify(`Uploaded ${file.name}. Processing runs in the background.`);
    } catch (e) { notify(`Upload failed for ${file.name}: ${e.message}`, true); }
  }
  loadDocuments();
}

function setupDocuments() {
  const zone = $("#dropzone");
  ["dragenter", "dragover"].forEach((type) => zone.addEventListener(type, (event) => { event.preventDefault(); zone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((type) => zone.addEventListener(type, (event) => { event.preventDefault(); zone.classList.remove("dragging"); }));
  zone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));
  $("#file-input").addEventListener("change", (event) => { uploadFiles(event.target.files); event.target.value = ""; });
  $("#doc-rows").addEventListener("click", async (event) => {
    if (!event.target.classList.contains("save-meta")) return;
    const row = event.target.closest("tr");
    const body = {};
    row.querySelectorAll("input[data-field]").forEach((input) => { body[input.dataset.field] = input.value.trim(); });
    try {
      await api(`/api/documents/${encodeURIComponent(row.dataset.id)}/metadata`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      notify("Metadata saved. Citations everywhere now use the new values.");
      loadDocuments();
    } catch (e) { notify(`Could not save metadata: ${e.message}`, true); }
  });
}

// ---------------------------------------------------------------- knowledge map
function fillDocumentFilter(docs) {
  const select = $("#filter-document");
  const current = select.value;
  select.innerHTML = `<option value="">All documents</option>` + docs.filter((doc) => doc.pages)
    .map((doc) => `<option value="${escapeHtml(doc.id)}">${escapeHtml(doc.title || doc.filename)}</option>`).join("");
  select.value = current;
}

function graphQuery(focus) {
  const params = new URLSearchParams({ limit: $("#filter-limit").value, min_weight: $("#filter-weight").value });
  if ($("#filter-document").value) params.set("document_id", $("#filter-document").value);
  if ($("#filter-type").value) params.set("type", $("#filter-type").value);
  if (focus) params.set("focus", focus);
  return `/api/graph?${params}`;
}

async function loadGraph(focus) {
  const data = await api(graphQuery(focus));
  const typeSelect = $("#filter-type");
  if (typeSelect.options.length <= 1) {
    typeSelect.innerHTML += data.types.map((type) => `<option value="${escapeHtml(type)}">${escapeHtml(type)}</option>`).join("");
  }
  const importances = data.elements.filter((el) => !el.data.source).map((el) => el.data.importance);
  const maxImportance = Math.max(...importances, 1e-9);
  for (const el of data.elements) {
    if (!el.data.source) {
      el.data.size = 18 + 52 * Math.sqrt(el.data.importance / maxImportance);
      el.data.color = PALETTE[el.data.community % PALETTE.length];
    }
  }
  cy.elements().remove();
  cy.add(data.elements);
  cy.layout({ name: "cose", animate: false, nodeRepulsion: 9000, idealEdgeLength: 90, randomize: true, fit: true, padding: 30 }).run();
  $("#graph-info").textContent = `Showing ${data.node_count} of ${data.total_nodes} concepts and ${data.edge_count} of ${data.total_edges} relationships.`;
  if (focus) selectNode(focus);
}

function ensureGraph() {
  if (cy) return;
  cy = cytoscape({
    container: $("#cy"),
    minZoom: 0.15,
    maxZoom: 3,
    style: [
      { selector: "node", style: {
        "background-color": "data(color)", width: "data(size)", height: "data(size)", label: "data(label)",
        "font-size": 10, "text-valign": "center", "text-halign": "center", "text-wrap": "wrap", "text-max-width": 90,
        color: "#1d2b2a", "text-outline-color": "#ffffff", "text-outline-width": 2, "border-width": 1, "border-color": "#ffffff" } },
      { selector: "node:selected", style: { "border-width": 4, "border-color": "#1f4e4a" } },
      { selector: "edge", style: {
        width: "mapData(weight, 0, 15, 1, 6)", "line-color": "#c9c2b6", "curve-style": "haystack", opacity: 0.7 } },
      { selector: "edge[relation != 'related_to']", style: {
        "line-color": "#1f4e4a", "curve-style": "bezier", "target-arrow-shape": "triangle", "target-arrow-color": "#1f4e4a",
        label: "data(relation)", "font-size": 8, color: "#1f4e4a", "text-rotation": "autorotate" } },
      { selector: ".faded", style: { opacity: 0.15 } },
    ],
  });
  cy.on("tap", "node", (event) => selectNode(event.target.id()));
  cy.on("tap", (event) => { if (event.target === cy) cy.elements().removeClass("faded"); });
  loadGraph().catch((e) => { $("#graph-info").textContent = `Could not load the graph: ${e.message}`; });
}

async function selectNode(conceptId) {
  const node = cy.getElementById(conceptId);
  if (node.nonempty()) {
    cy.elements().addClass("faded");
    node.closedNeighborhood().removeClass("faded");
    cy.$(":selected").unselect();
    node.select();
    cy.animate({ center: { eles: node }, duration: 300 });
  }
  const panel = $("#panel");
  panel.innerHTML = `<p class="muted">Loading…</p>`;
  try {
    const concept = await api(`/api/concepts/${encodeURIComponent(conceptId)}`);
    panel.innerHTML = renderConcept(concept);
  } catch (e) { panel.innerHTML = `<p class="muted">Could not load concept: ${escapeHtml(e.message)}</p>`; }
}

function renderConcept(concept) {
  const description = concept.description
    ? `<p class="quote">${escapeHtml(concept.description.quote)}</p>${citationButton(concept.description.citation, concept.description.content_unit_id, concept.description.quote)}`
    : `<p class="muted">No description sentence found.</p>`;
  const aliases = concept.aliases.length
    ? `<div class="chips">${concept.aliases.slice(0, 30).map((alias) => `<span class="chip">${escapeHtml(alias)}</span>`).join("")}</div>`
    : `<p class="muted">None</p>`;
  const related = concept.related.length ? `<ul class="list">${concept.related.map((item) => {
    const evidence = item.evidence[0];
    return `<li><div class="related-row"><button type="button" class="chip" data-concept="${escapeHtml(item.id)}">${escapeHtml(item.name)}</button>
      <span class="relation">${escapeHtml(item.relation_type)}${item.direction === "in" && item.relation_type !== "related_to" ? " (inverse)" : ""}, confidence ${item.confidence}</span></div>
      ${evidence ? citationButton(evidence.citation, evidence.content_unit_id, evidence.quote) : ""}</li>`;
  }).join("")}</ul>` : `<p class="muted">None</p>`;
  const sources = `<ul class="list">${concept.sources.slice(0, 25).map((source) =>
    `<li>${citationButton(source.citation, source.content_unit_id, source.quote ? source.quote.quote : "")}</li>`).join("")}</ul>`;
  const images = concept.images.length ? `<h3>Images (${concept.images.length})</h3><div class="thumbs">${concept.images.slice(0, 12).map((image) =>
    `<img src="/api/images/${encodeURIComponent(image.id)}" alt="${escapeHtml(image.caption || "Image")}" title="${escapeHtml((image.caption || "Image") + " " + image.citation)}">`).join("")}</div>` : "";
  return `<h2>${escapeHtml(concept.name)}</h2>
    <span class="chip type-chip">${escapeHtml(concept.type || "concept")}</span>
    <span class="muted small"> ${concept.mention_count} mentions on ${concept.page_count} pages</span>
    <h3>Description</h3>${description}
    <h3>Aliases</h3>${aliases}
    <h3>Related concepts (${concept.related.length})</h3>${related}
    <h3>Sources (${concept.sources.length})</h3>${sources}${images}`;
}

function setupMap() {
  const input = $("#concept-search");
  const list = $("#suggestions");
  let timer = null;
  let items = [];
  let active = -1;
  const close = () => { list.hidden = true; active = -1; };
  const choose = async (item) => {
    input.value = item.name;
    close();
    await loadGraph(item.id);
  };
  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (!q) { close(); return; }
    timer = setTimeout(async () => {
      try { items = await api(`/api/concepts/autocomplete?q=${encodeURIComponent(q)}&limit=10`); } catch (e) { items = []; }
      list.innerHTML = items.map((item, index) => `<li data-index="${index}"><span>${escapeHtml(item.name)}</span><span class="muted small">${escapeHtml(item.type || "")}</span></li>`).join("")
        || `<li class="muted">No matching concept</li>`;
      list.hidden = false;
    }, 150);
  });
  input.addEventListener("keydown", (event) => {
    if (list.hidden || !items.length) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      active = (active + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      list.querySelectorAll("li").forEach((li, index) => li.classList.toggle("active", index === active));
    } else if (event.key === "Enter") {
      event.preventDefault();
      choose(items[Math.max(active, 0)]);
    } else if (event.key === "Escape") close();
  });
  list.addEventListener("click", (event) => {
    const li = event.target.closest("li[data-index]");
    if (li) choose(items[Number(li.dataset.index)]);
  });
  document.addEventListener("click", (event) => { if (!event.target.closest(".search-box")) close(); });
  $("#filter-weight").addEventListener("input", () => { $("#weight-value").textContent = $("#filter-weight").value; });
  ["#filter-document", "#filter-type", "#filter-weight", "#filter-limit"].forEach((selector) =>
    $(selector).addEventListener("change", () => loadGraph()));
}

// ------------------------------------------------------------------------- ask
function renderAnswer(result) {
  if (!result.answer_found) {
    return `<div class="answer-card"><p class="not-found">${escapeHtml(result.message || NOT_FOUND)}</p></div>`;
  }
  const groups = result.groups.map((group) => `<div class="source-group">
      ${group.sentences.map((sentence) => `<p class="sentence">${highlighted(sentence.quote, sentence.highlights)} ${citationButton(sentence.citation, sentence.content_unit_id, sentence.quote)}</p>`).join("")}
    </div>`).join("");
  const chips = result.concepts.length ? `<h3 class="muted small">Related concepts</h3><div class="chips">${result.concepts.map((concept) =>
    `<button type="button" class="chip" data-concept="${escapeHtml(concept.id)}">${escapeHtml(concept.name)}</button>`).join("")}</div>` : "";
  return `<div class="answer-card"><h2>Answer from your documents</h2>${groups}</div>
    <div class="answer-card">${chips || '<p class="muted">No related concepts.</p>'}<p class="muted small">Answered in ${result.took_ms} ms. Every sentence above is copied word for word from the cited page.</p></div>`;
}

function renderSearch(result) {
  const concepts = result.concepts.length ? `<div class="chips">${result.concepts.map((concept) =>
    `<button type="button" class="chip" data-concept="${escapeHtml(concept.id)}">${escapeHtml(concept.name)}</button>`).join("")}</div>` : "";
  const rows = result.results.map((item) => `<div class="result"><p class="sentence">${highlighted(item.quote, item.highlights)}</p>
    ${citationButton(item.citation, item.content_unit_id, item.quote)} <span class="muted small">score ${item.score}</span></div>`).join("");
  return `<div class="answer-card"><h2>Page search</h2>${concepts}${rows || `<p class="not-found">${NOT_FOUND}</p>`}
    <p class="muted small">${result.results.length} pages in ${result.took_ms} ms.</p></div>`;
}

function setupAsk() {
  const answer = $("#answer");
  $("#ask-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = $("#question").value.trim();
    if (!question) return;
    answer.innerHTML = `<p class="muted">Searching your documents…</p>`;
    try {
      const result = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
      answer.innerHTML = renderAnswer(result);
    } catch (e) { answer.innerHTML = `<p class="muted">Error: ${escapeHtml(e.message)}</p>`; }
  });
  $("#search-button").addEventListener("click", async () => {
    const q = $("#question").value.trim();
    if (!q) return;
    answer.innerHTML = `<p class="muted">Searching…</p>`;
    try { answer.innerHTML = renderSearch(await api(`/api/search?q=${encodeURIComponent(q)}&limit=10`)); }
    catch (e) { answer.innerHTML = `<p class="muted">Error: ${escapeHtml(e.message)}</p>`; }
  });
}

// ------------------------------------------------------------- shared handlers
function openPage(unitId, label, quote) {
  const params = new URLSearchParams();
  if (quote) params.set("quote", quote);
  $("#modal-title").textContent = label;
  $("#modal-image").src = `/api/pages/${encodeURIComponent(unitId)}/image?${params}`;
  $("#modal").hidden = false;
}

document.addEventListener("click", (event) => {
  const citation = event.target.closest(".citation");
  if (citation) { openPage(citation.dataset.unit, citation.dataset.label, citation.dataset.quote); return; }
  const chip = event.target.closest("[data-concept]");
  if (chip) {
    location.hash = "#map";
    showScreen("map");
    loadGraph(chip.dataset.concept);
  }
});
$("#modal-close").addEventListener("click", () => { $("#modal").hidden = true; $("#modal-image").removeAttribute("src"); });
$("#modal").addEventListener("click", (event) => { if (event.target.id === "modal") $("#modal-close").click(); });
document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !$("#modal").hidden) $("#modal-close").click(); });

setupDocuments();
setupMap();
setupAsk();
loadDocuments();
loadStats();
showScreen(location.hash.slice(1) || "documents");
