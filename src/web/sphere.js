// Knowledge Sphere screens: overview, governance review, research answers and governed concepts.
// Loaded after app.js and uses its helpers (api, escapeHtml, citationButton, highlighted, Types).
"use strict";

const Sphere = (() => {
  const STATUS_LABEL = { verified: "verified", corroborated: "corroborated", adjudicated: "adjudicated",
                         needs_review: "needs review", candidate: "candidate", rejected: "rejected" };

  const status = (value) =>
    `<span class="chip status-chip status-${escapeHtml(value || "unvalidated")}">${escapeHtml(STATUS_LABEL[value] || value || "unvalidated")}</span>`;

  const evidenceLabel = (e) => `${e.document_id.slice(0, 12)}… p. ${e.page_number ?? "?"}`;

  function evidenceList(items) {
    if (!items || !items.length) return `<p class="muted small">No evidence quote.</p>`;
    return `<ul class="list">${items.map((e) => `<li><p class="quote">${escapeHtml(e.quote)}</p>
      ${citationButton(e.citation || evidenceLabel(e), e.content_unit_id, e.quote)}
      ${e.method ? `<span class="muted small"> ${escapeHtml(e.method)}${e.model ? " · " + escapeHtml(e.model) : ""}</span>` : ""}
      ${e.verified_in_source === false ? `<span class="chip status-chip status-rejected">quote not in source</span>` : ""}</li>`).join("")}</ul>`;
  }

  function reviewer() {
    let name = "";
    try { name = localStorage.getItem("km-reviewer") || ""; } catch (e) { /* storage unavailable */ }
    if (!name) {
      name = (window.prompt("Your name, recorded with your review decisions:") || "").trim();
      if (name) { try { localStorage.setItem("km-reviewer", name); } catch (e) { /* ignore */ } }
    }
    return name;
  }

  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify(body) });

  // ---------------------------------------------------------------- claims
  function claimRow(c) {
    const flags = [
      c.kind && c.kind !== "statement" ? `<span class="chip">${escapeHtml(c.kind)}</span>` : "",
      c.polarity === "negated" ? `<span class="chip">negated</span>` : "",
      ...(c.uncertainty || []).map((u) => `<span class="chip uncertain">${escapeHtml(u)}</span>`),
      c.citation_valid === false ? `<span class="chip status-chip status-rejected">citation failed</span>` : "",
    ].join("");
    const conflicts = c.conflicts && c.conflicts.length
      ? `<p class="conflict">Conflicts: ${c.conflicts.map(escapeHtml).join("; ")}</p>` : "";
    const e = (c.evidence || [])[0];
    return `<li><p>${escapeHtml(c.statement)} ${status(c.status)}</p><div class="chips">${flags}</div>${conflicts}
      ${e ? `<p class="quote small">${escapeHtml(e.quote)}</p>${citationButton(evidenceLabel(e), e.content_unit_id, e.quote)}` : ""}</li>`;
  }

  function claimGroups(groups) {
    const sections = [["facts", "Facts (accepted, stated plainly)"], ["interpretations", "Interpretations (the source infers or hedges)"],
                      ["hypotheses", "Hypotheses (the source proposes)"], ["unconfirmed", "Unconfirmed (waiting for review or contested)"]];
    const html = sections.filter(([key]) => groups[key] && groups[key].length)
      .map(([key, title]) => `<h3>${title} (${groups[key].length})</h3><ul class="list claims">${groups[key].map(claimRow).join("")}</ul>`).join("");
    return html || `<p class="muted">No claims about this yet.</p>`;
  }

  // ---------------------------------------------------------------- research and hybrid search
  function renderPassages(passages) {
    return passages.map((p) => `<div class="result"><p class="sentence">${highlighted(p.quote, p.highlights)}</p>
      ${citationButton(p.citation, p.content_unit_id, p.quote)}
      <span class="muted small"> found by ${p.methods.map(escapeHtml).join(" + ")}</span></div>`).join("");
  }

  function renderResearch(r) {
    if (r.abstained) {
      return `<div class="answer-card"><p class="not-found">${escapeHtml(r.message || NOT_FOUND)}</p>
        <p class="muted small">No passage in this project supports an answer, so none is given.</p></div>`;
    }
    const sentences = r.sentences && r.sentences.length ? r.sentences.map((s) =>
      `<p class="sentence">${highlighted(s.quote, s.highlights)} ${citationButton(s.citation, s.content_unit_id, s.quote)}</p>`).join("")
      : `<p class="muted">No single sentence answers this directly; see the passages below.</p>`;
    const conflicts = r.conflicts.length ? `<div class="answer-card warn"><h2>Unresolved conflicts (${r.conflicts.length})</h2>
      <ul class="list claims">${r.conflicts.map(claimRow).join("")}</ul></div>` : "";
    const calculations = r.calculations.length ? `<div class="answer-card"><h2>Converted values</h2><table class="mini"><tr><th>Statement</th><th>Stated</th><th>Converted</th><th>Standing</th></tr>
      ${r.calculations.map((c) => `<tr><td>${escapeHtml(c.statement)}</td><td>${escapeHtml(c.stated)}</td><td>${Number(c.value.toPrecision(6))} ${escapeHtml(c.unit)}</td><td>${escapeHtml(c.standing)}</td></tr>`).join("")}</table>
      <p class="muted small">Computed from the quoted numbers with fixed unit factors, not by a model.</p></div>` : "";
    const concepts = r.concepts && r.concepts.length ? `<div class="chips">${r.concepts.map((c) =>
      `<button type="button" class="chip" data-concept="${escapeHtml(c.id)}">${Types.dot(c.type)}${escapeHtml(c.name)}</button>`).join("")}</div>` : "";
    return `<div class="answer-card"><h2>From your documents</h2>${sentences}${concepts}</div>
      ${conflicts}
      <div class="answer-card"><h2>What the knowledge base says</h2>${claimGroups(r.claims)}</div>
      ${calculations}
      <div class="answer-card"><h2>Supporting passages</h2>${renderPassages(r.passages)}
      <p class="muted small">Methods: ${r.methods.join(", ")}. ${r.took_ms} ms. Every quote is copied from the cited page and checked against it.</p></div>`;
  }

  function renderHybrid(r) {
    return `<div class="answer-card"><h2>Hybrid search</h2>${renderPassages(r.results) || `<p class="not-found">${NOT_FOUND}</p>`}
      <p class="muted small">${r.results.length} pages in ${r.took_ms} ms, methods: ${r.methods.join(", ")}.</p></div>`;
  }

  // ---------------------------------------------------------------- governed concept panel
  function renderKnowledgeConcept(c) {
    const aliases = c.aliases.length ? `<div class="chips">${c.aliases.map((a) => `<span class="chip">${escapeHtml(a)}</span>`).join("")}</div>` : `<p class="muted">None</p>`;
    const edges = c.relationships.length ? `<ul class="list">${c.relationships.map((e) => {
      const other = e.source === c.id ? e.target : e.source;
      const otherName = e.source === c.id ? e.target_name : e.source_name;
      const ev = e.evidence[0];
      return `<li><span class="relation">${e.source === c.id ? "" : "(inverse) "}${escapeHtml(e.type)}</span>
        <button type="button" class="chip" data-concept="${escapeHtml(other)}">${escapeHtml(otherName || other)}</button> ${status(e.status)}
        ${ev ? `<p class="quote small">${escapeHtml(ev.quote)}</p>${citationButton(evidenceLabel(ev), ev.content_unit_id, ev.quote)}` : ""}</li>`;
    }).join("")}</ul>` : `<p class="muted">None</p>`;
    return `<h2>${escapeHtml(c.name)}</h2>${Types.badge(c.type)} ${status(c.status)}
      <span class="muted small"> ${c.documents.length} document(s), ${c.degree} connection(s)</span>
      ${c.description ? `<h3>Description</h3><p>${escapeHtml(c.description)}</p>` : ""}
      <h3>Aliases</h3>${aliases}
      <h3>Claims</h3>${claimGroups(c.claims)}
      <h3>Relationships (${c.relationships.length})</h3>${edges}
      <h3>Evidence</h3>${evidenceList(c.evidence)}`;
  }

  // ---------------------------------------------------------------- overview
  async function loadOverview() {
    const box = $("#overview");
    box.innerHTML = `<p class="muted">Loading…</p>`;
    try {
      const [ctx, diag, activity] = await Promise.all([
        api(`/api/projects/${encodeURIComponent(currentProject)}/context`),
        api(`/api/projects/${encodeURIComponent(currentProject)}/diagnostics`),
        api("/api/agents/activity?limit=15")]);
      const counts = (byStatus) => Object.entries(byStatus || {}).map(([k, v]) => `${status(k)} ${v}`).join(" ") || `<span class="muted">none</span>`;
      const kg = ctx.status.concept_first || {};
      box.innerHTML = `
        <div class="cards">
          <div class="card"><strong>${ctx.documents.length}</strong><span>documents</span></div>
          <div class="card"><strong>${ctx.concepts}</strong><span>concepts (${escapeHtml(ctx.graph_source.replace("_", "-"))})</span></div>
          <div class="card"><strong>${ctx.relationships}</strong><span>relationships</span></div>
          <div class="card"><strong>${ctx.claims}</strong><span>claims</span></div>
          <div class="card ${ctx.conflicts ? "warn" : ""}"><strong>${ctx.conflicts}</strong><span>conflicts</span></div>
          <div class="card ${ctx.review_queue ? "warn" : ""}"><strong>${ctx.review_queue}</strong><span>waiting for review</span></div>
        </div>
        ${diag.problems.length ? `<div class="notice">${diag.problems.map(escapeHtml).join("<br>")}</div>` : ""}
        <div class="two-col">
          <div class="answer-card"><h2>Central concepts</h2><div class="chips">${ctx.central_concepts.map((c) =>
            `<button type="button" class="chip" data-concept="${escapeHtml(c.id)}">${Types.dot(c.type)}${escapeHtml(c.name)}</button>`).join("") || '<span class="muted">None yet</span>'}</div>
            <h3>Governed knowledge</h3>
            <p>Concepts: ${counts(kg.concept)}</p><p>Relationships: ${counts(kg.relationship)}</p><p>Claims: ${counts(kg.claim)}</p>
            <h3>Documents</h3><p>${Object.entries(diag.document_kinds).map(([k, v]) => `${v} ${escapeHtml(k)}`).join(", ") || "none"}</p>
            <p class="muted small">Concept extraction: ${diag.concept_first.extracted} of ${diag.concept_first.chunks} chunks. Semantic index: ${diag.semantic_index_ready ? "ready" : "not built"}.</p></div>
          <div class="answer-card"><h2>Recent decisions</h2>${historyList(ctx.recent_activity)}
            <h2>Agent activity</h2>${activity.length ? `<ul class="list small">${activity.map((a) =>
              `<li><strong>${escapeHtml(a.agent)}</strong> ${escapeHtml(a.tool)} <span class="chip status-chip status-${a.outcome === "ok" ? "verified" : "rejected"}">${escapeHtml(a.outcome)}</span>
               <span class="muted">${escapeHtml(String(a.at).slice(0, 19))} · ${a.ms} ms</span></li>`).join("")}</ul>` : '<p class="muted">No agent has called the MCP server in this project.</p>'}</div>
        </div>`;
    } catch (e) { box.innerHTML = `<p class="muted">Could not load the overview: ${escapeHtml(e.message)}</p>`; }
  }

  function historyList(entries) {
    if (!entries || !entries.length) return `<p class="muted">No decisions yet.</p>`;
    return `<ul class="list small">${entries.map((h) => `<li><strong>${escapeHtml(h.actor)}</strong> ${escapeHtml(h.action.replace("_", " "))}
      ${h.details && h.details.decision ? escapeHtml(h.details.decision) : ""} ${h.details && h.details.names ? escapeHtml(h.details.names.join(" / ")) : ""}
      <span class="muted">${escapeHtml(String(h.at).slice(0, 19))}</span>${h.details && h.details.note ? `<br><span class="muted">${escapeHtml(h.details.note)}</span>` : ""}</li>`).join("")}</ul>`;
  }

  // ---------------------------------------------------------------- review
  async function loadReview() {
    const box = $("#review");
    box.innerHTML = `<p class="muted">Loading…</p>`;
    try {
      const [queue, merges, proposals, history] = await Promise.all([
        api("/api/knowledge/review"), api("/api/knowledge/merges"), api("/api/proposals"), api("/api/history?limit=30")]);
      box.innerHTML = `
        <div class="answer-card"><h2>Waiting for review (${queue.length})</h2>
          ${queue.length ? `<ul class="list review-list">${queue.map((item) => `<li data-item="${escapeHtml(item.id)}">
            <p><span class="chip">${escapeHtml(item.item_type)}</span> <strong>${escapeHtml(item.label)}</strong></p>
            <p class="muted small">Checks not passed: ${escapeHtml(item.failed_checks.join(", ") || "none")}</p>
            ${item.possible_duplicates.length ? `<p class="small">May be the same as: ${item.possible_duplicates.map((d) =>
              `<button type="button" class="small" data-merge="${escapeHtml(d)}" data-decision="same">Same concept</button>
               <button type="button" class="small" data-merge="${escapeHtml(d)}" data-decision="different">Keep apart</button>
               <code>${escapeHtml(d.slice(0, 22))}</code>`).join(" ")}</p>` : ""}
            <div class="row-actions"><button type="button" class="small" data-evidence="1">Evidence</button>
              <button type="button" class="small primary" data-decide="accept">Accept</button>
              <button type="button" class="small danger" data-decide="reject">Reject</button></div>
            <div class="evidence-box"></div></li>`).join("")}</ul>` : `<p class="muted">Nothing is waiting.</p>`}</div>
        <div class="two-col">
          <div class="answer-card"><h2>Merge decisions (${merges.length})</h2>${merges.length ? `<ul class="list small">${merges.map((m) =>
            `<li>${escapeHtml(m.decision === "same" ? "Merged" : "Kept apart")}: ${escapeHtml(m.keys.join(" / "))} by ${escapeHtml(m.reviewer)}
             <button type="button" class="small" data-revert="${escapeHtml(m.id)}">Revert</button></li>`).join("")}</ul>` : '<p class="muted">No reviewer merge decisions.</p>'}
            <h2>Proposals (${proposals.length})</h2>${proposals.length ? `<ul class="list small">${proposals.map((p) =>
            `<li><strong>${escapeHtml(p.agent)}</strong> ${escapeHtml(p.kind)}: ${escapeHtml(p.data.name || p.data.statement || [p.data.source, p.data.type, p.data.target].join(" "))}
             ${status(p.status === "submitted" ? "candidate" : "rejected")} ${p.issues.length ? `<span class="muted">${escapeHtml(p.issues.join("; "))}</span>` : ""}</li>`).join("")}</ul>` : '<p class="muted">No proposals.</p>'}</div>
          <div class="answer-card"><h2>Decision history</h2>${historyList(history)}</div>
        </div>`;
    } catch (e) { box.innerHTML = `<p class="muted">Could not load the review queue: ${escapeHtml(e.message)}</p>`; }
  }

  async function reviewAction(event) {
    const li = event.target.closest("li[data-item]");
    const button = event.target.closest("button");
    if (!button) return;
    try {
      if (button.dataset.revert) {
        const name = reviewer(); if (!name) return;
        button.disabled = true;
        await post(`/api/knowledge/merges/${encodeURIComponent(button.dataset.revert)}/revert`, { reviewer: name });
        return loadReview();
      }
      if (!li) return;
      const item = li.dataset.item;
      if (button.dataset.evidence) {
        const box = li.querySelector(".evidence-box");
        const data = await api(`/api/knowledge/items/${encodeURIComponent(item)}/evidence`);
        box.innerHTML = evidenceList(data.evidence);
        return;
      }
      const name = reviewer(); if (!name) return;
      if (button.dataset.decide) {
        const note = window.prompt(`Note for "${button.dataset.decide}" (optional):`) ?? "";
        button.disabled = true;
        await post(`/api/knowledge/${encodeURIComponent(item)}/adjudicate`, { reviewer: name, decision: button.dataset.decide, note });
      } else if (button.dataset.merge) {
        button.disabled = true;
        await post("/api/knowledge/merges", { a: item, b: button.dataset.merge, decision: button.dataset.decision, reviewer: name });
      }
      loadReview();
    } catch (e) { window.alert(`Not saved: ${e.message}`); button.disabled = false; }
  }

  function setup() {
    $("#review").addEventListener("click", reviewAction);
    $("#refresh-overview").addEventListener("click", loadOverview);
  }

  return { setup, loadOverview, loadReview, renderResearch, renderHybrid, renderKnowledgeConcept };
})();
