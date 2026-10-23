const state = { documents: [], busy: false, toastTimer: null };
const byId = (id) => document.getElementById(id);

function showNotice(id, message, isError = false) {
  const notice = byId(id);
  notice.textContent = message;
  notice.classList.toggle("is-error", isError);
  notice.hidden = false;
}

function clearNotice(id) {
  const notice = byId(id);
  notice.hidden = true;
  notice.textContent = "";
}

function showToast(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 3200);
}

function setBusy(button, busy, busyText, idleText) {
  button.disabled = busy;
  button.dataset.idleText ??= idleText;
  const label = button.querySelector("span");
  if (label) label.textContent = busy ? busyText : button.dataset.idleText;
}

function setDot(dot, kind) {
  dot.classList.remove("is-ready", "is-warning", "is-error");
  if (kind) dot.classList.add(kind);
}

function formatModel(model) {
  if (!model) return "Not configured";
  return model.replace("sentence-transformers/", "");
}

function renderStatus(status) {
  state.documents = status.documents || [];
  const count = state.documents.length;
  byId("metric-documents").textContent = count;
  byId("metric-chunks").textContent = status.chunk_count ?? 0;
  byId("metric-embedding").textContent = formatModel(status.embedding_model);
  byId("metric-generation").textContent = status.model || "Not configured";
  byId("sidebar-model").textContent = status.index_ready ? "Dense + BM25 indexed" : "Hybrid retrieval";
  byId("nav-document-count").textContent = count;
  byId("library-count").textContent = `${count} ${count === 1 ? "source" : "sources"}`;

  const ready = status.ready;
  const indexReady = status.index_ready;
  const statusText = ready ? "Ready to query" : indexReady ? "API key required" : "Index required";
  byId("sidebar-status-text").textContent = statusText;
  setDot(byId("sidebar-status-dot"), ready ? "is-ready" : "is-warning");
  const pill = byId("readiness-pill");
  pill.querySelector("span:last-child").textContent = ready ? "Ready to query" : statusText;
  setDot(pill.querySelector(".status-dot"), ready ? "is-ready" : "is-warning");
  byId("index-state-text").textContent = indexReady
    ? `${status.chunk_count} chunks · dense and BM25`
    : count ? "Documents need indexing" : "No indexed documents yet";
  byId("library-index-state").textContent = `Index status: ${indexReady ? "ready" : "not built"}`;
  byId("pipeline-model").textContent = `${status.model || "Gemini"} · citation checked`;
  renderDocuments(state.documents);
}

function renderDocuments(documents) {
  const rows = byId("source-rows");
  rows.replaceChildren();
  byId("table-empty").hidden = documents.length > 0;
  for (const document of documents) {
    const row = document.createElement("tr");
    const nameCell = document.createElement("td");
    const name = document.createElement("div");
    name.className = "source-name";
    const mark = document.createElement("span");
    mark.className = "file-mark";
    mark.textContent = document.file_type.toUpperCase();
    const detail = document.createElement("span");
    detail.textContent = document.name;
    const path = document.createElement("span");
    path.className = "source-path";
    path.textContent = document.source_path;
    detail.append(path);
    name.append(mark, detail);
    nameCell.append(name);

    const typeCell = document.createElement("td");
    typeCell.textContent = document.file_type.toUpperCase();
    const statusCell = document.createElement("td");
    const statusLabel = document.createElement("span");
    const loaded = document.status === "loaded";
    statusLabel.className = `status-label${loaded ? " is-loaded" : document.status === "failed" ? " is-failed" : ""}`;
    statusLabel.textContent = loaded ? "Indexed" : document.status === "failed" ? "Failed" : "Not indexed";
    statusCell.append(statusLabel);
    const chunksCell = document.createElement("td");
    chunksCell.textContent = document.chunk_count || "—";
    row.append(nameCell, typeCell, statusCell, chunksCell);
    rows.append(row);
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    if (!response.ok) throw new Error(status.detail || "Could not read workspace status.");
    renderStatus(status);
  } catch (error) {
    byId("sidebar-status-text").textContent = "Service unavailable";
    byId("index-state-text").textContent = error.message;
    showNotice("workspace-notice", error.message, true);
  }
}

function switchView(viewName) {
  const isSources = viewName === "sources";
  byId("workspace-view").hidden = isSources;
  byId("sources-view").hidden = !isSources;
  byId("workspace-view").classList.toggle("is-visible", !isSources);
  byId("sources-view").classList.toggle("is-visible", isSources);
  document.querySelectorAll(".nav-link[data-view]").forEach((link) => {
    link.classList.toggle("is-active", link.dataset.view === viewName);
  });
  byId("breadcrumb-current").textContent = isSources ? "Source library" : "Ask documents";
}

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});

byId("question-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = byId("question-input").value.trim();
  if (!question || state.busy) return;
  clearNotice("workspace-notice");
  const button = byId("ask-button");
  state.busy = true;
  setBusy(button, true, "Searching…", "Ask question");
  byId("answer-content").hidden = true;
  byId("empty-state").hidden = true;
  showNotice("workspace-notice", "Searching dense and keyword indexes, then preparing a cited answer…");
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, diagnostics: true }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The question could not be answered.");
    clearNotice("workspace-notice");
    renderAnswer(payload);
  } catch (error) {
    showNotice("workspace-notice", error.message, true);
    byId("empty-state").hidden = false;
  } finally {
    state.busy = false;
    setBusy(button, false, "Searching…", "Ask question");
  }
});

byId("question-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    byId("question-form").requestSubmit();
  }
});

function renderAnswer(payload) {
  const area = byId("answer-content");
  area.replaceChildren();
  const block = document.createElement("div");
  block.className = "answer-block";
  const meta = document.createElement("div");
  meta.className = "answer-meta";
  const label = document.createElement("span");
  label.textContent = "GROUNDED RESPONSE";
  const time = document.createElement("span");
  time.textContent = "· Hybrid retrieval";
  meta.append(label, time);
  const answer = document.createElement("div");
  answer.className = "answer-body";
  answer.textContent = payload.answer;
  block.append(meta, answer);

  if (payload.sources?.length) {
    const citations = document.createElement("div");
    citations.className = "answer-citations";
    payload.sources.forEach((source) => {
      const chip = document.createElement("span");
      chip.className = "citation-chip";
      chip.textContent = `${source.citation_label} ${source.file_name}${source.page_number ? ` · p. ${source.page_number}` : ""}`;
      chip.title = source.source_path;
      citations.append(chip);
    });
    block.append(citations);
  }

  const heading = document.createElement("div");
  heading.className = "evidence-heading";
  const headingLabel = document.createElement("strong");
  headingLabel.textContent = "Retrieved evidence";
  const headingCount = document.createElement("span");
  headingCount.textContent = `${payload.results.length} passages · dense + BM25`;
  heading.append(headingLabel, headingCount);
  block.append(heading);

  payload.results.forEach((result) => {
    const item = document.createElement("article");
    item.className = "evidence-item";
    const top = document.createElement("div");
    top.className = "evidence-top";
    const source = document.createElement("span");
    source.className = "evidence-source";
    source.textContent = result.file_name;
    const rank = document.createElement("span");
    rank.className = "evidence-rank";
    rank.textContent = `Fused rank ${result.rank}`;
    top.append(source, rank);
    const text = document.createElement("p");
    text.className = "evidence-text";
    text.textContent = result.text;
    const details = document.createElement("div");
    details.className = "evidence-details";
    const location = result.page_number ? `Page ${result.page_number}` : result.page_range ? `Pages ${result.page_range.join("–")}` : result.source_path;
    const ranks = `Dense ${result.dense_rank ?? "—"} · BM25 ${result.bm25_rank ?? "—"}`;
    [location, ranks].forEach((value) => {
      const detail = document.createElement("span");
      detail.textContent = value;
      details.append(detail);
    });
    item.append(top, text, details);
    block.append(item);
  });

  area.append(block);
  area.hidden = false;
}

const zone = byId("upload-zone");
const fileInput = byId("file-input");
byId("add-files-button").addEventListener("click", () => fileInput.click());
byId("browse-files-button").addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});
zone.addEventListener("click", (event) => {
  if (event.target.id !== "browse-files-button") fileInput.click();
});
zone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFiles(fileInput.files);
  fileInput.value = "";
});
for (const eventName of ["dragenter", "dragover"]) {
  zone.addEventListener(eventName, (event) => {
    event.preventDefault();
    zone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  zone.addEventListener(eventName, (event) => {
    event.preventDefault();
    zone.classList.remove("is-dragging");
  });
}
zone.addEventListener("drop", (event) => {
  if (event.dataTransfer.files.length) uploadFiles(event.dataTransfer.files);
});

async function uploadFiles(files) {
  const formData = new FormData();
  [...files].forEach((file) => formData.append("files", file));
  clearNotice("sources-notice");
  showNotice("sources-notice", `Uploading ${files.length} ${files.length === 1 ? "document" : "documents"} and building the indexes… This can take a little while for the first run.`);
  zone.setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/api/ingest", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Document ingestion failed.");
    await refreshStatus();
    const message = payload.failed.length
      ? `Indexed ${payload.document_count} documents with ${payload.failed.length} file error(s).`
      : `Index ready · ${payload.document_count} documents · ${payload.chunk_count} chunks.`;
    showNotice("sources-notice", message, payload.failed.length > 0);
    showToast(message);
  } catch (error) {
    showNotice("sources-notice", error.message, true);
  } finally {
    zone.removeAttribute("aria-busy");
  }
}

refreshStatus();