const urlForm = document.querySelector("#extract-form");
const urlInput = document.querySelector("#video-url");
const submitButton = document.querySelector("#submit-button");
const formMessage = document.querySelector("#form-message");
const uploadForm = document.querySelector("#upload-form");
const fileInput = document.querySelector("#media-file");
const uploadButton = document.querySelector("#upload-button");
const uploadMessage = document.querySelector("#upload-message");
const dropZone = document.querySelector("#drop-zone");
const jobsContainer = document.querySelector("#jobs");
const refreshButton = document.querySelector("#refresh-button");

const activeStatuses = new Set(["queued", "processing"]);
let pollHandle = null;

urlForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setButton(submitButton, true, "Starting…", "Extract audio");
  setMessage(formMessage, "Creating URL import job…", false);

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: urlInput.value.trim() }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "The URL job could not be created.");
    }
    urlInput.value = "";
    setMessage(formMessage, "Import started. This page will update automatically.", false);
    await loadJobs();
  } catch (error) {
    setMessage(formMessage, error.message, true);
  } finally {
    setButton(submitButton, false, "Starting…", "Extract audio");
  }
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setMessage(uploadMessage, "Select a local media file.", true);
    return;
  }
  setButton(uploadButton, true, "Uploading…", "Upload and prepare");
  setMessage(uploadMessage, `Uploading ${file.name}…`, false);

  try {
    const body = new FormData();
    body.append("file", file, file.name);
    const response = await fetch("/api/uploads", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "The local file could not be imported.");
    }
    fileInput.value = "";
    updateDropZone();
    setMessage(uploadMessage, "Upload saved. WAV preparation has started.", false);
    await loadJobs();
  } catch (error) {
    setMessage(uploadMessage, error.message, true);
  } finally {
    setButton(uploadButton, false, "Uploading…", "Upload and prepare");
  }
});

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}
dropZone.addEventListener("drop", (event) => {
  if (event.dataTransfer.files.length) {
    fileInput.files = event.dataTransfer.files;
    updateDropZone();
  }
});
fileInput.addEventListener("change", updateDropZone);
refreshButton.addEventListener("click", loadJobs);

function updateDropZone() {
  const selected = fileInput.files[0];
  const strong = dropZone.querySelector("strong");
  strong.textContent = selected ? selected.name : "Drop a file here";
}

async function loadJobs() {
  try {
    const response = await fetch("/api/jobs", { cache: "no-store" });
    if (!response.ok) throw new Error("Could not load import history.");
    const jobs = await response.json();
    renderJobs(jobs);
    schedulePolling(jobs.some((job) => activeStatuses.has(job.status)));
  } catch (error) {
    jobsContainer.innerHTML = `<p class="empty-state error">${escapeHtml(error.message)}</p>`;
  }
}

function renderJobs(jobs) {
  if (!jobs.length) {
    jobsContainer.innerHTML = `
      <div class="empty-state">
        <strong>No saved media yet.</strong>
        <span>Upload a file or submit a video URL above.</span>
      </div>`;
    return;
  }

  jobsContainer.innerHTML = jobs.map((job) => {
    const title = job.title || job.original_filename || "Waiting for source metadata";
    const sourceLabel = job.source_type === "upload" ? "Local upload" : safeHost(job.source_url);
    const statusLabel = capitalize(job.status);
    const stageLabel = capitalize((job.stage || "queued").replaceAll("_", " "));
    const fileLinks = job.files.length
      ? `<div class="files">${job.files.map((file) => `
          <div class="file-row">
            <div>
              <strong>${escapeHtml(file.label)}</strong>
              <small>${escapeHtml(file.name)} · ${formatBytes(file.size_bytes)}</small>
            </div>
            <div class="file-actions">
              ${file.preview_url ? `<audio controls preload="none" src="${file.preview_url}"></audio>` : ""}
              <a href="${file.download_url}">Download</a>
            </div>
          </div>`).join("")}</div>`
      : "";
    const error = job.error
      ? `<p class="job-error">${escapeHtml(job.error)}</p>`
      : "";
    const detail = [
      job.uploader,
      formatDuration(job.duration_seconds),
      job.source_format,
      job.sample_rate ? `${job.sample_rate} Hz` : "",
      job.channel_count ? `${job.channel_count} channel${job.channel_count === 1 ? "" : "s"}` : "",
      sourceLabel,
    ].filter(Boolean).map(escapeHtml).join(" · ");

    return `
      <article class="job-card">
        <div class="job-topline">
          <span class="status status-${job.status}">${statusLabel}</span>
          <time>${formatDate(job.created_at)}</time>
        </div>
        <h3>${escapeHtml(title)}</h3>
        <p class="job-detail">${detail || "Source details pending"}</p>
        <p class="stage-detail"><strong>${escapeHtml(stageLabel)}</strong> · ${escapeHtml(job.message || "")}</p>
        <div class="progress-track" aria-label="${job.progress}% complete">
          <span style="width: ${Math.max(2, job.progress)}%"></span>
        </div>
        ${error}
        ${fileLinks}
      </article>`;
  }).join("");
}

function schedulePolling(shouldPoll) {
  if (pollHandle) window.clearTimeout(pollHandle);
  pollHandle = shouldPoll ? window.setTimeout(loadJobs, 1500) : null;
}

function setButton(button, submitting, pendingLabel, readyLabel) {
  button.disabled = submitting;
  button.textContent = submitting ? pendingLabel : readyLabel;
}

function setMessage(element, message, isError) {
  element.textContent = message;
  element.classList.toggle("error", isError);
}

function safeHost(value) {
  try { return new URL(value).hostname; } catch { return ""; }
}

function capitalize(value) {
  return value ? value[0].toUpperCase() + value.slice(1) : "";
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "";
  const rounded = Math.round(seconds);
  const minutes = Math.floor(rounded / 60);
  const remainder = String(rounded % 60).padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadJobs();
