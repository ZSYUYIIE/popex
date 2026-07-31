const form = document.querySelector("#extract-form");
const urlInput = document.querySelector("#video-url");
const submitButton = document.querySelector("#submit-button");
const formMessage = document.querySelector("#form-message");
const jobsContainer = document.querySelector("#jobs");
const refreshButton = document.querySelector("#refresh-button");

const activeStatuses = new Set(["queued", "processing"]);
let pollHandle = null;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setSubmitting(true);
  setMessage("Creating extraction job…", false);

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: urlInput.value.trim() }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "The extraction job could not be created.");
    }
    urlInput.value = "";
    setMessage("Extraction started. This page will update automatically.", false);
    await loadJobs();
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    setSubmitting(false);
  }
});

refreshButton.addEventListener("click", loadJobs);

async function loadJobs() {
  try {
    const response = await fetch("/api/jobs", { cache: "no-store" });
    if (!response.ok) throw new Error("Could not load extraction history.");
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
        <strong>No saved audio yet.</strong>
        <span>Submit your first video URL above.</span>
      </div>`;
    return;
  }

  jobsContainer.innerHTML = jobs.map((job) => {
    const title = job.title || "Waiting for source metadata";
    const sourceHost = safeHost(job.source_url);
    const statusLabel = job.status[0].toUpperCase() + job.status.slice(1);
    const fileLinks = job.files.length
      ? `<div class="files">${job.files.map((file) => `
          <a href="${file.download_url}">
            <span>${escapeHtml(file.name)}</span>
            <small>${formatBytes(file.size_bytes)}</small>
          </a>`).join("")}</div>`
      : "";
    const error = job.error
      ? `<p class="job-error">${escapeHtml(job.error)}</p>`
      : "";
    const detail = [job.uploader, formatDuration(job.duration_seconds), sourceHost]
      .filter(Boolean)
      .map(escapeHtml)
      .join(" · ");

    return `
      <article class="job-card">
        <div class="job-topline">
          <span class="status status-${job.status}">${statusLabel}</span>
          <time>${formatDate(job.created_at)}</time>
        </div>
        <h3>${escapeHtml(title)}</h3>
        <p class="job-detail">${detail || "Source details pending"}</p>
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

function setSubmitting(submitting) {
  submitButton.disabled = submitting;
  submitButton.textContent = submitting ? "Starting…" : "Extract audio";
}

function setMessage(message, isError) {
  formMessage.textContent = message;
  formMessage.classList.toggle("error", isError);
}

function safeHost(value) {
  try { return new URL(value).hostname; } catch { return ""; }
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
