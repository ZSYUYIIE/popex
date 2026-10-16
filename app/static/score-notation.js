/* Render only PopEx's own saved MusicXML, locally and without changing it. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const form = $("notation-controls"), view = $("notation-view"), part = $("notation-part");
  const from = $("notation-from"), through = $("notation-through"), zoom = $("notation-zoom");
  const renderButton = $("notation-render"), retry = $("notation-retry"), paper = $("notation-paper");
  const match = /^\/jobs\/([a-f0-9]{32})\/notation$/.exec(location.pathname);
  let details = null, busy = false, initialized = false, restoreFocus = null;
  const text = (value, fallback = "") => typeof value === "string" ? value.slice(0, 400) : fallback;
  function status(message, error = false) {
    $("notation-status").textContent = message;
    $("notation-status").dataset.error = String(error);
  }
  function lock(value) {
    if (value) {
      const active = document.activeElement;
      restoreFocus = active && (form.contains(active) || active === retry) ? active : null;
    }
    busy = value;
    for (const control of form.elements) control.disabled = value;
    retry.disabled = value;
    form.setAttribute("aria-busy", String(value));
    if (!value) {
      renderButton.disabled = !details;
      if (restoreFocus?.isConnected) {
        const target = restoreFocus.disabled ? retry : restoreFocus;
        target.focus({preventScroll: true});
      }
      restoreFocus = null;
    }
  }
  function endpoint(kind, selectedView = view.value, selectedPart = part.value) {
    const base = `/api/jobs/${match[1]}/score/saved`;
    const query = new URLSearchParams({view: selectedView === "original" ? "original" : "corrected"});
    if (kind === "details") query.set("includeMeasures", "false");
    else {
      query.set("format", kind);
      if (kind === "musicxml" && selectedPart) query.set("part", selectedPart);
    }
    return `${base}${kind === "details" ? "" : "/download"}?${query}`;
  }
  async function response(url) {
    const result = await fetch(url, {cache: "no-store", credentials: "same-origin", mode: "same-origin", redirect: "error", signal: AbortSignal.timeout(30000)});
    if (!result.ok) throw new Error("The saved score could not be loaded. Return to recordings to check its status, or retry.");
    return result;
  }
  async function boundedXml(result) {
    const limit = 8 * 1024 * 1024;
    if (Number(result.headers.get("Content-Length")) > limit || !result.body) throw new Error("This MusicXML is too large for the browser view. Download it instead.");
    const reader = result.body.getReader(), chunks = [];
    let length = 0;
    try {
      while (true) {
        const item = await reader.read();
        if (item.done) break;
        length += item.value.byteLength;
        if (length > limit) { await reader.cancel(); throw new Error("This MusicXML is too large for the browser view. Download it instead."); }
        chunks.push(item.value);
      }
    } finally { reader.releaseLock(); }
    const bytes = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    return new TextDecoder("utf-8", {fatal: true}).decode(bytes);
  }
  function updateDownloads() {
    for (const kind of ["musicxml", "midi"]) {
      const link = $(`notation-${kind}`);
      link.href = endpoint(kind);
      link.hidden = false;
    }
  }
  async function metadata() {
    const [job, saved] = await Promise.all([
      response(`/api/jobs/${match[1]}`).then(r => r.json()),
      response(endpoint("details")).then(r => r.json()),
    ]);
    if (!saved || saved.available !== true || !Number.isInteger(saved.counts?.measures) || saved.counts.measures < 1) throw new Error("Build a saved draft score from recordings before opening notation.");
    details = saved;
    $("recording-title").textContent = text(job.title, "Recording") + " · draft notation";
    const version = [text(job.version?.label), text(job.version?.kind)].filter(Boolean).join(" · ");
    $("recording-identity").textContent = `Recording ${match[1]}${version ? " · " + version : ""} · saved ${text(saved.createdAt, "at an unknown time")} · ${saved.counts.measures} bars`;
    const selection = part.value;
    part.replaceChildren(new Option("Full score", ""));
    for (const item of Array.isArray(saved.exportParts) ? saved.exportParts.slice(0, 64) : []) {
      if (item && typeof item.id === "string" && /^[A-Za-z0-9_-]{1,64}$/.test(item.id)) part.add(new Option(text(item.name, item.id), item.id));
    }
    const partExists = Array.from(part.options).some(option => option.value === selection);
    part.value = partExists ? selection : "";
    from.max = through.max = String(saved.counts.measures);
    if (!initialized) { through.value = String(Math.min(128, saved.counts.measures)); initialized = true; }
    const warnings = $("notation-warning-list");
    warnings.replaceChildren();
    const messages = Array.isArray(saved.warnings) ? saved.warnings.slice(0, 32).map(value => text(value)).filter(Boolean) : [];
    if (saved.stale === true) messages.unshift("This saved draft is out of date. It remains readable; rebuild from recordings if you want the newer evidence.");
    for (const message of messages) { const item = document.createElement("li"); item.textContent = message; warnings.append(item); }
    $("notation-warnings").hidden = messages.length === 0;
    updateDownloads();
    if (!partExists) throw new Error("That part no longer exists in this view. Choose an available part or the full score, then render again.");
  }
  async function reload() {
    if (busy || !match) return;
    lock(true);
    status("Loading saved-score information…");
    try { await metadata(); status("Choose a score view, part and bar range, then Render notation."); }
    catch (error) { details = null; status(text(error.message, "Score information could not be loaded. Retry or return to recordings."), true); }
    finally { lock(false); renderButton.disabled = !details; }
  }
  async function render(event) {
    event.preventDefault();
    if (busy || !match || !form.reportValidity()) return;
    const start = Number(from.value), end = Number(through.value);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start || end - start >= 128) { status("Choose a range of 1 to 128 bars, with the end at or after the start.", true); return; }
    lock(true);
    status("Loading and rendering the saved MusicXML locally…");
    const stage = document.createElement("div");
    stage.className = "notation-stage";
    paper.append(stage);
    try {
      await metadata();
      if (end > details.counts.measures) throw new Error("That bar range no longer exists. Choose bars within the latest saved score.");
      const result = await response(endpoint("musicxml"));
      if (result.headers.get("X-PopEx-Score-Created-At") !== details.createdAt
          || result.headers.get("X-PopEx-Correction-Revision") !== String(details.corrections?.revision)
          || result.headers.get("X-PopEx-Score-View") !== view.value) throw new Error("The score changed while loading. Render again to use the latest saved score and corrections.");
      const xml = await boundedXml(result);
      if (/<!DOCTYPE|<!ENTITY/i.test(xml)) throw new Error("This notation document contains unsupported declarations. Use the MusicXML download instead.");
      const document = new DOMParser().parseFromString(xml, "application/xml");
      if (document.documentElement.localName !== "score-partwise" || document.querySelector("parsererror")) throw new Error("The MusicXML could not be read. Its saved source and downloads are unchanged.");
      if (document.querySelectorAll("note").length > 20000) throw new Error("This score has too many notes for the browser view. Select an individual part or use the complete download.");
      const Renderer = globalThis.opensheetmusicdisplay?.OpenSheetMusicDisplay;
      if (typeof Renderer !== "function") throw new Error("The local notation renderer is unavailable. Reload this page or use the MusicXML download.");
      const renderer = new Renderer(stage, {backend: "svg", autoResize: false, drawTitle: false, disableCursor: true, drawFromMeasureNumber: start, drawUpToMeasureNumber: end});
      renderer.Zoom = Number(zoom.value);
      await renderer.load(document);
      renderer.render();
      if (!stage.querySelector("svg")) throw new Error("No notation was rendered. Try another part or download the MusicXML.");
      stage.className = "notation-canvas";
      paper.replaceChildren(stage);
      const label = part.options[part.selectedIndex]?.textContent || "Full score";
      $("notation-rendered").textContent = `${label} · ${view.value === "original" ? "original prediction" : "with corrections"} · bars ${start}–${end} of ${details.counts.measures} · correction revision ${details.corrections?.revision ?? 0}`;
      status("Notation rendered locally. Review the transcription warnings and check it by ear.");
    } catch (error) {
      stage.remove();
      status(text(error.message, "Notation could not be rendered. Retry or use the MusicXML download.") + " Any previous render remains shown; saved files and corrections are unchanged.", true);
    } finally { lock(false); renderButton.disabled = !details; }
  }
  form.addEventListener("submit", render);
  retry.addEventListener("click", reload);
  view.addEventListener("change", reload);
  part.addEventListener("change", () => { updateDownloads(); status("Selection changed. Press Render notation to update the displayed score."); });
  for (const control of [from, through, zoom]) control.addEventListener("change", () => status("Selection changed. Press Render notation to update the displayed score."));
  if (match) reload();
  else { status("This notation link is not a recording. Return to recordings and open a saved score.", true); renderButton.disabled = retry.disabled = true; }
})();
