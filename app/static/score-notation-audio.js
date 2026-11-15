/* Recording review is bound to a successfully displayed score, never selectors. */
(() => {
  "use strict";
  function snapshot(value) {
    if (!value || !/^[a-f0-9]{32}$/.test(value.jobId)) return null;
    const prefix = `/api/jobs/${value.jobId}/files/`;
    const file = typeof value.sourceUrl === "string" && value.sourceUrl.startsWith(prefix) ? value.sourceUrl.slice(prefix.length) : "";
    if (!/^[A-Za-z0-9_-][A-Za-z0-9_.-]*\.wav$/.test(file)) return null;
    const start = value.start, end = value.end;
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start || end - start >= 128) return null;
    if (!Array.isArray(value.measures) || value.measures.length > 20000) return null;
    const measures = value.measures.filter(item => Number.isInteger(item?.measureIndex) && item.measureIndex >= start - 1 && item.measureIndex < end);
    if (measures.length !== end - start + 1) return null;
    let previousEnd = -Infinity;
    const copy = [];
    for (let i = 0; i < measures.length; i++) {
      const item = measures[i], a = item.startSeconds, b = item.endSeconds;
      if (item.measureIndex !== start - 1 + i || !Number.isFinite(a) || !Number.isFinite(b)
          || a < -1e6 || b > 1e6 || b <= Math.max(0, a) || a < previousEnd - 1e-6) return null;
      copy.push({measureIndex: item.measureIndex, startSeconds: Math.max(0, a), endSeconds: b});
      previousEnd = b;
    }
    return {sourceUrl: value.sourceUrl, start, end, measures: copy};
  }
  function barAt(measures, seconds) {
    return measures.find(item => seconds >= item.startSeconds && seconds < item.endSeconds) || null;
  }
  class Review {
    constructor(get = id => document.getElementById(id)) {
      this.get = get;
      this.audio = get("notation-audio");
      this.range = null;
      this.score = null;
      this.epoch = 0;
      this.intent = false;
      this.timer = null;
      this.lastCue = "";
      get("notation-play-range").addEventListener("click", () => this.playRange());
      get("notation-play-bar").addEventListener("click", () => this.playRange(Number(get("notation-audio-bar").value)));
      get("notation-audio-stop").addEventListener("click", () => this.stop());
      get("notation-audio-repeat").addEventListener("change", () => {
        if (this.range?.whole) this.range.loop = get("notation-audio-repeat").checked;
      });
      for (const event of ["timeupdate", "seeked", "loadedmetadata", "durationchange", "ratechange"]) this.audio.addEventListener(event, () => this.tick());
      this.audio.addEventListener("play", () => { this.intent = true; this.tick(); });
      this.audio.addEventListener("playing", () => this.tick());
      this.audio.addEventListener("waiting", () => this.clearTimer());
      this.audio.addEventListener("pause", () => { this.intent = false; this.clearTimer(); this.updateCue(); });
      this.audio.addEventListener("ended", () => this.finish());
      this.audio.addEventListener("error", () => {
        this.stop(false);
        this.message("The recording could not be played. Check it from recordings, or render again to retry. Notation and downloads are unchanged.");
      });
    }
    message(value) { this.get("notation-audio-status").textContent = value; }
    clearTimer() { if (this.timer !== null) clearTimeout(this.timer); this.timer = null; }
    stop(announce = true) {
      this.epoch++;
      this.intent = false;
      this.range = null;
      this.clearTimer();
      this.audio.pause();
      this.updateCue();
      if (announce) this.message("Recording paused. Notation and corrections are unchanged.");
    }
    setScore(value, onBar = () => {}) {
      this.stop(false);
      if (this.onBar) this.onBar(null, false);
      this.onBar = onBar;
      this.score = snapshot(value);
      this.get("notation-audio-controls").disabled = !this.score;
      const select = this.get("notation-audio-bar");
      select.replaceChildren();
      this.lastCue = "";
      if (!this.score) {
        this.audio.removeAttribute("src");
        this.audio.load();
        this.audio.hidden = true;
        this.message("Recording review is unavailable for this displayed score. Use the structured review or downloads.");
        this.updateCue();
        return;
      }
      // Reload even the same URL: a failed audio request must remain retryable.
      this.audio.src = this.score.sourceUrl;
      this.audio.load();
      this.audio.hidden = false;
      for (const item of this.score.measures) select.add(new Option(`Bar ${item.measureIndex + 1}`, String(item.measureIndex + 1)));
      select.value = String(this.score.start);
      this.message(`Recording ready for displayed bars ${this.score.start}–${this.score.end}. Playback starts only when you choose Play.`);
      this.updateCue();
    }
    async playRange(bar = null) {
      if (!this.score) return;
      const selected = bar === null ? this.score.measures : this.score.measures.filter(item => item.measureIndex + 1 === bar);
      if (!selected.length) { this.message("Choose a bar within the displayed notation."); return; }
      this.stop(false);
      const start = selected[0].startSeconds, end = selected[selected.length - 1].endSeconds;
      if (Number.isFinite(this.audio.duration) && start >= this.audio.duration) {
        this.message("That bar starts after the recording ends. Check the draft timing in structured review.");
        return;
      }
      const token = this.epoch;
      this.range = {start, end, whole: bar === null, loop: bar === null && this.get("notation-audio-repeat").checked};
      this.intent = true;
      try {
        this.audio.currentTime = start;
        await this.audio.play();
        if (token !== this.epoch) { if (!this.intent) this.audio.pause(); return; }
        if (this.audio.paused) { this.intent = false; this.updateCue(); return; }
        this.message(bar === null ? "Playing shown bars. Stop pauses playback; repeat applies only to this range." : `Playing bar ${bar}. Playback stops at its end.`);
        this.tick();
      } catch (_) {
        if (token !== this.epoch) return;
        this.stop(false);
        this.message("Playback could not start. Use the recording’s Play control or try again. Notation and downloads are unchanged.");
      }
    }
    finish() {
      if (!this.range) { this.updateCue(); return; }
      const again = this.range.whole && this.range.loop;
      this.stop(false);
      if (again) this.playRange();
      else this.message("Requested playback finished. Play again to review the passage.");
    }
    updateCue() {
      const item = this.score ? barAt(this.score.measures, this.audio.currentTime) : null;
      const playing = !this.audio.paused && !this.audio.ended;
      const cue = item ? `${playing ? "Playing" : "Paused at"} bar ${item.measureIndex + 1} · displayed bars ${this.score.start}–${this.score.end}` : "Playback position is outside the displayed bars or in a timing gap.";
      this.get("notation-audio-position").textContent = this.score ? cue : "No displayed bar selected.";
      const key = `${item?.measureIndex ?? "none"}:${playing}`;
      if (key !== this.lastCue) {
        this.lastCue = key;
        if (this.onBar) this.onBar(item?.measureIndex ?? null, playing);
      }
    }
    tick() {
      this.clearTimer();
      if (this.range && this.intent && !this.audio.paused) {
        const end = Number.isFinite(this.audio.duration) ? Math.min(this.range.end, this.audio.duration) : this.range.end;
        if (this.audio.currentTime >= end) { this.finish(); return; }
        const delay = Math.min(1000, Math.max(10, (end - this.audio.currentTime) * 1000 / Math.max(0.1, this.audio.playbackRate || 1)));
        this.timer = setTimeout(() => this.tick(), delay);
      }
      this.updateCue();
    }
  }
  // Cursor failure is cosmetic; it must never interrupt the recording or SVG.
  function measureCue(renderer) {
    let previous = null;
    return (index, playing) => {
      const cursor = renderer?.cursor;
      if (!cursor) return;
      try {
        if (index === null || !playing) { cursor.hide(); previous = null; return; }
        if (index === previous) return;
        cursor.hide();
        if (cursor.Iterator.CurrentMeasureIndex > index) cursor.reset();
        let steps = 0;
        while (!cursor.Iterator.EndReached && cursor.Iterator.CurrentMeasureIndex < index && steps++ < 40000) cursor.next();
        if (cursor.Iterator.CurrentMeasureIndex === index && !cursor.Iterator.EndReached) cursor.show();
        previous = index;
      } catch (_) { try { cursor.hide(); } catch (_) {} }
    };
  }
  globalThis.PopexNotationAudio = {Review, snapshot, barAt, measureCue};
})();
