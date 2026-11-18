"""One action that drafts a whole score: every stage in order, on its own.

A musician should not have to press Separate, Transcribe, Interpret,
Harmonize and Build score in the right order. A draft chain runs those stages
one after another in the background for one recording:

- it waits for source preparation and audio analysis to finish;
- a stage that already finished is reused, unless an earlier stage in the same
  chain ran again, in which case everything after it runs again too;
- optional stages (stem separation, interpretation, harmony) may fail or be
  skipped without stopping the chain; the score is then built from what exists;
- required stages (transcription, score) stop the chain when they fail;
- each stage keeps its own saved results and retry behaviour exactly as when
  it is started by hand.

The chain itself keeps no files. Its progress lives in memory, so after a
restart it simply shows as not running; every stage it finished stays saved.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

OUTCOMES = frozenset({"ran", "reused", "skipped", "failed"})
_POLL_SECONDS = 1.0


@dataclass(frozen=True)
class StepResult:
    outcome: str
    note: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"Unknown step outcome: {self.outcome}")


@dataclass(frozen=True)
class DraftStep:
    """One stage of the chain.

    ``run(job_id, force)`` starts the stage, waits for it to finish, and
    reports what happened. ``force`` is true when an earlier stage in this
    chain ran again, so saved results of this stage are out of date.
    """

    key: str
    label: str
    required: bool
    run: Callable[[str, bool], StepResult]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DraftTracker:
    """Starts draft chains and remembers their progress, one chain per recording."""

    def __init__(self, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self._lock = threading.Lock()
        self._chains: dict[str, dict[str, Any]] = {}
        self._sleep = sleep

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            chain = self._chains.get(job_id)
            return bool(chain and chain["state"] in {"waiting", "running"})

    def claim(self, job_id: str, steps: Sequence[DraftStep]) -> bool:
        """Reserve the chain for a recording. False when one is already running."""
        with self._lock:
            chain = self._chains.get(job_id)
            if chain and chain["state"] in {"waiting", "running"}:
                return False
            self._chains[job_id] = {
                "state": "waiting",
                "step": None,
                "message": "Waiting for the recording to be prepared and analysed.",
                "startedAt": _now(),
                "finishedAt": None,
                "steps": [
                    {"key": step.key, "label": step.label, "outcome": None, "note": None}
                    for step in steps
                ],
            }
            return True

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            chain = self._chains.get(job_id)
            if chain is None:
                return None
            return {
                **{key: value for key, value in chain.items() if key != "steps"},
                "steps": [dict(step) for step in chain["steps"]],
            }

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._chains.pop(job_id, None)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            chain = self._chains.get(job_id)
            if chain is not None:
                chain.update(fields)

    def _record_step(self, job_id: str, index: int, outcome: str | None, note: str | None) -> None:
        with self._lock:
            chain = self._chains.get(job_id)
            if chain is not None:
                chain["steps"][index].update(outcome=outcome, note=note)

    def run(
        self,
        job_id: str,
        steps: Sequence[DraftStep],
        *,
        readiness: Callable[[str], tuple[str, str | None]],
        timeout_seconds: float,
    ) -> None:
        """Run a claimed chain to the end. Never raises.

        ``readiness(job_id)`` returns ``("ready", None)``, ``("waiting", None)``
        or ``("failed", message)`` for source preparation and analysis.
        """
        try:
            self._run(job_id, steps, readiness=readiness, timeout_seconds=timeout_seconds)
        except Exception:  # the chain is a background task; it must not escape
            logging.exception("The draft chain stopped unexpectedly for job %s", job_id)
            self._update(
                job_id,
                state="failed",
                step=None,
                message="Drafting stopped unexpectedly. Saved results are kept; try again.",
                finishedAt=_now(),
            )

    def _run(
        self,
        job_id: str,
        steps: Sequence[DraftStep],
        *,
        readiness: Callable[[str], tuple[str, str | None]],
        timeout_seconds: float,
    ) -> None:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            state, message = readiness(job_id)
            if state == "ready":
                break
            if state == "failed":
                self._update(
                    job_id,
                    state="failed",
                    message=message or "The recording could not be prepared or analysed.",
                    finishedAt=_now(),
                )
                return
            if time.monotonic() >= deadline:
                self._update(
                    job_id,
                    state="failed",
                    message="Preparing and analysing the recording took too long; try again.",
                    finishedAt=_now(),
                )
                return
            self._sleep(_POLL_SECONDS)

        self._update(job_id, state="running")
        force = False
        notes: list[str] = []
        for index, step in enumerate(steps):
            self._update(job_id, step=step.key, message=f"{step.label}…")
            try:
                result = step.run(job_id, force)
            except Exception:
                logging.exception("Draft step %s failed for job %s", step.key, job_id)
                result = StepResult("failed", f"{step.label} stopped unexpectedly.")
            self._record_step(job_id, index, result.outcome, result.note)
            if result.outcome == "ran":
                force = True
            if result.outcome == "failed":
                if step.required:
                    self._update(
                        job_id,
                        state="failed",
                        step=step.key,
                        message=(result.note or f"{step.label} could not be completed.")
                        + " Earlier results are kept; try again.",
                        finishedAt=_now(),
                    )
                    return
            if result.outcome in {"failed", "skipped"} and result.note:
                notes.append(result.note)
        summary = "The draft score is ready to review."
        if notes:
            summary += " " + " ".join(notes)
        self._update(job_id, state="completed", step=None, message=summary, finishedAt=_now())


def wait_for_status(
    get_status: Callable[[], Mapping[str, Any] | None],
    field: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    timeout_seconds: float = 7200.0,
) -> str | None:
    """Poll a job record until ``field`` leaves ``processing``; return the final value."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        record = get_status()
        value = record.get(field) if record else None
        if value != "processing":
            return value
        if time.monotonic() >= deadline:
            return "processing"
        sleep(_POLL_SECONDS)


__all__ = ["DraftStep", "DraftTracker", "OUTCOMES", "StepResult", "wait_for_status"]
