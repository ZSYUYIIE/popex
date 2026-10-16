"""Recording playback follows the timing of the displayed saved score."""
from __future__ import annotations

from test_harmony_api import make_settings
from test_score_corrections import build, client_for, saved, score_file_digest
from test_score_percussion import create_drum_job


def test_notation_can_request_compact_exact_saved_measure_timing(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        digest = score_file_digest(settings, job_id)
        document = saved(client, job_id)
        base = f"/api/jobs/{job_id}/score/saved"
        assert "measureTiming" not in client.get(base).json()
        response = client.get(base + "?includeMeasureTiming=true")
        assert response.status_code == 200
        assert response.json()["measureTiming"] == [
            {key: measure[key] for key in ("measureIndex", "startSeconds", "endSeconds")}
            for measure in document["measures"]
        ]
        assert "measures" not in response.json()
        assert score_file_digest(settings, job_id) == digest


def test_local_notation_has_explicit_recording_review_controls(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        page = client.get(f"/jobs/{job_id}/notation")
    assert page.status_code == 200
    assert 'id="notation-audio"' in page.text
    assert "Play shown bars" in page.text
    assert "Repeat shown bars" in page.text
    assert "score-notation-audio.js?v=" in page.text
    assert "autoplay" not in page.text
