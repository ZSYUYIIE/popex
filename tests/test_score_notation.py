"""Local score notation reuses saved MusicXML without changing predictions."""
from test_harmony_api import make_settings
from test_score_corrections import build, client_for
from test_score_percussion import create_drum_job


def test_saved_score_has_local_notation_page(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        page = client.get(f"/jobs/{job_id}/notation")
    assert page.status_code == 200
    assert "Render notation" in page.text
    assert "score-notation.js" in page.text
