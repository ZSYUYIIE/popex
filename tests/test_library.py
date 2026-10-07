"""Private song library: arrangements, recording versions, comparison (Cycle 11)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.library import LibraryError, clean_title, clean_version, compare_versions
from app.main import create_app
from test_frontend_score import _run_node
from test_harmony_api import create_job, make_settings
from test_score_percussion import create_drum_job
from test_score_tablature import create_stem_job


def client_for(settings) -> TestClient:
    return TestClient(create_app(settings))


def new_song(client: TestClient, title: str = "Midnight Train", **extra) -> dict:
    response = client.post("/api/compositions", json={"title": title, **extra})
    assert response.status_code == 201, response.text
    return response.json()


def job_json(client: TestClient, job_id: str) -> dict:
    return client.get(f"/api/jobs/{job_id}").json()


# ---------------------------------------------------------------------------
# Validation and migration


@pytest.mark.parametrize("value", ["", "   ", "a\nb", "x" * 161, "<b>Song</b>", 5])
def test_titles_are_bounded_single_line_text(value) -> None:
    with pytest.raises(LibraryError):
        clean_title(value)


def test_titles_collapse_whitespace_and_kinds_are_known() -> None:
    assert clean_title("  Midnight   Train ") == "Midnight Train"
    assert clean_version(" Live  2019 ", "live") == ("Live 2019", "live")
    with pytest.raises(LibraryError):
        clean_version("x", "bootleg")


def test_old_databases_gain_library_tables_and_columns(tmp_path: Path) -> None:
    database = tmp_path / "old.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, source_url TEXT NOT NULL DEFAULT '', "
            "status TEXT NOT NULL DEFAULT 'queued', progress REAL NOT NULL DEFAULT 0, title TEXT, "
            "uploader TEXT, duration_seconds REAL, error TEXT, created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO jobs (id, created_at, updated_at) VALUES ('a', '2026-01-01', '2026-01-01')"
        )
    db.init_database(database)
    db.init_database(database)  # idempotent
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"arrangement_id", "version_label", "version_kind"} <= columns
    assert {"compositions", "arrangements", "score_corrections"} <= tables
    assert db.get_job(database, "a")["arrangement_id"] is None


# ---------------------------------------------------------------------------
# API


def test_song_and_arrangement_lifecycle(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        created = new_song(client, credits="Words and music: A. Writer", arrangementName="Album band")
        song_id, first = created["id"], created["arrangementId"]
        library = created["library"]
        assert [song["title"] for song in library["compositions"]] == ["Midnight Train"]
        assert library["compositions"][0]["credits"] == "Words and music: A. Writer"
        assert [item["name"] for item in library["compositions"][0]["arrangements"]] == ["Album band"]
        assert [item["jobId"] for item in library["unassigned"]] == [job_id]
        assert {kind["id"] for kind in library["versionKinds"]} >= {"studio", "live", "radio_edit"}

        default = new_song(client, title="Second song")
        names = [
            arrangement["name"]
            for song in default["library"]["compositions"]
            if song["id"] == default["id"]
            for arrangement in song["arrangements"]
        ]
        assert names == ["Main arrangement"]

        added = client.post(f"/api/compositions/{song_id}/arrangements", json={"name": "Acoustic"})
        assert added.status_code == 201
        acoustic = added.json()["id"]
        renamed = client.patch(f"/api/arrangements/{acoustic}", json={"name": "Unplugged"})
        assert "Unplugged" in json.dumps(renamed.json())
        retitled = client.patch(f"/api/compositions/{song_id}", json={"title": "Midnight Train (Remastered)"})
        assert "Midnight Train (Remastered)" in json.dumps(retitled.json())

        assigned = client.put(
            f"/api/jobs/{job_id}/version",
            json={"arrangementId": first, "label": "Studio album 2014", "kind": "studio"},
        )
        assert assigned.status_code == 200
        assert assigned.json()["version"]["arrangementId"] == first
        assert "arrangement_id" not in assigned.json() and "version_label" not in assigned.json()

        assert client.delete(f"/api/arrangements/{first}").status_code == 409
        assert client.delete(f"/api/compositions/{song_id}").status_code == 409
        assert client.delete(f"/api/arrangements/{acoustic}").status_code == 200

        ungrouped = client.put(
            f"/api/jobs/{job_id}/version", json={"arrangementId": None, "label": "kept?", "kind": "live"}
        ).json()
        assert ungrouped["version"]["arrangementId"] is None
        assert ungrouped["version"]["label"] is None and ungrouped["version"]["kind"] is None
        assert client.delete(f"/api/compositions/{song_id}").status_code == 200
        remaining = client.get("/api/library").json()
    assert [song["title"] for song in remaining["compositions"]] == ["Second song"]


@pytest.mark.parametrize(
    "method,path,body,expected",
    [
        ("post", "/api/compositions", {"title": ""}, 422),
        ("post", "/api/compositions", {"title": "x", "owner": "me"}, 422),
        ("post", "/api/compositions", {"title": "x", "credits": "a\tb"}, 422),
        ("patch", "/api/compositions/" + "0" * 32, {"title": "x"}, 404),
        ("patch", "/api/compositions/not-an-id", {"title": "x"}, 404),
        ("post", "/api/compositions/" + "0" * 32 + "/arrangements", {"name": "x"}, 404),
        ("delete", "/api/arrangements/" + "0" * 32, None, 404),
        ("get", "/api/compositions/" + "0" * 32 + "/comparison", None, 404),
    ],
)
def test_library_requests_are_validated(tmp_path: Path, method, path, body, expected) -> None:
    settings = make_settings(tmp_path)
    create_job(settings)
    with client_for(settings) as client:
        response = getattr(client, method)(path, **({"json": body} if body is not None else {}))
    assert response.status_code == expected


def test_version_assignment_is_validated(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        arrangement = new_song(client)["arrangementId"]
        bad_kind = client.put(f"/api/jobs/{job_id}/version", json={"arrangementId": arrangement, "kind": "bootleg"})
        missing = client.put(f"/api/jobs/{job_id}/version", json={"arrangementId": "f" * 32})
        malformed = client.put(f"/api/jobs/{job_id}/version", json={"arrangementId": "../x"})
        no_job = client.put(f"/api/jobs/{'0' * 32}/version", json={"arrangementId": arrangement})
        long_label = client.put(
            f"/api/jobs/{job_id}/version", json={"arrangementId": arrangement, "label": "x" * 81}
        )
    assert [r.status_code for r in (bad_kind, missing, malformed, no_job, long_label)] == [422, 422, 422, 404, 422]
    assert db.get_job(settings.database_path, job_id)["arrangement_id"] is None


def test_comparison_keeps_versions_separate(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    studio = create_stem_job(settings)
    live = create_drum_job(settings)
    pending = create_job(settings)
    with client_for(settings) as client:
        song = new_song(client, arrangementName="Album band")
        acoustic = client.post(f"/api/compositions/{song['id']}/arrangements", json={"name": "Live band"}).json()["id"]
        for job_id, arrangement, label, kind in (
            (studio, song["arrangementId"], "Album", "studio"),
            (live, acoustic, "Tokyo 2019", "live"),
            (pending, acoustic, None, "live"),
        ):
            assert client.put(
                f"/api/jobs/{job_id}/version",
                json={"arrangementId": arrangement, "label": label, "kind": kind},
            ).status_code == 200
        for job_id in (studio, live):
            assert client.post(f"/api/jobs/{job_id}/score/construct").status_code == 202
        # A chord correction on the live version only.
        client.post(
            f"/api/jobs/{live}/score/corrections",
            json={"expectedRevision": 0,
                  "operation": {"op": "set_chord", "target": {"measureIndex": 0}, "symbol": "Bb7"}},
        )
        comparison = client.get(f"/api/compositions/{song['id']}/comparison").json()
        live_score = client.get(f"/api/jobs/{live}/score/saved").json()
        studio_score = client.get(f"/api/jobs/{studio}/score/saved").json()

    assert "never merges parts" in comparison["note"]
    by_job = {item["jobId"]: item for item in comparison["versions"]}
    assert [item["jobId"] for item in comparison["versions"]] == [studio, live, pending]
    assert by_job[studio]["arrangementName"] == "Album band" and by_job[studio]["versionKind"] == "studio"
    assert by_job[live]["versionLabel"] == "Tokyo 2019"
    assert by_job[pending]["score"] is None
    assert by_job[studio]["tempoBpm"] is None or isinstance(by_job[studio]["tempoBpm"], float)
    # Each version reports exactly its own score's counts.
    assert by_job[live]["score"]["notes"] == live_score["counts"]["notes"]
    assert by_job[live]["score"]["percussionHits"] == live_score["counts"]["percussionHits"]
    assert by_job[studio]["score"]["notes"] == studio_score["counts"]["notes"]
    assert by_job[studio]["score"]["fingeredTabNotes"] == studio_score["counts"]["fingeredTabNotes"]
    assert by_job[studio]["score"]["percussionHits"] == 0
    assert by_job[live]["score"]["correctionsActive"] == 1
    assert "Bb7" in by_job[live]["score"]["chordsOnlyHere"]
    assert "Bb7" not in by_job[studio]["score"]["chordsUsed"]


def test_compare_versions_lists_chords_unique_to_each_version() -> None:
    summaries = [
        {"score": {"chordsUsed": ["C", "G", "Am"]}},
        {"score": {"chordsUsed": ["C", "G", "F"]}},
        {"score": None},
    ]
    result = compare_versions(summaries)
    assert result[0]["score"]["chordsOnlyHere"] == ["Am"]
    assert result[1]["score"]["chordsOnlyHere"] == ["F"]


# ---------------------------------------------------------------------------
# Library panel


LIBRARY = {
    "versionKinds": [{"id": "studio", "label": "Studio"}, {"id": "live", "label": "Live"}],
    "compositions": [
        {
            "id": "a" * 32, "title": "Midnight Train", "credits": "A. Writer",
            "url": "/api/compositions/" + "a" * 32,
            "comparisonUrl": "/api/compositions/" + "a" * 32 + "/comparison",
            "arrangements": [
                {"id": "b" * 32, "name": "Album band", "url": "/api/arrangements/" + "b" * 32,
                 "versions": [{"jobId": "job1", "title": "Studio take", "label": "Album", "kind": "studio"}]},
                {"id": "c" * 32, "name": "Acoustic", "url": "/api/arrangements/" + "c" * 32, "versions": []},
            ],
        }
    ],
    "unassigned": [{"jobId": "job2", "title": "Rehearsal"}],
}


def test_library_and_version_controls_render() -> None:
    result = _run_node(
        f"""
t.setFetch(async () => ({{ok:true,status:200,json:async()=>({json.dumps(LIBRARY)})}}));
await loadLibrary();
const library=t.getElement("#library").innerHTML;
const job=t.renderJob({{id:"job1",status:"completed",files:[],version:{{arrangementId:"{"b" * 32}",label:"Album",kind:"studio",url:"/api/jobs/job1/version"}}}});
console.log(JSON.stringify({{library,job}}));
"""
    )
    library, job = result["library"], result["job"]
    assert '<h3 id="song-' in library and "Midnight Train</h3>" in library
    assert '<a href="#job-job1-title">Studio take</a> <span class="detail-note">Album · Studio</span>' in library
    assert "Acoustic</strong> <span class=\"detail-note\">no recordings yet</span>" in library
    assert 'aria-label="Delete arrangement Acoustic"' in library
    assert ">Compare versions (1)</button>" in library and 'aria-expanded="false"' in library
    assert "Delete song" not in library  # a song with recordings cannot be deleted
    assert '<label for="arrangement-' in library and ">Add arrangement</button>" in library
    assert "1 recording is not grouped yet." in library
    assert "<summary>Recording version: Midnight Train · Album band · Album · Studio</summary>" in job
    assert '<optgroup label="Midnight Train">' in job
    assert f'<option value="{"b" * 32}" selected>Album band</option>' in job
    assert '<option value="studio" selected>Studio</option>' in job
    assert 'data-action="save-version"' in job


def test_comparison_table_renders_side_by_side() -> None:
    comparison = {
        "note": "Each recording version keeps its own analysis, score, and corrections.",
        "versions": [
            {"jobId": "job1", "title": "Studio take", "versionLabel": "Album", "versionKind": "studio",
             "arrangementName": "Album band", "durationSeconds": 212.0, "tempoBpm": 118.2,
             "tempoConfidence": 0.8, "keySymbol": "A minor", "keyConfidence": 0.6,
             "score": {"stale": False, "beatsPerMeasure": 4, "meterSource": "analysis", "measures": 104,
                       "notes": 480, "chordSymbols": 60, "percussionHits": 900, "fingeredTabNotes": 210,
                       "correctionsActive": 3, "chordsUsed": ["Am", "F", "C"], "chordsOnlyHere": ["F"]}},
            {"jobId": "job3", "title": "Live", "versionLabel": None, "versionKind": "live",
             "arrangementName": "Live band", "durationSeconds": None, "tempoBpm": None,
             "tempoConfidence": None, "keySymbol": None, "keyConfidence": None, "score": None},
        ],
    }
    result = _run_node(
        f"""
t.setFetch(async (url) => {{return {{ok:true,status:200,json:async()=>(url.includes("comparison")?{json.dumps(comparison)}:{json.dumps(LIBRARY)})}}}});
await loadLibrary();
const button={{dataset:{{action:"compare-versions",songId:"{"a" * 32}",url:"/api/compositions/{"a" * 32}/comparison",focusKey:"k"}}}};
button.closest=()=>button;
await t.getListener("#library","click")({{target:button}});
console.log(JSON.stringify({{html:t.getElement("#library").innerHTML}}));
"""
    )
    html = result["html"]
    assert 'aria-expanded="true"' in html and ">Hide comparison</button>" in html
    assert '<table class="comparison-table">' in html and "<caption>" in html
    assert '<th scope="col"><a href="#job-job1-title">Studio take</a><br><span class="detail-note">Album · Studio</span></th>' in html
    assert '<th scope="row">Tempo</th><td>118 BPM <span class="detail-note">(80%)</span></td>' in html
    assert '<td><span class="detail-note">not built</span></td>' in html
    assert "<td>F</td>" in html  # chords only in the studio version
    assert html.count('<span class="detail-note">not available</span>') >= 5
    assert 'role="region" aria-label="Version comparison table" tabindex="0"' in html
