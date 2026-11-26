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


def test_version_summary_counts_half_bar_chords_as_chords_used() -> None:
    from app.library import version_summary

    document = {
        "timing": {}, "counts": {},
        "measures": [
            {"chordSymbol": "C", "chordChanges": []},
            {"chordSymbol": None, "chordChanges": [{"beat": 0.0, "symbol": "Am"}, {"beat": 2.0, "symbol": "F"}]},
            {"chordSymbol": "G", "chordChanges": [{"beat": 0.0, "symbol": "E"}, {"beat": 2.0, "symbol": "B"}]},
        ],
    }
    summary = version_summary({"id": "job1"}, document, stale=False, corrections_active=0)
    assert summary["score"]["chordsUsed"] == ["C", "Am", "F", "G"]  # a whole-bar chord wins over a pair


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
    assert '<a href="#job-job1-title">Studio take</a> <span class="version-kind">Album · Studio</span>' in library
    assert '<span class="detail-note">no draft yet</span>' in library
    assert "Acoustic</strong> <span class=\"detail-note\">no recordings yet</span>" in library
    assert 'aria-label="Delete arrangement Acoustic"' in library
    assert ">Compare versions (1)</button>" in library and 'aria-expanded="false"' in library
    assert "Delete song" not in library  # a song with recordings cannot be deleted
    assert '<label for="arrangement-' in library and ">Add arrangement</button>" in library
    assert '<h3 id="unfiled-title">Not in a song yet</h3>' in library
    assert 'data-open-filing="job2">Add to library</a>' in library
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


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Taylor Swift - Shake It Off (Official Video)",
         {"title": "Shake It Off", "credits": "Taylor Swift", "kind": None, "label": None}),
        ("Coldplay - Yellow (Live in Buenos Aires).mp3",
         {"title": "Yellow", "credits": "Coldplay", "kind": "live", "label": "Live in Buenos Aires"}),
        ("Yellow - Acoustic", {"title": "Yellow", "credits": None, "kind": "acoustic", "label": "Acoustic"}),
        ("Artist - Song - Live", {"title": "Song", "credits": "Artist", "kind": "live", "label": "Live"}),
        ("Some_Band_-_Track_(mp3.pm).mp3", {"title": "Track", "credits": "Some Band", "kind": None, "label": None}),
        ("Song (Remix) [feat. Guest]", {"title": "Song", "credits": None, "kind": "remix", "label": "Remix"}),
        ("周杰伦 - 晴天 (Live)",
         {"title": "晴天", "credits": "周杰伦", "kind": "live", "label": "Live"}),
        ("Liveson - Song", {"title": "Song", "credits": "Liveson", "kind": None, "label": None}),
        ("", {"title": None, "credits": None, "kind": None, "label": None}),
        (None, {"title": None, "credits": None, "kind": None, "label": None}),
    ],
)
def test_filing_suggestions_read_only_the_recording_name(name, expected) -> None:
    from app.library import suggest_filing

    suggestion = suggest_filing(name, [])
    assert {key: suggestion[key] for key in expected} == expected
    assert suggestion["matchingSongId"] is None


def test_filing_suggestions_match_existing_songs_loosely() -> None:
    from app.library import song_key, suggest_filing

    songs = [{"id": "a" * 32, "title": "Shake It Off"}, {"id": "b" * 32, "title": "The Yellow Song"}]
    assert suggest_filing("Taylor Swift - shake it off!", songs)["matchingSongId"] == "a" * 32
    assert suggest_filing("Shake It Of", songs)["matchingSongId"] == "a" * 32  # a typo still matches
    assert suggest_filing("Yellow Song (Live)", songs)["matchingSongId"] == "b" * 32  # leading article ignored
    assert suggest_filing("Something Else", songs)["matchingSongId"] is None
    assert song_key("The  Yellow-Song") == song_key("yellow song") == "yellowsong"


def test_unfiled_recordings_carry_a_suggestion_and_file_in_one_step(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    first = create_job(settings)
    second = create_job(settings)
    db.update_job(settings.database_path, first, title="Night Owls - Midnight Train (Live at the Hall)")
    db.update_job(settings.database_path, second, title="Midnight train")
    with client_for(settings) as client:
        library = client.get("/api/library").json()
        entry = next(item for item in library["unassigned"] if item["jobId"] == first)
        assert entry["suggestion"] == {
            "title": "Midnight Train", "credits": "Night Owls", "kind": "live",
            "label": "Live at the Hall", "matchingSongId": None,
        }
        assert entry["fileUrl"] == f"/api/jobs/{first}/library"
        assert entry["notationUrl"] is None and entry["downloadUrls"] is None

        created = client.post(entry["fileUrl"], json={
            "title": "Midnight Train", "credits": "Night Owls", "kind": "live", "label": "Live at the Hall",
        })
        assert created.status_code == 200
        body = created.json()
        assert body["createdSong"] is True
        assert body["job"]["version"]["arrangementId"] == body["arrangementId"]
        assert body["job"]["version"]["kind"] == "live"
        songs = body["library"]["compositions"]
        assert [song["title"] for song in songs] == ["Midnight Train"]
        assert songs[0]["credits"] == "Night Owls"
        assert songs[0]["arrangements"][0]["name"] == "Main arrangement"

        # The second recording's suggestion now points at that song, and filing reuses it.
        library = client.get("/api/library").json()
        entry = next(item for item in library["unassigned"] if item["jobId"] == second)
        assert entry["suggestion"]["matchingSongId"] == body["songId"]
        reused = client.post(entry["fileUrl"], json={"title": "midnight  train!", "kind": "studio"}).json()
        assert reused["createdSong"] is False and reused["songId"] == body["songId"]
        assert reused["arrangementId"] == body["arrangementId"]
        versions = reused["library"]["compositions"][0]["arrangements"][0]["versions"]
        assert {item["jobId"] for item in versions} == {first, second}
        assert reused["library"]["unassigned"] == []

        assert client.post(f"/api/jobs/{'0' * 32}/library", json={"title": "X"}).status_code == 404
        assert client.post(f"/api/jobs/{first}/library", json={"title": "  "}).status_code == 422
        assert client.post(f"/api/jobs/{first}/library", json={"title": "X", "kind": "bootleg"}).status_code == 422
        assert client.post(f"/api/jobs/{first}/library", json={"title": "X", "extra": 1}).status_code == 422


def test_library_rows_link_scored_versions_to_notation_and_downloads(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        assert client.post(f"/api/jobs/{job_id}/score/construct").status_code == 202
        for _ in range(200):
            if job_json(client, job_id)["score"]["status"] in {"completed", "failed"}:
                break
        assert job_json(client, job_id)["score"]["status"] == "completed"
        entry = client.get("/api/library").json()["unassigned"][0]
    assert entry["notationUrl"] == f"/jobs/{job_id}/notation"
    assert entry["downloadUrls"] == {
        "musicxml": f"/api/jobs/{job_id}/score/saved/download?format=musicxml",
        "midi": f"/api/jobs/{job_id}/score/saved/download?format=midi",
    }
    assert isinstance(entry["bars"], int) and entry["bars"] >= 1
    # Short fixture: no sections; its tonal context is too uncertain to name, so the analysis key is shown as such.
    assert entry["sectionForm"] is None
    assert entry["keySource"] in {"score", "analysis", None}


def test_score_summaries_for_library_rows() -> None:
    from app.song_sections import score_section_form
    from app.tonal_context import score_key_name

    tonality = {"keySignature": {"fifths": 1, "mode": "major"}, "primaryCandidate": {"displayName": "G Ionian (major)"}}
    assert score_key_name({"tonality": tonality}) == "G Ionian (major)"
    assert score_key_name({"tonality": {**tonality, "keySignature": None}}) is None
    assert score_key_name({}) is None
    sections = {"sections": [{"label": "A"}, {"label": "B"}, {"label": "A"}]}
    assert score_section_form({"sections": sections}) == "A B A"
    assert score_section_form({"sections": {"sections": []}}) is None and score_section_form({}) is None


def test_filing_form_search_and_rich_rows_render() -> None:
    library = {
        "versionKinds": [{"id": "studio", "label": "Studio"}, {"id": "live", "label": "Live"}],
        "compositions": [
            {"id": "a" * 32, "title": "Midnight Train", "credits": "Night Owls", "url": "/api/compositions/" + "a" * 32,
             "comparisonUrl": "/api/compositions/" + "a" * 32 + "/comparison",
             "arrangements": [{"id": "b" * 32, "name": "Main arrangement", "url": "/api/arrangements/" + "b" * 32,
                               "versions": [{"jobId": "job1", "title": "Studio take", "label": None, "kind": "studio",
                                             "durationSeconds": 212.4, "tempoBpm": 118.2, "keySymbol": "A minor",
                                             "bars": 104, "notationUrl": "/jobs/job1/notation",
                                             "downloadUrls": {"musicxml": "/api/x?format=musicxml",
                                                              "midi": "https://evil.example/m"}}]}]},
            {"id": "c" * 32, "title": "Other Song", "credits": None, "url": "/api/compositions/" + "c" * 32,
             "comparisonUrl": "/api/compositions/" + "c" * 32 + "/comparison", "arrangements": []},
        ],
        "unassigned": [{"jobId": "job2", "title": "Night Owls - Midnight Train (Live)", "fileUrl": "/api/jobs/job2/library",
                        "suggestion": {"title": "Midnight Train", "credits": "Night Owls <b>", "kind": "live",
                                       "label": "Live", "matchingSongId": "a" * 32}}],
    }
    result = _run_node(
        f"""
t.setFetch(async () => ({{ok:true,status:200,json:async()=>({json.dumps(library)})}}));
await loadLibrary();
const all=t.getElement("#library").innerHTML;
const form=t.renderJob({{id:"job2",status:"completed",files:[],version:{{arrangementId:null,label:null,kind:null,url:"/api/jobs/job2/version"}}}});
t.setLibraryQuery("live owls");renderLibrary();const searched=t.getElement("#library").innerHTML;
t.setLibraryQuery("nothing here");renderLibrary();const none=t.getElement("#library").innerHTML;
console.log(JSON.stringify({{all,form,searched,none}}));
"""
    )
    all_html, form = result["all"], result["form"]
    assert '<span class="detail-note library-meta">3:32 · 118 BPM · A minor · 104 bars</span>' in all_html
    assert '<a href="/jobs/job1/notation">Open notation</a>' in all_html
    assert '<a href="/api/x?format=musicxml" download>MusicXML</a>' in all_html
    assert "evil.example" not in all_html
    assert 'data-open-filing="job2">Add to library</a>' in all_html
    assert '</article></li></ul><section class="unfiled"' in all_html  # after the songs, not inside one
    assert '<details class="library-file" data-open-key="file-job2">' in form
    assert "Matches the song “Midnight Train” in your library" in form
    assert 'name="file-title" maxlength="160" required value="Midnight Train" list="file-songs-job2"' in form
    assert 'value="Night Owls &lt;b&gt;"' in form and "<b>" not in form
    assert '<option value="live" selected>Live</option>' in form
    assert 'data-action="file-library" data-job-id="job2" data-url="/api/jobs/job2/library"' in form
    assert "Midnight Train" in result["searched"] and "Other Song" not in result["searched"]
    assert "No song or version matches “nothing here”." in result["none"]


def test_new_recordings_can_be_filed_while_adding() -> None:
    library = {
        "versionKinds": [{"id": "studio", "label": "Studio"}, {"id": "live", "label": "Live"}],
        "compositions": [{"id": "a" * 32, "title": "Midnight <Train>", "credits": None, "arrangements": []}],
        "unassigned": [],
    }
    filed = {"createdSong": False, "songId": "a" * 32, "library": library}
    result = _run_node(
        f"""
const calls=[];
t.setFetch(async (url, options) => {{calls.push([url, options && options.body ? JSON.parse(options.body) : null]);
  return {{ok:true,status:200,json:async()=>(url.includes("/library")&&options&&options.method==="POST"?{json.dumps(filed)}:{json.dumps(library)})}};}});
await loadLibrary();
const kinds=t.getElement("#add-kind").innerHTML, songs=t.getElement("#add-song-list").innerHTML;
const empty=await fileNewRecording({{id:"job9"}});
t.getElement("#add-song").value="  Midnight Train ";
t.getElement("#add-kind").value="live";
t.getElement("#add-label").value="Live at the Hall";
const message=await fileNewRecording({{id:"job9"}});
console.log(JSON.stringify({{kinds,songs,empty,message,calls,cleared:t.getElement("#add-song").value}}));
"""
    )
    assert '<option value="live">Live</option>' in result["kinds"]
    assert '<option value="Midnight &lt;Train&gt;"></option>' in result["songs"]
    assert result["empty"] == ""
    posted = [call for call in result["calls"] if call[1] is not None]
    assert posted == [["/api/jobs/job9/library",
                       {"title": "Midnight Train", "credits": None, "kind": "live", "label": "Live at the Hall"}]]
    assert result["message"] == " Filed in your library beside the other versions of “Midnight Train”."
    assert result["cleared"] == ""


def test_library_refreshes_when_a_drafted_recording_changes() -> None:
    # The library rows show each version's tempo, bars and links from its
    # score; they must update when a draft finishes, not only on reload.
    result = _run_node(
        f"""
let jobs=[{{id:"job1",status:"completed",files:[],analysis:{{status:"completed"}},score:{{status:"processing"}}}}];
let library={json.dumps(LIBRARY)};
const calls=[];
t.setFetch(async (url) => {{calls.push(url);return {{ok:true,status:200,json:async()=>(url.includes("/api/library")?library:jobs)}}}});
await loadLibrary();
await loadJobs();
const baseline=calls.filter(url=>url.includes("/api/library")).length;
await loadJobs();
const unchanged=calls.filter(url=>url.includes("/api/library")).length;
jobs=[{{id:"job1",status:"completed",files:[],analysis:{{status:"completed"}},score:{{status:"completed",createdAt:"2026-01-01T00:00:00Z"}}}}];
await loadJobs();
await new Promise(resolve=>setImmediate(resolve));
const afterDraft=calls.filter(url=>url.includes("/api/library")).length;
library={{...library,compositions:[]}};
await loadLibrary();
const emptyStart=calls.filter(url=>url.includes("/api/library")).length;
jobs=[{{id:"job1",status:"completed",files:[],analysis:{{status:"completed"}},score:{{status:"completed",createdAt:"2026-02-02T00:00:00Z"}}}}];
await loadJobs();
await new Promise(resolve=>setImmediate(resolve));
const emptyAfter=calls.filter(url=>url.includes("/api/library")).length;
console.log(JSON.stringify({{baseline,unchanged,afterDraft,emptyStart,emptyAfter}}));
"""
    )
    assert result["unchanged"] == result["baseline"]
    assert result["afterDraft"] == result["baseline"] + 1
    assert result["emptyAfter"] == result["emptyStart"]  # no songs: nothing to refresh
