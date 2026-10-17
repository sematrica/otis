"""Stage 4 end to end with a real model: a "YouTube" project retrieved by the
real worker with real yt-dlp from a local page (YouTube itself is not
reachable from the development container), transcribed, compared with its
published captions, searched and exported. With ``VERBATIM_TEST_YOUTUBE_URL``
(set in CI) the same is tried against YouTube itself."""

from __future__ import annotations

import os
import time

import pytest

from verbatim_engine.errors import EngineError, InvalidParams
from verbatim_engine.paths import EnginePaths
from verbatim_engine.rpc.service import EngineService
from verbatim_engine.storage import repo

from tests.test_stage4_youtube import media_site  # noqa: F401 - fixture

pytestmark = pytest.mark.model


def _wait(service, job_id, timeout=900):
    deadline = time.monotonic() + timeout
    while repo.get_job(service.db, job_id)["status"] not in ("completed", "failed", "cancelled", "interrupted"):
        assert time.monotonic() < deadline
        time.sleep(0.2)
    return repo.get_job(service.db, job_id)


@pytest.fixture()
def service(data_dir):
    svc = EngineService(EnginePaths.create(data_dir), lambda *_: None)
    svc.start()
    yield svc
    svc.shutdown()


def test_import_from_a_video_page_with_captions(service, fw_model_dir, media_site):  # noqa: F811
    call = service.methods()
    inspected = call["youtube.inspect"]({"url": media_site})
    assert inspected["refusal"] is None and "not part of that file" in inspected["ads_note"]
    options = {"backend": "faster-whisper", "model_path": str(fw_model_dir), "mode": "balanced"}
    with pytest.raises(InvalidParams):  # the user must confirm they may retrieve it
        call["projects.createFromYouTube"]({"url": media_site, "options": options})
    created = call["projects.createFromYouTube"]({
        "url": media_site, "authorized": True, "options": options, "media": "video",
        "captions": [{"language": "en", "kind": "manual"}]})
    project, job = created["project"], created["job"]
    assert job["kind"] == "youtube" and job["stages"][0]["stage"] == "download"
    assert project["source"]["kind"] == "url" and project["source"]["path"] is None
    assert project["source_meta"]["authorization"]["statement"].startswith("I am authorized")
    job = _wait(service, job["id"])
    assert job["status"] == "completed", job["error"]
    stages = {s["stage"]: s for s in job["stages"]}
    assert stages["download"]["status"] == "completed" and "caption track" in stages["download"]["detail"]
    project = repo.get_project(service.db, project["id"])
    size = project["source"]["size_bytes"]
    assert stages["download"]["progress"] == {"done": size, "total": size, "unit": "bytes"}
    media_dir = service.paths.media_dir(project["id"])
    assert project["source"]["path"].startswith(str(media_dir)) and project["has_video"]
    assert project["source_meta"]["retrieved"]["captions"][0]["kind"] == "manual"
    text = " ".join(s["text"] for s in repo.get_transcript(service.db, project["id"])["segments"])
    assert "fellow Americans" in text
    assert job["effective"]["source"]["captions_compared"][0]["language"] == "en"

    # captions are kept as published, separately from the transcript
    tracks = call["captions.list"]({"project_id": project["id"]})["captions"]
    assert [(t["language"], t["kind"], t["cue_count"]) for t in tracks] == [("en", "manual", 2)]
    srt = call["captions.export"]({"project_id": project["id"], "caption_id": tracks[0]["id"], "format": "srt"})
    assert srt["content"].startswith("1\n00:00:01,000 --> 00:00:03,500\nAnd so, my fellow Americans,")
    exported = call["export.render"]({"project_id": project["id"], "format": "json"})["content"]
    assert media_site in exported and "fellow Americans" in exported
    assert call["search.transcripts"]({"query": "americans"})["results"][0]["project_id"] == project["id"]

    # transcribing again uses the retrieved file; deleting the project removes it
    again = service.jobs_start({"project_id": project["id"], "options": options})
    assert _wait(service, again["id"])["status"] == "completed"
    service.projects_delete({"project_id": project["id"]})
    assert not media_dir.exists()


def test_real_youtube_video(service, fw_model_dir):
    """Import from YouTube itself. Opt-in with ``VERBATIM_TEST_YOUTUBE_URL``:
    it needs YouTube to answer this machine, and data-centre addresses are
    often asked to sign in, which Verbatim never does. The video's metadata is
    always checked; its media is retrieved only when YouTube lists it under a
    Creative Commons licence, or when ``VERBATIM_TEST_YOUTUBE_AUTHORIZED=1``
    says the person running the test may retrieve it (their own video, or
    with the rights holder's permission)."""
    url = os.environ.get("VERBATIM_TEST_YOUTUBE_URL")
    if not url:
        pytest.skip("VERBATIM_TEST_YOUTUBE_URL not set")
    call = service.methods()
    unreachable = ("youtube_blocked", "youtube_network")
    try:
        inspected = call["youtube.inspect"]({"url": url})
    except EngineError as exc:
        if exc.code in unreachable:
            pytest.skip(f"YouTube did not serve this machine: {exc.code}: {exc.message}")
        raise
    print({k: inspected[k] for k in ("title", "channel", "duration", "license", "captions", "approx_bytes",
                                     "refusal")})
    assert inspected["refusal"] is None and inspected["duration"]
    licensed = "creative commons" in (inspected["license"] or "").lower()
    if not licensed and os.environ.get("VERBATIM_TEST_YOUTUBE_AUTHORIZED") != "1":
        pytest.skip(f"metadata checked; media not retrieved (licence: {inspected['license']!r})")

    options = {"backend": "faster-whisper", "model_path": str(fw_model_dir), "mode": "fast"}
    captions = [{"language": t["language"], "kind": t["kind"]} for t in inspected["captions"]][:1]
    created = call["projects.createFromYouTube"]({"url": url, "authorized": True, "options": options,
                                                  "media": "audio", "captions": captions})
    job = _wait(service, created["job"]["id"], timeout=1800)
    if job["status"] == "failed" and (job["error"] or {}).get("code") in unreachable:
        pytest.skip(f"YouTube described the video but did not serve its media: {job['error']}")
    assert job["status"] == "completed", job["error"]
    project = repo.get_project(service.db, created["project"]["id"])
    retrieved = project["source_meta"]["retrieved"]
    print({"retrieved": retrieved, "segments": project["segment_count"], "download": job["stages"][0],
           "source": job["effective"].get("source")})
    assert retrieved["has_video"] is False and abs(retrieved["duration"] - inspected["duration"]) <= 2.0
    assert len(call["captions.list"]({"project_id": project["id"]})["captions"]) == len(captions)
