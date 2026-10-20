import json
import time

import pytest
from fastapi.testclient import TestClient

from terrasignal.api import create_app
from terrasignal.fetch import NoPairError


def fake_runner(bbox, before, after, out, method="auto", window_days=14):
    if bbox[0] == 10.0:
        raise NoPairError("no same-orbit Sentinel-1 pair")
    if bbox[0] == 11.0:
        raise RuntimeError("boom")
    out.mkdir(parents=True, exist_ok=True)
    (out / "flood.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    s = {"method": method, "files": ["flood.geojson", "summary.json"], "areas_km2": {"open_water_flood": 1.0}}
    (out / "summary.json").write_text(json.dumps(s))
    return s


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(runner=fake_runner, jobs_dir=tmp_path))


def wait(client, jid, timeout=5):
    t = time.time()
    while time.time() - t < timeout:
        j = client.get(f"/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


BODY = {"bbox": [80.52, 16.45, 80.75, 16.65], "before": "2024-08-20", "after": "2024-09-01"}


def test_job_lifecycle_and_files(client):
    r = client.post("/analyze", json=BODY)
    assert r.status_code == 202 and 500 < r.json()["aoi_km2"] < 600
    j = wait(client, r.json()["job_id"])
    assert j["status"] == "done" and j["summary"]["areas_km2"]["open_water_flood"] == 1.0
    f = client.get(j["links"]["flood.geojson"])
    assert f.status_code == 200 and f.json()["type"] == "FeatureCollection"
    assert client.get(f"/jobs/{j['id']}/files/confidence.tif").status_code == 404  # not produced


def test_rejects_bad_requests(client):
    assert client.post("/analyze", json={**BODY, "bbox": [60, 20, 70, 30]}).status_code == 400   # too big
    assert client.post("/analyze", json={**BODY, "bbox": [10, 5, 9, 6]}).status_code == 400      # inverted
    assert client.post("/analyze", json={**BODY, "after": "2024-08-01"}).status_code == 422      # after < before
    assert client.post("/analyze", json={**BODY, "bbox": [1, 2, 3]}).status_code == 422          # 3 numbers


def test_failures_are_reported_not_fatal(client):
    j = wait(client, client.post("/analyze", json={**BODY, "bbox": [10.0, 5, 10.1, 5.1]}).json()["job_id"])
    assert j["status"] == "failed" and j["error_type"] == "no_imagery" and "pair" in j["error"]
    j = wait(client, client.post("/analyze", json={**BODY, "bbox": [11.0, 5, 11.1, 5.1]}).json()["job_id"])
    assert j["error_type"] == "internal" and "trace" not in j
    assert wait(client, client.post("/analyze", json=BODY).json()["job_id"])["status"] == "done"  # worker alive


def test_file_whitelist_and_unknown_job(client):
    jid = wait(client, client.post("/analyze", json=BODY).json()["job_id"])["id"]
    assert client.get(f"/jobs/{jid}/files/..%2F..%2Fsecret.txt").status_code == 404
    assert client.get(f"/jobs/{jid}/files/job.json").status_code == 404
    assert client.get("/jobs/doesnotexist").status_code == 404


def test_cases_and_health(client):
    assert "vijayawada_2024" in client.get("/cases").json()
    assert client.get("/health").json()["status"] == "ok"
