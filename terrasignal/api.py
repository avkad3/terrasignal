"""HTTP API. A run takes from under a minute to several, so analysis is a job:

  POST /analyze            {bbox, before, after, method?} -> 202 {job_id}
  GET  /jobs/{id}          status, and the summary once done
  GET  /jobs/{id}/files/x  flood.tif, confidence.tif, vv_before.tif,
                           vv_after.tif, flood.geojson, summary.json
  GET  /cases              curated case studies
  GET  /health

One worker runs jobs in order: one GPU and 8 GB of RAM fit one scene at a time.

  uvicorn terrasignal.api:app --port 8000
"""
from __future__ import annotations

import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from .aoi import AOI, DEFAULT_MAX_AREA_KM2, AOIError
from .fetch import NoPairError

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "data" / "jobs"
OUTPUT_FILES = {
    "flood.tif": "image/tiff", "confidence.tif": "image/tiff", "vv_before.tif": "image/tiff",
    "vv_after.tif": "image/tiff", "flood.geojson": "application/geo+json", "summary.json": "application/json",
}


class AnalyzeRequest(BaseModel):
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[west, south, east, north] in degrees")
    before: date
    after: date
    method: Literal["auto", "hybrid", "baseline"] = "auto"
    window_days: int = Field(14, ge=1, le=60)

    @model_validator(mode="after")
    def _dates(self):
        if self.before >= self.after:
            raise ValueError("before must be earlier than after")
        if self.after > date.today():
            raise ValueError("after date is in the future")
        return self


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Jobs:
    def __init__(self, root: Path, runner):
        self.root, self.runner = root, runner
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="terrasignal")

    def _save(self, job: dict) -> None:
        d = self.root / job["id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(job, indent=2, default=str))

    def _set(self, jid: str, **kw) -> None:
        with self.lock:
            self.jobs[jid].update(kw)
            self._save(self.jobs[jid])

    def submit(self, req: AnalyzeRequest) -> dict:
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "status": "queued", "request": req.model_dump(mode="json"), "created": _now()}
        with self.lock:
            self.jobs[jid] = job
            self._save(job)
        self.pool.submit(self._run, jid, req)
        return job

    def _run(self, jid: str, req: AnalyzeRequest) -> None:
        self._set(jid, status="running", started=_now())
        try:
            summary = self.runner(req.bbox, req.before, req.after, self.root / jid, method=req.method,
                                  window_days=req.window_days)
            self._set(jid, status="done", finished=_now(), summary=summary)
        except NoPairError as e:
            self._set(jid, status="failed", finished=_now(), error_type="no_imagery", error=str(e))
        except (AOIError, ValueError) as e:
            self._set(jid, status="failed", finished=_now(), error_type="invalid_request", error=str(e))
        except Exception as e:  # keep the worker alive; record what happened
            self._set(jid, status="failed", finished=_now(), error_type="internal",
                      error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=5))

    def get(self, jid: str) -> dict | None:
        with self.lock:
            if jid in self.jobs:
                return dict(self.jobs[jid])
        f = self.root / jid / "job.json"  # jobs from an earlier server run
        if len(jid) == 12 and jid.isalnum() and f.exists():
            return json.loads(f.read_text())
        return None


def _default_runner(bbox, before, after, out, method="auto", window_days=14):
    from .analyze import analyze
    return analyze(bbox, before, after, out, method=method, window_days=window_days)


def create_app(runner=_default_runner, jobs_dir: Path = JOBS_DIR) -> FastAPI:
    app = FastAPI(title="TerraSignal", version="0.1.0",
                  description="Pick any place on Earth and two dates; see what flooded and how confident it is.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
    jobs = Jobs(jobs_dir, runner)
    app.state.jobs = jobs

    @app.get("/health")
    def health():
        return {"status": "ok", "max_area_km2": DEFAULT_MAX_AREA_KM2}

    @app.get("/cases")
    def cases():
        return yaml.safe_load((ROOT / "cases" / "cases.yaml").read_text())

    @app.post("/analyze", status_code=202)
    def analyze(req: AnalyzeRequest):
        try:
            aoi = AOI.from_list(req.bbox)
            aoi.check_size()
        except AOIError as e:
            raise HTTPException(400, str(e))
        job = jobs.submit(req)
        return {"job_id": job["id"], "status": job["status"], "aoi_km2": round(aoi.area_km2, 1),
                "links": {"status": f"/jobs/{job['id']}"}}

    @app.get("/jobs/{jid}")
    def job(jid: str):
        j = jobs.get(jid)
        if j is None:
            raise HTTPException(404, "unknown job")
        j.pop("trace", None)
        if j["status"] == "done":
            j["links"] = {f: f"/jobs/{jid}/files/{f}" for f in j["summary"].get("files", []) if f in OUTPUT_FILES}
        return j

    @app.get("/jobs/{jid}/files/{name}")
    def file(jid: str, name: str):
        if name not in OUTPUT_FILES:  # fixed whitelist: no path traversal
            raise HTTPException(404, "unknown file")
        j = jobs.get(jid)
        if j is None or j["status"] != "done":
            raise HTTPException(404, "job not finished")
        path = jobs_dir / jid / name
        if not path.exists():
            raise HTTPException(404, "file not produced for this job")
        return FileResponse(path, media_type=OUTPUT_FILES[name], filename=name)

    return app


app = create_app()
