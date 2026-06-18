from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

import asyncio
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.pipeline.orchestrator import PipelineOrchestrator
from backend.utils.config import get_config, load_config

app = FastAPI(
    title="Tennis Match Analysis API",
    description="AI-powered tennis match analysis pipeline (17 stages)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict[str, Any]] = {}
_config = get_config()


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    stage: str
    stage_num: int = 0
    match_id: str | None = None
    error: str | None = None


class AnalyzeRequest(BaseModel):
    video_path: str
    click_x: float | None = None
    click_y: float | None = None
    max_frames: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/analyze/upload", response_model=JobStatus)
async def upload_and_analyze(
    file: UploadFile = File(...),
    click_x: float | None = Form(None),
    click_y: float | None = Form(None),
    max_frames: int | None = Form(None),
):
    raw_dir = Path(_config["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    click = (click_x, click_y) if click_x is not None and click_y is not None else None
    return _start_job(str(dest), click, max_frames)


@app.post("/analyze/path", response_model=JobStatus)
def analyze_path(req: AnalyzeRequest):
    if not Path(req.video_path).exists():
        raise HTTPException(404, "Video file not found")
    click = (req.click_x, req.click_y) if req.click_x is not None and req.click_y is not None else None
    return _start_job(req.video_path, click, req.max_frames)


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    j = _jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=j["status"],
        progress=j["progress"],
        stage=j["stage"],
        stage_num=j.get("stage_num", 0),
        match_id=j.get("match_id"),
        error=j.get("error"),
    )


@app.get("/jobs/{job_id}/progress")
async def job_progress_sse(job_id: str):
    """SSE stream: {"stage": 6, "label": "Tracking ball", "pct": 35}"""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    async def event_stream():
        last_pct = -1.0
        while True:
            j = _jobs.get(job_id)
            if j is None:
                yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
                break
            pct = j.get("progress", 0.0) * 100 if j.get("progress", 0.0) <= 1 else j.get("progress", 0.0)
            payload = {
                "stage": j.get("stage_num", 0),
                "label": j.get("stage", "queued"),
                "pct": round(pct, 1),
                "status": j["status"],
                "match_id": j.get("match_id"),
            }
            if pct != last_pct or j["status"] in ("completed", "failed"):
                yield f"data: {json.dumps(payload)}\n\n"
                last_pct = pct
            if j["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/results/{match_id}")
def get_results(match_id: str):
    report_dir = Path(_config["paths"]["data_reports"]) / match_id
    json_path = report_dir / "match_stats.json"
    if not json_path.exists():
        raise HTTPException(404, "Results not found")
    import json

    return json.loads(json_path.read_text(encoding="utf-8"))


@app.get("/results/{match_id}/report")
def get_report(match_id: str):
    report_path = Path(_config["paths"]["data_reports"]) / match_id / "report.md"
    if not report_path.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(report_path, media_type="text/markdown")


@app.get("/results/{match_id}/viz/{name}")
def get_viz(match_id: str, name: str):
    viz_dir = Path(_config["paths"]["data_reports"]) / "viz"
    path = viz_dir / name
    if not path.exists():
        raise HTTPException(404, "Visualization not found")
    return FileResponse(path)


def _start_job(
    video_path: str,
    click: tuple[float, float] | None,
    max_frames: int | None,
) -> JobStatus:
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "stage": "queued",
        "stage_num": 0,
        "video_path": video_path,
    }

    def _run():
        _jobs[job_id]["status"] = "running"

        def progress(cur: int, total: int, stage: str):
            _jobs[job_id]["progress"] = cur / max(total, 1)
            _jobs[job_id]["stage"] = stage

        def stage_cb(stage_num: int, label: str, pct: float):
            _jobs[job_id]["stage_num"] = stage_num
            _jobs[job_id]["stage"] = label
            _jobs[job_id]["progress"] = pct / 100.0

        try:
            orch = PipelineOrchestrator(load_config())
            stats = orch.run(
                video_path,
                target_click=click,
                progress_callback=progress,
                stage_callback=stage_cb,
                max_frames=max_frames,
            )
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["progress"] = 1.0
            _jobs[job_id]["stage"] = "done"
            _jobs[job_id]["stage_num"] = 17
            _jobs[job_id]["match_id"] = stats.match_id
        except Exception as exc:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)

    threading.Thread(target=_run, daemon=True).start()
    return JobStatus(job_id=job_id, status="queued", progress=0.0, stage="queued")


_frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
