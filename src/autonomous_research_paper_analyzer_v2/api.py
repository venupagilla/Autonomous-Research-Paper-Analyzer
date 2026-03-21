import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from autonomous_research_paper_analyzer_v2.main import run_analysis


class AnalyzeRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=300)
    max_papers: int = Field(default=30, ge=1, le=200)
    analysis_papers: int = Field(default=20, ge=3, le=60)
    report_words: int = Field(default=800, ge=500, le=1200)


class AsyncAnalyzeResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str


_executor = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_json_block(raw_text: str) -> dict[str, Any]:
    """Parse JSON from plain text or markdown code block safely."""
    if not raw_text:
        return {}

    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _parse_year(published_value: Any) -> int:
    if not published_value:
        return 0

    text = str(published_value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])

    return 0


def _parse_score(score_value: Any) -> int:
    try:
        return int(score_value)
    except (TypeError, ValueError):
        return 0


def _prioritize_recent_papers(papers: list[Any], max_papers: int, current_year: int) -> list[Any]:
    if not papers:
        return []

    recent_start_year = current_year - 3

    def paper_sort_key(paper: Any) -> tuple[int, int]:
        year = _parse_year((paper or {}).get("published"))
        score = _parse_score((paper or {}).get("relevance_score"))
        return (year, score)

    recent_papers: list[Any] = []
    older_papers: list[Any] = []

    for paper in papers:
        year = _parse_year((paper or {}).get("published"))
        if year >= recent_start_year:
            recent_papers.append(paper)
        else:
            older_papers.append(paper)

    recent_papers.sort(key=paper_sort_key, reverse=True)
    older_papers.sort(key=paper_sort_key, reverse=True)

    selected = recent_papers[:max_papers]
    if len(selected) < max_papers:
        selected.extend(older_papers[: max_papers - len(selected)])

    return selected


def _extract_report_arxiv_ids(report_markdown: str) -> set[str]:
    if not report_markdown:
        return set()
    return set(re.findall(r"\b\d{4}\.\d{4,5}\b", report_markdown))


def _is_groq_role_error(exc: Exception) -> bool:
    message = str(exc)
    return "last message role must be 'user'" in message and "GroqException" in message


def _run_analysis_with_fallback(
    topic: str,
    max_papers: int,
    analysis_papers: int,
    report_words: int,
) -> Any:
    """Run analysis and retry once with safer model routing for known Groq role errors."""
    try:
        return run_analysis(
            topic=topic,
            max_papers=max_papers,
            analysis_papers=analysis_papers,
            report_words=report_words,
        )
    except Exception as exc:
        if not _is_groq_role_error(exc):
            raise

        strong_model = os.getenv("STRONG_LLM_MODEL", "").strip()
        if not strong_model:
            raise

        original_small_model = os.getenv("SMALL_LLM_MODEL")
        os.environ["SMALL_LLM_MODEL"] = strong_model
        try:
            return run_analysis(
                topic=topic,
                max_papers=max_papers,
                analysis_papers=analysis_papers,
                report_words=report_words,
            )
        finally:
            if original_small_model is None:
                os.environ.pop("SMALL_LLM_MODEL", None)
            else:
                os.environ["SMALL_LLM_MODEL"] = original_small_model


def _build_analysis_payload(
    crew_output: Any,
    topic: str,
    max_papers: int,
    analysis_papers: int,
    report_words: int,
) -> dict[str, Any]:
    tasks_output = getattr(crew_output, "tasks_output", []) or []

    fetch_output_raw = ""
    if len(tasks_output) > 0:
        fetch_output_raw = getattr(tasks_output[0], "raw", "") or ""

    papers = _extract_json_block(fetch_output_raw).get("papers", [])
    papers = _prioritize_recent_papers(
        papers=papers if isinstance(papers, list) else [],
        max_papers=max_papers,
        current_year=datetime.now().year,
    )
    fetched_ids = {
        str((paper or {}).get("arxiv_id", "")).strip()
        for paper in papers
        if str((paper or {}).get("arxiv_id", "")).strip()
    }

    report_markdown = ""
    if len(tasks_output) > 0:
        report_markdown = getattr(tasks_output[-1], "raw", "") or ""
    if not report_markdown:
        report_markdown = getattr(crew_output, "raw", "") or str(crew_output)

    referenced_ids = _extract_report_arxiv_ids(report_markdown)
    unknown_ids = sorted(referenced_ids - fetched_ids)
    validation_warnings: list[str] = []
    report_is_grounded = True
    if unknown_ids:
        validation_warnings.append(
            "Report referenced arXiv IDs not present in fetched papers: " + ", ".join(unknown_ids)
        )
        report_is_grounded = False

    if len(papers) < 3:
        validation_warnings.append(
            "Limited evidence: fewer than 3 papers were fetched; downstream conclusions may be unstable."
        )

    if len(papers) == 0:
        validation_warnings.append(
            "No papers were fetched in a parseable format. Grounded report rendering is disabled for this run."
        )
        report_markdown = (
            "## Grounding Warning\n"
            "No evidence-backed report can be shown because no papers were fetched in a parseable format.\n\n"
            "Please rerun the query, reduce max papers, or broaden the topic wording."
        )
        report_is_grounded = False

    return {
        "topic": topic,
        "max_papers": max_papers,
        "analysis_papers": analysis_papers,
        "report_words": report_words,
        "papers": papers,
        "report_markdown": report_markdown,
        "report_is_grounded": report_is_grounded,
        "validation_warnings": validation_warnings,
    }


def _run_async_job(job_id: str, payload: AnalyzeRequest) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        if job.get("cancel_requested"):
            job["status"] = "canceled"
            job["progress"] = 100
            job["message"] = "Canceled before execution."
            job["updated_at"] = _utc_now_iso()
            return
        job["status"] = "running"
        job["progress"] = 15
        job["message"] = "Running analysis workflow."
        job["updated_at"] = _utc_now_iso()

    try:
        analysis_papers = min(payload.analysis_papers, payload.max_papers)
        crew_output = _run_analysis_with_fallback(
            topic=payload.topic,
            max_papers=payload.max_papers,
            analysis_papers=analysis_papers,
            report_words=payload.report_words,
        )

        with _jobs_lock:
            finalizing_job = _jobs.get(job_id)
            if finalizing_job and not finalizing_job.get("cancel_requested"):
                finalizing_job["status"] = "finalizing"
                finalizing_job["progress"] = 90
                finalizing_job["message"] = "Finalizing report output."
                finalizing_job["updated_at"] = _utc_now_iso()

        result = _build_analysis_payload(
            crew_output,
            payload.topic,
            payload.max_papers,
            analysis_papers,
            payload.report_words,
        )
    except Exception as exc:
        with _jobs_lock:
            failed_job = _jobs.get(job_id)
            if not failed_job:
                return
            failed_job["status"] = "failed"
            failed_job["progress"] = 100
            failed_job["message"] = str(exc)
            failed_job["updated_at"] = _utc_now_iso()
        return

    with _jobs_lock:
        done_job = _jobs.get(job_id)
        if not done_job:
            return
        if done_job.get("cancel_requested"):
            done_job["status"] = "canceled"
            done_job["progress"] = 100
            done_job["message"] = "Cancellation requested. Background task may have already finished."
            done_job["updated_at"] = _utc_now_iso()
            return
        done_job["status"] = "completed"
        done_job["progress"] = 100
        done_job["message"] = "Analysis completed."
        done_job["result"] = result
        done_job["updated_at"] = _utc_now_iso()


app = FastAPI(title="Autonomous Research Paper Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    try:
        analysis_papers = min(payload.analysis_papers, payload.max_papers)
        crew_output = _run_analysis_with_fallback(
            topic=payload.topic,
            max_papers=payload.max_papers,
            analysis_papers=analysis_papers,
            report_words=payload.report_words,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_analysis_payload(
        crew_output,
        payload.topic,
        payload.max_papers,
        analysis_papers,
        payload.report_words,
    )


@app.post("/api/analyze/async", response_model=AsyncAnalyzeResponse)
def analyze_async(payload: AnalyzeRequest) -> AsyncAnalyzeResponse:
    job_id = str(uuid4())
    now = _utc_now_iso()

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Job queued.",
            "created_at": now,
            "updated_at": now,
            "cancel_requested": False,
            "result": None,
        }

    _executor.submit(_run_async_job, job_id, payload)
    return AsyncAnalyzeResponse(job_id=job_id, status="queued", poll_url=f"/api/jobs/{job_id}")


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job["status"] in {"completed", "failed", "canceled"}:
            return {
                "job_id": job_id,
                "status": job["status"],
                "message": "Job already finalized.",
            }

        job["cancel_requested"] = True
        if job["status"] == "queued":
            job["status"] = "canceled"
            job["progress"] = 100
            job["message"] = "Canceled before execution."
        else:
            job["message"] = "Cancellation requested. Job may finish if already running."
        job["updated_at"] = _utc_now_iso()

        return {
            "job_id": job_id,
            "status": job["status"],
            "message": job["message"],
        }
