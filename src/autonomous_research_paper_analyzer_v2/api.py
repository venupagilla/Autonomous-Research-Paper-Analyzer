import json
import os
import re
from hashlib import sha1
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
MAX_UPLOAD_FILES = 25
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_EXTRACTED_TEXT_CHARS = 22000
MAX_SUMMARY_SOURCE_CHARS = 9000
MAX_SUMMARY_WORDS = 120


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
    return set(re.findall(r"\b(?:\d{4}\.\d{4,5}|upl-[a-f0-9]{10})\b", report_markdown))


def _safe_topic(topic: str | None) -> str:
    value = (topic or "").strip()
    if len(value) >= 3:
        return value
    return "Uploaded research papers"


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _fallback_summary(extracted_text: str) -> str:
    words = extracted_text.split()
    summary_words = words[:80]
    return " ".join(summary_words).strip() or "No summary could be extracted."


def _is_llm_pdf_summary_enabled() -> bool:
    value = os.getenv("ENABLE_LLM_PDF_SUMMARY", "1").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _summarize_with_llm_single_pass(extracted_text: str, topic: str) -> str:
    """Generate a grounded single-pass summary for one paper using an LLM call."""
    try:
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError("litellm is unavailable for LLM summarization.") from exc

    model = os.getenv("PDF_SUMMARY_MODEL", "").strip() or os.getenv("SMALL_LLM_MODEL", "gpt-4o-mini")
    source_text = extracted_text[:MAX_SUMMARY_SOURCE_CHARS]

    response = completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize research papers from extracted PDF text. "
                    "Do not invent claims. Use only provided content."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize this research paper in 90-120 words. "
                    "Include objective, method, key findings, and one limitation. "
                    "Use plain text only and no bullets. "
                    f"Topic context: {topic}\n\n"
                    "Paper text:\n"
                    f"{source_text}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=220,
        timeout=30,
    )

    summary = ""
    choices = (response or {}).get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message", {})
            summary = str(message.get("content", "")).strip()
        else:
            message = getattr(first_choice, "message", None)
            summary = str(getattr(message, "content", "")).strip()

    if not summary:
        raise RuntimeError("LLM returned an empty summary.")

    return _truncate_words(summary, MAX_SUMMARY_WORDS)


def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Missing PDF dependency. Install pypdf to enable uploaded-PDF analysis.") from exc

    reader = PdfReader(BytesIO(file_bytes))
    chunks: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            chunks.append(text)

    extracted = "\n".join(chunks).strip()
    if not extracted:
        raise ValueError("No extractable text found in PDF")
    return extracted[:MAX_EXTRACTED_TEXT_CHARS]


def _normalize_uploaded_text_to_paper(
    filename: str,
    extracted_text: str,
    index: int,
    topic: str,
) -> dict[str, Any]:
    if _is_llm_pdf_summary_enabled():
        try:
            snippet = _summarize_with_llm_single_pass(extracted_text, topic)
        except Exception:
            snippet = _fallback_summary(extracted_text)
    else:
        snippet = _fallback_summary(extracted_text)

    line_candidates = [line.strip() for line in extracted_text.splitlines() if line.strip()]
    title = line_candidates[0][:180] if line_candidates else os.path.splitext(filename)[0]
    if len(title) < 5:
        title = os.path.splitext(filename)[0] or f"Uploaded Paper {index + 1}"

    digest = sha1((filename + extracted_text[:400]).encode("utf-8", errors="ignore")).hexdigest()[:10]
    synthetic_id = f"upl-{digest}"

    return {
        "title": title,
        "arxiv_id": synthetic_id,
        "published": datetime.now().strftime("%Y-%m-%d"),
        "authors": ["uploaded-document"],
        "abstract_snippet": snippet,
        "categories": ["uploaded-pdf"],
        "pdf_url": "",
        "relevance_score": 85,
        "source": "uploaded-pdf",
        "source_filename": filename,
    }


async def _normalize_uploaded_papers(files: list[UploadFile], topic: str) -> list[dict[str, Any]]:
    if not files:
        raise HTTPException(status_code=422, detail="At least one PDF file is required.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=422, detail=f"A maximum of {MAX_UPLOAD_FILES} files is supported.")

    normalized: list[dict[str, Any]] = []

    for index, upload in enumerate(files):
        filename = upload.filename or f"uploaded_{index + 1}.pdf"
        content_type = (upload.content_type or "").lower()
        if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
            raise HTTPException(status_code=422, detail=f"{filename}: only PDF files are supported.")

        file_bytes = await upload.read()
        await upload.close()
        if not file_bytes:
            raise HTTPException(status_code=422, detail=f"{filename}: file is empty.")
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            raise HTTPException(status_code=422, detail=f"{filename}: file exceeds {max_mb}MB limit.")

        try:
            extracted_text = _extract_pdf_text(file_bytes)
            normalized.append(_normalize_uploaded_text_to_paper(filename, extracted_text, index, topic))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"{filename}: failed to parse PDF ({exc}).") from exc

    return normalized


def _is_groq_role_error(exc: Exception) -> bool:
    message = str(exc)
    return "last message role must be 'user'" in message and "GroqException" in message


def _run_analysis_with_fallback(
    topic: str,
    max_papers: int,
    analysis_papers: int,
    report_words: int,
    uploaded_papers: list[dict[str, Any]] | None = None,
) -> Any:
    """Run analysis and retry once with safer model routing for known Groq role errors."""
    try:
        return run_analysis(
            topic=topic,
            max_papers=max_papers,
            analysis_papers=analysis_papers,
            report_words=report_words,
            uploaded_papers=uploaded_papers,
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
                uploaded_papers=uploaded_papers,
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


def _run_async_uploaded_job(
    job_id: str,
    topic: str,
    max_papers: int,
    analysis_papers: int,
    report_words: int,
    uploaded_papers: list[dict[str, Any]],
) -> None:
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
        job["progress"] = 35
        job["message"] = "Uploaded papers normalized. Running analysis workflow."
        job["updated_at"] = _utc_now_iso()

    try:
        bounded_analysis = min(analysis_papers, max_papers)
        crew_output = _run_analysis_with_fallback(
            topic=topic,
            max_papers=max_papers,
            analysis_papers=bounded_analysis,
            report_words=report_words,
            uploaded_papers=uploaded_papers,
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
            topic,
            max_papers,
            bounded_analysis,
            report_words,
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


@app.post("/api/analyze/upload")
async def analyze_uploaded(
    files: list[UploadFile] = File(...),
    topic: str | None = Form(default=None),
    max_papers: int = Form(default=30),
    analysis_papers: int = Form(default=20),
    report_words: int = Form(default=800),
) -> dict[str, Any]:
    if max_papers < 1 or max_papers > 200:
        raise HTTPException(status_code=422, detail="max_papers must be between 1 and 200.")
    if analysis_papers < 3 or analysis_papers > 60:
        raise HTTPException(status_code=422, detail="analysis_papers must be between 3 and 60.")
    if report_words < 500 or report_words > 1200:
        raise HTTPException(status_code=422, detail="report_words must be between 500 and 1200.")

    safe_topic = _safe_topic(topic)
    normalized_papers = await _normalize_uploaded_papers(files, safe_topic)

    try:
        bounded_analysis = min(analysis_papers, max_papers)
        crew_output = _run_analysis_with_fallback(
            topic=safe_topic,
            max_papers=max_papers,
            analysis_papers=bounded_analysis,
            report_words=report_words,
            uploaded_papers=normalized_papers,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_analysis_payload(
        crew_output,
        safe_topic,
        max_papers,
        bounded_analysis,
        report_words,
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


@app.post("/api/analyze/upload/async", response_model=AsyncAnalyzeResponse)
async def analyze_uploaded_async(
    files: list[UploadFile] = File(...),
    topic: str | None = Form(default=None),
    max_papers: int = Form(default=30),
    analysis_papers: int = Form(default=20),
    report_words: int = Form(default=800),
) -> AsyncAnalyzeResponse:
    if max_papers < 1 or max_papers > 200:
        raise HTTPException(status_code=422, detail="max_papers must be between 1 and 200.")
    if analysis_papers < 3 or analysis_papers > 60:
        raise HTTPException(status_code=422, detail="analysis_papers must be between 3 and 60.")
    if report_words < 500 or report_words > 1200:
        raise HTTPException(status_code=422, detail="report_words must be between 500 and 1200.")

    job_id = str(uuid4())
    now = _utc_now_iso()
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 5,
            "message": "Queued. Extracting and normalizing uploaded PDFs.",
            "created_at": now,
            "updated_at": now,
            "cancel_requested": False,
            "result": None,
        }

    safe_topic = _safe_topic(topic)
    normalized_papers = await _normalize_uploaded_papers(files, safe_topic)

    with _jobs_lock:
        job = _jobs.get(job_id)
        if job and not job.get("cancel_requested"):
            job["status"] = "running"
            job["progress"] = 25
            job["message"] = "Uploaded PDFs parsed successfully. Queueing analysis."
            job["updated_at"] = _utc_now_iso()

    _executor.submit(
        _run_async_uploaded_job,
        job_id,
        safe_topic,
        max_papers,
        analysis_papers,
        report_words,
        normalized_papers,
    )
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
