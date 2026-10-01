"""Interview Intelligence Generator - single-page Flask app.

Run:  OPENAI_API_KEY=sk-...  python app.py   ->  http://localhost:5000
The API key stays on the server; the browser only uploads files and receives the PDF/ZIP.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from pipeline.analysis import Inputs, run
from pipeline.extract import ExtractionError, extract_text
from pipeline.llm import LLMError, OpenAIClient
from pipeline.package import build_all

try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
JOB_TTL_SECONDS = 24 * 3600

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "80")) * 1024 * 1024

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _log(msg: str) -> None:
    """Logging must never break a job."""
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _set(job_id: str, **kw):
    with LOCK:
        JOBS[job_id].update(kw)


def _cleanup_old_jobs():
    now = time.time()
    for f in (OUTPUT / "_cache").glob("*.json") if (OUTPUT / "_cache").exists() else []:
        if now - f.stat().st_mtime > JOB_TTL_SECONDS:
            f.unlink(missing_ok=True)
    for d in OUTPUT.iterdir():
        if d.name == "_cache":
            continue
        if d.is_dir() and now - d.stat().st_mtime > JOB_TTL_SECONDS:
            shutil.rmtree(d, ignore_errors=True)
            JOBS.pop(d.name, None)


def _read_upload(storage) -> tuple[str, str]:
    name = os.path.basename(storage.filename or "file")
    return name, extract_text(name, storage.read())


def _worker(job_id: str, inputs: Inputs):
    started = time.time()
    _log(f"[job {job_id[:8]}] started: {inputs.company} / {inputs.designation} / {inputs.target_round}, "
          f"{len(inputs.transcripts)} transcript file(s), JD={'yes' if inputs.jd else 'no'}, "
          f"resume={'yes' if inputs.resume else 'no'}")
    try:
        llm = OpenAIClient()
        packet = run(inputs, llm, progress=lambda m: _set(job_id, stage=m), cache_dir=OUTPUT / "_cache")
        t_model = time.time() - started
        _set(job_id, stage="Building and checking the one-page PDF")
        out = build_all(packet, OUTPUT / job_id)
        calls = len(getattr(llm, "used_models", []))
        _log(f"[job {job_id[:8]}] done in {time.time() - started:.1f}s (analysis {t_model:.1f}s, "
              f"PDF {time.time() - started - t_model:.1f}s), API calls: {calls}, model: "
              f"{packet['meta'].get('model')}, checks passed: "
              f"{sum(c['passed'] for c in out['checks'])}/{len(out['checks'])}")
        _set(job_id, status="done", stage="Done", result={
            "pdf": out["pdf"], "zip": out["zip"], "checks": out["checks"], "fixes": out["fixes"],
            "interviews": packet["n"], "files": len(packet["meta"]["transcript_files"]),
            "duplicates": len(packet["meta"]["duplicates"]),
            "rejected": packet["meta"]["rejected_files"],
        })
    except (LLMError, ValueError, ExtractionError) as e:
        _log(f"[job {job_id[:8]}] FAILED after {time.time() - started:.1f}s: {e}")
        _set(job_id, status="error", error=str(e))
    except BaseException as e:  # pragma: no cover - a job must never stay "running" forever
        _log(f"[job {job_id[:8]}] CRASHED after {time.time() - started:.1f}s: {e!r}")
        try:
            traceback.print_exc()
        except Exception:
            pass
        _set(job_id, status="error", error=f"Unexpected error: {e}")


@app.get("/")
def index():
    return render_template("index.html", key_ok=bool(os.environ.get("OPENAI_API_KEY")))


@app.get("/fonts/<path:name>")
def fonts(name):
    if not re.fullmatch(r"Inter-[A-Za-z]+\.ttf", name):
        abort(404)
    return send_from_directory(ROOT / "fonts", name, max_age=86400)


@app.post("/api/generate")
def generate():
    _cleanup_old_jobs()
    company = (request.form.get("company") or "").strip()
    designation = (request.form.get("designation") or "").strip()
    target_round = (request.form.get("target_round") or "").strip()
    dates = (request.form.get("interview_dates") or "").strip()
    missing = [n for n, v in (("Company", company), ("Designation", designation),
                              ("Target round", target_round)) if not v]
    if missing:
        return jsonify(error=f"Fill in: {', '.join(missing)}."), 400

    # Transcripts are optional. Without them the packet is role-based (designation, round, JD, resume).
    uploads = [f for f in request.files.getlist("transcripts") if f and f.filename]
    transcripts, rejected = [], []
    for f in uploads:
        try:
            transcripts.append(_read_upload(f))
        except ExtractionError as e:
            rejected.append((os.path.basename(f.filename), str(e)))
    if uploads and not transcripts:
        return jsonify(error="None of the transcripts could be read. " +
                       "; ".join(f"{n}: {r}" for n, r in rejected) +
                       " Remove them to generate a role-based packet instead."), 400

    def optional(field):
        f = request.files.get(field)
        if not f or not f.filename:
            return None
        try:
            return _read_upload(f)
        except ExtractionError as e:
            raise ValueError(f"{field.upper() if field == 'jd' else 'Resume'} file could not be read: {e}")

    try:
        jd_file, resume = optional("jd"), optional("resume")
    except ValueError as e:
        return jsonify(error=str(e)), 400

    # JD: pasted text, an uploaded file, or both (combined).
    jd_text = (request.form.get("jd_text") or "").strip()
    if jd_file and jd_text:
        jd = (f"{jd_file[0]} + pasted text", f"{jd_file[1]}\n\n--- pasted JD text ---\n{jd_text}")
    elif jd_file:
        jd = jd_file
    elif jd_text:
        jd = ("pasted text", jd_text)
    else:
        jd = None

    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify(error="The server has no OPENAI_API_KEY set. Add it and restart the app."), 500

    job_id = uuid.uuid4().hex
    (OUTPUT / job_id).mkdir(parents=True)
    with LOCK:
        JOBS[job_id] = {"status": "running", "stage": "Reading files", "error": None, "result": None}
    inputs = Inputs(company, designation, target_round, transcripts, jd, resume, dates, rejected)
    threading.Thread(target=_worker, args=(job_id, inputs), daemon=True).start()
    return jsonify(job_id=job_id)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify(error="Job not found. Generate again."), 404
        return jsonify(job)


@app.get("/files/<job_id>/<path:name>")
def files(job_id, name):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id) or "/" in name or name.startswith("."):
        abort(404)
    as_download = request.args.get("download") == "1"
    return send_from_directory(OUTPUT / job_id, name, as_attachment=as_download,
                               download_name=name if as_download else None)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Upload is too large. Remove some files or raise MAX_UPLOAD_MB."), 413


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. Generation will fail until it is.")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")),
            threaded=True, debug=False)
