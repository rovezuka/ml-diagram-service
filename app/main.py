from __future__ import annotations

import time
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader, select_autoescape

from core.pipeline import parse_image_bytes
from core.output_format import build_output
from core.text_render import steps_to_text
from core.render import text_to_mermaid
from core.eval import parse_ground_truth_txt, evaluate_predictions
from core.llm_client import llm_refine_steps, LLMError
from core.settings import settings

app = FastAPI(title="Diagram → Algorithm (macOS/CPU)", version="6.0.0")

TEMPLATES = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html"])
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    tpl = TEMPLATES.get_template("index.html")
    return tpl.render({
        "llm_enabled": settings.LLM_ENABLED,
        "example_gt": "### file.png\n1. Шаг | Роль\n2. Шаг | Роль\n"
    })


@app.post("/ui/parse", response_class=HTMLResponse)
async def ui_parse(file: UploadFile = File(...), use_llm: bool = Query(False), engine: str = Query("cv")):
    started = time.time()
    _validate_image(file.filename)
    data = await file.read()

    raw = parse_image_bytes(data, hard_timeout_s=20.0, engine=engine)
    raw["output"] = build_output(raw["graph"], raw["algorithm"])
    raw["algorithm_text"] = steps_to_text(raw["output"]["bpmn"]["steps"], with_role_header=True)
    raw["meta"]["latency_ms"] = int((time.time() - started) * 1000)
    raw["meta"]["filename"] = file.filename

    if use_llm:
        try:
            llm = llm_refine_steps(raw)
            if llm:
                raw["llm"] = llm
                raw["llm_text"] = steps_to_text(llm["steps"], with_role_header=True)
        except LLMError as e:
            raw["llm_error"] = str(e)

    tpl = TEMPLATES.get_template("result.html")
    return tpl.render({"raw_json": json.dumps(raw, ensure_ascii=False, indent=2), "raw": raw})


@app.post("/v1/parse")
async def parse(file: UploadFile = File(...), use_llm: bool = Query(False), engine: str = Query("cv")):
    started = time.time()
    _validate_image(file.filename)
    data = await file.read()

    raw = parse_image_bytes(data, hard_timeout_s=20.0, engine=engine)
    raw["output"] = build_output(raw["graph"], raw["algorithm"])
    raw["algorithm_text"] = steps_to_text(raw["output"]["bpmn"]["steps"], with_role_header=True)

    if use_llm:
        try:
            llm = llm_refine_steps(raw)
            if llm:
                raw["llm"] = llm
                raw["llm_text"] = steps_to_text(llm["steps"], with_role_header=True)
        except LLMError as e:
            raw["llm_error"] = str(e)

    raw["meta"]["latency_ms"] = int((time.time() - started) * 1000)
    raw["meta"]["filename"] = file.filename
    return JSONResponse(raw)


@app.post("/v1/parse_many")
async def parse_many(files: List[UploadFile] = File(...), use_llm: bool = Query(False), engine: str = Query("cv")):
    started = time.time()
    results = []
    for f in files:
        if not _is_supported_image(f.filename):
            results.append({"file": f.filename, "error": "unsupported_type"})
            continue
        try:
            data = await f.read()
            raw = parse_image_bytes(data, hard_timeout_s=20.0, engine=engine)
            raw["output"] = build_output(raw["graph"], raw["algorithm"])
            raw["algorithm_text"] = steps_to_text(raw["output"]["bpmn"]["steps"], with_role_header=True)
            raw["meta"]["filename"] = f.filename

            if use_llm:
                try:
                    llm = llm_refine_steps(raw)
                    if llm:
                        raw["llm"] = llm
                        raw["llm_text"] = steps_to_text(llm["steps"], with_role_header=True)
                except LLMError as e:
                    raw["llm_error"] = str(e)

            results.append(raw)
        except Exception as e:
            results.append({"file": f.filename, "error": str(e)})

    return JSONResponse({
        "meta": {"count": len(results), "latency_ms": int((time.time() - started) * 1000)},
        "results": results
    })


class RenderRequest(BaseModel):
    text: str


@app.post("/v1/render")
def render(req: RenderRequest):
    return {"mermaid": text_to_mermaid(req.text)}


@app.post("/v1/evaluate")
async def evaluate(files: List[UploadFile] = File(...), ground_truth: UploadFile = File(...), engine: str = "cv"):
    if not (ground_truth.filename or "").lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="ground_truth must be .txt")

    gt_text = (await ground_truth.read()).decode("utf-8", errors="ignore")
    gt_map = parse_ground_truth_txt(gt_text)

    preds_map = {}
    for f in files:
        if not _is_supported_image(f.filename):
            continue
        data = await f.read()
        raw = parse_image_bytes(data, hard_timeout_s=20.0, engine=engine)
        out = build_output(raw["graph"], raw["algorithm"])
        preds_map[f.filename] = out["bpmn"]["steps"]

    report = evaluate_predictions(preds_map, gt_map)
    return JSONResponse(report)


def _is_supported_image(name: str) -> bool:
    n = (name or "").lower()
    return n.endswith((".png", ".jpg", ".jpeg", ".webp"))


def _validate_image(name: str):
    if not _is_supported_image(name):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PNG/JPG/WEBP.")
