from __future__ import annotations
import json
import requests
from core.settings import settings

class LLMError(RuntimeError):
    pass

def llm_refine_steps(raw: dict) -> dict | None:
    if not settings.LLM_ENABLED:
        return None
    if not settings.LLM_API_KEY:
        raise LLMError("LLM_ENABLED=true but LLM_API_KEY is empty.")

    url = settings.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.LLM_API_KEY}", "Content-Type": "application/json"}

    graph = raw.get("graph", {})
    algo = raw.get("algorithm", {})
    output = raw.get("output", {})

    payload = {
        "model": settings.LLM_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты помощник, который превращает распознанную диаграмму процесса в аккуратный список шагов.\n"
                    "Очисти OCR-мусор, нормализуй формулировки, убери дубликаты.\n"
                    "Верни JSON строго в формате:\n"
                    "{\n"
                    '  \"steps\": [{\"action\": \"...\", \"role\": \"\"}],\n'
                    '  \"notes\": \"кратко: что было исправлено\"\n'
                    "}\n"
                    "Роль оставляй пустой строкой, если её нельзя уверенно определить.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Вот результат CV/OCR распознавания. На его основе верни очищенный список шагов.\n\n"
                    f"graph.nodes (первые 60): {json.dumps(graph.get('nodes', [])[:60], ensure_ascii=False)}\n"
                    f"graph.edges (первые 120): {json.dumps(graph.get('edges', [])[:120], ensure_ascii=False)}\n"
                    f"algorithm: {json.dumps(algo, ensure_ascii=False)}\n"
                    f"normalized_output_now: {json.dumps(output, ensure_ascii=False)}\n"
                ),
            },
        ],
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=settings.LLM_TIMEOUT_S)
    except requests.RequestException as e:
        raise LLMError(f"LLM request failed: {e}") from e

    if r.status_code >= 400:
        raise LLMError(f"LLM HTTP {r.status_code}: {r.text[:500]}")

    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
        obj = json.loads(content)
        if "steps" not in obj or not isinstance(obj["steps"], list):
            raise ValueError("bad schema")
        for s in obj["steps"]:
            if not isinstance(s, dict) or "action" not in s:
                raise ValueError("bad step schema")
            s.setdefault("role", "")
        obj.setdefault("notes", "")
        return obj
    except Exception as e:
        raise LLMError(f"Could not parse LLM response as JSON object: {e}. Raw: {data}") from e
