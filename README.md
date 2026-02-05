# Diagram → Algorithm (macOS/CPU) — FastAPI service (v5)

**База (логика распознавания не менялась):** OpenCV (contours/lines) + Tesseract OCR + rules + graph traversal.  
В v5 добавлены:
- **Нормальный выход**: JSON + текстовый алгоритм (`Шаг | Роль`)
- **Batch** `/v1/parse_many`
- **Оценка** `/v1/evaluate` по `ground_truth.txt`
- **Улучшение качества**: fallback на spatial ordering + фильтр OCR-мусора + дедупликация
- **LLM (опционально)** для улучшения текста/ролей
- **UI** `/ui` для загрузки файла и получения 2 ответов

## Установка (macOS)

### 1) Tesseract (+ языки)
```bash
brew install tesseract tesseract-lang
tesseract --list-langs
```

### 2) venv
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-macos.txt
```

### 3) Запуск
```bash
export TESS_LANG=rus+eng
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Swagger: http://localhost:8080/docs  
UI: http://localhost:8080/ui

## API
- `GET /health`
- `POST /v1/parse` — 1 изображение (`use_llm=true` опционально)
- `POST /v1/parse_many` — несколько изображений
- `POST /v1/evaluate` — несколько изображений + `ground_truth.txt` → метрики
- `POST /v1/render` — (доп.) текст → mermaid (упрощённо)

## Текстовый формат (пример как в эталонах)
```
Шаг | Роль
1. Создание запроса | Инициатор
2. ...             | Координатор
```

## Формат ground_truth.txt (универсальный)
- одна строка = один шаг
- допускаются варианты:
  - `1<TAB>описание`
  - `1. описание`
  - `описание | роль` (роль опционально)
- разные файлы можно разделять блоками: `### filename.png`

## LLM (опционально)
OpenAI‑совместимый backend (OpenAI API или локальный proxy). По умолчанию выключен.

Env:
```bash
export LLM_ENABLED=true
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=...
export LLM_MODEL=gpt-4o-mini
export LLM_TIMEOUT_S=12
```

Использование:
- `POST /v1/parse?use_llm=true`
- UI: галочка "Использовать LLM"
## v6: YOLO BPMN + Swimlanes + Arrow parser (optional)
Добавлен движок `engine=yolo_bpmn`:
- детекция BPMN блоков через YOLO (`model/best.pt`)
- извлечение swimlane (роль) + OCR названий дорожек
- парсинг стрелок (Hough + connections)

### Включение
1) Положите веса: `model/best.pt`
2) Установите зависимости (опционально):
```bash
pip install -r requirements-yolo.txt
```
3) Используйте:
- API: `POST /v1/parse?engine=yolo_bpmn`
- UI: выбрать `yolo_bpmn`
