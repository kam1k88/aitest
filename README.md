# Yandex AI Studio: мини-пайплайн (RAG)

Этот репозиторий содержит минимальный, но практичный **RAG-пайплайн** на базе Yandex AI Studio:

- **Retrieval (локально)**: индексирует файлы (BM25 по чанкам), находит топ-N фрагментов.
- **Generation (облако)**: отправляет вопрос + найденный контекст в **Responses API** (`POST /v1/responses`).

Подходит как “скелет” для бота/ассистента с базой знаний без внешних векторных БД.

## Подготовка

1) Установите зависимости:

```bash
python -m pip install -r requirements.txt
```

2) Задайте переменные окружения (PowerShell):

```powershell
$env:YANDEX_API_KEY = "<секретная_часть_API-ключа>"
$env:YANDEX_FOLDER_ID = "<folder_id_каталога_YC>"
```

Важно:

- `YANDEX_API_KEY` — **секретное значение** API-ключа. Никогда не коммитьте его и не отправляйте в чаты/issue.
- `YANDEX_FOLDER_ID` — **реальный** `folder_id` каталога Yandex Cloud (например, строка вида `b1g...`), **не** `"default"`.

Опционально:

- `YANDEX_MODEL` — модель (по умолчанию `aliceai-llm`)
- `YANDEX_BASE_URL` — base URL (по умолчанию `https://ai.api.cloud.yandex.net/v1`)

3) Положите документы в папку `kb/` (любой текст: `.md`, `.txt` и т.п.).

## Как получить `folder_id`

Самый простой способ — через Yandex Cloud CLI (`yc`):

```powershell
yc init
yc resource-manager cloud list
yc resource-manager folder list --cloud-id <CLOUD_ID>
```

Берите значение `ID` из списка folders — это и есть `YANDEX_FOLDER_ID`.

## Запуск

### Быстрая проверка соединения с API (без RAG)

```powershell
python pipeline.py "Привет! Ответь одним словом." --no-rag
```

### RAG-запрос по вашей базе знаний

```bash
python pipeline.py "Какой эндпоинт используется для Responses API?"
```

С явным указанием путей к базе знаний:

```bash
python pipeline.py "Что такое AI-агенты в AI Studio?" --kb kb yandex_ai_studio_docs.md --top-k 8
```

## Параметры

- `--kb ...` — пути к файлам/папкам базы знаний (можно несколько).
- `--top-k N` — сколько чанков подмешивать в контекст.
- `--no-rag` — отключить retrieval и отправить вопрос напрямую (удобно для диагностики).
- `--model ...` — модель (например `aliceai-llm`).

## Как это связано с документацией

В вашей `yandex_ai_studio_docs.md` есть пример вызова Responses API:

- `POST https://ai.api.cloud.yandex.net/v1/responses`
- заголовок `Authorization: Api-Key ...`
- поле `modelUri` вида `gpt://<folder_id>/<model>`

Скрипт делает ровно это, только добавляет retrieval-шаг перед генерацией. Для совместимости он отправляет и `model`, и `modelUri`.

## Troubleshooting

### `Missing env vars...`

В текущей сессии PowerShell не заданы `YANDEX_API_KEY` / `YANDEX_FOLDER_ID`. Выставьте их заново (переменные не сохраняются между разными терминалами/сессиями).

### HTTP 500 от `/v1/responses`

1) Сначала проверьте `--no-rag`:

```powershell
python pipeline.py "ping" --no-rag
```

2) Убедитесь, что `YANDEX_FOLDER_ID` — реальный `b1g...`, а не имя папки/`default`.
3) Если ошибка повторяется — смотрите `traceId`/`request-id` в выводе.


