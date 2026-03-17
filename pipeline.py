import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import requests


DEFAULT_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
DEFAULT_MODEL = "aliceai-llm"
_FOLDER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{4,63}$", re.IGNORECASE)


def _iter_text_files(paths: Sequence[str]) -> Iterable[str]:
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for name in files:
                    yield os.path.join(root, name)
        else:
            yield p


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    step = max(1, chunk_chars - overlap)
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


_token_re = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)


def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _token_re.findall(s)]


@dataclass(frozen=True)
class DocChunk:
    source: str
    text: str
    tokens: Tuple[str, ...]


class BM25Index:
    def __init__(self, chunks: Sequence[DocChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        self._doc_lens = [len(c.tokens) for c in self.chunks]
        self._avgdl = (sum(self._doc_lens) / len(self._doc_lens)) if self._doc_lens else 0.0

        df = {}
        for c in self.chunks:
            seen = set(c.tokens)
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self._df = df
        self._N = len(self.chunks)

        # IDF with BM25+ style smoothing to avoid negative values on very frequent terms
        self._idf = {
            t: (max(0.0, (self._N - n + 0.5) / (n + 0.5))) for t, n in self._df.items()
        }

        # term frequencies per document (compact dicts)
        self._tfs = []
        for c in self.chunks:
            tf = {}
            for t in c.tokens:
                tf[t] = tf.get(t, 0) + 1
            self._tfs.append(tf)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, DocChunk]]:
        q = _tokenize(query)
        if not q or not self.chunks:
            return []

        scores = []
        for i, c in enumerate(self.chunks):
            dl = self._doc_lens[i]
            denom_base = self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0.0))
            tf = self._tfs[i]
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self._idf.get(term, 0.0)
                s += idf * ((f * (self.k1 + 1.0)) / (f + denom_base))
            if s > 0:
                scores.append((s, c))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[: max(1, top_k)]


def build_index(kb_paths: Sequence[str]) -> BM25Index:
    chunks: List[DocChunk] = []
    for path in _iter_text_files(kb_paths):
        if not os.path.isfile(path):
            continue
        if os.path.getsize(path) == 0:
            continue
        text = _read_text(path)
        for chunk in _chunk_text(text):
            toks = tuple(_tokenize(chunk))
            if toks:
                chunks.append(DocChunk(source=path, text=chunk, tokens=toks))
    return BM25Index(chunks)


def call_yandex_responses(
    *,
    api_key: str,
    folder_id: str,
    model: str,
    input_text: str,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.2,
    max_output_tokens: int = 800,
    timeout_s: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/responses"
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }
    model_ref = f"gpt://{folder_id}/{model}"
    payload = {
        # В документации встречаются оба варианта:
        # - OpenAI-совместимый: "model"
        # - FoundationModels-стиль: "modelUri"
        # Чтобы избежать 500 из-за несовпадения схемы, отправляем оба.
        "model": model_ref,
        "modelUri": model_ref,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "input": input_text,
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
    if not r.ok:
        req_id = (
            r.headers.get("x-request-id")
            or r.headers.get("x-requestid")
            or r.headers.get("x-correlation-id")
        )
        body = r.text.strip()
        hint = (
            "Hint: YANDEX_FOLDER_ID должен быть реальным folder_id каталога Yandex Cloud "
            "(в modelUri он используется как gpt://<folder_id>/<model>). Значение 'default' не подойдет."
        )
        raise RuntimeError(
            f"Yandex AI Studio error: HTTP {r.status_code} for {url}"
            + (f" (request-id: {req_id})" if req_id else "")
            + (f"\nResponse body:\n{body}" if body else "\n(no response body)")
            + f"\n{hint}"
        )

    data = r.json()

    # Docs show: print(response.output[0].content[0].text)
    try:
        return data["output"][0]["content"][0]["text"]
    except Exception:
        # fallback: best-effort dump
        return json.dumps(data, ensure_ascii=False, indent=2)


def make_prompt(question: str, retrieved: Sequence[Tuple[float, DocChunk]]) -> str:
    context_blocks = []
    for score, c in retrieved:
        context_blocks.append(
            f"[source: {c.source} | score: {score:.4f}]\n{c.text}".strip()
        )
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(нет контекста)"

    return (
        "Ты — помощник, который отвечает строго по предоставленному контексту.\n"
        "Если в контексте нет ответа, скажи, что информации недостаточно, и уточни, чего не хватает.\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Контекст:\n{context}\n\n"
        "Ответ:"
    )


def main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(description="Yandex AI Studio RAG pipeline (BM25 + Responses API)")
    p.add_argument("question", help="User question")
    p.add_argument("--kb", nargs="+", default=["kb"], help="Knowledge base paths (files or directories)")
    p.add_argument("--top-k", type=int, default=5, help="How many chunks to retrieve")
    p.add_argument("--model", default=os.getenv("YANDEX_MODEL", DEFAULT_MODEL), help="Model name, e.g. aliceai-llm")
    p.add_argument("--base-url", default=os.getenv("YANDEX_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-output-tokens", type=int, default=800)
    p.add_argument(
        "--no-rag",
        action="store_true",
        help="Do not retrieve context; send question directly (useful for API connectivity check)",
    )

    args = p.parse_args(list(argv))

    api_key = os.getenv("YANDEX_API_KEY") or os.getenv("YC_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID") or os.getenv("YC_FOLDER_ID")
    if not api_key or not folder_id:
        print(
            "Missing env vars. Set YANDEX_API_KEY and YANDEX_FOLDER_ID (or YC_API_KEY / YC_FOLDER_ID).",
            file=sys.stderr,
        )
        return 2
    if folder_id.lower() == "default" or not _FOLDER_ID_RE.match(folder_id):
        print(
            "YANDEX_FOLDER_ID выглядит неверно. Нужен реальный folder_id каталога Yandex Cloud "
            "(не 'default').",
            file=sys.stderr,
        )
        return 2

    if args.no_rag:
        prompt = args.question
    else:
        index = build_index(args.kb)
        retrieved = index.search(args.question, top_k=args.top_k)
        prompt = make_prompt(args.question, retrieved)

    try:
        answer = call_yandex_responses(
            api_key=api_key,
            folder_id=folder_id,
            model=args.model,
            input_text=prompt,
            base_url=args.base_url,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 3

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
