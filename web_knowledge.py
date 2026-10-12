"""İsteğe bağlı, sınırlı ve kaynak gösteren Türkçe Vikipedi bilgi istemcisi.

Arama motoru ya da keyfi URL indirici değildir. Yalnızca sabit HTTPS Wikipedia
uçlarına gider, yönlendirmeleri takip etmez, yanıt boyutunu sınırlar ve alınan
metni talimat değil alıntı belge olarak döndürür.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .retrieval import RetrievedPassage


API = "https://tr.wikipedia.org/w/api.php"
SUMMARY = "https://tr.wikipedia.org/api/rest_v1/page/summary/"
MAX_RESPONSE_BYTES = 1_000_000


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise ValueError("Web istemcisi yönlendirmeleri takip etmez.")


def _json(url: str, timeout: float) -> dict:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "HyperAI-local-retrieval/0.1"})
    try:
        with build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, socket.timeout) as exc:
        raise RuntimeError(f"Vikipedi isteği başarısız: {exc}") from exc
    if len(payload) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Vikipedi yanıtı güvenli boyut sınırını aştı.")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Vikipedi geçerli JSON yanıtı vermedi.") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Vikipedi beklenmeyen yanıt biçimi döndürdü.")
    return value


def wikipedia_summary(query: str, timeout: float = 8.0) -> RetrievedPassage | None:
    """Türkçe Vikipedi'deki en yakın sayfanın kısa özetini, varsa döndürür."""
    if not query.strip() or not 0 < timeout <= 30:
        raise ValueError("Sorgu boş olamaz ve timeout 0..30 saniye arasında olmalıdır.")
    search_url = API + "?" + urlencode({"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json", "utf8": 1})
    search_response = _json(search_url, timeout)
    query_result = search_response.get("query")
    search = query_result.get("search", []) if isinstance(query_result, dict) else []
    if not isinstance(search, list) or not search or not isinstance(search[0], dict) or not isinstance(search[0].get("title"), str):
        return None
    title = search[0]["title"]
    summary = _json(SUMMARY + quote(title, safe=""), timeout)
    extract = summary.get("extract")
    content_urls = summary.get("content_urls")
    desktop_urls = content_urls.get("desktop") if isinstance(content_urls, dict) else None
    page_url = desktop_urls.get("page") if isinstance(desktop_urls, dict) else None
    if not isinstance(extract, str) or not extract.strip() or not isinstance(page_url, str):
        return None
    return RetrievedPassage(source=page_url, title=title, text=" ".join(extract.split()), score=1.0)
