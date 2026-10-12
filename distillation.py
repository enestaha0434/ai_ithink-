"""Uzaktan öğretmen model yanıtlarını güvenli biçimde alma yardımcıları."""

from __future__ import annotations

import json
import socket
from urllib.parse import urlparse
from urllib.request import Request, urlopen


MAX_RESPONSE_BYTES = 1_000_000


def validate_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Öğretmen endpoint'i HTTPS ve tam bir alan adı olmalıdır.")
    return endpoint


def extract_chat_content(response: dict) -> str:
    """OpenAI-uyumlu chat-completions JSON yanıtını sıkı biçimde doğrular."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("Öğretmen API yanıtında choices[0] yok.")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Öğretmen API boş yanıt verdi.")
    return content.strip()


def request_chat_completion(endpoint: str, api_key: str, model: str, system: str, user: str, timeout: float) -> str:
    """Tek öğretmen cevabı ister; anahtarı diske/loga yazmaz."""
    validate_endpoint(endpoint)
    if not api_key.strip():
        raise ValueError("Öğretmen API anahtarı boş olamaz.")
    if not 0 < timeout <= 60:
        raise ValueError("Timeout 0..60 saniye arasında olmalıdır.")
    body = json.dumps({
        "model": model,
        "temperature": 0.35,
        "max_tokens": 220,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }, ensure_ascii=False).encode("utf-8")
    request = Request(endpoint, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json",
        "User-Agent": "HyperAI-distillation/0.1",
    })
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, socket.timeout) as exc:
        raise RuntimeError(f"Öğretmen API isteği başarısız: {exc}") from exc
    if len(payload) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Öğretmen API yanıtı güvenli boyut sınırını aştı.")
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Öğretmen API geçerli JSON döndürmedi.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Öğretmen API beklenmeyen JSON biçimi döndürdü.")
    return extract_chat_content(parsed)
