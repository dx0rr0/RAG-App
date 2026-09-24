from urllib.parse import urlparse

from rag_app.domain.errors import ValidationError


def validate_youtube_channel_url(value):
    url = str(value or "").strip()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in {
        "youtube.com", "www.youtube.com", "m.youtube.com", "www.youtube-nocookie.com"
    }:
        raise ValidationError("Introduce una URL pública de canal de YouTube.")
    if not (parsed.path.startswith("/@") or "/channel/" in parsed.path or "/c/" in parsed.path or "/user/" in parsed.path):
        raise ValidationError("La URL debe apuntar a un canal, por ejemplo youtube.com/@canal.")
    return url.rstrip("/")
