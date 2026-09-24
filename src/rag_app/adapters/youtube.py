from datetime import datetime
from pathlib import Path
from rag_app.domain.errors import ProviderError, ValidationError
from rag_app.domain.validation import validate_youtube_channel_url


class YouTubeAdapter:
    """Metadata and audio download adapter; it performs no speech inference."""

    def discover(self, channel_url):
        normalized = validate_youtube_channel_url(channel_url)
        try:
            from pytubefix import Channel
            channel = Channel(normalized)
            videos = []
            for video in channel.videos:
                published = getattr(video, "publish_date", None)
                if isinstance(published, datetime):
                    published = published.isoformat()
                video_id = str(getattr(video, "video_id", "") or "")
                if not video_id:
                    continue
                videos.append({
                    "id": video_id,
                    "channel_id": str(getattr(channel, "channel_id", "") or normalized),
                    "title": str(getattr(video, "title", "Untitled video") or "Untitled video"),
                    "url": "https://www.youtube.com/watch?v=" + video_id,
                    "duration_seconds": max(0, int(getattr(video, "length", 0) or 0)),
                    "published_at": published,
                })
            channel_id = str(getattr(channel, "channel_id", "") or normalized)
            for video in videos:
                video["channel_id"] = channel_id
            return ({"id": channel_id, "url": normalized,
                     "title": str(getattr(channel, "channel_name", "") or normalized)}, videos)
        except ValidationError:
            raise
        except Exception as exc:
            raise ProviderError(f"No se pudo leer el canal de YouTube ({type(exc).__name__}).") from None

    def download_audio(self, video_url, output_dir):
        try:
            from pytubefix import YouTube
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            video = YouTube(video_url)
            stream = video.streams.filter(only_audio=True).order_by("abr").desc().first()
            if stream is None:
                raise ProviderError("YouTube no ofrece una pista de audio descargable.")
            downloaded = Path(stream.download(output_path=str(target), filename="source-audio"))
            suffix = downloaded.suffix.lower().lstrip(".")
            if suffix not in {"wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"}:
                raise ProviderError(f"Formato de audio descargado no compatible: {suffix or 'desconocido'}.")
            return downloaded
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"No se pudo descargar el audio ({type(exc).__name__}).") from None
