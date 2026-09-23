import hashlib
import os
import re
from pathlib import Path


_INVALID_FILENAME = re.compile(r'[\\/*?:"<>|\x00-\x1f]')
_VIDEO_URL_ID = re.compile(r"(?:[?&]v=|youtu\.be/)([A-Za-z0-9_-]{6,})")


def sanitize_filename(title, max_length=100):
    cleaned = _INVALID_FILENAME.sub("_", str(title))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    cleaned = cleaned[:max_length].rstrip(" .")
    if not cleaned or cleaned.upper() in {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }:
        return "video"
    return cleaned


def video_identifier(video):
    for attribute in ("video_id", "id"):
        value = getattr(video, attribute, None)
        if value:
            return sanitize_filename(value, max_length=80)
    url = str(getattr(video, "watch_url", "") or getattr(video, "url", "") or "")
    match = _VIDEO_URL_ID.search(url)
    if match:
        return match.group(1)
    title = str(getattr(video, "title", "video"))
    return hashlib.sha256(f"{url}\n{title}".encode("utf-8")).hexdigest()[:12]


def video_file_stem(video):
    identifier = video_identifier(video)
    title = sanitize_filename(getattr(video, "title", "video"))
    return f"{identifier}__{title}"


def resolve_whisper_device(device="auto", cuda_available=None):
    if cuda_available is None:
        try:
            import ctranslate2
            cuda_available = ctranslate2.get_cuda_device_count() > 0
        except (ImportError, AttributeError, RuntimeError):
            cuda_available = False
    if device in (None, "auto"):
        return "cuda" if cuda_available else "cpu"
    if device == "cuda" and not cuda_available:
        return "cpu"
    return device


def format_timestamp(seconds):
    total_seconds = max(0, int(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_timestamped_transcription(segments):
    lines = []
    for segment in segments:
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        start = getattr(segment, "start", 0)
        lines.append(f"[{format_timestamp(start)}] {text}")
    return "\n".join(lines).strip()


def _load_channel(channel_url):
    try:
        from pytubefix import Channel
    except ImportError as exc:
        raise RuntimeError("YouTube download needs pytubefix. Install requirements.txt.") from exc
    return Channel(channel_url)


def _download_video(video, audio_folder):
    audio_folder = Path(audio_folder)
    audio_folder.mkdir(parents=True, exist_ok=True)
    stem = video_file_stem(video)
    audio_path = audio_folder / f"{stem}.mp3"
    if audio_path.exists():
        print(f"Audio already downloaded, skipping: {audio_path.name}")
        return audio_path

    audio_stream = video.streams.filter(only_audio=True).first()
    if not audio_stream:
        print(f"No audio stream found for video: {video.title}")
        return None
    audio_stream.download(output_path=str(audio_folder), filename=audio_path.name)
    print(f"Downloaded: {audio_path.name}")
    return audio_path


def _load_whisper_model(model_name, device="auto", compute_type=None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Transcription needs faster-whisper. Install requirements.txt.") from exc

    options = {"device": resolve_whisper_device(device)}
    if compute_type:
        options["compute_type"] = compute_type
    return WhisperModel(model_name, **options)


def _transcribe_file(model, audio_path, transcription_path, language=None):
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=language,
    )
    text = format_timestamped_transcription(segments)
    transcription_path = Path(transcription_path)
    transcription_path.parent.mkdir(parents=True, exist_ok=True)
    transcription_path.write_text(text, encoding="utf-8")
    detected_language = getattr(info, "language", None)
    if language:
        print(f"Transcription saved to {transcription_path} ({language})")
    elif detected_language:
        print(f"Transcription saved to {transcription_path} (detected {detected_language})")
    else:
        print(f"Transcription saved to {transcription_path}")
    return text


def download_audio_from_youtube_channel(
    channel_url,
    output_path,
    max_videos=10,
    max_duration=1800,
):
    if max_videos < 0:
        raise ValueError("max_videos must be non-negative")
    output_folder = Path(output_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    channel = _load_channel(channel_url)

    for video in list(channel.videos)[:max_videos]:
        print(f"Nombre del video: {video.title}\tDuración: {video.length}")
        if video.length <= max_duration:
            _download_video(video, output_folder)
        else:
            print(f"Skipped (longer than {max_duration / 60:.2f} minutes): {video.title}")
    print("All eligible videos downloaded.")


def transcribe_audios_from_folder(
    audio_folder="audiosBorjaBandera/",
    transcription_folder="transcriptionsBorjaBandera/",
    model_name="Systran/faster-whisper-small",
    device="auto",
    compute_type=None,
    language=None,
):
    audio_folder = Path(audio_folder)
    if not audio_folder.is_dir():
        raise FileNotFoundError(f"Audio folder not found: {audio_folder}")
    output_folder = Path(transcription_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    model = _load_whisper_model(model_name, device=device, compute_type=compute_type)

    for audio_path in sorted(audio_folder.glob("*.mp3")):
        transcription_path = output_folder / f"{audio_path.stem}.txt"
        if transcription_path.exists():
            print(f"Transcription already exists, skipping: {transcription_path.name}")
            continue
        print(f"Transcribing file {audio_path.name} ...")
        _transcribe_file(model, audio_path, transcription_path, language=language)


def download_and_transcribe_videos_from_youtube_channel(
    channel_url,
    output_path,
    max_videos=10,
    max_duration=1800,
    transcription_folder="transcriptionsBorjaBandera/",
    audio_folder=None,
    model_name="Systran/faster-whisper-small",
    device="auto",
    compute_type=None,
    language=None,
):
    if max_videos < 0:
        raise ValueError("max_videos must be non-negative")
    audio_directory = Path(audio_folder or output_path)
    transcription_directory = Path(transcription_folder)
    audio_directory.mkdir(parents=True, exist_ok=True)
    transcription_directory.mkdir(parents=True, exist_ok=True)
    model = _load_whisper_model(model_name, device=device, compute_type=compute_type)
    channel = _load_channel(channel_url)

    for video in list(channel.videos)[:max_videos]:
        print(f"Nombre del video: {video.title}\tDuración: {video.length}")
        if video.length > max_duration:
            print(f"Skipped (longer than {max_duration / 60:.2f} minutes): {video.title}")
            continue

        stem = video_file_stem(video)
        audio_path = audio_directory / f"{stem}.mp3"
        transcription_path = transcription_directory / f"{stem}.txt"
        if not audio_path.exists():
            audio_path = _download_video(video, audio_directory)
        if audio_path and audio_path.is_file() and not transcription_path.exists():
            _transcribe_file(model, audio_path, transcription_path, language=language)
        elif transcription_path.exists():
            print(f"Transcription already exists, skipping: {transcription_path.name}")

    print("All eligible videos downloaded and transcribed.")
