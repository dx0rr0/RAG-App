"""Run a small, cost-bounded OpenRouter transcription comparison.

The reference transcript is only read locally. Audio is sent once to each
selected model; returned transcripts are not persisted by default.
"""

import argparse
import base64
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from api_config import get_openrouter_api_key

TRANSCRIPTION_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
MODEL_RATES_USD_PER_SECOND = {
    # Catalog snapshot on 2026-09-23. Actual response usage.cost takes priority.
    "microsoft/mai-transcribe-2": 0.10 / 3600,
    "qwen/qwen3-asr-0.6b": 0.000003,
    "qwen/qwen3-asr-1.7b": 0.000008,
    # Use Groq's higher listed rate for a conservative Whisper preflight.
    "openai/whisper-large-v3-turbo": 0.000011,
}
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"}
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def _normalize(text):
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    chars = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith(("P", "S")):
            chars.append(" ")
        else:
            chars.append(char)
    return " ".join("".join(chars).split())


def edit_distance(left, right):
    """Levenshtein distance using O(min(len(left), len(right))) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_item in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def transcription_metrics(reference, hypothesis):
    reference_words = _normalize(reference).split()
    hypothesis_words = _normalize(hypothesis).split()
    reference_chars = " ".join(reference_words)
    hypothesis_chars = " ".join(hypothesis_words)
    if not reference_words or not reference_chars:
        raise ValueError("Reference transcript is empty after normalization.")
    return {
        "reference_word_count": len(reference_words),
        "wer": edit_distance(reference_words, hypothesis_words) / len(reference_words),
        "cer": edit_distance(reference_chars, hypothesis_chars) / len(reference_chars),
    }


def _extract_cost(result):
    usage = result.get("usage") or {}
    for key in ("cost", "total_cost", "estimated_cost"):
        value = usage.get(key)
        if value is not None:
            try:
                return float(value), "response usage"
            except (TypeError, ValueError):
                pass
    return None, None


def _transcribe(api_key, model, audio_bytes, audio_format, language, timeout):
    payload = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": audio_format,
        },
        "language": language,
    }
    request = urllib.request.Request(
        TRANSCRIPTION_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
            generation_id = response.headers.get("X-Generation-Id")
    except urllib.error.HTTPError as exc:
        detail = exc.read(2000).decode("utf-8", errors="replace")
        raise RuntimeError(f"{model}: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{model}: request failed: {exc.reason}") from exc
    elapsed = time.perf_counter() - started
    try:
        result = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{model}: API returned invalid JSON") from exc
    transcript = str(result.get("text") or "").strip()
    if not transcript:
        raise RuntimeError(f"{model}: API returned an empty transcript")
    actual_cost, cost_source = _extract_cost(result)
    return {
        "transcript": transcript,
        "latency_seconds": elapsed,
        "cost_usd": actual_cost,
        "cost_source": cost_source,
        "usage": result.get("usage") or {},
        "generation_id": generation_id,
    }


def _render_markdown_table(results):
    lines = [
        "| Modelo | WER | CER | Latencia (s) | Coste (USD) | ID de generación | Estado |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        if result.get("error"):
            error = str(result["error"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{result['model']}` | — | — | — | — | — | Error: {error} |"
            )
            continue
        metrics = result["metrics"]
        cost = result.get("cost_usd")
        cost_source = result.get("cost_source")
        if cost is None:
            cost_text = "sin coste devuelto"
        elif cost_source == "response usage":
            cost_text = f"${cost:.6f} (API)"
        else:
            cost_text = f"~${cost:.6f} (estimado)"
        lines.append(
            f"| `{result['model']}` | {metrics['wer']:.4f} | "
            f"{metrics['cer']:.4f} | {result['latency_seconds']:.2f} | "
            f"{cost_text} | `{result.get('generation_id') or '—'}` | OK |"
        )
    return "\n".join(lines)


def _append_measurement_log(log_path, audio_path, duration_seconds, language, results):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "",
        f"### {timestamp} — Comparativa corta de transcripción",
        "",
        f"- Audio: `{audio_path.name}` ({duration_seconds:.1f} s); idioma fijado: `{language}`.",
        "- La referencia se usó solo localmente; el audio se envió una vez a cada modelo. "
        "No se guardaron transcripciones de salida.",
        "- WER/CER se calcularon tras normalizar Unicode, mayúsculas y puntuación; "
        "los acentos y los espacios entre palabras se conservan.",
        "",
        _render_markdown_table(results),
        "",
        "Los resultados corresponden solo a esta muestra y no generalizan al corpus.",
        "",
    ]
    with Path(log_path).open("a", encoding="utf-8", newline="\n") as log_file:
        log_file.write("\n".join(lines))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare MAI, Qwen ASR, and Whisper on one audio sample."
    )
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--language", default="es")
    parser.add_argument("--duration-seconds", required=True, type=float,
                        help="Audio duration for a cost preflight; obtain from the media player.")
    parser.add_argument("--max-duration-seconds", default=300, type=float)
    parser.add_argument("--max-estimated-cost-usd", default=0.02, type=float)
    parser.add_argument("--timeout-seconds", default=120, type=float)
    parser.add_argument("--log", type=Path, default=Path("MEASUREMENTS.md"))
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_RATES_USD_PER_SECOND),
        default=[
            "microsoft/mai-transcribe-2",
            "qwen/qwen3-asr-0.6b",
            "openai/whisper-large-v3-turbo",
        ],
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.duration_seconds <= 0:
        raise SystemExit("--duration-seconds must be greater than zero")
    if args.duration_seconds > args.max_duration_seconds:
        raise SystemExit(
            f"Sample duration {args.duration_seconds:g}s exceeds the "
            f"{args.max_duration_seconds:g}s safety limit."
        )
    if args.max_estimated_cost_usd <= 0:
        raise SystemExit("--max-estimated-cost-usd must be greater than zero")
    if not args.audio.is_file() or not args.reference.is_file():
        raise SystemExit("Both --audio and --reference must point to existing files.")
    audio_format = args.audio.suffix.lower().lstrip(".")
    if audio_format not in SUPPORTED_FORMATS:
        raise SystemExit(f"Unsupported audio format: {audio_format or '(none)'}")
    if args.audio.stat().st_size > MAX_AUDIO_BYTES:
        raise SystemExit("Audio exceeds the 25 MiB upload safety limit.")

    estimated_total = sum(
        MODEL_RATES_USD_PER_SECOND[model] * args.duration_seconds
        for model in args.models
    )
    if estimated_total > args.max_estimated_cost_usd:
        raise SystemExit(
            f"Estimated total ${estimated_total:.4f} exceeds the "
            f"${args.max_estimated_cost_usd:.4f} limit; shorten the sample or "
            "raise the limit explicitly."
        )

    api_key = get_openrouter_api_key()
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in the environment; do not put it in the repo.")
    reference = args.reference.read_text(encoding="utf-8")
    # Fail before any billable request if the local reference is unusable.
    if not _normalize(reference).split():
        raise SystemExit("Reference transcript is empty after normalization.")

    print(f"Audio: {args.audio.name} ({args.duration_seconds:g}s); language: {args.language}")
    print(f"Estimated total for {len(args.models)} calls: ${estimated_total:.6f} (catalog snapshot)")
    print("Running one request per model; outputs are held in memory and not saved.")
    audio_bytes = args.audio.read_bytes()
    results = []
    for model in args.models:
        print(f"Transcribing with {model} ...", flush=True)
        try:
            response = _transcribe(
                api_key=api_key,
                model=model,
                audio_bytes=audio_bytes,
                audio_format=audio_format,
                language=args.language,
                timeout=args.timeout_seconds,
            )
            metrics = transcription_metrics(reference, response["transcript"])
            cost = response["cost_usd"]
            cost_source = response["cost_source"]
            if cost is None:
                cost = MODEL_RATES_USD_PER_SECOND[model] * args.duration_seconds
                cost_source = "catalog estimate"
            result = {
                "model": model,
                "metrics": metrics,
                "latency_seconds": response["latency_seconds"],
                "cost_usd": cost,
                "cost_source": cost_source,
                "generation_id": response["generation_id"],
            }
            print(
                f"  WER={metrics['wer']:.4f}, CER={metrics['cer']:.4f}, "
                f"latency={result['latency_seconds']:.2f}s, cost=${cost:.6f} "
                f"({cost_source})"
            )
        except (RuntimeError, ValueError) as exc:
            result = {"model": model, "error": str(exc)}
            print(f"  {exc}", file=sys.stderr)
        results.append(result)

    _append_measurement_log(
        args.log, args.audio, args.duration_seconds, args.language, results
    )
    print("\n" + _render_markdown_table(results))
    print(f"\nResults appended to {args.log}")
    return 0 if any(not result.get("error") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
