import base64
import hashlib
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

from rag_app.domain.errors import ProviderError


API_ROOT = "https://openrouter.ai/api/v1"
JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"
JEV_INPUT_USD_PER_MILLION = 0.042
JEV_PROMPT_VERSION = "direct-evidence-v1"
SUPPORTED_AUDIO = {"wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"}


def _reported_cost(payload):
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if isinstance(usage, dict):
        for name in ("cost", "total_cost", "estimated_cost"):
            value = usage.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


class OpenRouterAdapter:
    """Single no-retry client for every billable model call."""

    def __init__(self, settings, jev_cache_path=None, timeout_seconds=90):
        self.settings = settings
        self.timeout_seconds = timeout_seconds
        self.jev_cache_path = Path(jev_cache_path or settings.database_path.with_suffix(".jev-cache.json"))
        self._jev_cache = self._read_jev_cache()

    def _post_json(self, endpoint, payload, timeout=None):
        if not self.settings.openrouter_api_key:
            raise ProviderError("Falta OPENROUTER_API_KEY en el entorno o en .env.")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": "Bearer " + self.settings.openrouter_api_key,
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "RAG-App",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout_seconds) as response:
                raw = response.read()
                generation_id = response.headers.get("X-Generation-Id")
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"OpenRouter respondió HTTP {exc.code}; no se reintentó la llamada.") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(f"No se pudo contactar con OpenRouter ({type(exc).__name__}); no se reintentó la llamada.") from None
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise ProviderError("OpenRouter devolvió una respuesta JSON inválida; no se reintentó.") from None
        if not isinstance(result, dict) or result.get("error"):
            raise ProviderError("OpenRouter rechazó la solicitud; no se reintentó la llamada.")
        result["_generation_id"] = generation_id
        return result

    def embed(self, texts):
        items = [str(text) for text in texts]
        if not items:
            return {"vectors": [], "cost_usd": 0.0, "usage": {}}
        response = self._post_json(
            API_ROOT + "/embeddings",
            {"model": self.settings.embedding_model, "input": items},
        )
        rows = response.get("data")
        if not isinstance(rows, list) or len(rows) != len(items):
            raise ProviderError("OpenRouter no devolvió un embedding por texto.")
        ordered = sorted(rows, key=lambda row: row.get("index", 0))
        vectors = [row.get("embedding") for row in ordered]
        if any(not isinstance(vector, list) or not vector for vector in vectors):
            raise ProviderError("OpenRouter devolvió un embedding vacío o inválido.")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ProviderError("OpenRouter devolvió embeddings con dimensiones distintas.")
        reported_cost = _reported_cost(response)
        estimated_cost = math.ceil(sum(len(text) for text in items) / 4) * self.settings.embedding_usd_per_million_tokens / 1_000_000
        return {"vectors": vectors, "cost_usd": reported_cost if reported_cost is not None else estimated_cost,
                "cost_is_estimate": reported_cost is None, "usage": response.get("usage") or {}}

    def transcribe(self, audio_path, language="es", duration_seconds=None):
        path = Path(audio_path)
        audio_format = path.suffix.lower().lstrip(".")
        if audio_format not in SUPPORTED_AUDIO:
            raise ProviderError(f"Formato de audio no compatible: {audio_format or 'sin extensión'}.")
        if path.stat().st_size > self.settings.max_audio_bytes:
            raise ProviderError("El audio supera 25 MiB; se rechaza antes de llamar a OpenRouter.")
        payload = {
            "model": self.settings.transcription_model,
            "input_audio": {
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                "format": audio_format,
            },
            "language": language,
        }
        response = self._post_json(API_ROOT + "/audio/transcriptions", payload, timeout=180)
        transcript = str(response.get("text") or "").strip()
        if not transcript:
            raise ProviderError("Qwen ASR devolvió una transcripción vacía.")
        cost = _reported_cost(response)
        cost_is_estimate = cost is None and duration_seconds is not None
        if cost_is_estimate:
            cost = float(duration_seconds) * self.settings.asr_usd_per_second
        segments = response.get("segments")
        return {
            "text": transcript,
            "segments": segments if isinstance(segments, list) else [],
            "cost_usd": cost,
            "cost_is_estimate": cost_is_estimate,
            "usage": response.get("usage") or {},
            "generation_id": response.get("_generation_id"),
        }

    def complete(self, messages, max_tokens=512, json_mode=False, purpose="chat"):
        payload = {
            "model": self.settings.chat_model,
            "reasoning_effort": self.settings.reasoning_effort,
            "messages": list(messages),
            "max_tokens": int(max_tokens),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = self._post_json(API_ROOT + "/chat/completions", payload)
        choices = response.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("El modelo devolvió una respuesta vacía.")
        return {
            "text": content.strip(),
            "cost_usd": _reported_cost(response),
            "usage": response.get("usage") or {},
            "generation_id": response.get("_generation_id"),
            "purpose": purpose,
        }

    def rerank_jev(self, query, passages, max_cost_usd=None):
        candidates = [str(text) for text in passages]
        if not candidates:
            return {"scores": [], "cost_usd": 0.0, "cache_hit": True}
        payload = {
            "model": JEV_MODEL,
            "state": {"query": str(query)},
            "questions": {
                f"candidate_{index}": {
                    "type": "noul",
                    "instructions": {
                        "candidate_passage": text,
                        "question": "Does this passage directly provide evidence answering the query? Judge the passage alone.",
                    },
                }
                for index, text in enumerate(candidates)
            },
        }
        fingerprint = hashlib.sha256(json.dumps(
            {"version": JEV_PROMPT_VERSION, "payload": payload},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        cached = self._jev_cache.get(fingerprint)
        if isinstance(cached, list) and len(cached) == len(candidates):
            return {"scores": cached, "cost_usd": 0.0, "cache_hit": True}
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        estimate = math.ceil(len(serialized) * 1.5) * JEV_INPUT_USD_PER_MILLION / 1_000_000
        limit = self.settings.jev_max_estimated_usd if max_cost_usd is None else max_cost_usd
        if estimate > limit:
            raise ProviderError(f"Estimación Jev ${estimate:.6f} supera el límite ${limit:.6f}.")
        response = self._post_json(JEV_ENDPOINT, payload)
        answers = response.get("answers")
        if not isinstance(answers, dict) or len(answers) != len(candidates):
            raise ProviderError("Jev no devolvió una puntuación por candidato.")
        scores = []
        for index in range(len(candidates)):
            answer = answers.get(f"candidate_{index}") or {}
            score = answer.get("noul")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise ProviderError("Jev devolvió una puntuación inválida.")
            scores.append(float(score))
        self._jev_cache[fingerprint] = scores
        self._write_jev_cache()
        reported_cost = _reported_cost(response)
        return {"scores": scores, "cost_usd": reported_cost if reported_cost is not None else estimate,
                "cost_is_estimate": reported_cost is None, "cache_hit": False}

    def _read_jev_cache(self):
        try:
            payload = json.loads(self.jev_cache_path.read_text(encoding="utf-8"))
            if payload.get("version") == 1 and payload.get("model") == JEV_MODEL:
                return payload.get("scores", {})
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            pass
        return {}

    def _write_jev_cache(self):
        self.jev_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "model": JEV_MODEL, "scores": self._jev_cache}
        temporary = self.jev_cache_path.with_suffix(self.jev_cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.jev_cache_path)
