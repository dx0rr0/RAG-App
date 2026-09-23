"""Optional Jev reranker client for OpenRouter's decisions endpoint."""

import hashlib
import json
import math
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

from api_config import get_openrouter_api_key


JEV_MODEL = "typesafe/jev-1.13"
JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
JEV_INPUT_USD_PER_MILLION = 0.042
JEV_TIMEOUT_SECONDS = 45
JEV_PROMPT_VERSION = "direct-evidence-v1"
JEV_QUESTION = (
    "Does `candidate_passage` explicitly state evidence that answers `query`, "
    "rather than only mentioning the same topic? Judge this passage alone."
)
DEFAULT_CACHE_PATH = Path.cwd() / ".rag-app-cache" / "jev-reranker-v1.json"


class JevRerankerError(RuntimeError):
    """A safe-to-display Jev error that never includes credentials or source text."""


class JevReranker:
    """Score a small candidate shortlist for direct answer evidence.

    The disk cache stores only a SHA-256 request fingerprint and score list. It
    never stores the query, candidate passages, API key, or raw API response.
    """

    def __init__(
        self,
        api_key=None,
        cache_path=DEFAULT_CACHE_PATH,
        max_estimated_cost_usd=0.001,
    ):
        if max_estimated_cost_usd < 0:
            raise ValueError("max_estimated_cost_usd must be non-negative")
        self.api_key = api_key if api_key is not None else get_openrouter_api_key()
        self.cache_path = Path(cache_path)
        self.max_estimated_cost_usd = float(max_estimated_cost_usd)
        self._cache = self._load_cache()

    def __call__(self, query, candidate_texts):
        candidates = [str(text) for text in candidate_texts]
        if not candidates:
            return []
        payload = self._build_payload(query, candidates)
        fingerprint = self._fingerprint(payload)
        cached_scores = self._cache.get(fingerprint)
        if cached_scores is not None:
            if isinstance(cached_scores, dict) and cached_scores.get("error"):
                raise JevRerankerError(
                    "Una petición Jev previa para esta consulta no produjo scores válidos; "
                    "no se repite para evitar otro posible cargo. Borra el archivo de caché "
                    "solo si quieres volver a intentarlo."
                )
            if (
                not isinstance(cached_scores, list)
                or len(cached_scores) != len(candidates)
                or any(
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                    or not 0.0 <= float(score) <= 1.0
                    for score in cached_scores
                )
            ):
                raise JevRerankerError(
                    "La entrada de caché Jev no es válida; revísala antes de continuar."
                )
            print("[Jev] Reutilizando scores en caché; coste API $0.", file=sys.stderr)
            return list(cached_scores)

        estimated_cost = self._estimate_cost(payload)
        if estimated_cost > self.max_estimated_cost_usd:
            raise JevRerankerError(
                "Estimación Jev de "
                f"${estimated_cost:.6f} supera el límite por consulta de "
                f"${self.max_estimated_cost_usd:.6f}; se conserva el orden RRF."
            )
        if not self.api_key:
            raise JevRerankerError(
                "Falta OPENROUTER_API_KEY para Jev; se conserva el orden RRF."
            )

        started = time.perf_counter()
        response = self._post_json(payload)
        latency = time.perf_counter() - started
        try:
            scores = self._parse_scores(response, len(candidates))
        except JevRerankerError:
            # Do not make another potentially billable request for the same bad response.
            self._cache[fingerprint] = {"error": "invalid-response"}
            try:
                self._save_cache()
            except OSError:
                print(
                    "[Jev] No se pudo guardar el fallo en caché; repetir la consulta "
                    "podría volver a generar coste API.",
                    file=sys.stderr,
                )
            raise
        self._cache[fingerprint] = scores
        try:
            self._save_cache()
        except OSError:
            print(
                "[Jev] No se pudo guardar la caché; repetir la consulta podría "
                "volver a generar coste API.",
                file=sys.stderr,
            )

        usage = response.get("usage") or {}
        reported_cost = usage.get("cost")
        if isinstance(reported_cost, (int, float)) and not isinstance(reported_cost, bool):
            cost_text = f"${float(reported_cost):.8f} reportados"
        else:
            input_tokens = usage.get("input_tokens")
            if isinstance(input_tokens, int) and input_tokens >= 0:
                estimated = input_tokens * JEV_INPUT_USD_PER_MILLION / 1_000_000
                cost_text = f"~${estimated:.8f} estimados (sin coste reportado)"
            else:
                cost_text = "coste no informado por OpenRouter"
        print(
            f"[Jev] {len(candidates)} candidatos; {cost_text}; "
            f"{latency:.2f} s.",
            file=sys.stderr,
        )
        return scores

    @staticmethod
    def _build_payload(query, candidate_texts):
        questions = {}
        for index, text in enumerate(candidate_texts):
            questions[f"candidate_{index}"] = {
                "type": "noul",
                "instructions": {
                    "candidate_passage": text,
                    "question": JEV_QUESTION,
                },
            }
        return {
            "model": JEV_MODEL,
            "state": {"query": str(query)},
            "questions": questions,
        }

    @staticmethod
    def _fingerprint(payload):
        cache_material = {
            "model": JEV_MODEL,
            "prompt_version": JEV_PROMPT_VERSION,
            "payload": payload,
        }
        serialized = json.dumps(
            cache_material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    @staticmethod
    def _estimate_cost(payload):
        # Deliberately conservative preflight: 1.5 input tokens per serialized
        # character, charged at the published Jev input rate.
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        estimated_tokens = math.ceil(len(serialized) * 1.5)
        return estimated_tokens * JEV_INPUT_USD_PER_MILLION / 1_000_000

    def _load_cache(self):
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise JevRerankerError(
                "La caché Jev local no se puede leer; revísala antes de continuar."
            ) from exc
        if (
            not isinstance(data, dict)
            or data.get("version") != 1
            or data.get("model") != JEV_MODEL
            or not isinstance(data.get("scores"), dict)
        ):
            raise JevRerankerError(
                "La caché Jev tiene una versión incompatible; revísala antes de continuar."
            )
        return data["scores"]

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            {"version": 1, "model": JEV_MODEL, "scores": self._cache},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        temporary_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary_path.write_text(content + "\n", encoding="utf-8")
        temporary_path.replace(self.cache_path)

    def _post_json(self, payload):
        request = urllib.request.Request(
            JEV_ENDPOINT,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=JEV_TIMEOUT_SECONDS
            ) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise JevRerankerError(
                f"OpenRouter Jev devolvió HTTP {exc.code}; se conserva el orden RRF."
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise JevRerankerError(
                f"No se pudo contactar con OpenRouter Jev ({type(exc).__name__}); "
                "se conserva el orden RRF."
            ) from None
        except (UnicodeError, json.JSONDecodeError):
            raise JevRerankerError(
                "OpenRouter Jev devolvió una respuesta no JSON; se conserva el orden RRF."
            ) from None
        if not isinstance(result, dict) or result.get("error"):
            raise JevRerankerError(
                "OpenRouter Jev rechazó la petición; se conserva el orden RRF."
            )
        return result

    @staticmethod
    def _parse_scores(response, expected_count):
        answers = response.get("answers")
        if not isinstance(answers, dict) or len(answers) != expected_count:
            raise JevRerankerError(
                "Jev no devolvió un score por candidato; se conserva el orden RRF."
            )
        scores = []
        for index in range(expected_count):
            answer = answers.get(f"candidate_{index}")
            score = answer.get("noul") if isinstance(answer, dict) else None
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise JevRerankerError(
                    "Jev devolvió un score inválido; se conserva el orden RRF."
                )
            scores.append(float(score))
        return scores
