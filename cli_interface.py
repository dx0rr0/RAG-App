import argparse
import sys
import textwrap

from api_config import get_openrouter_api_key
from hybrid_retrieval import BM25Index, load_cross_encoder
from jev_reranker import JevReranker, JevRerankerError
from llm_interaction import ask, load_model, openrouter_faithfulness_checker
from vector_db_manager import (
    DEFAULT_EMBEDDING_MODEL,
    embeddings_tensor,
    get_embedding_function,
    load_or_create_vector_db,
)


def print_wrapped(text, wrap_length=80):
    print(textwrap.fill(text, wrap_length))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Consulta transcripciones con retrieval augmented generation."
    )
    parser.add_argument("--transcriptions", default="transcriptionsBorjaBandera")
    parser.add_argument("--index", default="vector_db.parquet")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-backend", choices=("local", "openrouter"), default="local")
    parser.add_argument("--llm-model")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="low",
        help="Reasoning effort for the OpenRouter backend.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval-mode",
        choices=("vector", "hybrid"),
        default="vector",
        help="Use vector retrieval or BM25 plus vector fusion.",
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument(
        "--reranker",
        choices=("none", "jev"),
        default="none",
        help="Optional Jev direct-evidence reranker for hybrid retrieval.",
    )
    parser.add_argument(
        "--reranker-candidate-k",
        type=int,
        default=9,
        help="Maximum fused candidates sent to the reranker (default: 9).",
    )
    parser.add_argument(
        "--jev-max-estimated-cost-usd",
        type=float,
        default=0.001,
        help="Per-query Jev preflight estimate ceiling; over-limit queries keep RRF order.",
    )
    parser.add_argument(
        "--reranker-model",
        help="Optional sentence-transformers cross-encoder model ID for hybrid mode.",
    )
    parser.add_argument("--min-evidence-score", type=float, default=0.35)
    parser.add_argument(
        "--check-faithfulness",
        action="store_true",
        help=(
            "Verify each generated answer against retrieved text with GPT-6 Luna via OpenRouter "
            "(adds roughly $0.0002-$0.001 per checked answer; disabled by default)."
        ),
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the index even when the corpus and configuration match.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Build or reuse the index, then exit without loading the chat model.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (
        (args.llm_backend == "openrouter" or args.check_faithfulness)
        and not args.index_only
        and not get_openrouter_api_key()
    ):
        raise SystemExit(
            "Set OPENROUTER_API_KEY before using OpenRouter generation or --check-faithfulness."
        )
    if args.top_k < 0:
        raise SystemExit("--top-k must be greater than or equal to zero")
    if args.top_k == 0 and not args.index_only:
        raise SystemExit("--top-k must be at least 1 when running interactive chat")
    if args.candidate_k < 1:
        raise SystemExit("--candidate-k must be at least 1")
    if args.rrf_constant < 1:
        raise SystemExit("--rrf-constant must be at least 1")
    if not -1.0 <= args.min_evidence_score <= 1.0:
        raise SystemExit("--min-evidence-score must be between -1 and 1")
    if args.reranker_model and args.retrieval_mode != "hybrid":
        raise SystemExit("--reranker-model requires --retrieval-mode hybrid")
    if args.reranker != "none" and args.retrieval_mode != "hybrid":
        raise SystemExit("--reranker jev requires --retrieval-mode hybrid")
    if args.reranker != "none" and args.reranker_model:
        raise SystemExit("Choose either --reranker jev or --reranker-model, not both")
    if (args.reranker == "jev" or args.reranker_model) and args.reranker_candidate_k < args.top_k:
        raise SystemExit("--reranker-candidate-k must be at least --top-k")
    if (args.reranker == "jev" or args.reranker_model) and args.reranker_candidate_k < 1:
        raise SystemExit("--reranker-candidate-k must be at least 1")
    if args.jev_max_estimated_cost_usd < 0:
        raise SystemExit("--jev-max-estimated-cost-usd must be non-negative")
    if args.reranker == "jev" and not args.index_only and not get_openrouter_api_key():
        raise SystemExit("Set OPENROUTER_API_KEY before using --reranker jev.")

    embedding_function = None
    if not args.index_only:
        embedding_function = get_embedding_function(
            model_name=args.embedding_model,
            device=args.device,
        )
    frame = load_or_create_vector_db(
        directory_path=args.transcriptions,
        save_path=args.index,
        embedding_function=embedding_function,
        embedding_model=args.embedding_model,
        device=args.device,
        force_rebuild=args.rebuild_index,
    )
    if args.index_only:
        print(f"Index ready: {len(frame)} chunks in {args.index}")
        return 0

    if embedding_function is None:
        embedding_function = get_embedding_function(
            model_name=args.embedding_model,
            device=args.device,
        )
    embeddings = embeddings_tensor(frame)
    if args.llm_backend == "openrouter":
        llm_model = None
        tokenizer = None
        api_model = args.llm_model or "openai/gpt-6-luna"
    else:
        local_model_id = args.llm_model or "UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3"
        llm_model, tokenizer = load_model(local_model_id, device=args.device)
        api_model = "openai/gpt-6-luna"
    bm25_index = None
    reranker = None
    reranker_candidate_k = None
    if args.retrieval_mode == "hybrid":
        bm25_index = BM25Index(frame["page_content"].fillna("").astype(str).tolist())
        if args.reranker_model:
            reranker = load_cross_encoder(args.reranker_model, device=args.device)
        elif args.reranker == "jev":
            try:
                jev = JevReranker(
                    max_estimated_cost_usd=args.jev_max_estimated_cost_usd
                )
            except JevRerankerError as exc:
                print(f"[Jev] {exc} Se usará RRF.", file=sys.stderr)
            else:
                reranker_candidate_k = args.reranker_candidate_k

                def reranker(query, candidate_texts):
                    try:
                        return jev(query, candidate_texts)
                    except JevRerankerError as exc:
                        print(f"[Jev] {exc}", file=sys.stderr)
                        # Preserve the incoming RRF order when Jev cannot rerank.
                        return list(range(len(candidate_texts), 0, -1))
    print("Welcome! Type 'exit' to quit.")
    while True:
        try:
            query = input("Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() == "exit":
            break
        if not query:
            print("Escribe una consulta o 'exit' para salir.")
            continue

        response = ask(
            query=query,
            embeddings=embeddings,
            embedding_function=embedding_function,
            llm_model=llm_model,
            tokenizer=tokenizer,
            df=frame,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            bm25_index=bm25_index,
            candidate_k=args.candidate_k,
            rrf_constant=args.rrf_constant,
            reranker=reranker,
            reranker_candidate_k=reranker_candidate_k,
            min_evidence_score=args.min_evidence_score,
            faithfulness_checker=(
                openrouter_faithfulness_checker if args.check_faithfulness else None
            ),
            llm_backend=args.llm_backend,
            api_model=api_model,
            reasoning_effort=args.reasoning_effort,
        )
        print()
        print_wrapped(response)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
