import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
INDEX_SCHEMA_VERSION = 2
_VIDEO_FILENAME = re.compile(r"^([A-Za-z0-9_-]{6,})__(.+)$")


def resolve_device(device="auto", cuda_available=None):
    """Choose a usable embedding device without importing torch at module import."""
    if cuda_available is None:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

    if device in (None, "auto"):
        return "cuda" if cuda_available else "cpu"
    if device == "cuda" and not cuda_available:
        return "cpu"
    return device


def get_embedding_function(model_name=DEFAULT_EMBEDDING_MODEL, device="auto"):
    try:
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "Embedding dependencies are missing. Install the packages in requirements.txt."
        ) from exc

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": resolve_device(device),
            "trust_remote_code": True,
        },
        encode_kwargs={"normalize_embeddings": False},
    )


def load_documents(data_path):
    try:
        from langchain_community.document_loaders import TextLoader
    except ImportError as exc:
        raise RuntimeError(
            "Document-loader dependencies are missing. Install requirements.txt."
        ) from exc
    return TextLoader(data_path, encoding="utf-8").load()


def _enrich_source_metadata(metadata, source):
    details = dict(metadata or {})
    path = Path(str(source or details.get("source", "")))
    stem = path.stem
    match = _VIDEO_FILENAME.match(stem)
    video_id = str(details.get("video_id") or (match.group(1) if match else ""))
    title = str(details.get("title") or (match.group(2) if match else stem))
    source_url = str(details.get("source_url") or details.get("url") or "")
    if not source_url and video_id:
        source_url = f"https://www.youtube.com/watch?v={video_id}"

    timestamp = details.get(
        "timestamp",
        details.get("start_timestamp", details.get("start", "")),
    )
    try:
        chunk_index = int(details.get("chunk_index", 0))
    except (TypeError, ValueError):
        chunk_index = 0

    return {
        "source": str(source or details.get("source", "")),
        "title": title,
        "video_id": video_id,
        "source_url": source_url,
        "timestamp": timestamp if timestamp is not None else "",
        "chunk_index": chunk_index,
    }


def load_directory(directory_path):
    root = Path(directory_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {root}")
    try:
        from langchain_community.document_loaders import DirectoryLoader, TextLoader
    except ImportError as exc:
        raise RuntimeError(
            "Document-loader dependencies are missing. Install requirements.txt."
        ) from exc

    loader = DirectoryLoader(
        str(root),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    documents = loader.load()
    resolved_root = root.resolve()
    for document in documents:
        metadata = dict(getattr(document, "metadata", {}) or {})
        absolute_source = Path(str(metadata.get("source", "")))
        try:
            relative_source = absolute_source.resolve().relative_to(resolved_root).as_posix()
        except (OSError, ValueError):
            relative_source = absolute_source.name
        metadata.update(_enrich_source_metadata(metadata, relative_source))
        document.metadata = metadata
    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=80):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError(
            "Text-splitting dependencies are missing. Install requirements.txt."
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    per_source = {}
    for chunk in chunks:
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        source = str(metadata.get("source", ""))
        chunk_index = per_source.get(source, 0)
        metadata["chunk_index"] = chunk_index
        if not metadata.get("timestamp"):
            timestamp = re.search(
                r"\[(\d{2}:\d{2}:\d{2})\]",
                str(getattr(chunk, "page_content", "")),
            )
            if timestamp:
                metadata["timestamp"] = timestamp.group(1)
        per_source[source] = chunk_index + 1
        metadata.update(_enrich_source_metadata(metadata, source))
        chunk.metadata = metadata
    return chunks


def _document_fields(document, position=0):
    page_content = getattr(document, "page_content", None)
    metadata = getattr(document, "metadata", None)
    if isinstance(document, dict):
        page_content = document.get("page_content", page_content)
        metadata = document.get("metadata", metadata)
    metadata = metadata if isinstance(metadata, dict) else {}
    source = str(metadata.get("source", ""))
    fields = _enrich_source_metadata(metadata, source)
    if "chunk_index" not in metadata:
        fields["chunk_index"] = position
    return str(page_content or ""), fields


def _embed_texts(embedding_function, texts, batch_size=32):
    texts = list(texts)
    batch_size = max(1, int(batch_size))
    if callable(getattr(embedding_function, "embed_documents", None)):
        vectors = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(
                embedding_function.embed_documents(texts[start:start + batch_size])
            )
    else:
        vectors = [embedding_function.embed_query(text) for text in texts]
    if len(vectors) != len(texts):
        raise ValueError(
            f"Embedding model returned {len(vectors)} vectors for {len(texts)} chunks."
        )
    return vectors


def create_vector_db(
    splitted_doc,
    save_path,
    embedding_function=None,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    device="auto",
    batch_size=32,
):
    """Embed chunks in batches and atomically persist a Parquet table."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is missing. Install requirements.txt.") from exc

    documents = list(splitted_doc)
    if embedding_function is None:
        embedding_function = get_embedding_function(embedding_model, device)

    page_content = []
    metadata_rows = []
    for position, document in enumerate(documents):
        content, metadata = _document_fields(document, position)
        page_content.append(content)
        metadata_rows.append(metadata)

    vectors = _embed_texts(embedding_function, page_content, batch_size)

    vector_db = pd.DataFrame.from_dict({
        "page_content": page_content,
        "source": [item["source"] for item in metadata_rows],
        "title": [item["title"] for item in metadata_rows],
        "video_id": [item["video_id"] for item in metadata_rows],
        "source_url": [item["source_url"] for item in metadata_rows],
        "timestamp": [item["timestamp"] for item in metadata_rows],
        "chunk_index": [item["chunk_index"] for item in metadata_rows],
        "embeddings": vectors,
    })
    save_vector_db(vector_db, save_path)
    return vector_db


def save_vector_db(vector_db, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=path.suffix or ".parquet",
        dir=str(path.parent),
    )
    os.close(file_descriptor)
    try:
        vector_db.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    except Exception:
        Path(temporary_path).unlink(missing_ok=True)
        raise


def load_vector_db_frame(save_path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is missing. Install requirements.txt.") from exc
    return pd.read_parquet(save_path)


def embeddings_tensor(vector_db):
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and NumPy are required for retrieval. Install requirements.txt."
        ) from exc

    if len(vector_db) == 0:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.tensor(
        np.asarray(vector_db["embeddings"].tolist(), dtype=np.float32),
        dtype=torch.float32,
    )


def load_vector_db(save_path):
    """Legacy helper returning only the embedding tensor."""
    return embeddings_tensor(load_vector_db_frame(save_path))


def _corpus_fingerprint(directory_path, model_name, chunk_size, chunk_overlap):
    """Return a content-based signature; transcript text is not written to the manifest."""
    root = Path(directory_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {root}")

    entries = []
    for path in sorted(root.rglob("*.txt")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"path": relative_path, "sha256": digest})

    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "embedding_model": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "files": entries,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    payload["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _manifest_path(save_path):
    path = Path(save_path)
    return path.with_suffix(path.suffix + ".manifest.json")


def _manifest_matches(manifest_path, expected):
    try:
        with Path(manifest_path).open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    return manifest == expected


def _write_manifest(manifest_path, manifest):
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".json",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
    except Exception:
        Path(temporary_path).unlink(missing_ok=True)
        raise


def load_or_create_vector_db(
    directory_path,
    save_path,
    embedding_function=None,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    device="auto",
    chunk_size=800,
    chunk_overlap=80,
    batch_size=32,
    force_rebuild=False,
):
    expected_manifest = _corpus_fingerprint(
        directory_path,
        embedding_model,
        chunk_size,
        chunk_overlap,
    )
    index_path = Path(save_path)
    manifest_path = _manifest_path(index_path)
    required_columns = {
        "page_content", "source", "title", "video_id", "source_url",
        "timestamp", "chunk_index", "embeddings",
    }

    if (
        not force_rebuild
        and index_path.is_file()
        and _manifest_matches(manifest_path, expected_manifest)
    ):
        try:
            cached = load_vector_db_frame(index_path)
            if required_columns.issubset(set(cached.columns)) and len(cached) > 0:
                return cached
        except Exception:
            # A missing/corrupt or old-schema cache is safely rebuilt below.
            pass

    documents = load_directory(directory_path)
    if not documents:
        raise ValueError(f"No .txt transcripts found in {directory_path!r}.")
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"No text chunks were produced from {directory_path!r}.")

    if embedding_function is None:
        embedding_function = get_embedding_function(embedding_model, device)
    frame = create_vector_db(
        splitted_doc=chunks,
        save_path=index_path,
        embedding_function=embedding_function,
        embedding_model=embedding_model,
        device=device,
        batch_size=batch_size,
    )
    _write_manifest(manifest_path, expected_manifest)
    return frame
