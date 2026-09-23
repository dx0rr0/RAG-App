import tempfile
import sys
import unittest
import types
from pathlib import Path
from unittest.mock import patch

from vector_db_manager import (
    _corpus_fingerprint,
    _embed_texts,
    _enrich_source_metadata,
    _manifest_matches,
    load_or_create_vector_db,
    split_documents,
)


class BatchedEmbeddingTests(unittest.TestCase):
    def test_sixty_five_chunks_use_three_embedding_batches(self):
        class FakeBatchEmbedder:
            def __init__(self):
                self.batch_sizes = []

            def embed_documents(self, texts):
                self.batch_sizes.append(len(texts))
                return [[float(len(text))] for text in texts]

        embedder = FakeBatchEmbedder()
        vectors = _embed_texts(embedder, [str(i) for i in range(65)], batch_size=32)

        self.assertEqual(len(vectors), 65)
        self.assertEqual(embedder.batch_sizes, [32, 32, 1])


class IndexFingerprintTests(unittest.TestCase):
    def test_split_documents_carries_timestamp_marker_to_metadata(self):
        class FakeDocument:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

        class FakeSplitter:
            def __init__(self, **kwargs):
                pass

            def split_documents(self, documents):
                return [FakeDocument("[00:00:12] frase", {"source": "clip.txt"})]

        fake_module = types.ModuleType("langchain_text_splitters")
        fake_module.RecursiveCharacterTextSplitter = FakeSplitter
        with patch.dict(sys.modules, {"langchain_text_splitters": fake_module}):
            chunks = split_documents([FakeDocument("original", {"source": "clip.txt"})])
        self.assertEqual(chunks[0].metadata["timestamp"], "00:00:12")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)

    def test_fingerprint_changes_with_content_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "transcripts"
            corpus.mkdir()
            transcript = corpus / "video.txt"
            transcript.write_text("frase original", encoding="utf-8")

            original = _corpus_fingerprint(corpus, "embed-v1", 800, 80)
            transcript.write_text("frase editada", encoding="utf-8")
            changed_content = _corpus_fingerprint(corpus, "embed-v1", 800, 80)
            changed_model = _corpus_fingerprint(corpus, "embed-v2", 800, 80)
            changed_split = _corpus_fingerprint(corpus, "embed-v1", 500, 50)

            self.assertNotEqual(original["fingerprint"], changed_content["fingerprint"])
            self.assertNotEqual(changed_content["fingerprint"], changed_model["fingerprint"])
            self.assertNotEqual(changed_content["fingerprint"], changed_split["fingerprint"])
            self.assertNotIn("frase editada", str(changed_content))

    def test_video_filename_yields_title_and_source_link(self):
        metadata = _enrich_source_metadata({}, "abcDEF_1234__Título de vídeo.txt")
        self.assertEqual(metadata["video_id"], "abcDEF_1234")
        self.assertEqual(metadata["title"], "Título de vídeo")
        self.assertEqual(
            metadata["source_url"],
            "https://www.youtube.com/watch?v=abcDEF_1234",
        )

    def test_manifest_match_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.parquet.manifest.json"
            path.write_text('{"schema_version": 2}', encoding="utf-8")
            self.assertTrue(_manifest_matches(path, {"schema_version": 2}))
            self.assertFalse(_manifest_matches(path, {"schema_version": 3}))


class IndexReuseTests(unittest.TestCase):
    def test_second_run_reuses_index_and_content_change_rebuilds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "transcripts"
            corpus.mkdir()
            transcript = corpus / "video.txt"
            transcript.write_text("primera versión", encoding="utf-8")
            index_path = root / "index.parquet"
            columns = [
                "page_content", "source", "title", "video_id", "source_url",
                "timestamp", "chunk_index", "embeddings",
            ]
            class FakeFrame:
                def __init__(self):
                    self.columns = columns

                def __len__(self):
                    return 1

            frame = FakeFrame()
            build_calls = []

            def fake_create(**kwargs):
                build_calls.append(kwargs)
                Path(kwargs["save_path"]).write_bytes(b"fake parquet marker")
                return frame

            with patch("vector_db_manager.load_directory", return_value=[object()]), \
                 patch("vector_db_manager.split_documents", return_value=[object()]), \
                 patch("vector_db_manager.create_vector_db", side_effect=fake_create):
                first = load_or_create_vector_db(
                    corpus, index_path, embedding_function=object()
                )
            self.assertIs(first, frame)
            self.assertEqual(len(build_calls), 1)

            with patch("vector_db_manager.load_vector_db_frame", return_value=frame) as load, \
                 patch("vector_db_manager.create_vector_db") as create:
                second = load_or_create_vector_db(
                    corpus, index_path, embedding_function=object()
                )
            self.assertIs(second, frame)
            load.assert_called_once_with(index_path)
            create.assert_not_called()

            transcript.write_text("segunda versión", encoding="utf-8")
            with patch("vector_db_manager.load_directory", return_value=[object()]), \
                 patch("vector_db_manager.split_documents", return_value=[object()]), \
                 patch("vector_db_manager.create_vector_db", side_effect=fake_create):
                third = load_or_create_vector_db(
                    corpus, index_path, embedding_function=object()
                )
            self.assertIs(third, frame)
            self.assertEqual(len(build_calls), 2)


if __name__ == "__main__":
    unittest.main()
