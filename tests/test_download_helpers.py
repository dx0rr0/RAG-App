import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path

from download_channel import (
    format_timestamped_transcription,
    _transcribe_file,
    resolve_whisper_device,
    video_file_stem,
)


class DownloadHelperTests(unittest.TestCase):
    def test_video_id_prevents_sanitized_title_collisions(self):
        first = SimpleNamespace(id="video_A1", title="A/B: Title")
        second = SimpleNamespace(id="video_B2", title="A?B: Title")
        self.assertNotEqual(video_file_stem(first), video_file_stem(second))
        self.assertTrue(video_file_stem(first).startswith("video_A1__"))

    def test_transcription_format_keeps_segment_timestamps(self):
        segments = [
            SimpleNamespace(start=12.9, text="  Primera frase. "),
            SimpleNamespace(start=3723, text=" Segunda frase."),
        ]
        self.assertEqual(
            format_timestamped_transcription(segments),
            "[00:00:12] Primera frase.\n[01:02:03] Segunda frase.",
        )

    def test_transcription_autodetect_writes_timecoded_segments(self):
        class FakeModel:
            def __init__(self):
                self.language = "not-called"

            def transcribe(self, path, beam_size, language):
                self.language = language
                return [SimpleNamespace(start=5, text=" texto")], SimpleNamespace(language="es")

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "clip.txt"
            model = FakeModel()
            _transcribe_file(model, "audio.mp3", output_path, language=None)
            self.assertIsNone(model.language)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "[00:00:05] texto")

    def test_whisper_device_falls_back_when_cuda_is_unavailable(self):
        self.assertEqual(resolve_whisper_device("auto", cuda_available=False), "cpu")
        self.assertEqual(resolve_whisper_device("cuda", cuda_available=False), "cpu")
        self.assertEqual(resolve_whisper_device("auto", cuda_available=True), "cuda")

    def test_import_does_not_load_downloader_or_transcription_packages(self):
        import sys

        self.assertNotIn("pytubefix", sys.modules)
        self.assertNotIn("faster_whisper", sys.modules)


if __name__ == "__main__":
    unittest.main()
