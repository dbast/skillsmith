"""Encoding round-trip tests for the transcript pipeline."""

from __future__ import annotations

from pathlib import Path


def test_cleaned_files_preserve_unicode(tmp_path: Path) -> None:
    """Ensure German umlauts and special chars survive SRT-style cleaning."""
    from skillsmith.transcripts_clean import SRTCleaner

    srt_dir = tmp_path / "srt"
    srt_dir.mkdir()
    srt_file = srt_dir / "test.de.srt"
    # Minimal SRT with German text and HTML tags.
    srt_file.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\n"
        '<font color="white">Grüße aus der Käseküche für Anfänger</font>\n\n'
        "2\n00:00:05,000 --> 00:00:07,000\n"
        "Öl im grünen Käfig - öffentliche Übung\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "cleaned"
    cleaner = SRTCleaner(str(srt_dir), str(out_dir))
    metadata = cleaner.extract_all_files()

    assert metadata["total_files"] == 1
    cleaned = (out_dir / "test.de.txt").read_text(encoding="utf-8")
    assert "Grüße aus der Käseküche für Anfänger" in cleaned
    assert "Öl im grünen Käfig - öffentliche Übung" in cleaned
    assert "<font" not in cleaned
    assert "00:00" not in cleaned
    assert "Käseküche" in cleaned


def test_batch_summary_round_trip(tmp_path: Path) -> None:
    """Verify BatchSummary model can serialize and deserialize unicode."""
    from skillsmith.models import BatchSummary

    original = BatchSummary(
        batch_name="batch_01",
        main_topics=["Bunte Türen", "Grüne Tassen"],
        key_concepts={
            "Käseküche": {
                "de": "Raum mit grünen Türen",
                "en": "Room with green doors",
                "importance": "high",
            }
        },
        learning_prerequisites=["Umlaute lesen"],
        important_paragraphs=["Kapitel 9"],
        practical_examples=["Beispiel mit Öl und Käse"],
        key_takeaways=["Umlaute bleiben erhalten"],
        suggested_chapter_title="Bunte Türen und grüne Tassen",
        related_topics=["Küchenübungen"],
        content_summary="Überblick über Käseküchen",
        confidence_level=0.9,
    )

    path = tmp_path / "summary.json"
    import json

    path.write_text(
        json.dumps(original.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    loaded = BatchSummary.from_dict(json.loads(path.read_text(encoding="utf-8")))

    assert loaded.batch_name == original.batch_name
    assert loaded.main_topics == original.main_topics
    assert loaded.key_concepts == original.key_concepts
    assert loaded.suggested_chapter_title == original.suggested_chapter_title


def test_batch_summary_model_validate_json_preserves_unicode() -> None:
    """Typed summary validation should preserve German characters."""
    from skillsmith.models import BatchSummary

    raw = (
        '{"batch_name": "batch_01", '
        '"main_topics": ["Öffentliche Übung"], '
        '"key_concepts": {"Übung": {"de": "öffentliche Übung"}}, '
        '"content_summary": "Öl im grünen Käfig - öffentliche Übung"}'
    )
    parsed = BatchSummary.model_validate_json(raw)
    assert parsed.content_summary == "Öl im grünen Käfig - öffentliche Übung"


def test_batch_pipeline_artifacts_use_utf8(tmp_path: Path) -> None:
    """Pipeline artifacts should be written with UTF-8 and read back correctly."""
    from skillsmith.batches_process import BatchPipeline
    from skillsmith.models import BatchSummary

    (tmp_path / "manifest.json").write_text('{"batch_01": {}}', encoding="utf-8")

    runner = _dummy_runner()
    pipeline = BatchPipeline(batch_dir=tmp_path, runner=runner)

    summary = BatchSummary(
        batch_name="batch_01",
        main_topics=["Grundlagen"],
        key_concepts={"ABC": {"de": "Äpfel, Birnen, Käse", "importance": "high"}},
        learning_prerequisites=[],
        important_paragraphs=[],
        practical_examples=[],
        key_takeaways=[],
        suggested_chapter_title="Grundlagen der Käseküche",
        related_topics=[],
        content_summary="Einführung",
        confidence_level=0.8,
    )

    pipeline.save_artifacts(
        summaries={"batch_01": summary},
        duration=1.23,
    )

    summaries_path = tmp_path / "batch_summaries.json"
    meta_path = tmp_path / "processing_meta.json"

    # All files must exist and parse back correctly.
    assert summaries_path.exists()
    assert meta_path.exists()

    import json

    summaries = json.loads(summaries_path.read_text(encoding="utf-8"))
    assert summaries["batch_01"]["suggested_chapter_title"] == "Grundlagen der Käseküche"

    meta_loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta_loaded["successful"] == 1


def test_batch_summary_model_handles_escaped_german() -> None:
    """Pydantic validation should decode escaped unicode in JSON summaries."""
    from skillsmith.models import BatchSummary

    raw = (
        '{"batch_name": "batch_01", '
        '"main_topics": ["Käseküche"], '
        '"key_concepts": {}, '
        '"content_summary": "Gr\\u00fcne T\\u00fcr mit K\\u00e4se"}'
    )
    parsed = BatchSummary.model_validate_json(raw)
    assert parsed.content_summary == "Grüne Tür mit Käse"


async def test_batch_pipeline_records_fetch_errors(tmp_path: Path) -> None:
    """One failed model call should not abort the whole batch run."""
    import json

    from skillsmith.batches_process import BatchPipeline

    (tmp_path / "manifest.json").write_text('{"batch_01": {}}', encoding="utf-8")
    (tmp_path / "batch_01.txt").write_text("Text", encoding="utf-8")

    pipeline = BatchPipeline(batch_dir=tmp_path, runner=_failing_runner())
    results = await pipeline.fetch_typed_responses()

    assert results == []
    error_path = tmp_path / "cleaned_json" / "batch_01.json"
    saved = json.loads(error_path.read_text(encoding="utf-8"))
    assert saved["batch_name"] == "batch_01"
    assert "RuntimeError: upstream 502" in saved["error"]

    parsed = pipeline.parse_typed_responses()
    assert parsed["batch_01"]["error"] == saved["error"]


def _dummy_runner():
    class Dummy:
        provider = "dummy"
        model = "dummy-model"
        concurrency = 1

        async def run(self, *, system, user, output_type=str):
            raise NotImplementedError

    return Dummy()


def _failing_runner():
    class Failing:
        provider = "dummy"
        model = "dummy-model"
        concurrency = 1

        async def run(self, *, system, user, output_type=str):
            raise RuntimeError("upstream 502")

    return Failing()
