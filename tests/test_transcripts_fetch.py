from __future__ import annotations

import subprocess
from pathlib import Path


def test_download_single_subtitle_stops_after_first_language(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from skillsmith import transcripts_fetch

    calls: list[str] = []

    def fake_download(yt_dlp: str, video_id: str, out_dir: Path, language: str):
        calls.append(language)
        if language == "de":
            (out_dir / f"{video_id}.de.srt").write_text("Hallo", encoding="utf-8")
        return subprocess.CompletedProcess(args=[], returncode=0, stderr="")

    monkeypatch.setattr(transcripts_fetch, "_download_with_language", fake_download)

    result = transcripts_fetch._download_single_subtitle(
        "yt-dlp",
        "abc12345678",
        tmp_path,
        ["de", "en"],
    )

    assert result.success is True
    assert result.language == "de"
    assert calls == ["de"]
