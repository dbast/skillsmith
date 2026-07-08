"""Prompt configuration tests."""

from __future__ import annotations

from pathlib import Path

from skillsmith.config import load_config
from skillsmith.prompts import build_batch_prompt


def _write_demo_config(path: Path) -> Path:
    path.write_text(
        """
project:
  slug: demo-project
  language: en
  domain: Demo domain
  output_language: English

source:
  type: youtube_channel
  channel_url: https://www.youtube.com/@demo
  subtitle_languages: [en]

paths:
  raw_subtitles: output/org
  cleaned_transcripts: output/cleaned_transcripts
  batches: output/cleaned_transcripts/batches
  final_skill: output/demo-skill

prompts:
  system_role: Demo system role.
  source_description: Demo source material.

skill:
  name: demo-skill
  title: Demo Skill
  description: Helps agents use the demo source material.
  source_channel: Demo channel
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_prompts_use_configured_domain_and_language(tmp_path: Path) -> None:
    config_path = _write_demo_config(tmp_path / "project.yml")
    config = load_config(config_path)

    batch_prompt = build_batch_prompt("batch_01", "demo transcript", config)

    assert "Demo domain" in batch_prompt
    assert "English" in batch_prompt
