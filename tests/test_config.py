"""Configuration loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from skillsmith.config import load_config


def _write_config(path: Path, *, skill_name: str = "demo-skill") -> Path:
    path.write_text(
        f"""
project:
  slug: demo-project
  language: de
  domain: Demo domain
  output_language: Deutsch

source:
  type: youtube_channel
  channel_url: https://www.youtube.com/@demo
  subtitle_languages: [de, en]

paths:
  raw_subtitles: output/org
  cleaned_transcripts: output/cleaned_transcripts
  batches: output/cleaned_transcripts/batches
  final_skill: output/{skill_name}

batching:
  target_chars: 300000

models:
  summarization:
    provider: openrouter
    model: minimax/minimax-m2.7
  skill_builder:
    provider: openrouter
    model: anthropic/claude-sonnet-4.5

prompts:
  system_role: Demo system role.
  source_description: Demo source material.

skill:
  name: {skill_name}
  title: Demo Skill
  description: Helps agents use the demo source material.
  source_channel: Demo channel
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_valid_config_loads() -> None:
    config = load_config(Path("config_template.yml"))

    assert config.project.slug == "example-channel-skill"
    assert config.source.channel_url == "https://www.youtube.com/@your-channel"
    assert config.models.summarization.model == "minimax/minimax-m2.7"
    assert config.models.skill_builder.model == "anthropic/claude-sonnet-4.5"
    assert config.skill.name == "example-channel-skill"


def test_relative_paths_resolve_against_config_dir(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "project.yml")

    config = load_config(config_path)

    assert config.raw_subtitles_dir == tmp_path / "output/org"
    assert config.cleaned_transcripts_dir == tmp_path / "output/cleaned_transcripts"
    assert config.batches_dir == tmp_path / "output/cleaned_transcripts/batches"
    assert config.final_skill_dir == tmp_path / "output/demo-skill"


def test_invalid_skill_name_fails(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "project.yml", skill_name="Bad Skill Name")

    with pytest.raises(ValidationError):
        load_config(config_path)
