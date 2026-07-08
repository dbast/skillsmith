"""Typed project configuration for reusable channel-to-skill pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer
import yaml
from pydantic import BaseModel, Field, field_validator
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False, help=__doc__)


class ProjectSection(BaseModel):
    """High-level project identity."""

    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9\-]*[a-z0-9]$")
    language: str = "de"
    domain: str
    output_language: str = "Deutsch"


class SourceSection(BaseModel):
    """Source data configuration."""

    type: Literal["youtube_channel"] = "youtube_channel"
    channel_url: str
    subtitle_languages: list[str] = Field(default_factory=lambda: ["de", "en"])

    @field_validator("subtitle_languages")
    @classmethod
    def validate_languages(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("subtitle_languages must not be empty")
        for lang in value:
            if len(lang) < 2 or not lang.replace("-", "").isalpha():
                raise ValueError(f"invalid subtitle language: {lang}")
        return value


class PathsSection(BaseModel):
    """Project paths. Relative paths resolve against the config file directory."""

    raw_subtitles: Path = Path("output/org")
    cleaned_transcripts: Path = Path("output/cleaned_transcripts")
    batches: Path = Path("output/cleaned_transcripts/batches")
    final_skill: Path


class BatchingSection(BaseModel):
    """Transcript batching behavior."""

    target_chars: int = Field(default=300_000, ge=10_000)


class ProviderModel(BaseModel):
    """A configured LLM provider/model pair."""

    provider: Literal["openrouter"] = "openrouter"
    model: str


class ModelsSection(BaseModel):
    """Model choices per pipeline phase."""

    summarization: ProviderModel = ProviderModel(
        provider="openrouter",
        model="minimax/minimax-m2.7",
    )
    skill_builder: ProviderModel = ProviderModel(
        provider="openrouter",
        model="anthropic/claude-sonnet-4.5",
    )


class PromptsSection(BaseModel):
    """Prompt variables used by prompt builders."""

    system_role: str
    source_description: str


class SkillSection(BaseModel):
    """agentskills.io package metadata."""

    name: str = Field(pattern=r"^[a-z0-9](?:[a-z0-9\-]{0,62}[a-z0-9])?$")
    title: str
    description: str = Field(max_length=1024)
    source_channel: str
    output_dir: Path | None = None


class ProjectConfig(BaseModel):
    """Complete reusable project config."""

    project: ProjectSection
    source: SourceSection
    paths: PathsSection
    batching: BatchingSection = BatchingSection()
    models: ModelsSection = ModelsSection()
    prompts: PromptsSection
    skill: SkillSection
    config_path: Path | None = Field(default=None, exclude=True)

    @property
    def base_dir(self) -> Path:
        if self.config_path is None:
            return Path.cwd()
        return self.config_path.resolve().parent

    def resolve_path(self, path: Path) -> Path:
        """Resolve config-relative paths."""
        return path if path.is_absolute() else self.base_dir / path

    @property
    def raw_subtitles_dir(self) -> Path:
        return self.resolve_path(self.paths.raw_subtitles)

    @property
    def cleaned_transcripts_dir(self) -> Path:
        return self.resolve_path(self.paths.cleaned_transcripts)

    @property
    def batches_dir(self) -> Path:
        return self.resolve_path(self.paths.batches)

    @property
    def final_skill_dir(self) -> Path:
        return self.resolve_path(self.skill.output_dir or self.paths.final_skill)


def load_config(config_path: Path = Path("config.yml")) -> ProjectConfig:
    """Load and validate a YAML config file."""
    path = config_path.resolve()
    if not path.exists():
        raise typer.BadParameter(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Config must be a mapping: {path}")
    config = ProjectConfig.model_validate(data)
    config.config_path = path
    return config


def print_config_summary(config: ProjectConfig) -> None:
    """Print a human-readable config summary."""
    console.print(f"[bold]Project:[/] {config.project.slug} ({config.project.domain})")
    console.print(f"[bold]Channel:[/] {config.source.channel_url}")
    console.print(f"[bold]Subtitle languages:[/] {', '.join(config.source.subtitle_languages)}")
    console.print(f"[bold]Raw subtitles:[/] {config.raw_subtitles_dir}")
    console.print(f"[bold]Cleaned transcripts:[/] {config.cleaned_transcripts_dir}")
    console.print(f"[bold]Batches:[/] {config.batches_dir}")
    console.print(f"[bold]Batch target chars:[/] {config.batching.target_chars:,}")
    console.print(
        f"[bold]Summarization model:[/] "
        f"{config.models.summarization.provider}:{config.models.summarization.model}"
    )
    console.print(
        f"[bold]Skill builder model:[/] "
        f"{config.models.skill_builder.provider}:{config.models.skill_builder.model}"
    )
    console.print(f"[bold]Skill output:[/] {config.final_skill_dir}")


@app.command()
def main(
    config: Path = typer.Option(
        Path("config.yml"),
        "--config",
        help="Path to project config YAML.",
    ),
) -> None:
    """Validate and summarize a project config."""
    project_config = load_config(config)
    print_config_summary(project_config)
    console.print("[green]Config valid.[/]")


if __name__ == "__main__":
    app()
