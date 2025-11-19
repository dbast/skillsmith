"""Domain models for the channel-to-skill pipeline."""

from __future__ import annotations

import os
from typing import Any, Literal

import typer
from pydantic import BaseModel, Field


def ensure_openrouter_api_key() -> None:
    """Fail fast before constructing OpenRouter-backed agents."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise typer.BadParameter("OPENROUTER_API_KEY environment variable not set")


class BatchSummary(BaseModel):
    """Structured summary for one transcript batch."""

    batch_name: str = ""
    main_topics: list[str] = Field(default_factory=list)
    key_concepts: dict[str, Any] = Field(default_factory=dict)
    learning_prerequisites: list[str] = Field(default_factory=list)
    important_paragraphs: list[str] = Field(default_factory=list)
    practical_examples: list[str] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)
    suggested_chapter_title: str = ""
    related_topics: list[str] = Field(default_factory=list)
    content_summary: str = ""
    confidence_level: float = Field(default=0.0, ge=0.0, le=1.0)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchSummary:
        return cls.model_validate(data)


class TypedResponse(BaseModel):
    """Cached typed model response for one batch."""

    batch_name: str
    typed_response: BatchSummary
    finish_reason: str = "unknown"
    provider: str
    model: str

    def summary(self) -> BatchSummary:
        return self.typed_response


class PipelineMeta(BaseModel):
    """Metadata persisted alongside processing results."""

    schema_version: int
    generated_at: str
    provider: str
    model: str
    total_batches: int
    successful: int
    failed: int
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ReferenceSpec(BaseModel):
    """A discoverable reference file inside an Agent Skill."""

    path: str = Field(
        pattern=r"^references/[a-z0-9_\-]+\.md$",
        description="One-level relative path under references/.",
    )
    title: str = Field(max_length=120, description="Human-readable reference title.")
    purpose: str = Field(
        max_length=350,
        description="What this reference helps the agent decide or do.",
    )
    triggers: list[str] = Field(
        min_length=2,
        max_length=8,
        description="User words, goals, or facts that should load it.",
    )
    client_archetypes: list[str] = Field(default_factory=list, max_length=6)
    source_topics: list[str] = Field(default_factory=list, max_length=12)
    key_paragraphs: list[str] = Field(default_factory=list, max_length=10)
    dependencies: list[str] = Field(default_factory=list, max_length=5)
    priority: Literal["core", "advanced", "supporting"] = "core"


class AssetSpec(BaseModel):
    """A deterministic helper/template bundled with the skill."""

    path: str = Field(pattern=r"^assets/[a-z0-9_\-]+\.md$")
    title: str = Field(max_length=120)
    purpose: str = Field(max_length=300)


class SkillPlan(BaseModel):
    """Plan for a full agentskills.io-compatible skill package."""

    name: str = Field(
        pattern=r"^[a-z0-9](?:[a-z0-9\-]{0,62}[a-z0-9])?$",
        description="Skill directory/name; lowercase letters, numbers, hyphens only.",
    )
    description: str = Field(max_length=1024)
    version: str = "1.0.0"
    source_channel: str = "source channel"
    activation_examples: list[str] = Field(default_factory=list, min_length=4, max_length=12)
    routing_steps: list[str] = Field(default_factory=list, min_length=4, max_length=8)
    references: list[ReferenceSpec] = Field(min_length=8, max_length=16)
    assets: list[AssetSpec] = Field(default_factory=list, max_length=8)


class GeneratedReference(BaseModel):
    """Generated content for one reference markdown file."""

    path: str = Field(pattern=r"^references/[a-z0-9_\-]+\.md$")
    title: str
    markdown: str = Field(description="Complete markdown body for the reference file.")


class TopicHit(BaseModel):
    """Search result over batch summaries."""

    batch_name: str
    topic: str
    reason: str
