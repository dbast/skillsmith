"""Compositional prompt builders for transcript summarization.

Separates the *structure* of the prompts (system role, JSON schema, rules)
from the *content* (batch text, aggregated topics) so both are easy to inspect
and dry-run without calling an API.
"""

from __future__ import annotations

from typing import Any


def _output_language(project_config: Any | None) -> str:
    return (
        project_config.project.output_language
        if project_config
        else "the configured output language"
    )


def _domain(project_config: Any | None, fallback: str = "the source domain") -> str:
    return project_config.project.domain if project_config else fallback


def _summarization_rules(project_config: Any | None) -> str:
    domain = _domain(project_config)
    output_language = _output_language(project_config)
    return f"""Rules:
- Do NOT explain your reasoning, think out loud, or include a chain-of-thought.
- Stay faithful to the transcript text; do not invent facts, numbers, examples, or references.
- Prefer terminology from the transcript and write extracted labels in {output_language} where useful.
- If the transcript text is fragmented or repetitive, still extract the most
  salient {domain} topics and practical examples you can find.
"""


def build_batch_prompt(
    batch_name: str,
    batch_text: str,
    project_config: Any | None = None,
) -> str:
    """Prompt used in Step 1 to extract structured topics from a batch."""
    source_description = (
        project_config.prompts.source_description.strip()
        if project_config is not None
        else "transcribed content from video tutorials"
    )
    domain = _domain(project_config)
    output_language = _output_language(project_config)
    return f"""Analyze this {source_description}.

BATCH: {batch_name}
TEXT:
{batch_text}

Extract these fields for the structured output:
- batch_name: use the exact batch name above.
- main_topics: salient {domain} topics covered in this batch.
- key_concepts: important concepts with definitions in {output_language} and importance levels.
- learning_prerequisites: prerequisite concepts a reader/agent should know.
- important_paragraphs: important references, source citations, standards, laws, or named resources.
- practical_examples: concrete cases and structuring examples.
- key_takeaways: concise implementation-relevant takeaways.
- suggested_chapter_title: a fitting chapter or reference title in {output_language}.
- related_topics: cross-links to adjacent topics.
- content_summary: a compact factual summary of the batch content.
- confidence_level: 0.0-1.0 based on transcript clarity and coverage.

{_summarization_rules(project_config)}
"""
