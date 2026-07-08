"""Build an agentskills.io package from transcript batch summaries."""

from __future__ import annotations

import asyncio
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from rich.console import Console

from .config import ProjectConfig, load_config
from .models import (
    BatchSummary,
    GeneratedReference,
    SkillPlan,
    TopicHit,
    ensure_openrouter_api_key,
)

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


@dataclass
class SkillBuilderDeps:
    """Dependencies exposed to Pydantic AI tools."""

    summaries: dict[str, BatchSummary]
    topic_to_batches: dict[str, list[str]]
    project_config: ProjectConfig


def load_summaries(batch_dir: Path) -> dict[str, BatchSummary]:
    """Load validated batch summaries from the batch processing output."""
    path = batch_dir / "batch_summaries.json"
    if not path.exists():
        raise typer.BadParameter(f"Missing {path}; run transcripts:process first")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: BatchSummary.from_dict(obj) for name, obj in raw.items()}


def build_topic_index(summaries: dict[str, BatchSummary]) -> dict[str, list[str]]:
    """Map topics and related topics to their source batches."""
    topic_to_batches: dict[str, list[str]] = defaultdict(list)
    for batch_name, summary in summaries.items():
        for topic in [*summary.main_topics, *summary.related_topics]:
            if batch_name not in topic_to_batches[topic]:
                topic_to_batches[topic].append(batch_name)
    return dict(topic_to_batches)


def build_source_digest(
    summaries: dict[str, BatchSummary],
    topic_to_batches: dict[str, list[str]],
    *,
    topic_limit: int = 160,
    concept_limit: int = 80,
) -> str:
    """Compact source digest for planning without flooding the model."""
    concept_counter: Counter[str] = Counter()
    paragraph_counter: Counter[str] = Counter()
    examples: list[str] = []
    for summary in summaries.values():
        concept_counter.update(summary.key_concepts.keys())
        paragraph_counter.update(summary.important_paragraphs)
        examples.extend(summary.practical_examples[:2])

    topics = sorted(topic_to_batches)[:topic_limit]
    concepts = concept_counter.most_common(concept_limit)
    paragraphs = paragraph_counter.most_common(80)
    example_sample = examples[:80]

    return "\n".join(
        [
            f"Total batches: {len(summaries)}",
            f"Total indexed topics: {len(topic_to_batches)}",
            "",
            "Representative topics:",
            *[f"- {topic} (batches: {', '.join(topic_to_batches[topic])})" for topic in topics],
            "",
            "Frequent concepts:",
            *[f"- {name} ({count}x)" for name, count in concepts],
            "",
            "Frequent paragraphs:",
            *[f"- {name} ({count}x)" for name, count in paragraphs],
            "",
            "Representative practical examples:",
            *[f"- {example}" for example in example_sample],
        ]
    )


def make_planner_agent(
    model_name: str,
    project_config: ProjectConfig,
) -> Any:
    """Create the typed planning agent."""
    ensure_openrouter_api_key()
    agent = Agent(
        OpenRouterModel(model_name),
        deps_type=SkillBuilderDeps,
        output_type=SkillPlan,
        retries={"output": 3, "tools": 2},
        model_settings=ModelSettings(temperature=0.1, max_tokens=20_000),
        instructions=(
            f"You design agentskills.io packages for AI agents in this domain: "
            f"{project_config.project.domain}. Produce a concrete, "
            "validated SkillPlan. Prefer one-level reference paths under references/. "
            "Use enough references to make the source knowledge discoverable, but keep "
            "the root SKILL.md small. Keep every field concise; do not write prose "
            "paragraphs inside the plan."
        ),
    )

    @agent.tool
    def search_topics(
        ctx: RunContext[SkillBuilderDeps], query: str, limit: int = 12
    ) -> list[TopicHit]:
        """Search indexed transcript topics for a planning keyword."""
        query_words = {part.lower() for part in query.split() if part.strip()}
        hits: list[TopicHit] = []
        for topic, batches in ctx.deps.topic_to_batches.items():
            topic_lower = topic.lower()
            if query.lower() in topic_lower or any(word in topic_lower for word in query_words):
                hits.extend(
                    TopicHit(batch_name=batch, topic=topic, reason=f"matched query {query!r}")
                    for batch in batches[:3]
                )
        return hits[:limit]

    @agent.tool
    def load_batch_summary(ctx: RunContext[SkillBuilderDeps], batch_name: str) -> dict[str, Any]:
        """Load one compact batch summary by batch name."""
        summary = ctx.deps.summaries.get(batch_name)
        if summary is None:
            return {"error": f"unknown batch {batch_name}"}
        return {
            "batch_name": summary.batch_name,
            "main_topics": summary.main_topics,
            "key_concepts": list(summary.key_concepts)[:30],
            "important_paragraphs": summary.important_paragraphs,
            "practical_examples": summary.practical_examples[:8],
            "key_takeaways": summary.key_takeaways[:8],
            "content_summary": summary.content_summary[:2500],
        }

    return agent


def make_reference_agent(
    model_name: str,
    project_config: ProjectConfig,
) -> Any:
    """Create the reference-file generation agent."""
    ensure_openrouter_api_key()
    agent = Agent(
        OpenRouterModel(model_name),
        deps_type=SkillBuilderDeps,
        output_type=str,
        retries={"output": 3, "tools": 2},
        model_settings=ModelSettings(temperature=0.2, max_tokens=14_000),
        instructions=(
            f"Generate one agentskills.io reference markdown file for "
            f"{project_config.project.domain}. Write for an AI agent, not for a "
            "long-form human reader. "
            "Include trigger conditions, facts to collect, rules, checklists, examples, "
            "and cross-references. Return only Markdown, not JSON."
        ),
    )

    @agent.tool
    def search_topics(
        ctx: RunContext[SkillBuilderDeps], query: str, limit: int = 10
    ) -> list[TopicHit]:
        """Find source batches and topics for this reference."""
        query_words = {part.lower() for part in query.split() if part.strip()}
        hits: list[TopicHit] = []
        for topic, batches in ctx.deps.topic_to_batches.items():
            topic_lower = topic.lower()
            if query.lower() in topic_lower or any(word in topic_lower for word in query_words):
                hits.extend(
                    TopicHit(batch_name=batch, topic=topic, reason=f"matched query {query!r}")
                    for batch in batches[:3]
                )
        return hits[:limit]

    @agent.tool
    def load_batch_summary(ctx: RunContext[SkillBuilderDeps], batch_name: str) -> dict[str, Any]:
        """Load source details for one transcript batch."""
        summary = ctx.deps.summaries.get(batch_name)
        if summary is None:
            return {"error": f"unknown batch {batch_name}"}
        return summary.to_dict()

    return agent


async def create_skill_plan(
    deps: SkillBuilderDeps,
    model_name: str,
    source_digest: str,
) -> SkillPlan:
    """Ask Pydantic AI to produce the skill package plan."""
    project_config = deps.project_config
    agent = make_planner_agent(model_name, project_config)
    result = await agent.run(
        "\n".join(
            [
                f"Create a complete agentskills.io SkillPlan named {project_config.skill.name}.",
                f"Skill title: {project_config.skill.title}",
                f"Skill description: {project_config.skill.description}",
                f"Domain: {project_config.project.domain}",
                f"Source channel: {project_config.skill.source_channel}",
                "The skill must help an agent analyze newly described cases, route to the",
                "right references, ask clarifying questions, and apply concepts from the",
                "source material.",
                "",
                "Plan exactly 12 reference files. Keep each field concise: 2-8 triggers,",
                "up to 6 client archetypes, up to 12 source topics, up to 10 paragraphs,",
                "and up to 5 dependencies per reference.",
                "",
                "Cover the major case families, entity types, jurisdictions, process risks,",
                "and edge cases that are supported by the digest.",
                "",
                "Source digest:",
                source_digest,
            ]
        ),
        deps=deps,
    )
    return result.output.model_copy(
        update={
            "name": project_config.skill.name,
            "description": project_config.skill.description,
            "source_channel": project_config.skill.source_channel,
        }
    )


def render_skill_md(plan: SkillPlan, project_config: ProjectConfig) -> str:
    """Render root SKILL.md deterministically from the validated plan."""
    references = "\n".join(
        f"- `{ref.path}` - {ref.title}: {ref.purpose}" for ref in plan.references
    )
    routing_steps = "\n".join(f"{i}. {step}" for i, step in enumerate(plan.routing_steps, 1))
    examples = "\n".join(f"- {example}" for example in plan.activation_examples)

    return f"""---
name: {plan.name}
description: {json.dumps(plan.description, ensure_ascii=False)}
metadata:
  version: {json.dumps(plan.version, ensure_ascii=False)}
  source_channel: {json.dumps(plan.source_channel, ensure_ascii=False)}
  generator: channel-to-skill-builder
---

# {project_config.skill.title}

## When To Use This Skill

Use this skill when the user asks about {project_config.project.domain} and the
case should be routed through source-derived references before answering.

Example triggers:

{examples}

## Agent Workflow

{routing_steps}

## Reference Routing Index

Read only the reference files needed for the user's case. Start with the most
specific matching reference, then load dependencies listed inside that file.

{references}

## Guardrails

- Do not present {project_config.project.domain} recommendations as professional advice;
  explain assumptions and ask for missing facts.
- Distinguish source-supported facts, assumptions, jurisdictions, and edge cases.
- Cite references only when the reference file supports them.
- Flag uncertainty and recommend professional review for implementation.
"""


async def generate_reference(
    deps: SkillBuilderDeps,
    model_name: str,
    plan: SkillPlan,
    ref_index: int,
) -> GeneratedReference:
    """Generate one reference file via Pydantic AI."""
    ref = plan.references[ref_index]
    agent = make_reference_agent(model_name, deps.project_config)
    result = await agent.run(
        "\n".join(
            [
                f"Generate the complete markdown file for {ref.path}.",
                f"Title: {ref.title}",
                f"Purpose: {ref.purpose}",
                f"Triggers: {', '.join(ref.triggers)}",
                f"Client archetypes: {', '.join(ref.client_archetypes)}",
                f"Source topics: {', '.join(ref.source_topics)}",
                f"Key paragraphs: {', '.join(ref.key_paragraphs)}",
                f"Dependencies: {', '.join(ref.dependencies)}",
                "",
                "Required markdown sections:",
                f"# <title in {deps.project_config.project.output_language}>",
                "## When To Use This Reference",
                "## Facts To Collect",
                "## Core Rules And Paragraphs",
                "## Decision Checklist",
                "## Structure Options",
                "## Example Cases",
                "## Related References",
                "",
                "Use tools if you need to inspect relevant source batches.",
                "Return ONLY the markdown file content. Do not wrap it in JSON.",
                f"The first line must be: # {ref.title}",
            ]
        ),
        deps=deps,
    )
    markdown = result.output.strip()
    if not markdown.startswith("#"):
        markdown = f"# {ref.title}\n\n{markdown}"
    return GeneratedReference(path=ref.path, title=ref.title, markdown=markdown)


def write_assets(output_dir: Path, plan: SkillPlan) -> None:
    """Write deterministic helper assets."""
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    (assets_dir / "client_intake_checklist.md").write_text(
        """# Case Intake Checklist

## User Goal
- Desired outcome
- Current context
- Constraints and deadlines
- Prior attempts or decisions

## Source-Relevant Facts
- Facts mentioned by the user
- Missing facts needed before applying a reference
- Terms, entities, tools, or processes involved
- Jurisdiction, environment, or version details where relevant

## Decision Goals
- Options to compare
- Trade-offs to explain
- Risks to flag
- Implementation steps to recommend

## Risk Facts
- Unverified assumptions
- Edge cases
- Compliance, safety, or review needs
""",
        encoding="utf-8",
    )

    (assets_dir / "structure_comparison_template.md").write_text(
        """# Structure Comparison Template

| Option | Benefits | Costs / Burden | Constraints | Risks | Best Fit |
|---|---|---|---|---|---|
| Option A |  |  |  |  |  |
| Option B |  |  |  |  |  |
| Option C |  |  |  |  |  |

## Recommendation Logic
1. Rule out unsupported or high-risk options.
2. Separate user goals from implementation constraints.
3. Check assumptions against source-backed references.
4. Compare cost, complexity, reversibility, and expected benefit.
""",
        encoding="utf-8",
    )

    for asset in plan.assets:
        path = output_dir / asset.path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {asset.title}\n\n{asset.purpose}\n", encoding="utf-8")


def zip_skill(output_dir: Path) -> Path:
    """Create a zip archive next to the skill directory."""
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir.parent))
    return zip_path


@app.command()
def main(
    config: Path = typer.Option(
        Path("config.yml"),
        "--config",
        help="Path to project config YAML.",
    ),
    batch_dir: Path | None = typer.Option(
        None,
        "--batch-dir",
        help="Override configured directory containing batch_summaries.json.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Override configured skill package directory to create.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Override configured provider. Skill builder currently supports openrouter.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Override configured OpenRouter model for Pydantic AI.",
    ),
    reference_limit: int | None = typer.Option(
        None,
        "--reference-limit",
        min=1,
        help="Generate only the first N references for testing.",
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Generate only skill_plan.json and SKILL.md, no reference files.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print source stats and planned output path without API calls.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Remove existing output directory before generating.",
    ),
) -> None:
    """Generate a full agentskills.io package from batch summaries."""
    project_config = load_config(config)
    resolved_batch_dir = batch_dir or project_config.batches_dir
    resolved_output_dir = output_dir or project_config.final_skill_dir
    skill_model = model or project_config.models.skill_builder.model
    skill_provider = provider or project_config.models.skill_builder.provider
    if skill_provider != "openrouter":
        raise typer.BadParameter("skill:build currently supports provider=openrouter only")

    summaries = load_summaries(resolved_batch_dir)
    topic_to_batches = build_topic_index(summaries)
    deps = SkillBuilderDeps(
        summaries=summaries,
        topic_to_batches=topic_to_batches,
        project_config=project_config,
    )
    source_digest = build_source_digest(summaries, topic_to_batches)

    console.print(f"[bold]Loaded {len(summaries)} summaries[/]")
    console.print(f"[bold]Indexed {len(topic_to_batches)} topics[/]")
    console.print(f"[bold]Output directory:[/] {resolved_output_dir}")

    if dry_run:
        console.print("[cyan]Dry run: no API calls, no files written.[/]")
        console.print(source_digest[:4000])
        return

    if force and resolved_output_dir.exists():
        shutil.rmtree(resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    (resolved_output_dir / "references").mkdir(exist_ok=True)

    async def _run() -> SkillPlan:
        plan = await create_skill_plan(deps, skill_model, source_digest)
        (resolved_output_dir / "skill_plan.json").write_text(
            plan.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        (resolved_output_dir / "SKILL.md").write_text(
            render_skill_md(plan, project_config),
            encoding="utf-8",
        )
        write_assets(resolved_output_dir, plan)

        refs = plan.references[: reference_limit or len(plan.references)]
        if not plan_only:
            for index, ref in enumerate(refs, 1):
                console.print(f"[bold cyan]Generating {index}/{len(refs)}:[/] {ref.path}")
                generated = await generate_reference(deps, skill_model, plan, index - 1)
                path = resolved_output_dir / generated.path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(generated.markdown.rstrip() + "\n", encoding="utf-8")

        zip_path = zip_skill(resolved_output_dir)
        console.print(f"[green]Wrote skill:[/] {resolved_output_dir}")
        console.print(f"[green]Wrote zip:[/] {zip_path}")
        return plan

    asyncio.run(_run())


if __name__ == "__main__":
    app()
