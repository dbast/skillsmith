"""Process transcript batches into typed summaries.

This is the single Pixi process step for batch summarization, offline typed
cache validation, and summary aggregation.

Examples:

    pixi run transcripts:process --dry-run
    pixi run transcripts:process --step 2
    pixi run transcripts:process --only batch_01,batch_02
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from .config import ProjectConfig, load_config
from .models import (
    BatchSummary,
    PipelineMeta,
    TypedResponse,
    ensure_openrouter_api_key,
)
from .prompts import build_batch_prompt

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()
SCHEMA_VERSION = 1
DEFAULT_MAX_TOKENS = 16_192
DEFAULT_CONCURRENCY = 6
OUTPUT_RETRIES = 3


class ModelRunner:
    """Small Pydantic AI runner for the configured OpenRouter model."""

    provider = "openrouter"

    def __init__(
        self,
        model: str,
        *,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model = model
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        ensure_openrouter_api_key()
        self.client = OpenRouterModel(model)

    async def run(
        self,
        *,
        system: str | None,
        user: str,
        output_type: Any = str,
    ) -> Any:
        agent = Agent(
            self.client,
            output_type=output_type,
            instructions=system,
            retries={"output": OUTPUT_RETRIES},
            model_settings=ModelSettings(temperature=0.7, max_tokens=self.max_tokens),
        )
        result = await agent.run(user)
        return result.output


class DryRunModelRunner:
    """Runner placeholder for code paths that never call a model."""

    concurrency = 16

    def __init__(self, provider: str = "dry-run", model: str = "none") -> None:
        self.provider = provider
        self.model = model

    async def run(self, *, system: str | None, user: str, output_type: Any = str) -> Any:
        raise NotImplementedError("dry-run runner should not be called")


class BatchPipeline:
    """End-to-end batch processor: fetch -> parse -> aggregate summaries."""

    def __init__(
        self,
        batch_dir: Path,
        runner: ModelRunner | DryRunModelRunner,
        concurrency: int | None = None,
        project_config: Any | None = None,
    ):
        self.batch_dir = batch_dir.resolve()
        self.runner = runner
        self.concurrency = concurrency or runner.concurrency
        self.project_config = project_config
        self.typed_responses_dir = self.batch_dir / "typed_responses"
        self.cleaned_json_dir = self.batch_dir / "cleaned_json"
        self.meta_path = self.batch_dir / "processing_meta.json"

        self.typed_responses_dir.mkdir(exist_ok=True)
        self.cleaned_json_dir.mkdir(exist_ok=True)

    def load_manifest(self) -> dict[str, Any]:
        """Load the batch manifest produced by ``transcripts_clean``."""
        manifest_path = self.batch_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def load_batch_text(self, batch_name: str) -> str:
        """Load the cleaned text for one batch."""
        batch_path = self.batch_dir / f"{batch_name}.txt"
        if not batch_path.exists():
            return ""
        return batch_path.read_text(encoding="utf-8")

    async def fetch_typed_responses(
        self,
        *,
        only: set[str] | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> list[TypedResponse]:
        """Call the LLM for every missing batch and persist typed responses."""
        manifest = self.load_manifest()
        batch_names = sorted(name for name in manifest if only is None or name in only)
        needing = [
            name
            for name in batch_names
            if force or not (self.typed_responses_dir / f"{name}.json").exists()
        ]

        console.print(
            f"[dim]Typed responses: {len(batch_names) - len(needing)} cached, "
            f"{len(needing)} to fetch[/]"
        )

        if dry_run:
            console.print("\n[bold cyan]Prompts that would be sent (dry-run):[/]")
            for name in needing:
                text = self.load_batch_text(name)
                console.print(f"\n--- {name} ---")
                console.print(build_batch_prompt(name, text, self.project_config))
            return []

        if not needing:
            console.print("[green]All typed responses already fetched.[/]")
            return []

        sem = asyncio.Semaphore(self.concurrency)
        results: list[TypedResponse] = []

        async def _fetch_one(name: str) -> TypedResponse | None:
            text = self.load_batch_text(name)
            prompt = build_batch_prompt(name, text, self.project_config)
            async with sem:
                t0 = time.monotonic()
                try:
                    output = await self.runner.run(
                        system=self._system_role(),
                        user=prompt,
                        output_type=BatchSummary,
                    )
                    if not isinstance(output, BatchSummary):
                        output = BatchSummary.model_validate(output)
                    summary = output.model_copy(update={"batch_name": name})
                    dt = time.monotonic() - t0
                    typed = TypedResponse(
                        batch_name=name,
                        typed_response=summary,
                        finish_reason="unknown",
                        provider=self.runner.provider,
                        model=self.runner.model,
                    )
                    self._save_typed_response(typed)
                    self._save_cleaned_summary(summary)
                    console.log(f"[dim]{name}: fetched in {dt:.1f}s ({typed.finish_reason})[/]")
                    return typed
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    self._save_error_summary(name, error)
                    console.log(f"[red]{name}: failed: {error}[/]")
                    return None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Fetching batch summaries[/]", total=len(needing))
            coros = [_fetch_one(name) for name in needing]
            for coro in asyncio.as_completed(coros):
                if result := await coro:
                    results.append(result)
                progress.advance(task)

        return results

    def _save_typed_response(self, typed: TypedResponse) -> None:
        path = self.typed_responses_dir / f"{typed.batch_name}.json"
        payload = typed.model_dump(mode="json") | {
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds")
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def _save_cleaned_summary(self, summary: BatchSummary) -> None:
        path = self.cleaned_json_dir / f"{summary.batch_name}.json"
        path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def _save_error_summary(self, batch_name: str, error: str) -> None:
        path = self.cleaned_json_dir / f"{batch_name}.json"
        payload = {"batch_name": batch_name, "error": error}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def parse_typed_responses(self, *, only: set[str] | None = None) -> dict[str, Any]:
        """Validate cached typed responses and mirror them to cleaned JSON."""
        manifest = self.load_manifest()
        batch_names = sorted(name for name in manifest if only is None or name in only)

        cleaned: dict[str, Any] = {}
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Parsing typed responses[/]", total=len(batch_names))
            for name in batch_names:
                typed = self._load_typed_response(name)
                if typed is None:
                    cleaned[name] = self._load_existing_error_summary(name) or {
                        "batch_name": name,
                        "error": "missing typed response",
                    }
                else:
                    try:
                        summary = typed.summary().model_copy(update={"batch_name": name})
                        cleaned[name] = summary.to_dict()
                    except ValueError as exc:
                        cleaned[name] = {
                            "batch_name": name,
                            "error": f"invalid typed response: {exc}",
                        }
                out_path = self.cleaned_json_dir / f"{name}.json"
                out_path.write_text(json.dumps(cleaned[name], indent=2, ensure_ascii=False) + "\n")
                progress.advance(task)

        return cleaned

    def _load_existing_error_summary(self, batch_name: str) -> dict[str, Any] | None:
        path = self.cleaned_json_dir / f"{batch_name}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if "error" in data else None

    def _load_typed_response(self, batch_name: str) -> TypedResponse | None:
        path = self.typed_responses_dir / f"{batch_name}.json"
        if not path.exists():
            return None
        return TypedResponse.model_validate_json(path.read_text(encoding="utf-8"))

    def build_summaries(self, *, only: set[str] | None = None) -> dict[str, BatchSummary]:
        """Validate cleaned JSON and produce typed ``BatchSummary`` objects."""
        manifest = self.load_manifest()
        summaries: dict[str, BatchSummary] = {}

        for name in sorted(manifest):
            if only is not None and name not in only:
                continue
            path = self.cleaned_json_dir / f"{name}.json"
            if not path.exists():
                continue
            obj = json.loads(path.read_text(encoding="utf-8"))
            if "error" in obj:
                continue
            try:
                summaries[name] = BatchSummary.from_dict(obj)
            except (TypeError, ValueError):
                continue

        return summaries

    def _system_role(self) -> str | None:
        """Return configured system instructions for model calls."""
        if self.project_config is None:
            return None
        return self.project_config.prompts.system_role

    def save_artifacts(
        self,
        summaries: dict[str, BatchSummary],
        duration: float | None,
    ) -> None:
        """Persist summaries and metadata."""
        successful = len(summaries)
        manifest = self.load_manifest()
        failed = len(manifest) - successful

        summaries_path = self.batch_dir / "batch_summaries.json"
        summaries_path.write_text(
            json.dumps(
                {name: summary.to_dict() for name, summary in summaries.items()},
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )

        meta = PipelineMeta(
            schema_version=SCHEMA_VERSION,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            provider=self.runner.provider,
            model=self.runner.model,
            total_batches=len(manifest),
            successful=successful,
            failed=failed,
            duration_seconds=duration,
        )
        self.meta_path.write_text(json.dumps(meta.to_dict(), indent=2) + "\n", encoding="utf-8")

        console.print(f"\n[dim]Saved {summaries_path}[/]")
        console.print(f"[dim]Saved {self.meta_path}[/]")


@app.command()
def main(
    config: Path = typer.Option(
        Path("config.yml"),
        "--config",
        help="Path to project config YAML.",
    ),
    step: int | None = typer.Option(
        None,
        "--step",
        min=1,
        max=3,
        help="Run only step 1 (fetch), 2 (parse), or 3 (aggregate). Omit for all.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Override configured API provider: openrouter",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Override the provider's default model.",
    ),
    batch_dir: Path | None = typer.Option(
        None,
        "--batch-dir",
        help="Override configured directory containing batch files.",
    ),
    concurrency: int = typer.Option(
        4,
        "--concurrency",
        min=1,
        max=16,
        help="Max parallel API calls.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print prompts instead of calling APIs.",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Comma-separated batch names to process (e.g. batch_01,batch_02).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-fetch typed responses even if they already exist.",
    ),
) -> None:
    """Process transcript batches into typed summaries for skill generation."""
    project_config = load_config(config)
    resolved_batch_dir = batch_dir or project_config.batches_dir
    only_set = {n.strip() for n in only.split(",") if n.strip()} if only else None
    configured_model = project_config.models.summarization

    runner = (
        DryRunModelRunner(configured_model.provider, configured_model.model)
        if dry_run or step in {2, 3}
        else _make_runner(project_config, provider, model, concurrency)
    )
    pipeline = BatchPipeline(
        batch_dir=resolved_batch_dir,
        runner=runner,
        project_config=project_config,
    )

    if dry_run:
        if step is None or step == 1:
            _run_step_1(pipeline, only_set, dry_run=True, force=force)
        if step == 3:
            _run_step_3(pipeline, only_set, dry_run=True)
    elif step == 1:
        _run_step_1(pipeline, only_set, dry_run=False, force=force)
    elif step == 2:
        _run_step_2(pipeline, only_set)
    elif step == 3:
        _run_step_3(pipeline, only_set, dry_run=False)
    else:
        _run_all(pipeline, only_set, force=force)


def _make_runner(
    project_config: ProjectConfig,
    provider_override: str | None,
    model_override: str | None,
    concurrency: int,
) -> ModelRunner:
    """Create the configured OpenRouter runner for a pipeline phase."""
    configured = project_config.models.summarization
    provider = provider_override or configured.provider
    if provider != "openrouter":
        raise typer.BadParameter("transcripts:process currently supports provider=openrouter only")
    return ModelRunner(
        model_override or configured.model,
        concurrency=min(concurrency, DEFAULT_CONCURRENCY),
    )


def _run_step_1(
    pipeline: BatchPipeline,
    only_set: set[str] | None,
    dry_run: bool,
    force: bool = False,
) -> None:
    if dry_run:
        console.print("[bold cyan]STEP 1 (dry-run)[/]")
    else:
        console.print("[bold]STEP 1: Fetching typed responses[/]")
    asyncio.run(pipeline.fetch_typed_responses(only=only_set, dry_run=dry_run, force=force))


def _run_step_2(pipeline: BatchPipeline, only_set: set[str] | None) -> None:
    console.print("[bold]STEP 2: Offline pre-parse[/]")
    pipeline.parse_typed_responses(only=only_set)


def _run_step_3(
    pipeline: BatchPipeline,
    only_set: set[str] | None,
    dry_run: bool,
) -> None:
    console.print("[bold]STEP 3: Aggregating batch summaries[/]")
    summaries = pipeline.build_summaries(only=only_set)
    if dry_run:
        console.print(f"[dim]Would save {len(summaries)} batch summaries.[/]")
        return
    pipeline.save_artifacts(summaries, duration=None)


def _run_all(
    pipeline: BatchPipeline,
    only_set: set[str] | None,
    force: bool = False,
) -> None:
    t0 = time.monotonic()

    _run_step_1(pipeline, only_set, dry_run=False, force=force)
    _run_step_2(pipeline, only_set)
    summaries = pipeline.build_summaries(only=only_set)
    pipeline.save_artifacts(summaries, duration=time.monotonic() - t0)


if __name__ == "__main__":
    app()
