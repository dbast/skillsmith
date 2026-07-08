# skillsmith

Reusable pipeline for turning a YouTube channel into structured transcript summaries and an agentskills.io-compatible skill package.

Core concept: use Pydantic and Pydantic AI as opinionated runtime boundaries between deterministic Python steps, LLM calls, typed artifacts, and final skill files.

## Flow

1. Configure: `config.yml` defines source URL, paths, batching, prompts, models, and skill metadata. Deterministic.
1. Download: fetch missing subtitles with `yt-dlp` and write raw SRT files. Deterministic.
1. Clean and batch: remove SRT noise, write cleaned transcripts, and group them into character-sized `batch_XX.txt` files. Deterministic.
1. Summarize: send each missing batch to the configured summarization model and request a typed `BatchSummary`. LLM.
1. Validate and aggregate: parse cached typed responses, validate them with Pydantic models, merge valid summaries, and write processing metadata. Deterministic.
1. Index: map topics, concepts, paragraphs, and examples back to source batches; create a compact source digest. Deterministic.
1. Plan: ask the skill-builder model for a typed `SkillPlan` with routing, assets, and planned reference/topic files. LLM.
1. Render: write `skill_plan.json`, root `SKILL.md`, and deterministic helper assets. Deterministic.
1. Generate references: create one Markdown reference file per planned topic under `references/`. LLM.
1. Package: zip the final skill directory next to the output folder. Deterministic.

## Boundaries

- `config.yml` is the project contract: source URL, paths, batching, models, prompt variables, and skill metadata.
- `skillsmith/models.py` holds shared Pydantic data models and OpenRouter / Pydantic AI helpers.
- `skillsmith/prompts.py` builds prompts from config plus batch text.
- `output/` contains generated pipeline data.
- Pixi owns environments and runnable tasks.

## Architecture Principles

- Separation of concerns: one process file owns one pipeline responsibility.
- Loose coupling: steps exchange files instead of hidden in-memory state.
- Typed boundaries: Pydantic validates config, LLM outputs, and persisted artifacts.
- Deterministic first: non-LLM work stays plain Python and reproducible.
- Incremental runs: existing outputs are reused unless `--force` asks otherwise.
- Checkpointed pipeline: each long-running phase writes inspectable artifacts.
- Progressive disclosure: `SKILL.md` stays small while details live in `references/` and `assets/`.
- Config as contract: `config.yml` defines sources, paths, models, prompts, and skill metadata.
- Fail-soft batches: one failed model call should not discard successful batches.
- Pixi runtime: tasks and environments run through one reproducible entrypoint layer.

## Configure

Start from `config_template.yml` for a new channel/project, then save it as `config.yml` or pass it with `--config`. Relative paths are resolved from the config file location.

Validate the config:

```bash
pixi run config:check
```

Important sections:

- `source.channel_url`: YouTube channel or handle URL.
- `source.subtitle_languages`: subtitle language fallback order.
- `paths`: raw subtitles, cleaned transcripts, batches, final skill output under `output/`.
- `batching.target_chars`: target batch size for LLM context windows.
- `models`: provider/model choices for summarization and skill building.
- `prompts`: reusable project prompt variables.
- `skill`: agentskills.io package metadata.

## Workflow

Fetch transcripts:

```bash
pixi run transcripts:fetch --dry-run
pixi run transcripts:fetch
```

Clean transcripts and build batches:

```bash
pixi run transcripts:clean
```

Summarize batches and write `batch_summaries.json`:

```bash
export OPENROUTER_API_KEY='sk-or-...'
pixi run transcripts:process
```

Build the skill package:

```bash
export OPENROUTER_API_KEY='sk-or-...'
pixi run skill:build --dry-run
pixi run skill:build --plan-only --force
pixi run skill:build --force
```

Use a different config file with any CLI:

```bash
pixi run transcripts:fetch --config path/to/config.yml
pixi run transcripts:clean --config path/to/config.yml
pixi run transcripts:process --config path/to/config.yml
pixi run skill:build --config path/to/config.yml
```

## Development

Run checks:

```bash
pixi run test
pixi run prek
```

## TODO

1. Add portable skill knowledge graph.
   - Generate `assets/knowledge_graph.yml` from `skill_plan.json` and `batch_summaries.json`.
   - Keep graph dependency-free and agent-readable.
   - Include topic/reference nodes, trigger phrases, source batches, and typed edges such as `depends_on`, `related_to`, `compare_with`, and `risk_for`.
   - Reference graph from `SKILL.md` as routing/index aid, not replacement for Markdown references.
2. Add provenance to generated references.
   - Include source batch IDs and source topics in each `references/*.md` file.
   - Prefer small frontmatter blocks or HTML comments that are easy to parse.
   - Use provenance for graph generation and evaluation.
3. Add deterministic skill evaluation.
   - Add a `skill:evaluate` Pixi task.
   - Check batch coverage, topic coverage, reference coverage, graph integrity, broken links, orphan nodes, and unsupported generated topics.
   - Write `output/eval/coverage_matrix.json` and `output/eval/coverage_report.md`.
4. Add optional LLM-assisted evaluation.
   - Use only behind an explicit flag such as `--llm`.
   - Judge whether references and graph edges are supported by source batch summaries.
   - Report likely lost concepts, hallucinated claims, weak source support, and routing failures.
5. Add RSS feed ingestion for fast-moving topics.
   - Support blogs, podcasts, newsletters, and other feeds where current knowledge is not yet covered by books or model training data.
   - Download feed items from configured RSS/Atom URLs with source metadata, publication dates, authors, canonical URLs, and stable IDs.
   - Extract readable article text or podcast transcript text where available.
   - Clean feed content with the same deterministic rules: remove boilerplate, markup, duplicated text, and empty items.
   - Add cleaned RSS items into the existing batching pipeline alongside YouTube transcripts.
   - Preserve source provenance so generated references can distinguish YouTube, blog, podcast, newsletter, and other feed-derived claims.
