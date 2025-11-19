"""Idempotent YouTube subtitle fetcher for configured channels.

Scans the configured raw subtitle directory for existing subtitle files, asks
the channel playlist for the current list of videos via ``yt-dlp``, and
downloads only the missing transcripts.

This is a Pixi-managed entry point:

    pixi run transcripts:fetch
    pixi run transcripts:fetch --dry-run
    pixi run transcripts:fetch --out-dir other_folder
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn

from .config import load_config

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

YT_DLP_TIMEOUT = 120  # seconds per individual subtitle fetch
SLEEP_BETWEEN_DOWNLOADS = 3.0  # seconds; helps avoid YouTube rate limits
SUBTITLE_LANGUAGES = ["de", "en"]  # priority order
SUBTITLE_PATTERNS = ("*.srt", "*.ttml")


@dataclass
class FetchResult:
    """Outcome for a single video download attempt."""

    video_id: str
    success: bool
    language: str | None = None
    error: str | None = None


@dataclass
class FetchSummary:
    """Aggregate outcome of a fetch run."""

    channel_url: str
    out_dir: Path
    total_channel_videos: int
    already_present: int
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    errors: list[tuple[str, str]]


def _extract_video_id(stem: str) -> str | None:
    """Return the base video ID from a filename stem like ``id.de``."""
    # yt-dlp can emit ``video_id.lang.ext`` or ``video_id.ext`` depending on
    # language flags; strip any two-letter language suffix.
    parts = stem.rsplit(".", 1)
    if len(parts) == 2 and len(parts[1]) == 2 and parts[1].isalpha():
        return parts[0]
    return stem


def _existing_video_ids(out_dir: Path) -> set[str]:
    """Return video IDs that already have any subtitle file in ``out_dir``."""
    return {video_id for p in _subtitle_files(out_dir) if (video_id := _extract_video_id(p.stem))}


def _subtitle_files(out_dir: Path) -> Iterable[Path]:
    if out_dir.exists():
        for pattern in SUBTITLE_PATTERNS:
            yield from out_dir.glob(pattern)


def _list_channel_video_ids(yt_dlp: str, channel_url: str) -> list[str]:
    """Return every video ID currently reachable on the channel."""
    cmd = [
        yt_dlp,
        "--flat-playlist",
        "--print",
        "%(id)s",
        channel_url,
    ]
    console.log(f"[dim]Listing videos from {channel_url}...[/]")
    # subprocess.run is intentionally used here to call the pixi-managed yt-dlp
    # binary; the only variable input is the configured channel URL.
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if result.returncode != 0:
        console.print(f"[red]yt-dlp playlist listing failed:[/] {result.stderr.strip()}")
        raise typer.Exit(code=2)

    # YouTube IDs are 11 characters; ignore blank/erroneous lines.
    return [video_id for line in result.stdout.splitlines() if len(video_id := line.strip()) == 11]


def _find_downloaded_file(
    video_id: str, out_dir: Path, language: str | None = None
) -> Path | None:
    """Return the subtitle file for ``video_id`` and optional language if present."""
    return next(
        (
            p
            for p in _subtitle_files(out_dir)
            if _extract_video_id(p.stem) == video_id and _matches_language(p, video_id, language)
        ),
        None,
    )


def _matches_language(path: Path, video_id: str, language: str | None) -> bool:
    """Check the subtitle filename language suffix emitted by yt-dlp."""
    if language is None:
        return True
    return path.stem == video_id or path.stem.endswith(f".{language}")


def _download_with_language(
    yt_dlp: str,
    video_id: str,
    out_dir: Path,
    language: str,
) -> subprocess.CompletedProcess[str]:
    """Run yt-dlp for a specific language and return the completed process."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    # subprocess.run is intentionally used here to call the pixi-managed yt-dlp
    # binary; the video ID has already been validated as an 11-character token.
    return subprocess.run(  # noqa: S603
        [
            yt_dlp,
            "--extractor-retries",
            "3",
            "--retry-sleep",
            "extractor:5",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            language,
            "--convert-subs",
            "srt",
            "--sub-format",
            "best",
            "-o",
            str(out_dir / "%(id)s.%(ext)s"),
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=YT_DLP_TIMEOUT,
    )


def _download_single_subtitle(
    yt_dlp: str,
    video_id: str,
    out_dir: Path,
    subtitle_languages: list[str],
) -> FetchResult:
    """Download subtitles for one video ID in configured language order."""
    last_error: str | None = None
    for language in subtitle_languages:
        result = _download_with_language(yt_dlp, video_id, out_dir, language)
        stderr = result.stderr.lower()
        # Rate-limit failures are transient: surface them immediately so the
        # caller can decide whether to retry later.
        if "429" in stderr or "too many requests" in stderr:
            return FetchResult(
                video_id=video_id,
                success=False,
                error="HTTP Error 429: Too Many Requests",
            )

        # yt-dlp may exit 0 even if no subtitle file is written. Verify that
        # this specific language exists before falling back to the next one.
        downloaded = _find_downloaded_file(video_id, out_dir, language)
        if downloaded is not None and downloaded.stat().st_size > 0:
            return FetchResult(
                video_id=video_id,
                success=True,
                language=language,
            )

        # Capture the most informative error for the final failure reason.
        stderr_snippet = result.stderr.strip()[:300].replace("\n", " ")
        if stderr_snippet:
            last_error = stderr_snippet

    return FetchResult(
        video_id=video_id,
        success=False,
        error=last_error or "no subtitle file written",
    )


def _make_summary(
    channel_url: str,
    out_dir: Path,
    channel_ids: list[str],
    existing_ids: set[str],
    missing_ids: list[str],
    results: list[FetchResult],
) -> FetchSummary:
    return FetchSummary(
        channel_url=channel_url,
        out_dir=out_dir,
        total_channel_videos=len(channel_ids),
        already_present=len(existing_ids),
        attempted=len(missing_ids),
        succeeded=sum(1 for r in results if r.success),
        failed=sum(1 for r in results if not r.success),
        skipped=0,
        errors=[(r.video_id, r.error or "unknown") for r in results if r.error],
    )


def _run_fetch(
    channel_url: str,
    out_dir: Path,
    dry_run: bool,
    subtitle_languages: list[str] | None = None,
) -> FetchSummary:
    """Orchestrate listing, diffing, and (optionally) downloading."""
    yt_dlp = "yt-dlp"
    subtitle_languages = subtitle_languages or SUBTITLE_LANGUAGES
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_ids = _list_channel_video_ids(yt_dlp, channel_url)
    if not channel_ids:
        console.print("[yellow]No videos found on channel.[/]")
        return _make_summary(channel_url, out_dir, channel_ids, set(), [], [])

    existing_ids = _existing_video_ids(out_dir)
    missing_ids = [vid for vid in channel_ids if vid not in existing_ids]

    console.log(
        f"[dim]Channel has {len(channel_ids)} videos; "
        f"{len(existing_ids)} already present; "
        f"{len(missing_ids)} to fetch.[/]"
    )

    if dry_run:
        if missing_ids:
            console.print("\n[bold cyan]Missing videos (dry-run):[/]")
            for vid in missing_ids:
                console.print(f"  - {vid}")
        return _make_summary(channel_url, out_dir, channel_ids, existing_ids, missing_ids, [])

    results: list[FetchResult] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            "[cyan]Downloading subtitles[/]",
            total=len(missing_ids),
        )
        for video_id in missing_ids:
            res = _download_single_subtitle(yt_dlp, video_id, out_dir, subtitle_languages)
            results.append(res)
            progress.advance(task, 1)
            # Back off slightly even on success; YouTube throttles rapid
            # subtitle requests.
            if len(missing_ids) > 1:
                time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    return _make_summary(channel_url, out_dir, channel_ids, existing_ids, missing_ids, results)


def _write_summary_json(summary: FetchSummary) -> Path | None:
    """Write a machine-readable summary next to the output directory."""
    payload: dict[str, Any] = {
        **summary.__dict__,
        "out_dir": str(summary.out_dir),
        "errors": [{"video_id": vid, "error": err} for vid, err in summary.errors],
    }
    summary_path = summary.out_dir / ".last_fetch_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return summary_path


def _print_summary(summary: FetchSummary) -> None:
    console.print("\n" + "=" * 60)
    console.print("[bold]Fetch Summary[/]")
    console.print("=" * 60)
    console.print(f"Channel:       {summary.channel_url}")
    console.print(f"Output dir:    {summary.out_dir}")
    console.print(f"Channel total: {summary.total_channel_videos}")
    console.print(f"Already local: {summary.already_present}")
    console.print(f"Attempted:     {summary.attempted}")
    console.print(
        f"[green]Succeeded:[/]     {summary.succeeded}"
        f" · [red]Failed:[/] {summary.failed}"
        f" · [yellow]Skipped:[/] {summary.skipped}"
    )
    if summary.errors:
        console.print("\n[red]First 10 errors:[/]")
        for vid, err in summary.errors[:10]:
            console.print(f"  {vid}: {err[:120]}")
    console.print("=" * 60)


@app.command()
def main(
    config_file: Path = typer.Option(
        Path("config.yml"),
        "--config",
        help="Path to project config YAML.",
    ),
    channel_url: str = typer.Option(
        "",
        "--channel-url",
        help="Override configured YouTube channel URL or handle URL to scan.",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Override configured directory to write SRT files to.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="List missing videos without downloading anything.",
    ),
) -> None:
    """Fetch missing transcripts for the configured YouTube channel."""
    project_config = load_config(config_file)
    resolved_channel_url = channel_url or project_config.source.channel_url
    resolved_out_dir = (out_dir or project_config.raw_subtitles_dir).resolve()
    summary = _run_fetch(
        resolved_channel_url,
        resolved_out_dir,
        dry_run,
        subtitle_languages=project_config.source.subtitle_languages,
    )
    _print_summary(summary)
    if not dry_run:
        summary_path = _write_summary_json(summary)
        console.print(f"\n[dim]Wrote summary: {summary_path}[/]")

    if summary.failed > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
