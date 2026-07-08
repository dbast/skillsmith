#!/usr/bin/env python3
"""
Clean SRT subtitle files and create transcript batches.
Removes timestamps, HTML formatting, and overlapping subtitles.
Creates clean transcripts and batch groups for AI processing.
"""

import json
import re
from pathlib import Path

import typer

from .config import load_config

app = typer.Typer(add_completion=False, help=__doc__)
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
COMBINED_TRANSCRIPT = "all_transcripts_combined.txt"


class SRTCleaner:
    """Clean and extract text from SRT subtitle files."""

    def __init__(self, input_dir: str | Path, output_dir: str | Path = "cleaned_transcripts"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir = self.output_dir / "batches"
        self.batch_dir.mkdir(parents=True, exist_ok=True)

    def clean_srt_text(self, srt_content: str) -> str:
        """
        Remove timestamps, HTML tags, and clean up SRT content.

        Args:
            srt_content: Raw SRT file content

        Returns:
            Clean text without timestamps or formatting
        """
        lines = (
            SPACE_RE.sub(" ", HTML_RE.sub("", line)).strip()
            for line in srt_content.splitlines()
            if line.strip() and "-->" not in line and not line.strip().isdigit()
        )
        return SPACE_RE.sub(" ", " ".join(line for line in lines if line)).strip()

    def extract_all_files(self) -> dict[str, any]:
        """
        Extract and clean all SRT files in input directory.

        Returns:
            Dictionary with processing statistics
        """
        srt_files = sorted(self.input_dir.glob("*.srt"))

        if not srt_files:
            print(f"❌ No SRT files found in {self.input_dir}")
            return {}

        print(f"🔍 Found {len(srt_files)} SRT files")
        print("📝 Processing and cleaning...\n")

        all_texts: list[str] = []
        file_data: dict[str, dict[str, str | int]] = {}
        total_text_size = 0

        for i, srt_path in enumerate(srt_files, 1):
            cleaned_text = self.clean_srt_text(srt_path.read_text(encoding="utf-8"))

            if cleaned_text:
                output_path = self.output_dir / f"{srt_path.stem}.txt"
                output_path.write_text(cleaned_text, encoding="utf-8")

                all_texts.append(cleaned_text)
                file_data[srt_path.stem] = {
                    "original_file": srt_path.name,
                    "cleaned_file": output_path.name,
                    "text_length": len(cleaned_text),
                }
                total_text_size += len(cleaned_text)

                # Progress indicator
                if i % 50 == 0 or i == len(srt_files):
                    print(f"✓ Processed {i}/{len(srt_files)} files")

        print(f"\n✅ Successfully processed {len(file_data)} files")
        print(f"📊 Total cleaned text: {total_text_size:,} characters")

        combined_path = self.output_dir / COMBINED_TRANSCRIPT
        combined_path.write_text("\n\n[VIDEO BREAK]\n\n".join(all_texts), encoding="utf-8")

        print(f"💾 Saved combined transcript: {combined_path}")

        # Save metadata
        metadata = {
            "total_files": len(file_data),
            "total_characters": total_text_size,
            "average_file_size": total_text_size // len(file_data) if file_data else 0,
            "files": file_data,
        }

        metadata_path = self.output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"📋 Saved metadata: {metadata_path}")

        return metadata

    def create_batches(
        self,
        target_chars: int = 300_000,
    ) -> dict[str, list[str]]:
        """
        Create batches of cleaned text files for AI processing.

        Instead of grouping by a fixed number of files, batches are built by
        accumulating files until adding the next file would exceed
        ``target_chars`` characters. This keeps batches roughly uniform in
        context-window usage. A single file that already exceeds ``target_chars``
        is placed in its own batch.

        Args:
            target_chars: Target number of characters per batch.

        Returns:
            Dictionary mapping batch names to lists of text content.
        """

        cleaned_files = sorted(self.output_dir.glob("*.txt"))
        cleaned_files = [f for f in cleaned_files if f.name != COMBINED_TRANSCRIPT]

        if not cleaned_files:
            print("❌ No cleaned files found. Run extract_all_files() first.")
            return {}

        print(f"\n📦 Creating batches (target: {target_chars:,} characters per batch)...")

        # Start fresh: remove any previously generated batch files so a change
        # in target_chars doesn't leave stale batches behind.
        for stale_path in self.batch_dir.glob("batch_*.txt"):
            stale_path.unlink()

        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_chars = 0

        for file_path in cleaned_files:
            content = file_path.read_text(encoding="utf-8")
            content_chars = len(content)

            # Start a new batch if the current one is non-empty and adding this
            # file would push us meaningfully over target. A single oversized
            # file is allowed to be its own batch.
            if current_batch and current_chars + content_chars > target_chars:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

            current_batch.append(content)
            current_chars += content_chars

        if current_batch:
            batches.append(current_batch)

        # Merge a tiny trailing batch (< 25 % of target) into the previous one
        # to avoid a runt batch at the end.
        if len(batches) >= 2 and sum(len(t) for t in batches[-1]) < target_chars * 0.25:
            last_batch = batches.pop()
            batches[-1].extend(last_batch)

        # Save batches
        batch_info = {}
        result: dict[str, list[str]] = {}
        for i, texts in enumerate(batches, 1):
            batch_key = f"batch_{i:02d}"
            batch_text = "\n\n[NEXT VIDEO]\n\n".join(texts)
            batch_path = self.batch_dir / f"{batch_key}.txt"

            batch_path.write_text(batch_text, encoding="utf-8")

            batch_info[batch_key] = {
                "file_count": len(texts),
                "total_characters": len(batch_text),
                "file_path": str(batch_path),
            }
            result[batch_key] = texts

            print(f"✓ {batch_key}: {len(texts)} files, {len(batch_text):,} characters")

        # Save batch manifest
        manifest_path = self.batch_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(batch_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"\n✅ Created {len(batches)} batches")
        print(f"📋 Saved manifest: {manifest_path}")

        return result


@app.command()
def main(
    input_dir: Path | None = typer.Argument(
        None,
        help="Override configured subtitle input directory.",
    ),
    output_dir: Path | None = typer.Argument(
        None,
        help="Override configured cleaned output directory.",
    ),
    target_chars: int | None = typer.Argument(
        None,
        help="Override target characters per batch.",
    ),
    config: Path = typer.Option(
        Path("config.yml"),
        "--config",
        help="Path to project config YAML.",
    ),
) -> None:
    """Main execution function."""
    project_config = load_config(config)
    resolved_input_dir = input_dir or project_config.raw_subtitles_dir
    resolved_output_dir = output_dir or project_config.cleaned_transcripts_dir
    resolved_target_chars = target_chars or project_config.batching.target_chars

    print("\n" + "=" * 60)
    print("🎬 SRT EXTRACTOR & CLEANER")
    print("=" * 60)
    print(f"Input: {resolved_input_dir}")
    print(f"Output: {resolved_output_dir}")
    print(f"Target chars per batch: {resolved_target_chars:,}")
    print("=" * 60 + "\n")

    # Create cleaner instance
    cleaner = SRTCleaner(resolved_input_dir, resolved_output_dir)

    # Extract all files
    metadata = cleaner.extract_all_files()

    if metadata:
        batches = cleaner.create_batches(target_chars=resolved_target_chars)

        print("✨ Extraction complete!")
        print(f"Processed {metadata['total_files']} files into {len(batches)} batches")
        print("Next step: Use batch files for AI processing")
    else:
        print("❌ Extraction failed")
        raise SystemExit(1)


if __name__ == "__main__":
    app()
