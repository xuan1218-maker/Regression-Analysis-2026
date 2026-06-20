from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import nbformat


SLIDE_TAGS = {"slide", "sub-slide", "fragment", "skip"}
VALID_SLIDE_TYPES = {"slide", "subslide", "fragment", "skip", "notes", "-"}
TAG_TO_SLIDE_TYPE = {
    "slide": "slide",
    "sub-slide": "subslide",
    "fragment": "fragment",
    "skip": "skip",
}
REVEAL_WIDTH = 1280
REVEAL_HEIGHT = 720
SCRIPT_HTML_CSS = """
<style id="slide-builder-overrides">
  :root {
    --slide-builder-bg: #f6f3ea;
    --slide-builder-panel: rgba(255, 255, 255, 0.88);
    --slide-builder-ink: #1f2937;
    --slide-builder-accent: #8b5e34;
    --slide-builder-muted: #475569;
    --slide-builder-border: rgba(139, 94, 52, 0.18);
    --slide-builder-canvas-width: 1280px;
    --slide-builder-canvas-height: 720px;
    --slide-builder-safe-image-height: 460px;
    --slide-builder-safe-code-height: 400px;
  }

  body {
    background:
      radial-gradient(circle at top left, rgba(203, 213, 225, 0.35), transparent 28rem),
      linear-gradient(180deg, #f7f3e8 0%, #f2efe7 100%);
  }

  .reveal {
    font-size: 140%;
    color: var(--slide-builder-ink);
  }

  .reveal .slides {
    text-align: left;
  }

  .reveal .slides section {
    padding: 0.2em 0.4em;
  }

  .reveal h1,
  .reveal h2,
  .reveal h3 {
    color: #0f172a;
    letter-spacing: -0.03em;
    text-transform: none;
  }

  .reveal h1 {
    font-size: 1.9em;
    margin-bottom: 0.35em;
  }

  .reveal h2 {
    font-size: 1.35em;
    margin-bottom: 0.4em;
  }

  .reveal p,
  .reveal li {
    line-height: 1.4;
  }

  .reveal ul,
  .reveal ol {
    display: block;
    margin-left: 1.1em;
  }

  .reveal strong {
    color: #7c2d12;
  }

  .reveal blockquote {
    width: auto;
    margin: 0.8em 0;
    padding: 0.65em 0.9em;
    background: rgba(255, 255, 255, 0.7);
    border-left: 0.24em solid var(--slide-builder-accent);
    box-shadow: none;
  }

  .reveal pre,
  .reveal div.jp-OutputArea-output > pre,
  .reveal div.highlight > pre {
    max-height: var(--slide-builder-safe-code-height);
    overflow: auto;
    border: 1px solid var(--slide-builder-border);
    border-radius: 14px;
    background: var(--slide-builder-panel);
    box-shadow: none;
    padding: 0.65em 0.8em;
  }

  .reveal table {
    border-collapse: collapse;
    font-size: 0.82em;
  }

  .reveal th,
  .reveal td {
    border: 1px solid rgba(148, 163, 184, 0.35);
    padding: 0.45em 0.6em;
  }

  .reveal th {
    background: rgba(226, 232, 240, 0.65);
  }

  .reveal img,
  .reveal section img {
    max-height: var(--slide-builder-safe-image-height);
    max-width: 100%;
    width: auto;
    margin: 0.5em auto;
    display: block;
    border-radius: 12px;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
  }

  .reveal .jp-RenderedHTMLCommon table {
    margin: 0.5em auto;
  }

  .reveal .jp-RenderedHTMLCommon p code,
  .reveal .jp-RenderedHTMLCommon li code {
    color: var(--slide-builder-muted);
    background: rgba(226, 232, 240, 0.7);
    border-radius: 6px;
    padding: 0.08em 0.28em;
  }

  .reveal .progress {
    color: var(--slide-builder-accent);
  }
</style>
"""


@dataclass
class Issue:
    severity: str
    message: str
    cell_index: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reveal.js slides and lightweight teacher notes from a notebook."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Sync, validate, and export a notebook.")
    build_parser.add_argument(
        "notebook",
        help="Path to a .ipynb or .py notebook, relative to the slides/ directory or absolute.",
    )
    build_parser.add_argument(
        "--output-dir",
        help="Directory for build artifacts. Defaults to the notebook directory.",
    )
    build_parser.add_argument(
        "--skip-execute",
        action="store_true",
        help="Skip notebook execution before export.",
    )
    build_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat validation warnings as build-stopping errors.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return build_command(args)
    raise ValueError(f"Unsupported command: {args.command}")


def build_command(args: argparse.Namespace) -> int:
    slides_root = Path(__file__).resolve().parent
    target = resolve_path(args.notebook, slides_root)
    if target.suffix not in {".ipynb", ".py"}:
        print(f"error: expected a .ipynb or .py file, got {target}", file=sys.stderr)
        return 2
    if not target.exists():
        print(f"error: file not found: {target}", file=sys.stderr)
        return 2

    output_dir = resolve_output_dir(args.output_dir, target, slides_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    ipynb_path = sync_notebook_pair(target, slides_root)
    if not args.skip_execute:
        execute_notebook(ipynb_path, slides_root)

    notebook = nbformat.read(ipynb_path, as_version=4)
    normalize_notebook_for_build(notebook)
    issues = validate_notebook(notebook)
    report_issues(issues)
    if should_fail(issues, strict=args.strict):
        return 1

    slides_html = export_slides(notebook, ipynb_path, output_dir, slides_root)
    postprocess_slides_html(slides_html)
    notes_path = export_teacher_notes(notebook, ipynb_path, output_dir)

    print(f"build complete: {slides_html}")
    print(f"teacher notes: {notes_path}")
    return 0


def resolve_path(raw_path: str, slides_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (slides_root / path).resolve()


def resolve_output_dir(raw_output_dir: str | None, target: Path, slides_root: Path) -> Path:
    if raw_output_dir is None:
        return target.parent
    return resolve_path(raw_output_dir, slides_root)


def sync_notebook_pair(target: Path, slides_root: Path) -> Path:
    if target.suffix == ".ipynb":
        # A human may edit the notebook directly. In that case the notebook is
        # deliberately the source for this build: write it back to the paired
        # Jupytext file instead of asking Jupytext to infer a direction from
        # modification timestamps.
        normalize_notebook_file_metadata(target)
        run_command(
            [
                "jupytext",
                "--to",
                "py:percent",
                str(target.relative_to(slides_root)),
                "--output",
                str(target.with_suffix(".py").relative_to(slides_root)),
            ],
            slides_root,
        )
        return target

    run_command(["jupytext", "--sync", str(target.relative_to(slides_root))], slides_root)
    return target.with_suffix(".ipynb")


def normalize_notebook_file_metadata(ipynb_path: Path) -> None:
    """Fill slideshow metadata implied by structural tags in notebook edits."""
    notebook = nbformat.read(ipynb_path, as_version=4)
    changed = False

    for cell in notebook.cells:
        tags = set(cell.metadata.get("tags", []))
        implied_type = next(
            (TAG_TO_SLIDE_TYPE[tag] for tag in TAG_TO_SLIDE_TYPE if tag in tags), None
        )
        if implied_type is None:
            continue
        slideshow = cell.metadata.setdefault("slideshow", {})
        if slideshow.get("slide_type") != implied_type:
            slideshow["slide_type"] = implied_type
            changed = True

    if changed:
        nbformat.write(notebook, ipynb_path)


def execute_notebook(ipynb_path: Path, slides_root: Path) -> None:
    run_command(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(ipynb_path.relative_to(slides_root)),
        ],
        slides_root,
    )


def export_slides(
    notebook: nbformat.NotebookNode, ipynb_path: Path, output_dir: Path, slides_root: Path
) -> Path:
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{ipynb_path.stem}.build.ipynb"
        nbformat.write(notebook, tmp_path)
        run_command(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "slides",
                str(tmp_path.relative_to(slides_root)),
                "--output",
                ipynb_path.stem,
                "--output-dir",
                str(output_dir.relative_to(slides_root)),
                "--TagRemovePreprocessor.enabled=True",
                "--TagRemovePreprocessor.remove_input_tags={'hide-input'}",
            ],
            slides_root,
        )
    return output_dir / f"{ipynb_path.stem}.slides.html"


def postprocess_slides_html(slides_html: Path) -> None:
    html = slides_html.read_text(encoding="utf-8")
    if "</head>" not in html:
        return
    html = html.replace("</head>", f"{SCRIPT_HTML_CSS}\n</head>", 1) if "slide-builder-overrides" not in html else html
    html = re.sub(
        r"Reveal\.initialize\(\{\s*controls: true,\s*progress: true,\s*history: true,\s*transition: \"slide\",\s*slideNumber: \"\",\s*plugins: \[RevealNotes\],\s*width: \d+,\s*height: \d+,\s*\}\);",
        (
            "Reveal.initialize({\n"
            "            controls: true,\n"
            "            progress: true,\n"
            "            history: true,\n"
            "            transition: \"slide\",\n"
            "            slideNumber: \"\",\n"
            "            plugins: [RevealNotes],\n"
            f"            width: {REVEAL_WIDTH},\n"
            f"            height: {REVEAL_HEIGHT},\n"
            "            margin: 0.04,\n"
            "            minScale: 0.2,\n"
            "            maxScale: 1.5,\n"
            "            center: false,\n"
            "        });"
        ),
        html,
        count=1,
        flags=re.S,
    )
    slides_html.write_text(html, encoding="utf-8")


def export_teacher_notes(notebook: nbformat.NotebookNode, ipynb_path: Path, output_dir: Path) -> Path:
    notes: list[str] = [
        f"# Teacher Notes: {ipynb_path.stem}",
        "",
        f"- source notebook: `{ipynb_path.name}`",
        "",
    ]
    section_title = "Opening"
    note_count = 0

    for cell in notebook.cells:
        tags = set(cell.metadata.get("tags", []))
        if cell.cell_type == "markdown":
            heading = extract_heading(cell.source)
            if heading:
                section_title = heading
        if not is_teacher_note_cell(cell):
            continue
        note_count += 1
        notes.extend(
            [
                f"## Note {note_count}: {section_title}",
                "",
                normalize_markdown(cell.source),
                "",
            ]
        )

    if note_count == 0:
        notes.extend(["_No `script` cells were found in this notebook._", ""])

    notes_path = output_dir / f"{ipynb_path.stem}.notes.md"
    notes_path.write_text("\n".join(notes), encoding="utf-8")
    return notes_path


def normalize_markdown(source: str) -> str:
    text = source.strip()
    if not text:
        return "_Empty note cell._"
    return text


def extract_heading(source: str) -> str | None:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return None


def validate_notebook(notebook: nbformat.NotebookNode) -> list[Issue]:
    issues: list[Issue] = []
    for index, cell in enumerate(notebook.cells, start=1):
        tags = set(cell.metadata.get("tags", []))
        slide_type = cell.metadata.get("slideshow", {}).get("slide_type")

        if tags & SLIDE_TAGS and slide_type not in VALID_SLIDE_TYPES:
            issues.append(
                Issue(
                    "error",
                    f"Missing or invalid slideshow.slide_type for tags {sorted(tags & SLIDE_TAGS)}.",
                    index,
                )
            )

        if "script" in tags and "skip" not in tags:
            issues.append(
                Issue(
                    "warning",
                    "A `script` cell should also carry the `skip` tag.",
                    index,
                )
            )

        if "teacher-only" in tags:
            issues.append(
                Issue(
                    "warning",
                    "Legacy `teacher-only` tag detected; migrate to `script` + `skip`.",
                    index,
                )
            )

        if cell.cell_type == "code":
            source_lines = len(cell.source.splitlines())
            outputs = cell.get("outputs", [])
            has_visible_output = any(output.get("output_type") != "stream" or output.get("text") for output in outputs)
            if has_visible_output and "hide-input" not in tags and "skip" not in tags:
                issues.append(
                    Issue(
                        "warning",
                        "Code cell has outputs but is missing `hide-input`; source will likely show in slides.",
                        index,
                    )
                )
            if source_lines > 45 and "skip" not in tags:
                issues.append(
                    Issue(
                        "warning",
                        "Code cell is long; consider splitting setup and stage cells.",
                        index,
                    )
                )

        if cell.cell_type == "markdown":
            lines = [line for line in cell.source.splitlines() if line.strip()]
            bullet_lines = sum(1 for line in lines if re.match(r"^(\s*[-*]|\s*\d+\.)\s+", line))
            if len(lines) > 16:
                issues.append(
                    Issue(
                        "warning",
                        "Markdown cell is long; it may overflow a single slide.",
                        index,
                    )
                )
            if bullet_lines > 8:
                issues.append(
                    Issue(
                        "warning",
                        "Markdown cell has many bullets; consider splitting into vertical slides.",
                        index,
                    )
                )
            if "skip" in tags and "script" not in tags and looks_like_teacher_note(cell.source):
                issues.append(
                    Issue(
                        "warning",
                        "This hidden markdown cell looks like a teacher note; consider tagging it `script` as well.",
                        index,
                    )
                )

    if not any(is_teacher_note_cell(cell) for cell in notebook.cells):
        issues.append(Issue("warning", "No teacher-note cells found; teacher notes export will be empty."))

    return issues


def normalize_notebook_for_build(notebook: nbformat.NotebookNode) -> None:
    for cell in notebook.cells:
        tags = list(cell.metadata.get("tags", []))
        implied_type = next(
            (TAG_TO_SLIDE_TYPE[tag] for tag in TAG_TO_SLIDE_TYPE if tag in tags), None
        )
        if implied_type is not None:
            cell.metadata.setdefault("slideshow", {})["slide_type"] = implied_type

        if cell.cell_type != "code":
            continue
        if "skip" in tags or "hide-input" in tags:
            continue
        outputs = cell.get("outputs", [])
        has_visible_output = any(output.get("output_type") != "stream" or output.get("text") for output in outputs)
        if has_visible_output:
            tags.append("hide-input")
            cell.metadata["tags"] = tags


def looks_like_teacher_note(source: str) -> bool:
    lowered = source.lower()
    return any(token in lowered for token in ["teacher note", "节奏建议", "如果学生", "先让学生", "提醒"])


def is_teacher_note_cell(cell: nbformat.NotebookNode) -> bool:
    tags = set(cell.metadata.get("tags", []))
    if cell.cell_type != "markdown":
        return False
    if "script" in tags or "teacher-only" in tags:
        return True
    return "skip" in tags and looks_like_teacher_note(cell.source)


def report_issues(issues: list[Issue]) -> None:
    if not issues:
        print("validation: no issues found")
        return

    print("validation:")
    for issue in issues:
        location = f" cell {issue.cell_index}" if issue.cell_index is not None else ""
        print(f"- {issue.severity.upper()}{location}: {issue.message}")


def should_fail(issues: list[Issue], strict: bool) -> bool:
    if any(issue.severity == "error" for issue in issues):
        return True
    if strict and issues:
        return True
    return False


def run_command(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
