"""Convert a review Markdown file into a reveal.js-ready Jupytext source file.

The Markdown heading hierarchy determines slide structure:
    #       opening slide
    ##      new horizontal chapter slide (with the matching original question)
    ###     vertical slide
    ####    vertical slide
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
JUPYTEXT_HEADER = """# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a review Markdown file into a Jupytext slide source."
    )
    parser.add_argument("--input", type=Path, default=HERE / "note_gemini.md")
    parser.add_argument("--questions", type=Path, default=HERE / "pics.md")
    parser.add_argument("--output", type=Path, default=HERE / "note_gemini_slides.py")
    return parser.parse_args()


def split_problem_blocks(text: str) -> list[str]:
    """Return the three original-question blocks separated by Markdown rules."""
    return [block.strip() for block in re.split(r"\n\s*---\s*\n", text) if block.strip()]


def split_heading_blocks(text: str) -> list[tuple[int, str, str]]:
    """Split Markdown by headings and retain each heading's level and body."""
    pattern = re.compile(r"(?m)^(#{1,4})\s+(.+?)\s*$")
    matches = list(pattern.finditer(text))
    blocks: list[tuple[int, str, str]] = []

    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((len(match.group(1)), match.group(2).strip(), text[body_start:body_end].strip()))
    return blocks


def split_numbered_items(text: str, anchor: str) -> list[str]:
    """Extract numbered subquestions after an anchor such as ``那么，`` or ``证明：``."""
    if anchor not in text:
        raise ValueError(f"Could not find {anchor!r} in a question block.")
    tail = text.split(anchor, maxsplit=1)[1].strip()
    items = re.split(r"(?m)^\s*(?=\d+\.\s)", tail)
    return [item.strip() for item in items if item.strip()]


def question_for_heading(
    chapter: int, level: int, title: str, question_blocks: list[str]
) -> str | None:
    """Return the original subquestion that should precede this heading, if any."""
    if chapter == 0 and level == 3:
        numbered = split_numbered_items(question_blocks[0], "那么，")
        mapping = {"第一小问": 0, "第二小问": 1, "第三小问": 2}
        for label, index in mapping.items():
            if label in title:
                return numbered[index]

    if chapter == 1 and level == 3 and "实际求解" in title:
        match = re.search(r"(?m)^求:\s*(.+)$", question_blocks[1])
        if match:
            return f"求：{match.group(1).strip()}"

    if chapter == 2 and level == 4:
        match = re.search(r"[（(](\d)[）)]", title)
        if match:
            anchor = "证明：" if "证明：" in question_blocks[2] else "证明:"
            numbered = split_numbered_items(question_blocks[2], anchor)
            return numbered[int(match.group(1)) - 1]

    return None


def slide_metadata(level: int) -> tuple[list[str], str]:
    if level <= 2:
        return ["slide"], "slide"
    return ["sub-slide"], "subslide"


def as_markdown_cell(source: str, level: int) -> list[str]:
    tags, slide_type = slide_metadata(level)
    lines = [
        f'# %% [markdown] tags={json.dumps(tags)} slideshow={{"slide_type": "{slide_type}"}}'
    ]
    lines.extend(f"# {line}" if line else "#" for line in source.splitlines())
    lines.append("")
    return lines


def split_body_for_pages(body: str, max_lines: int = 12, max_chars: int = 1050) -> list[str]:
    """Split long Markdown bodies at paragraph boundaries for readable slides."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    if not paragraphs:
        return []

    pages: list[str] = []
    current: list[str] = []
    current_lines = 0
    current_chars = 0

    for paragraph in paragraphs:
        paragraph_lines = len([line for line in paragraph.splitlines() if line.strip()])
        paragraph_chars = len(paragraph)
        would_overflow = current and (
            current_lines + paragraph_lines > max_lines
            or current_chars + paragraph_chars > max_chars
        )
        if would_overflow:
            pages.append("\n\n".join(current))
            current = []
            current_lines = 0
            current_chars = 0

        current.append(paragraph)
        current_lines += paragraph_lines
        current_chars += paragraph_chars

    if current:
        pages.append("\n\n".join(current))
    return pages


def build_notebook_source(markdown: str, questions: list[str]) -> str:
    output_lines = JUPYTEXT_HEADER.rstrip("\n").splitlines()
    chapter_index = 0

    for level, title, body in split_heading_blocks(markdown):
        source = f"{'#' * level} {title}"

        if level == 2:
            if chapter_index >= len(questions):
                raise ValueError("`pics.md` does not contain enough chapter questions.")
            source += f"\n\n**原题**\n\n{questions[chapter_index]}"
            chapter_index += 1

        subquestion = question_for_heading(chapter_index - 1, level, title, questions)
        if subquestion and body:
            prompt = f"{'#' * level} {title} · 题目\n\n**题目**\n\n{subquestion}"
            output_lines.extend(as_markdown_cell(prompt, level))
        elif subquestion:
            source += f"\n\n**题目**\n\n{subquestion}"

        pages = split_body_for_pages(body) if body else []
        if not pages:
            output_lines.extend(as_markdown_cell(source, level))
            continue

        source += f"\n\n{pages[0]}"
        output_lines.extend(as_markdown_cell(source, level))
        for continuation in pages[1:]:
            continuation_source = f"{'#' * level} {title}（续）\n\n{continuation}"
            output_lines.extend(as_markdown_cell(continuation_source, level))

    if chapter_index != len(questions):
        raise ValueError(
            "The number of `##` chapters in the note does not match the questions in `pics.md`."
        )
    return "\n".join(output_lines) + "\n"


def repair_escaped_jupytext(source_path: Path) -> str:
    """Recover a source whose newlines were accidentally written as ``\\n``."""
    raw = source_path.read_text(encoding="utf-8")
    decoded = raw.replace("\\n", "\n")
    marker = re.compile(
        r'^# %% \[markdown\] \{"slideshow": \{"slide_type": "(slide|subslide)"\}, '
        r'"tags": \["(?:slide|subslide)"\]\}$',
        flags=re.MULTILINE,
    )

    def replace_marker(match: re.Match[str]) -> str:
        slide_type = match.group(1)
        tag = "slide" if slide_type == "slide" else "sub-slide"
        return (
            f'# %% [markdown] tags=["{tag}"] '
            f'slideshow={{"slide_type": "{slide_type}"}}'
        )

    repaired, count = marker.subn(replace_marker, decoded)
    if count == 0:
        raise ValueError(f"Could not find Markdown cell markers in {source_path}.")
    return JUPYTEXT_HEADER + repaired.lstrip("\n")


def main() -> None:
    args = parse_args()
    if args.input.exists():
        markdown = args.input.read_text(encoding="utf-8")
        questions = split_problem_blocks(args.questions.read_text(encoding="utf-8"))
        source = build_notebook_source(markdown, questions)
    elif args.output.exists():
        source = repair_escaped_jupytext(args.output)
        print(f"recovered Markdown cells from {args.output}")
    else:
        raise FileNotFoundError(f"Input Markdown not found: {args.input}")
    args.output.write_text(source, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
