"""Shared section analysis used by multiple rule modules."""

from __future__ import annotations

import re
from pathlib import Path

from sciwrite_lint.models import SectionInfo
from sciwrite_lint.tex_parser import (
    body_without_bibliography,
    find_all_cite_keys,
    strip_comments,
    word_count,
)


def analyze_sections(tex_path: Path) -> list[SectionInfo]:
    """Analyze section structure: titles, depths, word counts, citation counts."""
    text = strip_comments(tex_path.read_text(encoding="utf-8"))
    body = body_without_bibliography(text)
    lines = body.split("\n")

    sections: list[SectionInfo] = []
    sec_pattern = re.compile(r"\\(section|subsection|subsubsection)\*?\{(.+?)\}")

    # Find all section headings with line numbers
    headings = []
    for i, line in enumerate(lines, 1):
        m = sec_pattern.search(line)
        if m:
            depth = {"section": 0, "subsection": 1, "subsubsection": 2}[m.group(1)]
            title = m.group(2)
            # Check for label on next few lines
            label = ""
            for j in range(i, min(i + 3, len(lines))):
                label_match = re.search(r"\\label\{([^}]+)\}", lines[j - 1])
                if label_match:
                    label = label_match.group(1)
                    break
            headings.append((i, depth, title, label))

    # Compute word counts between headings
    for idx, (start_line, depth, title, label) in enumerate(headings):
        end_line = headings[idx + 1][0] - 1 if idx + 1 < len(headings) else len(lines)
        section_text = "\n".join(lines[start_line:end_line])
        cite_count = len(find_all_cite_keys(section_text))
        sections.append(
            SectionInfo(
                label=label,
                title=title,
                depth=depth,
                start_line=start_line,
                end_line=end_line,
                word_count=word_count(section_text),
                cite_count=cite_count,
            )
        )

    return sections


def analyze_sections_with_text(tex_path: Path) -> list[tuple[SectionInfo, str]]:
    """Like analyze_sections but also returns raw text for each section."""
    text = strip_comments(tex_path.read_text(encoding="utf-8"))
    body = body_without_bibliography(text)
    lines = body.split("\n")

    sec_pattern = re.compile(r"\\(section|subsection|subsubsection)\*?\{(.+?)\}")

    headings = []
    for i, line in enumerate(lines, 1):
        m = sec_pattern.search(line)
        if m:
            depth = {"section": 0, "subsection": 1, "subsubsection": 2}[m.group(1)]
            title = m.group(2)
            label = ""
            for j in range(i, min(i + 3, len(lines))):
                label_match = re.search(r"\\label\{([^}]+)\}", lines[j - 1])
                if label_match:
                    label = label_match.group(1)
                    break
            headings.append((i, depth, title, label))

    result: list[tuple[SectionInfo, str]] = []
    for idx, (start_line, depth, title, label) in enumerate(headings):
        end_line = headings[idx + 1][0] - 1 if idx + 1 < len(headings) else len(lines)
        section_text = "\n".join(lines[start_line:end_line])
        cite_count = len(find_all_cite_keys(section_text))
        info = SectionInfo(
            label=label,
            title=title,
            depth=depth,
            start_line=start_line,
            end_line=end_line,
            word_count=word_count(section_text),
            cite_count=cite_count,
        )
        result.append((info, section_text))

    return result


def get_abstract_text(tex_path: Path) -> str:
    """Extract abstract text from a LaTeX file."""
    text = strip_comments(tex_path.read_text(encoding="utf-8"))
    m = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        text,
        re.DOTALL,
    )
    return m.group(1).strip() if m else ""
