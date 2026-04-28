"""LaTeX parsing utilities.

All functions are pure: they take strings and return results with no side effects.
"""

from __future__ import annotations

import re


def strip_comments(text: str) -> str:
    r"""Remove LaTeX comments (lines starting with % or trailing % comments).

    Preserves \% (escaped percent signs).
    """
    lines = []
    for line in text.split("\n"):
        result = []
        i = 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            result.append(line[i])
            i += 1
        lines.append("".join(result))
    return "\n".join(lines)


_VERBATIM_ENVS = ("lstlisting", "verbatim", "minted", "Verbatim", "alltt")

_VERBATIM_RE = re.compile(
    r"\\begin\{(" + "|".join(re.escape(e) for e in _VERBATIM_ENVS) + r")\}"
    r".*?"
    r"\\end\{\1\}",
    re.DOTALL,
)

# \verb|...| (any single-char delimiter, e.g. \verb+...+, \verb|...|)
_VERB_INLINE_RE = re.compile(r"\\verb(.)(.*?)\1")


def strip_verbatim(text: str) -> str:
    r"""Replace content of verbatim-like environments and \verb|...| with blanks.

    Preserves line count so that line numbers in downstream results stay correct.
    Strips: lstlisting, verbatim, minted, Verbatim, alltt environments and
    inline \verb|...| commands.
    """

    def _blank_lines(m: re.Match[str]) -> str:
        return "\n" * m.group(0).count("\n")

    text = _VERBATIM_RE.sub(_blank_lines, text)
    text = _VERB_INLINE_RE.sub("", text)
    return text


def extract_body(text: str) -> str:
    r"""Extract content between \begin{document} and \end{document}."""
    start = text.find("\\begin{document}")
    end = text.find("\\end{document}")
    if start == -1 or end == -1:
        return text
    start = text.index("\n", start) + 1
    return text[start:end]


def extract_bibliography(text: str) -> str:
    r"""Extract content inside \begin{thebibliography}...\end{thebibliography}."""
    match = re.search(
        r"\\begin\{thebibliography\}.*?\n(.*?)\\end\{thebibliography\}",
        text,
        re.DOTALL,
    )
    return match.group(1) if match else ""


def body_without_bibliography(text: str) -> str:
    """Return document body with bibliography section removed."""
    body = extract_body(text)
    body = re.sub(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
        "",
        body,
        flags=re.DOTALL,
    )
    return body


def find_all_commands(text: str, cmd: str) -> list[tuple[int, str]]:
    r"""Find all \cmd{arg} occurrences. Returns (line_number, arg) pairs.

    Handles arguments that span multiple lines. Line numbers are 1-indexed.
    """
    results = []
    pattern = re.compile(r"\\%s\{" % re.escape(cmd))
    for m in pattern.finditer(text):
        line_no = text[: m.start()].count("\n") + 1
        start = m.end()
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            arg = text[start : pos - 1]
            results.append((line_no, arg))
    return results


def find_all_cite_keys(text: str) -> list[tuple[int, str]]:
    r"""Find all citation keys from \cite, \citep, \citet, \citeyearpar, \citeunverified.

    Returns (line_number, key) pairs. Multi-key citations like
    \cite{a,b} produce separate entries for each key.
    Skips citations inside verbatim-like environments (lstlisting, verbatim, etc.).
    """
    text = strip_verbatim(text)
    results = []
    pattern = re.compile(r"\\cite(?:unverified|[tp]|yearpar)?\{([^}]+)\}")
    for line_no, line in enumerate(text.split("\n"), 1):
        for m in pattern.finditer(line):
            for key in m.group(1).split(","):
                key = key.strip()
                if key:
                    results.append((line_no, key))
    return results


def find_unverified_cite_keys(text: str) -> set[str]:
    r"""Find citation keys used with \citeunverified{...}."""
    keys = set()
    pattern = re.compile(r"\\citeunverified\{([^}]+)\}")
    for m in pattern.finditer(text):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


def find_bare_cite_keys(text: str) -> set[str]:
    r"""Find citation keys used with regular \cite (not \citeunverified)."""
    all_keys = set()
    unverified = find_unverified_cite_keys(text)

    pattern = re.compile(r"\\cite(?:[tp]|yearpar)?\{([^}]+)\}")
    for m in pattern.finditer(text):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                all_keys.add(key)

    return all_keys - unverified


def word_count(text: str) -> int:
    """Count words in text, excluding LaTeX commands and braces."""
    cleaned = re.sub(r"\\[a-zA-Z]+\*?", "", text)
    cleaned = re.sub(r"[{}~\\]", " ", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    return len(cleaned.split())


_PROSE_NON_ENV_NAMES = (
    "figure",
    "figure*",
    "table",
    "table*",
    "equation",
    "equation*",
    "align",
    "align*",
    "tikzpicture",
    "lstlisting",
    "verbatim",
    "minted",
)

_PROSE_NON_ENV_RE = re.compile(
    r"\\begin\{("
    + "|".join(re.escape(e) for e in _PROSE_NON_ENV_NAMES)
    + r")\}.*?\\end\{\1\}",
    re.DOTALL,
)


def strip_non_prose_environments_preserve_lines(text: str) -> str:
    """Blank out float/math environments while preserving line count.

    Used before paragraph splitting so environments do not fragment
    paragraphs or pollute prose content. Replaces each environment with
    as many newlines as it contained, so line numbers in the stripped
    text match line numbers in the source. Target envs are non-prose
    (figures, tables, math displays, verbatim) — their contents are not
    reviewable as prose regardless of which check is running.
    """

    def _blank(m: re.Match[str]) -> str:
        return "\n" * m.group(0).count("\n")

    return _PROSE_NON_ENV_RE.sub(_blank, text)


def split_paragraphs(text: str) -> list[tuple[int, str]]:
    """Split text into paragraphs with 1-indexed start line numbers.

    Paragraphs are separated by one or more blank lines. Each returned
    ``line_number`` points at the first non-empty line of the paragraph
    within ``text``. Useful for prose-level checks that need paragraph
    boundaries plus source line tracking.
    """
    results: list[tuple[int, str]] = []
    lines = text.split("\n")
    current: list[str] = []
    start_line: int | None = None
    for idx, line in enumerate(lines, 1):
        if line.strip():
            if start_line is None:
                start_line = idx
            current.append(line)
        else:
            if current and start_line is not None:
                results.append((start_line, "\n".join(current)))
                current = []
                start_line = None
    if current and start_line is not None:
        results.append((start_line, "\n".join(current)))
    return results


def split_sentences(text: str) -> list[tuple[int, str]]:
    """Split text into sentences with line numbers.

    Handles common abbreviations and LaTeX constructs.
    Returns (line_number, sentence_text) pairs.
    """
    line_map = []
    line_no = 1
    for ch in text:
        line_map.append(line_no)
        if ch == "\n":
            line_no += 1

    cleaned = re.sub(
        r"\\begin\{(?:tikzpicture|tabular|figure|table|equation|align)\}.*?"
        r"\\end\{(?:tikzpicture|tabular|figure|table|equation|align)\}",
        "",
        text,
        flags=re.DOTALL,
    )

    protected = cleaned
    for abbrev in [
        "et al.",
        "e.g.",
        "i.e.",
        "cf.",
        "vs.",
        "Dr.",
        "Mr.",
        "Mrs.",
        "Prof.",
        "Fig.",
        "Eq.",
        "Ref.",
        "Sec.",
        "Vol.",
        "No.",
        "U.S.",
        "U.K.",
        "N.J.",
    ]:
        protected = protected.replace(abbrev, abbrev.replace(".", "\xb7"))

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", protected)

    results = []
    pos = 0
    for part in parts:
        part = part.replace("\xb7", ".").strip()
        if not part:
            continue
        idx = text.find(part[:30], pos) if len(part) >= 30 else text.find(part, pos)
        if idx >= 0 and idx < len(line_map):
            results.append((line_map[idx], part))
            pos = idx + len(part)
        elif results:
            results.append((results[-1][0], part))
        else:
            results.append((1, part))

    return results


def is_in_environment(text: str, pos: int, env_name: str) -> bool:
    r"""Check if a character position is inside a \begin{env}...\end{env}."""
    before = text[:pos]
    opens = len(re.findall(r"\\begin\{%s\}" % re.escape(env_name), before))
    closes = len(re.findall(r"\\end\{%s\}" % re.escape(env_name), before))
    return opens > closes


def extract_author_block(text: str) -> str:
    r"""Extract content of \author{...} command, handling nested braces."""
    results = find_all_commands(text, "author")
    return results[0][1] if results else ""


def line_number_at(text: str, pos: int) -> int:
    """Return the 1-indexed line number for a character position."""
    return text[:pos].count("\n") + 1
