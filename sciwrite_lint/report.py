"""Output formatters: terminal (rich) and JSON."""

from __future__ import annotations

import json
import sys
from typing import TextIO

from sciwrite_lint.models import Finding

# ---------------------------------------------------------------------------
# Level styling
# ---------------------------------------------------------------------------

_LEVEL_STYLE = {
    "error": ("bold red", "red"),
    "warning": ("bold yellow", "yellow"),
    "info": ("bold blue", "blue"),
}

_LEVEL_LABEL = {
    "error": "ERROR",
    "warning": "WARN",
    "info": "INFO",
}


# ---------------------------------------------------------------------------
# Terminal (rich) output
# ---------------------------------------------------------------------------


def format_terminal(
    findings: list[Finding],
    file: str,
    color: bool = True,
    out: TextIO = sys.stdout,
) -> None:
    """Print findings using rich for styled terminal output."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console(file=out, force_terminal=color, no_color=not color)

    if not findings:
        console.print(f"{file} — no issues found.")
        return

    errors = sum(1 for f in findings if f.level == "error")
    warnings = sum(1 for f in findings if f.level == "warning")
    infos = sum(1 for f in findings if f.level == "info")

    # Build findings table
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("level", width=5, no_wrap=True)
    table.add_column("rule", style="bold", no_wrap=True)
    table.add_column("detail", ratio=1)

    for f in findings:
        label_style, border_style = _LEVEL_STYLE[f.level]
        label = Text(_LEVEL_LABEL[f.level], style=label_style)

        # Location suffix
        loc = ""
        if f.line:
            loc = f" [dim]line {f.line}[/dim]"
        elif f.file:
            loc = f" [dim]{f.file}[/dim]"

        rule_text = Text.from_markup(f"{f.rule_id}{loc}")

        # Message + context
        detail = Text(f.message)
        if f.context:
            detail.append("\n")
            detail.append(f.context, style="dim")

        table.add_row(label, rule_text, detail)

    # Panel border color: red if errors, yellow if only warnings, blue if only info
    if errors:
        border = "red"
    elif warnings:
        border = "yellow"
    else:
        border = "blue"

    # Summary subtitle
    parts = []
    if errors:
        parts.append(f"[red]{errors} error{'s' if errors != 1 else ''}[/red]")
    if warnings:
        parts.append(
            f"[yellow]{warnings} warning{'s' if warnings != 1 else ''}[/yellow]"
        )
    if infos:
        parts.append(f"[blue]{infos} info[/blue]")
    subtitle = " · ".join(parts)

    panel = Panel(
        table,
        title=f"[bold]{file}[/bold] — {len(findings)} issues",
        subtitle=subtitle,
        border_style=border,
        padding=(0, 1),
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def format_json(findings: list[Finding], file: str) -> str:
    """Return findings as JSON string."""
    data = {
        "file": file,
        "total": len(findings),
        "errors": sum(1 for f in findings if f.level == "error"),
        "warnings": sum(1 for f in findings if f.level == "warning"),
        "infos": sum(1 for f in findings if f.level == "info"),
        "findings": [
            {
                "level": f.level,
                "rule_id": f.rule_id,
                "message": f.message,
                "file": f.file,
                "line": f.line,
                "context": f.context,
            }
            for f in findings
        ],
    }
    return json.dumps(data, indent=2)


def format_findings(
    findings: list[Finding],
    label: str,
    fmt: str | None = None,
    color: bool = True,
) -> None:
    """Dispatch findings to the appropriate formatter.

    Args:
        findings: List of findings to output.
        label: File/paper label for the output header.
        fmt: Output format — "terminal" (default) or "json".
        color: Whether to use colored terminal output.
    """
    fmt = fmt or "terminal"
    if fmt == "json":
        print(format_json(findings, label))
    else:
        format_terminal(findings, label, color=color)
