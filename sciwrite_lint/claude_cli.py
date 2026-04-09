"""Shared helper for calling Claude CLI from Python.

All automated Claude CLI calls should go through ``run_claude()`` or
``run_claude_async()`` to keep model selection, tool blocking, and
error handling consistent.

Always uses ``--agent`` for strong identity (overrides Claude Code's
default "coder" persona).

For structured JSON responses, use ``run_claude_validated()`` or
``run_claude_async_validated()``: they parse JSON, validate against
a Pydantic model, and retry with the validation error as feedback
if the response doesn't match.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel, ValidationError

_T = TypeVar("_T", bound=BaseModel)

_BLOCKED_TOOLS = "Task,WebSearch,WebFetch,Bash,Write,Edit,NotebookEdit"


def _build_cmd(
    prompt: str,
    *,
    model: str = "sonnet",
    agent: str | Path | None = None,
    system_prompt: str | None = None,
    allowed_tools: str | None = None,
    budget: float = 0.50,
) -> tuple[list[str], Path | None]:
    """Build the claude CLI command list.

    Either ``agent`` (path to .md file) or ``system_prompt`` (text) must
    be provided. If ``system_prompt`` is given, a temporary agent file is
    created with the model in frontmatter.

    Returns (cmd, temp_file) where temp_file is a temporary agent file
    that the caller must clean up (or None if using an existing agent).
    """
    temp_path: Path | None = None

    if agent:
        agent_path = str(agent)
    elif system_prompt:
        tf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        tf.write(f"---\nmodel: {model}\n---\n\n{system_prompt}")
        tf.close()
        agent_path = tf.name
        temp_path = Path(tf.name)
    else:
        raise ValueError("Either agent or system_prompt must be provided")

    cmd = [
        "claude",
        "--agent",
        agent_path,
        "--model",
        model,
        "--print",
        "--no-session-persistence",
    ]

    # Block dangerous tools; allow specific ones if requested
    blocked = _BLOCKED_TOOLS
    if not allowed_tools:
        blocked += ",Read,Glob,Grep"
    cmd.extend(["--disallowed-tools", blocked])

    if allowed_tools:
        cmd.extend(["--tools", allowed_tools])

    cmd.extend(["--max-budget-usd", str(budget)])
    cmd.extend(["--", prompt])

    return cmd, temp_path


def run_claude(
    prompt: str,
    *,
    model: str = "sonnet",
    agent: str | Path | None = None,
    system_prompt: str | None = None,
    allowed_tools: str | None = None,
    budget: float = 0.50,
    timeout: int = 180,
    cwd: Path | None = None,
) -> str | None:
    """Run claude CLI synchronously. Returns stdout or None on error."""
    cmd, temp_path = _build_cmd(
        prompt,
        model=model,
        agent=agent,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        budget=budget,
    )
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd or Path.cwd()),
        )
    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timed out after {}s", timeout)
        return None
    except FileNotFoundError:
        raise RuntimeError(
            "claude CLI not found. Install: https://docs.anthropic.com/en/docs/claude-code"
        )
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error(
            "Claude CLI error (exit {}): {}", result.returncode, result.stderr[:200]
        )
        return None

    return result.stdout


async def run_claude_async(
    prompt: str,
    *,
    model: str = "sonnet",
    agent: str | Path | None = None,
    system_prompt: str | None = None,
    allowed_tools: str | None = None,
    budget: float = 0.50,
    timeout: int = 180,
    cwd: Path | None = None,
) -> str | None:
    """Run claude CLI asynchronously. Returns stdout or None on error."""
    cmd, temp_path = _build_cmd(
        prompt,
        model=model,
        agent=agent,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        budget=budget,
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd or Path.cwd()),
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            logger.error("Claude CLI timed out after {}s", timeout)
            return None
    except FileNotFoundError:
        raise RuntimeError(
            "claude CLI not found. Install: https://docs.anthropic.com/en/docs/claude-code"
        )
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)

    stdout = stdout_bytes.decode() if stdout_bytes else ""
    stderr = stderr_bytes.decode() if stderr_bytes else ""

    if proc.returncode != 0:
        logger.error("Claude CLI error (exit {}): {}", proc.returncode, stderr[:200])
        return None

    return stdout


# ---------------------------------------------------------------------------
# JSON extraction + Pydantic validation
# ---------------------------------------------------------------------------


def extract_json_from_response(text: str) -> dict[str, Any] | None:
    """Extract JSON from Claude CLI text response.

    Handles ``<thinking>`` blocks, markdown code fences, and stray text
    around the JSON object.
    """
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def validate_response(
    raw: str | None,
    model: type[_T],
) -> _T | None:
    """Parse Claude CLI output and validate against a Pydantic model.

    Returns a validated model instance, or None if parsing/validation fails.
    On failure, logs the error for diagnostics.
    """
    if raw is None:
        return None
    data = extract_json_from_response(raw)
    if data is None:
        logger.warning("Could not parse JSON from Claude response: {}", raw[:200])
        return None
    try:
        return model.model_validate(data)
    except ValidationError as e:
        logger.warning("Claude response failed validation: {}", e)
        return None


def run_claude_validated(
    prompt: str,
    response_model: type[_T],
    *,
    retries: int = 1,
    model: str = "sonnet",
    agent: str | Path | None = None,
    system_prompt: str | None = None,
    allowed_tools: str | None = None,
    budget: float = 0.50,
    timeout: int = 180,
    cwd: Path | None = None,
) -> _T | None:
    """Synchronous version of ``run_claude_async_validated``."""
    current_prompt = prompt
    for attempt in range(1 + retries):
        stdout = run_claude(
            current_prompt,
            model=model,
            agent=agent,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            budget=budget,
            timeout=timeout,
            cwd=cwd,
        )
        if stdout is None:
            return None

        data = extract_json_from_response(stdout)
        if data is None:
            if attempt < retries:
                current_prompt = (
                    f"{prompt}\n\n"
                    f"YOUR PREVIOUS RESPONSE COULD NOT BE PARSED AS JSON. "
                    f"Respond with ONLY a valid JSON object, no other text."
                )
                logger.warning(
                    "Claude response not valid JSON (attempt {}/{}), retrying",
                    attempt + 1,
                    1 + retries,
                )
                continue
            return None

        try:
            return response_model.model_validate(data)
        except ValidationError as e:
            if attempt < retries:
                error_msg = str(e)
                current_prompt = (
                    f"{prompt}\n\n"
                    f"YOUR PREVIOUS RESPONSE FAILED VALIDATION:\n{error_msg}\n\n"
                    f"Fix the errors and respond with ONLY a valid JSON object."
                )
                logger.warning(
                    "Claude response failed validation (attempt {}/{}): {}",
                    attempt + 1,
                    1 + retries,
                    error_msg[:200],
                )
            else:
                logger.warning(
                    "Claude response failed validation after {} attempts: {}",
                    1 + retries,
                    str(e)[:200],
                )

    return None


async def run_claude_async_validated(
    prompt: str,
    response_model: type[_T],
    *,
    retries: int = 1,
    model: str = "sonnet",
    agent: str | Path | None = None,
    system_prompt: str | None = None,
    allowed_tools: str | None = None,
    budget: float = 0.50,
    timeout: int = 180,
    cwd: Path | None = None,
) -> _T | None:
    """Run Claude CLI and validate the response against a Pydantic model.

    On validation failure, retries with the error message appended to the
    prompt so Claude can correct its response. Cloud APIs don't have
    constrained decoding, so validation + retry is the only way to
    guarantee schema compliance.

    Args:
        prompt: The user prompt.
        response_model: Pydantic model class to validate against.
        retries: Number of retry attempts on validation failure (default: 1).
        model: Claude model to use (default: sonnet).
        agent: Path to agent .md file.
        system_prompt: System prompt text (alternative to agent).
        allowed_tools: Comma-separated tool list to allow.
        budget: Max budget in USD.
        timeout: CLI timeout in seconds.
        cwd: Working directory for claude CLI.

    Returns:
        Validated model instance, or None if all attempts fail.
    """
    current_prompt = prompt
    for attempt in range(1 + retries):
        stdout = await run_claude_async(
            current_prompt,
            model=model,
            agent=agent,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            budget=budget,
            timeout=timeout,
            cwd=cwd,
        )
        if stdout is None:
            return None

        data = extract_json_from_response(stdout)
        if data is None:
            if attempt < retries:
                current_prompt = (
                    f"{prompt}\n\n"
                    f"YOUR PREVIOUS RESPONSE COULD NOT BE PARSED AS JSON. "
                    f"Respond with ONLY a valid JSON object, no other text."
                )
                logger.warning(
                    "Claude response not valid JSON (attempt {}/{}), retrying",
                    attempt + 1,
                    1 + retries,
                )
                continue
            return None

        try:
            return response_model.model_validate(data)
        except ValidationError as e:
            if attempt < retries:
                error_msg = str(e)
                current_prompt = (
                    f"{prompt}\n\n"
                    f"YOUR PREVIOUS RESPONSE FAILED VALIDATION:\n{error_msg}\n\n"
                    f"Fix the errors and respond with ONLY a valid JSON object."
                )
                logger.warning(
                    "Claude response failed validation (attempt {}/{}): {}",
                    attempt + 1,
                    1 + retries,
                    error_msg[:200],
                )
            else:
                logger.warning(
                    "Claude response failed validation after {} attempts: {}",
                    1 + retries,
                    str(e)[:200],
                )

    return None
