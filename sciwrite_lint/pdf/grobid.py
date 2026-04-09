"""GROBID client for academic PDF parsing.

GROBID extracts structured data from academic PDFs: sections with headings,
abstract, references with parsed fields (author, title, year, DOI).

Fully async with connection reuse:
- ``is_grobid_running()`` — health check with 60 s TTL cache; uses longer
  timeout + retry when GROBID was previously confirmed (handles load spikes)
- ``extract_title_from_header()`` — title extraction via processHeaderDocument;
  retries transient 500/503 errors up to 3 times with exponential backoff
- ``process_pdf()`` / ``process_pdf_to_markdown()`` — PDF → structured data / markdown;
  retries transient 500/503 errors up to 3 times with exponential backoff
- ``close_client()`` — call on app shutdown to release the connection pool

All functions share a single ``httpx.AsyncClient`` internally, so concurrent
PDF extractions (e.g. via ``asyncio.gather()``) reuse the same TCP pool.

Runs as a podman/docker container on localhost:8070.
Start with: sciwrite-lint containers start
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from defusedxml.ElementTree import fromstring as _safe_fromstring
from pydantic import BaseModel, Field
from pathlib import Path

import httpx

GROBID_URL = "http://localhost:8070"
TEI_NS = "http://www.tei-c.org/ns/1.0"

CONTAINER_NAME = "grobid"
CONTAINER_IMAGE = "docker.io/grobid/grobid:0.8.2.1-crf"
CONTAINER_RUNTIME = "podman"

_HEALTH_TTL = 60.0  # seconds

# Module-level shared state
_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None
_health_ok: bool = False
_health_checked_at: float = 0.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class GrobidSection(BaseModel):
    """A section extracted from a PDF."""

    title: str
    text: str
    level: int  # nesting depth (0 = top-level)
    index: int


class GrobidReference(BaseModel):
    """A parsed reference from the bibliography."""

    index: int
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    arxiv_id: str = ""
    pmid: str = ""
    isbn: str = ""
    lccn: str = ""
    raw: str = ""


class GrobidResult(BaseModel):
    """Full extraction result from a PDF."""

    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    sections: list[GrobidSection] = Field(default_factory=list)
    references: list[GrobidReference] = Field(default_factory=list)
    raw_tei: str = ""


# ---------------------------------------------------------------------------
# Shared async client
# ---------------------------------------------------------------------------


def _get_client() -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it lazily.

    Recreates the client when the event loop has changed (e.g. after
    a second ``asyncio.run()`` call), since httpx.AsyncClient is bound
    to the loop it was created in.

    Uses direct loop reference (``is``) instead of ``id()`` — CPython
    can reuse memory addresses for different loop objects after GC.
    """
    global _client, _client_loop
    current_loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not current_loop:
        _client = httpx.AsyncClient()
        _client_loop = current_loop
    return _client


async def close_client() -> None:
    """Close the shared AsyncClient. Call on app shutdown."""
    global _client, _client_loop, _health_ok, _health_checked_at
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
    _client_loop = None
    _health_ok = False
    _health_checked_at = 0.0


# ---------------------------------------------------------------------------
# Container management
# ---------------------------------------------------------------------------


async def is_grobid_running() -> bool:
    """Check if GROBID API is responding. Cached for 60s after success."""
    global _health_ok, _health_checked_at
    now = time.monotonic()
    if _health_ok and (now - _health_checked_at) < _HEALTH_TTL:
        return True
    # If GROBID was previously confirmed, use longer timeout and retry once —
    # it may be slow to respond while processing concurrent PDF requests.
    was_ok = _health_ok
    attempts = 2 if was_ok else 1
    timeout = 15.0 if was_ok else 3.0
    for attempt in range(attempts):
        try:
            resp = await _get_client().get(
                f"{GROBID_URL}/api/isalive",
                timeout=timeout,
            )
            _health_ok = resp.status_code == 200
            _health_checked_at = time.monotonic()
            return _health_ok
        except httpx.HTTPError:
            if attempt < attempts - 1:
                await asyncio.sleep(2)
    _health_ok = False
    return False


async def start_grobid(*, memory: str = "8g", image: str = CONTAINER_IMAGE) -> bool:
    """Start GROBID container via podman. Returns True if started.

    Args:
        memory: Container memory limit (e.g. "8g", "12g").
        image: Container image (e.g. "docker.io/grobid/grobid:0.8.2-crf").
            Override via ``[containers] grobid_image`` in config.
    """
    if await is_grobid_running():
        return True

    result = subprocess.run(
        [CONTAINER_RUNTIME, "container", "inspect", CONTAINER_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        subprocess.run(
            [CONTAINER_RUNTIME, "start", CONTAINER_NAME],
            capture_output=True,
        )
    else:
        subprocess.run(
            [
                CONTAINER_RUNTIME,
                "run",
                "-d",
                "--name",
                CONTAINER_NAME,
                "--init",
                "--ulimit",
                "core=0",
                "--memory",
                memory,
                "-p",
                "8070:8070",
                image,
            ],
            capture_output=True,
        )

    for _ in range(30):
        await asyncio.sleep(2)
        if await is_grobid_running():
            return True

    return False


def stop_grobid() -> None:
    """Stop GROBID container."""
    subprocess.run(
        [CONTAINER_RUNTIME, "stop", CONTAINER_NAME],
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# Container memory monitor
# ---------------------------------------------------------------------------

_MEMORY_WARN_THRESHOLD = 0.80  # warn at 80% of limit
_MEMORY_THROTTLE_THRESHOLD = 0.70  # start throttling at 70%
MAX_PARSE_CONCURRENCY = 4


def _resolve_cgroup_dir(
    runtime: str = CONTAINER_RUNTIME, container: str = CONTAINER_NAME
) -> Path | None:
    """Find the cgroup directory for a container.

    Reads the container's init PID via inspect (once), then returns the
    cgroup path accessible via ``/proc/<pid>/root/sys/fs/cgroup``.
    Returns None if the container isn't running.
    """
    result = subprocess.run(
        [runtime, "inspect", container, "--format", "{{.State.Pid}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    pid = result.stdout.strip()
    if not pid or pid == "0":
        return None
    cgroup = Path(f"/proc/{pid}/root/sys/fs/cgroup")
    if not (cgroup / "memory.current").exists():
        return None
    return cgroup


def _read_cgroup_memory(cgroup: Path) -> tuple[int, int | None]:
    """Read memory usage and limit from cgroup v2 files.

    Uses ``anon`` from ``memory.stat`` (actual heap/stack RSS) rather than
    ``memory.current`` which includes page cache and GPU driver mappings.

    Returns (used_bytes, limit_bytes). limit_bytes is None if no limit is set.
    """
    try:
        limit_str = (cgroup / "memory.max").read_text().strip()
        stat_text = (cgroup / "memory.stat").read_text()
    except OSError:
        return 0, None
    limit = None if limit_str == "max" else int(limit_str)
    # Parse "anon <bytes>" line from memory.stat
    used = 0
    for line in stat_text.splitlines():
        if line.startswith("anon "):
            used = int(line.split()[1])
            break
    return used, limit


def container_memory_status(
    runtime: str = CONTAINER_RUNTIME, container: str = CONTAINER_NAME
) -> str | None:
    """Return a human-readable memory status string for a container.

    Returns e.g. "2.6GB / 8.0GB (32%)" or "2.6GB (no limit)" or None
    if the container is not running.
    """
    cgroup = _resolve_cgroup_dir(runtime, container)
    if cgroup is None:
        return None
    used, limit = _read_cgroup_memory(cgroup)
    used_gb = used / (1024**3)
    if limit is not None:
        limit_gb = limit / (1024**3)
        pct = used / limit
        return f"{used_gb:.1f}GB / {limit_gb:.1f}GB ({pct:.0%})"
    return f"{used_gb:.1f}GB (no limit)"


def gpu_memory_status() -> tuple[int, int] | None:
    """Return GPU VRAM usage as ``(used_bytes, total_bytes)``.

    Returns None if nvidia-smi is unavailable.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        used_str, total_str = result.stdout.strip().split(",")
        used_mb, total_mb = int(used_str.strip()), int(total_str.strip())
    except (ValueError, IndexError):
        return None
    return used_mb * 1024 * 1024, total_mb * 1024 * 1024


def gpu_utilization() -> int | None:
    """Return GPU compute utilization as a percentage (0-100), or None."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


_PEAK_FILE = Path.home() / ".sciwrite-lint" / "peak_memory.json"


def _load_peaks() -> dict[str, float]:
    """Load peak memory records from disk."""
    import json

    try:
        return json.loads(_PEAK_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save_peak(container: str, peak_bytes: int) -> None:
    """Save peak memory for a container if it's a new high."""
    import json

    peaks = _load_peaks()
    prev = peaks.get(container, 0)
    if peak_bytes > prev:
        peaks[container] = peak_bytes
        _PEAK_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PEAK_FILE.write_text(json.dumps(peaks))


async def monitor_container_memory(
    sem: asyncio.Semaphore, interval: float = 10.0
) -> None:
    """Background task that monitors GROBID memory and throttles concurrency.

    - Tracks peak usage → ``~/.sciwrite-lint/peak_memory.json``
    - At 70% of limit: reduces parse concurrency from 4 → 2
    - At 80% of limit: reduces to 1 and logs a warning
    - When memory drops below 70%: restores full concurrency

    Throttling works by acquiring slots on ``sem`` (the parse semaphore),
    reducing what's available for actual parsing. The semaphore must be
    created by the caller (same event loop) and shared with the parse stage.

    Designed to be run as an asyncio task; cancel to stop.
    """
    from loguru import logger

    cgroup = _resolve_cgroup_dir()
    if cgroup is None:
        return  # container not running — nothing to monitor

    peak = 0
    reserved = 0  # how many semaphore slots we're holding

    try:
        while True:
            await asyncio.sleep(interval)
            used, limit = _read_cgroup_memory(cgroup)
            if used == 0:
                continue
            if used > peak:
                peak = used
                _save_peak(CONTAINER_NAME, peak)
            if limit is None:
                continue

            pct = used / limit

            # Determine how many slots to reserve (0–3, leaving at least 1)
            if pct >= _MEMORY_WARN_THRESHOLD:
                target_reserved = MAX_PARSE_CONCURRENCY - 1
            elif pct >= _MEMORY_THROTTLE_THRESHOLD:
                target_reserved = MAX_PARSE_CONCURRENCY - 2
            else:
                target_reserved = 0  # full concurrency

            # Acquire slots — if all are held by parsers, skip this cycle
            while reserved < target_reserved:
                try:
                    await asyncio.wait_for(sem.acquire(), timeout=0.5)
                    reserved += 1
                except TimeoutError:
                    break  # parsers hold all slots, retry next cycle

            # Release slots to restore concurrency
            while reserved > target_reserved:
                sem.release()
                reserved -= 1

            if reserved > 0:
                used_gb = used / (1024**3)
                limit_gb = limit / (1024**3)
                active = MAX_PARSE_CONCURRENCY - reserved
                if pct >= _MEMORY_WARN_THRESHOLD:
                    logger.warning(
                        f"GROBID memory: {used_gb:.1f}GB / {limit_gb:.1f}GB "
                        f"({pct:.0%}) — throttled to {active} concurrent parse(s)"
                    )
                else:
                    logger.info(
                        f"GROBID memory: {used_gb:.1f}GB / {limit_gb:.1f}GB "
                        f"({pct:.0%}) — throttled to {active} concurrent parse(s)"
                    )
    finally:
        # Release any reserved slots on cancellation
        for _ in range(reserved):
            sem.release()


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------


async def process_pdf(pdf_path: Path) -> GrobidResult:
    """Process a PDF through GROBID. Returns structured result."""
    if not await is_grobid_running():
        raise RuntimeError(
            "GROBID is not running. Start with:\n  sciwrite-lint containers start"
        )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    from loguru import logger
    from sciwrite_lint.usage import tracked

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        async with tracked("grobid"):
            resp = await _get_client().post(
                f"{GROBID_URL}/api/processFulltextDocument",
                files={"input": (pdf_path.name, pdf_bytes, "application/pdf")},
                data={"consolidateCitations": "1"},
                timeout=60.0,
            )

        if resp.status_code == 200:
            return _parse_tei(resp.text)

        if resp.status_code in (500, 503) and attempt < max_retries:
            delay = 2**attempt
            logger.warning(
                "GROBID returned {} for {} (attempt {}/{}), retrying in {}s",
                resp.status_code,
                pdf_path.name,
                attempt,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)
            continue

        raise RuntimeError(
            f"GROBID returned {resp.status_code} for {pdf_path.name}: {resp.text[:200]}"
        )

    raise RuntimeError(f"GROBID failed after {max_retries} retries for {pdf_path.name}")


async def extract_title_from_header(pdf_path: Path) -> str:
    """Extract the paper title via GROBID's processHeaderDocument endpoint.

    Lighter than processFulltextDocument — only parses the document header.
    Raises RuntimeError if GROBID is not running.
    """
    if not await is_grobid_running():
        raise RuntimeError(
            "GROBID is not running. Start with:\n  sciwrite-lint containers start"
        )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    from loguru import logger

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = await _get_client().post(
                f"{GROBID_URL}/api/processHeaderDocument",
                files={"input": (pdf_path.name, pdf_bytes, "application/pdf")},
                timeout=30.0,
            )
        except httpx.HTTPError as e:
            if attempt < max_retries:
                delay = 2**attempt
                logger.warning(
                    "GROBID header extraction failed for {} (attempt {}/{}), retrying in {}s: {}",
                    pdf_path.name,
                    attempt,
                    max_retries,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                continue
            raise RuntimeError(
                f"GROBID header extraction failed for {pdf_path.name}: {e}"
            ) from e

        if resp.status_code == 200:
            break

        if resp.status_code in (500, 503) and attempt < max_retries:
            delay = 2**attempt
            logger.warning(
                "GROBID header returned {} for {} (attempt {}/{}), retrying in {}s",
                resp.status_code,
                pdf_path.name,
                attempt,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)
            continue

        raise RuntimeError(
            f"GROBID header extraction returned {resp.status_code} for {pdf_path.name}"
        )

    # processHeaderDocument returns BibTeX by default — extract title field
    import re

    m = re.search(r"title\s*=\s*\{(.+?)\}", resp.text, re.DOTALL)
    if m:
        return _strip_license_prefix(m.group(1).strip())

    logger.debug("No title found in GROBID header response for {}", pdf_path.name)
    return ""


def _strip_license_prefix(title: str) -> str:
    """Remove license/permission preamble that GROBID sometimes prepends.

    Some arXiv PDFs (e.g. Google papers) have a license notice at the
    top of the first page.  GROBID's header extraction concatenates this
    with the real title, e.g.::

        "Provided proper attribution is provided, Google hereby grants
         permission to reproduce ... Attention Is All You Need"

    We detect sentences that look like license text (contain
    "permission", "granted", "license", "hereby") and strip them,
    keeping only the final segment as the actual title.
    """
    import re

    # Split on sentence boundaries (period/period+space before an uppercase letter)
    # Also split on ". " that precedes a capitalized word
    segments = re.split(r"\.\s+(?=[A-Z])", title)
    if len(segments) <= 1:
        return title

    license_keywords = {
        "permission",
        "granted",
        "grants",
        "license",
        "licence",
        "licensed",
        "hereby",
        "attribution",
        "redistribute",
        "reproduction",
        "copyright",
    }

    # Walk from the end to find the first non-license segment
    for i in range(len(segments) - 1, -1, -1):
        words = set(segments[i].lower().split())
        if not words & license_keywords:
            # Return this segment and everything after it
            result = ". ".join(segments[i:])
            return result

    # All segments look like license text — return original
    return title


async def process_pdf_to_markdown(pdf_path: Path) -> str:
    """Process PDF and return markdown with ## headings."""
    result = await process_pdf(pdf_path)

    lines = []
    if result.abstract:
        lines.append("## Abstract")
        lines.append("")
        lines.append(result.abstract)
        lines.append("")

    for sec in result.sections:
        prefix = "#" * (sec.level + 2)
        lines.append(f"{prefix} {sec.title}")
        lines.append("")
        lines.append(sec.text)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TEI XML parsing
# ---------------------------------------------------------------------------


def _parse_tei(xml_text: str) -> GrobidResult:
    """Parse GROBID TEI XML into structured result."""
    result = GrobidResult(raw_tei=xml_text)

    root = _safe_fromstring(xml_text)
    ns = {"tei": TEI_NS}

    # Title
    title_el = root.find(".//tei:teiHeader//tei:titleStmt/tei:title", ns)
    if title_el is not None and title_el.text:
        result.title = title_el.text.strip()

    # Authors
    for author_el in root.findall(".//tei:teiHeader//tei:author", ns):
        persname = author_el.find("tei:persName", ns)
        if persname is not None:
            parts = []
            for part in persname:
                if part.text:
                    parts.append(part.text.strip())
            if parts:
                result.authors.append(" ".join(parts))

    # Abstract
    abstract_el = root.find(".//tei:profileDesc/tei:abstract", ns)
    if abstract_el is not None:
        result.abstract = _get_all_text(abstract_el).strip()

    # Body sections
    body_el = root.find(".//tei:text/tei:body", ns)
    if body_el is not None:
        result.sections = _parse_sections(body_el, ns)

    # References
    for i, bs in enumerate(root.findall(".//tei:listBibl/tei:biblStruct", ns)):
        result.references.append(_parse_reference(bs, ns, i))

    return result


def _parse_sections(body_el, ns: dict, level: int = 0) -> list[GrobidSection]:
    """Recursively parse <div> sections from body."""
    sections = []
    idx = 0

    for div in body_el.findall("tei:div", ns):
        head = div.find("tei:head", ns)
        title = (
            head.text.strip()
            if head is not None and head.text
            else f"Section {idx + 1}"
        )

        paragraphs = []
        for p in div.findall("tei:p", ns):
            text = _get_all_text(p).strip()
            if text:
                paragraphs.append(text)

        if paragraphs or title:
            sections.append(
                GrobidSection(
                    title=title,
                    text="\n\n".join(paragraphs),
                    level=level,
                    index=idx,
                )
            )
            idx += 1

        nested = _parse_sections(div, ns, level + 1)
        for s in nested:
            s.index = idx
            idx += 1
        sections.extend(nested)

    return sections


def _parse_reference(bs_el, ns: dict, index: int) -> GrobidReference:
    """Parse a <biblStruct> into GrobidReference."""
    ref = GrobidReference(index=index)

    # Analytic title (article/chapter) is always the paper title.
    analytic_title = bs_el.find("tei:analytic/tei:title", ns)
    if analytic_title is not None and analytic_title.text:
        ref.title = analytic_title.text.strip()

    # Monogr titles: classify by @level attribute.
    # level="j" (journal) or "m" (monograph/book series) → venue.
    # level="a" or absent → paper title (only if analytic didn't supply one).
    # This prevents book series names (level="m") from being mistaken for the
    # paper title — a common GROBID issue with book chapters where the analytic
    # element is missing or empty.
    for monogr_title in bs_el.findall("tei:monogr/tei:title", ns):
        if not monogr_title.text:
            continue
        level = monogr_title.get("level", "")
        text = monogr_title.text.strip()
        if level in ("j", "m"):
            if not ref.venue:
                ref.venue = text
        elif not ref.title:
            ref.title = text

    for author_el in bs_el.findall(".//tei:author/tei:persName", ns):
        parts = []
        for part in author_el:
            if part.text:
                parts.append(part.text.strip())
        if parts:
            ref.authors.append(" ".join(parts))

    date_el = bs_el.find(".//tei:date[@type='published']", ns)
    if date_el is not None:
        ref.year = date_el.get("when", "")[:4]
    if not ref.year:
        date_el = bs_el.find(".//tei:date", ns)
        if date_el is not None:
            ref.year = date_el.get("when", "")[:4]

    doi_el = bs_el.find(".//tei:idno[@type='DOI']", ns)
    if doi_el is not None and doi_el.text:
        ref.doi = doi_el.text.strip()

    arxiv_el = bs_el.find(".//tei:idno[@type='arXiv']", ns)
    if arxiv_el is not None and arxiv_el.text:
        ref.arxiv_id = arxiv_el.text.strip()

    pmid_el = bs_el.find(".//tei:idno[@type='PMID']", ns)
    if pmid_el is not None and pmid_el.text:
        ref.pmid = pmid_el.text.strip()

    isbn_el = bs_el.find(".//tei:idno[@type='ISBN']", ns)
    if isbn_el is not None and isbn_el.text:
        ref.isbn = isbn_el.text.strip()

    lccn_el = bs_el.find(".//tei:idno[@type='LCCN']", ns)
    if lccn_el is not None and lccn_el.text:
        ref.lccn = lccn_el.text.strip()

    ptr_el = bs_el.find(".//tei:ptr[@type='url']", ns)
    if ptr_el is not None:
        ref.url = ptr_el.get("target", "")

    note_el = bs_el.find("tei:note[@type='raw_reference']", ns)
    if note_el is not None and note_el.text:
        ref.raw = note_el.text.strip()

    return ref


def _get_all_text(el) -> str:
    """Get all text content from an element, including nested elements."""
    return " ".join(el.itertext())
