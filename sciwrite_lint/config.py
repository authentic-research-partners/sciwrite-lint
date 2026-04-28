"""TOML-based configuration for sciwrite-lint.

Looks for .sciwrite-lint.toml in the current directory, then parent
directories (like .eslintrc). Falls back to built-in defaults.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


_DEFAULT_WE_VERBS = [
    "propose",
    "argue",
    "present",
    "define",
    "describe",
    "introduce",
    "summarize",
    "examine",
    "suggest",
    "discuss",
    "note",
    "show",
    "demonstrate",
    "identify",
    "call",
    "use",
    "further",
    "consider",
    "begin",
    "turn",
    "return",
]


class PaperWorkspace(BaseModel):
    """Per-paper directory structure for references and parsed files.

    Each paper gets an isolated workspace under references/{paper_name}/.
    The directory is created lazily on first pipeline run via ensure_dirs().
    Citation metadata is stored in workspace.db (parsed/workspace.db).
    """

    model_config = ConfigDict(frozen=True)

    root: Path  # references/{paper_name}/
    parsed: Path  # references/{paper_name}/parsed/
    source: Path  # references/{paper_name}/source/

    def ensure_dirs(self) -> None:
        """Create the workspace directories if they don't exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.parsed.mkdir(exist_ok=True)
        self.source.mkdir(exist_ok=True)

    def sub_workspace(self, ref_key: str) -> PaperWorkspace:
        """Workspace for a reference's own references (depth+1)."""
        sub = self.root / ref_key
        return PaperWorkspace(
            root=sub,
            parsed=sub / "parsed",
            source=sub / "source",
        )

    @property
    def source_manifest(self) -> Path:
        """Path to source.json — records what was analyzed."""
        return self.root / "source.json"

    def check_source(
        self, tex_path: Path, bib_path: Path | None = None
    ) -> tuple[bool, str]:
        """Check if the workspace source matches the current input.

        Returns (ok, reason). ok=True means workspace is compatible.
        Reasons for incompatibility:
        - Source type changed (tex→pdf or pdf→tex)
        - First run (no source.json yet) — always ok
        """
        import json

        if not self.source_manifest.exists():
            return True, "first_run"

        try:
            manifest = json.loads(self.source_manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return True, "corrupt_manifest"

        old_type = manifest.get("source_type", "")
        new_type = "pdf" if tex_path.suffix.lower() == ".pdf" else "tex"
        if old_type and old_type != new_type:
            return False, f"source_type_changed:{old_type}→{new_type}"

        return True, "ok"

    def save_source(self, tex_path: Path, bib_path: Path | None = None) -> None:
        """Copy source files into workspace and write source.json manifest.

        For LaTeX input, also copies all images referenced via
        \\includegraphics (respecting \\graphicspath), preserving the
        directory structure relative to the .tex file so paths still resolve.
        """
        import hashlib
        import json
        import shutil

        source_type = "pdf" if tex_path.suffix.lower() == ".pdf" else "tex"

        # Copy source file
        shutil.copy2(tex_path, self.source / tex_path.name)

        # Copy .bib if present
        bib_hash = ""
        if bib_path and bib_path.exists():
            shutil.copy2(bib_path, self.source / bib_path.name)
            bib_hash = hashlib.sha256(bib_path.read_bytes()).hexdigest()[:16]

        # Copy referenced images (LaTeX only)
        image_count = 0
        if source_type == "tex":
            from sciwrite_lint.vision.image_extraction import collect_image_paths

            for abs_path, rel_path in collect_image_paths(tex_path):
                dest = self.source / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(abs_path, dest)
                image_count += 1

        manifest = {
            "source_type": source_type,
            "source_file": tex_path.name,
            "source_hash": hashlib.sha256(tex_path.read_bytes()).hexdigest()[:16],
            "bib_file": bib_path.name if bib_path else "",
            "bib_hash": bib_hash,
            "image_count": image_count,
        }
        self.source_manifest.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


class PaperConfig(BaseModel):
    """Configuration for a single paper in the project."""

    name: str
    file_path: Path
    bib: Path | None = None  # explicit .bib path; None = auto-detect from .tex
    prohibited_terms: list[str] = Field(default_factory=list)
    # Per-paper override of the academic-source directory. Accepts
    # ``.pdf`` (primary) and ``.md`` (summaries). The key is named
    # ``local_pdfs_dir`` for historical reasons — it pre-dates markdown
    # summary support — and is read from the TOML unchanged.
    local_pdfs_dir: Path | None = None

    # Per-paper override of the web-capture directory. Accepts ``.md``
    # (hand-written or previously captured) and ``.mhtml`` / ``.mht``
    # (browser-saved web archives for JS-rendered pages).
    local_web_dir: Path | None = None

    def resolve_local_pdfs_dir(self, fallback: Path) -> Path:
        """Return the effective academic-source directory for this paper.

        Resolution order:

        1. ``self.local_pdfs_dir`` if explicitly configured.
        2. ``<file_path.parent>/Sources/full_text`` if it exists on disk
           (per-paper curated archive convention).
        3. ``fallback`` — typically ``LintConfig.local_pdfs_dir``.
        """
        if self.local_pdfs_dir is not None:
            return self.local_pdfs_dir
        auto = self.file_path.parent / "Sources" / "full_text"
        if auto.is_dir():
            return auto
        return fallback

    def resolve_local_web_dir(self, fallback: Path) -> Path:
        """Return the effective web-capture directory for this paper.

        Resolution order mirrors :meth:`resolve_local_pdfs_dir`:

        1. ``self.local_web_dir`` if explicitly configured.
        2. ``<file_path.parent>/Sources/full_text_web`` if it exists on
           disk (per-paper web-capture archive convention).
        3. ``fallback`` — typically ``LintConfig.local_web_dir``.
        """
        if self.local_web_dir is not None:
            return self.local_web_dir
        auto = self.file_path.parent / "Sources" / "full_text_web"
        if auto.is_dir():
            return auto
        return fallback


class LintConfig(BaseModel):
    """Resolved configuration for a sciwrite-lint run."""

    # Paths — the 'before' validator resolves None → CWD-relative defaults
    # before field validation, so these are always Path after construction.
    config_path: Path | None = None  # where the TOML was found (None if no TOML)
    project_dir: Path = Field(default=None)  # type: ignore[assignment]
    references_dir: Path = Field(default=None)  # type: ignore[assignment]
    results_dir: Path = Field(default=None)  # type: ignore[assignment]
    calibration_dir: Path = Field(default=None)  # type: ignore[assignment]
    benchmarks_dir: Path = Field(default=None)  # type: ignore[assignment]
    local_pdfs_dir: Path = Field(default=None)  # type: ignore[assignment]
    local_web_dir: Path = Field(default=None)  # type: ignore[assignment]

    @model_validator(mode="before")
    @classmethod
    def _resolve_default_paths(cls, data: dict) -> dict:  # type: ignore[override]
        """Resolve default paths relative to project_dir."""
        if not isinstance(data, dict):
            return data
        project_dir = data.get("project_dir") or Path.cwd()
        data["project_dir"] = project_dir
        refs = data.get("references_dir")
        if refs is None:
            refs = (project_dir / "references").resolve()
        data["references_dir"] = refs
        if data.get("results_dir") is None:
            data["results_dir"] = (project_dir / "results").resolve()
        if data.get("calibration_dir") is None:
            data["calibration_dir"] = (refs / "scilint-calibration").resolve()
        if data.get("benchmarks_dir") is None:
            data["benchmarks_dir"] = (project_dir / "benchmarks").resolve()
        if data.get("local_pdfs_dir") is None:
            data["local_pdfs_dir"] = (project_dir / "local_pdfs").resolve()
        if data.get("local_web_dir") is None:
            data["local_web_dir"] = (project_dir / "local_web").resolve()
        return data

    # Papers
    papers: list[PaperConfig] = Field(default_factory=list)

    # Style thresholds
    emdash_threshold: float = 3.0
    sentence_length_max: int = 50
    passive_voice_threshold: float = 0.4
    prohibited_terms: list[str] = Field(default_factory=list)
    we_allowed_verbs: list[str] = Field(default_factory=lambda: list(_DEFAULT_WE_VERBS))

    # Document size limit for LLM checks (~50 pages at ~3 500 chars/page).
    # Documents larger than this skip consistency and contribution checks.
    max_document_chars: int = 175_000

    # Rules
    disabled_rules: set[str] = Field(default_factory=set)
    severity_overrides: dict[str, str] = Field(default_factory=dict)

    # API
    polite_email: str = ""
    openalex_interval: float = 0.22
    s2_interval: float = 1.6
    crossref_interval: float = 0.12
    core_interval: float = 1.0
    unpaywall_interval: float = 0.1
    rw_cache_hours: int = 24  # Retraction Watch CSV re-download interval
    # How long (days) to cache a "definitively unavailable" reference —
    # one where every OA source returned not-found / no-match / size /
    # title-mismatch. Within the TTL the fetch stage skips such refs to
    # avoid re-running ~14 API calls per ref per pipeline invocation.
    # After the TTL, the ref is retried (OA status can change when a
    # preprint gets deposited or an embargo lifts). Transient failures
    # (timeouts, 5xx, connection errors) are never cached — they are
    # retried every run. ``--fresh`` recreates the workspace and so
    # bypasses this cache unconditionally.
    fetch_retry_ttl_days: int = 30

    # Output
    output_format: str = "terminal"
    color: bool = True

    # Logging
    log_level: str = "DEBUG"  # file sink level; stderr sink stays at INFO

    # LLM (required — fail if not available)
    llm_endpoint: str = "http://localhost:5001/v1"
    llm_model: str = "qwen3"
    llm_timeout: float = 300.0  # per-request timeout; batches queue in vLLM

    # Embeddings (for reference store semantic retrieval)
    embedding_model: str = "Snowflake/snowflake-arctic-embed-m-v2.0"
    embedding_dim: int = 768
    embedding_device: str = "auto"  # "auto" (cuda if available), "cpu", "cuda"

    # Vision (figure description)
    vision_backend: str = (
        "vllm"  # "vllm" (8B FP8 container, default) or "transformers" (2B subprocess)
    )
    vision_device: str = "auto"  # "auto", "cpu", "cuda" (transformers backend only)

    # Runtime context — set by pipeline before running checks.
    # Registered checks only receive (tex_path, config), so this is
    # how they know which paper's workspace to use.
    current_paper: str = ""

    # Manuscript context — set by build_pdf_context() for PDF input,
    # or by get_or_create_manuscript_context() for LaTeX.
    # Typed as Any to avoid circular import with manuscript_store.
    manuscript_context: Any = Field(default=None, exclude=True)

    # Last pipeline run stats — set by run_full_check(), read by eval runner.
    last_run_stats: Any = Field(default=None, exclude=True)

    @property
    def is_pdf(self) -> bool:
        """True when the current manuscript is a PDF (GROBID-parsed)."""
        ctx = self.manuscript_context
        return ctx is not None and getattr(ctx, "source_type", None) == "pdf"

    # Container config
    grobid_version: str = "0.8.2.1-crf"
    grobid_memory: str = "8g"
    vllm_version: str = "v0.18.0"
    vllm_memory: str = "4g"

    @property
    def grobid_image(self) -> str:
        return f"docker.io/grobid/grobid:{self.grobid_version}"

    @property
    def vllm_image(self) -> str:
        return f"docker.io/vllm/vllm-openai:{self.vllm_version}"

    def is_check_enabled(self, check_id: str) -> bool:
        """Check if a check is enabled."""
        return check_id not in self.disabled_rules

    def is_rule_enabled(self, check_id: str) -> bool:
        """Legacy alias for is_check_enabled."""
        return self.is_check_enabled(check_id)

    def effective_severity(self, check_id: str, default: str) -> str:
        """Return overridden severity or the check's default."""
        return self.severity_overrides.get(check_id, default)

    def get_paper(self, name: str) -> PaperConfig | None:
        """Look up a paper by name."""
        for p in self.papers:
            if p.name == name:
                return p
        return None

    def paper_workspace(self, paper_name: str) -> PaperWorkspace:
        """Return the per-paper workspace under references_dir."""
        root = self.references_dir / paper_name
        return PaperWorkspace(
            root=root,
            parsed=root / "parsed",
            source=root / "source",
        )

    def effective_references_dir(self) -> Path:
        """Per-paper references dir if current_paper is set, else global."""
        if self.current_paper:
            return self.paper_workspace(self.current_paper).root
        return self.references_dir

    def effective_local_pdfs_dir(self, paper_name: str = "") -> Path:
        """Return the academic-source directory to use for ``paper_name``.

        Accepts both ``.pdf`` (peer-reviewed articles) and ``.md``
        (summary notes). Per-paper override wins; otherwise auto-detects
        a sibling ``<file_path.parent>/Sources/full_text`` directory;
        otherwise returns the project-wide ``local_pdfs_dir``. When
        ``paper_name`` is empty or unknown, returns the project-wide
        default.
        """
        name = paper_name or self.current_paper
        if name:
            paper = self.get_paper(name)
            if paper is not None:
                return paper.resolve_local_pdfs_dir(self.local_pdfs_dir)
        return self.local_pdfs_dir

    def effective_local_web_dir(self, paper_name: str = "") -> Path:
        """Return the web-capture directory to use for ``paper_name``.

        Accepts ``.md`` and ``.mhtml`` / ``.mht``. Resolution order
        mirrors :meth:`effective_local_pdfs_dir`: per-paper override,
        then auto-detection of ``<file_path.parent>/Sources/full_text_web``,
        then the project-wide ``local_web_dir``.
        """
        name = paper_name or self.current_paper
        if name:
            paper = self.get_paper(name)
            if paper is not None:
                return paper.resolve_local_web_dir(self.local_web_dir)
        return self.local_web_dir


def is_wsl2() -> bool:
    """Detect WSL2 via /proc/version (contains 'microsoft' or 'WSL2')."""
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def find_config(start: Path | None = None) -> Path | None:
    """Walk up from start (default: cwd) looking for .sciwrite-lint.toml."""
    current = (start or Path.cwd()).resolve()
    while True:
        candidate = current / ".sciwrite-lint.toml"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def load_config(path: Path | None = None) -> LintConfig:
    """Load configuration from a TOML file, or use defaults.

    If path is None, searches for .sciwrite-lint.toml via find_config().
    """
    config = LintConfig()

    toml_path = path or find_config()
    if toml_path is None:
        return config

    config.config_path = toml_path
    config.project_dir = toml_path.parent
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    project_dir = toml_path.parent

    # Papers — [[papers]] array
    for paper_data in data.get("papers", []):
        fp = paper_data.get("file_path", "")
        if not fp:
            continue
        fp_resolved = (project_dir / fp).resolve()
        bib_path = None
        if paper_data.get("bib"):
            bib_path = (project_dir / paper_data["bib"]).resolve()
        local_pdfs_override: Path | None = None
        if paper_data.get("local_pdfs_dir"):
            local_pdfs_override = (project_dir / paper_data["local_pdfs_dir"]).resolve()
        local_web_override: Path | None = None
        if paper_data.get("local_web_dir"):
            local_web_override = (project_dir / paper_data["local_web_dir"]).resolve()
        config.papers.append(
            PaperConfig(
                name=paper_data.get("name", fp_resolved.stem),
                file_path=fp_resolved,
                bib=bib_path,
                prohibited_terms=paper_data.get("prohibited_terms", []),
                local_pdfs_dir=local_pdfs_override,
                local_web_dir=local_web_override,
            )
        )

    # Style section
    style = data.get("style", {})
    if "emdash_threshold" in style:
        config.emdash_threshold = float(style["emdash_threshold"])
    if "sentence_length_max" in style:
        config.sentence_length_max = int(style["sentence_length_max"])
    if "passive_voice_threshold" in style:
        config.passive_voice_threshold = float(style["passive_voice_threshold"])
    if "prohibited_terms" in style:
        config.prohibited_terms = list(style["prohibited_terms"])
    if "we_allowed_verbs" in style:
        config.we_allowed_verbs = list(style["we_allowed_verbs"])
    if "max_document_chars" in style:
        config.max_document_chars = int(style["max_document_chars"])

    # Rules section
    rules = data.get("rules", {})
    if "disable" in rules:
        config.disabled_rules = set(rules["disable"])
    if "severity_overrides" in rules:
        config.severity_overrides = dict(rules["severity_overrides"])

    # API section
    api = data.get("api", {})
    for key in (
        "polite_email",
        "openalex_interval",
        "s2_interval",
        "crossref_interval",
        "core_interval",
        "unpaywall_interval",
        "rw_cache_hours",
        "fetch_retry_ttl_days",
    ):
        if key in api:
            setattr(config, key, api[key])

    # Output section
    output = data.get("output", {})
    if "format" in output:
        config.output_format = output["format"]
    if "color" in output:
        config.color = output["color"]

    # Logging section
    logging = data.get("logging", {})
    if "level" in logging:
        config.log_level = logging["level"].upper()

    # LLM section
    llm = data.get("llm", {})
    if "endpoint" in llm:
        config.llm_endpoint = llm["endpoint"]
    if "model" in llm:
        config.llm_model = llm["model"]

    # Embeddings section
    _EMBEDDING_PRESETS = {
        "snowflake-m-v2": ("Snowflake/snowflake-arctic-embed-m-v2.0", 768),
        "snowflake-l-v2": ("Snowflake/snowflake-arctic-embed-l-v2.0", 1024),
    }
    emb = data.get("embeddings", {})
    if "model" in emb:
        model_val = emb["model"]
        if model_val in _EMBEDDING_PRESETS:
            config.embedding_model, config.embedding_dim = _EMBEDDING_PRESETS[model_val]
        else:
            config.embedding_model = model_val
            if "dimension" in emb:
                config.embedding_dim = int(emb["dimension"])
    if "dimension" in emb and "model" not in emb:
        config.embedding_dim = int(emb["dimension"])
    if "device" in emb:
        config.embedding_device = emb["device"]

    # Vision section
    vision = data.get("vision", {})
    if "backend" in vision:
        config.vision_backend = vision["backend"]
    if "device" in vision:
        config.vision_device = vision["device"]

    # Containers section
    containers = data.get("containers", {})
    if "grobid_version" in containers:
        config.grobid_version = containers["grobid_version"]
    if "grobid_memory" in containers:
        config.grobid_memory = containers["grobid_memory"]
    if "vllm_version" in containers:
        config.vllm_version = containers["vllm_version"]
    if "vllm_memory" in containers:
        config.vllm_memory = containers["vllm_memory"]

    # Paths (relative to config file location)
    if data.get("references_dir"):
        config.references_dir = (project_dir / data["references_dir"]).resolve()
    else:
        config.references_dir = (project_dir / "references").resolve()
    if data.get("results_dir"):
        config.results_dir = (project_dir / data["results_dir"]).resolve()
    else:
        config.results_dir = (project_dir / "results").resolve()
    if data.get("calibration_dir"):
        config.calibration_dir = (project_dir / data["calibration_dir"]).resolve()
    else:
        config.calibration_dir = (
            config.references_dir / "scilint-calibration"
        ).resolve()
    if data.get("benchmarks_dir"):
        config.benchmarks_dir = (project_dir / data["benchmarks_dir"]).resolve()
    else:
        config.benchmarks_dir = (project_dir / "benchmarks").resolve()
    if data.get("local_pdfs_dir"):
        config.local_pdfs_dir = (project_dir / data["local_pdfs_dir"]).resolve()
    else:
        config.local_pdfs_dir = (project_dir / "local_pdfs").resolve()
    if data.get("local_web_dir"):
        config.local_web_dir = (project_dir / data["local_web_dir"]).resolve()
    else:
        config.local_web_dir = (project_dir / "local_web").resolve()

    return config


def generate_default_toml(papers: list[dict[str, str]] | None = None) -> str:
    """Generate a default .sciwrite-lint.toml for `sciwrite-lint init`.

    If papers is provided, include [[papers]] entries for each.
    """
    lines = [
        "# sciwrite-lint configuration",
        "# See: https://github.com/authentic-research-partners/sciwrite-lint",
        "",
        '# local_pdfs_dir = "local_pdfs"   # academic sources you already have: PDFs (incl. OA pages that need a browser) + .md summaries',
        '# local_web_dir  = "local_web"    # web captures: .md or .mhtml (for JS-rendered pages)',
        "",
    ]

    # Papers section
    if papers:
        for p in papers:
            lines.append("[[papers]]")
            lines.append(f'name = "{p["name"]}"')
            lines.append(f'file_path = "{p["file_path"]}"')
            if p.get("bib"):
                lines.append(f'bib = "{p["bib"]}"')
            lines.append("# prohibited_terms = []")
            lines.append("")
    else:
        lines.extend(
            [
                "# Register your papers here. Each [[papers]] entry is one manuscript.",
                "# sciwrite-lint check              — checks all papers",
                "# sciwrite-lint check --paper name — checks one paper",
                "",
                "# [[papers]]",
                '# name = "my-paper"',
                '# file_path = "paper.tex"          # path to .tex or .pdf file',
                '# bib = "references.bib"          # optional, auto-detected from \\bibliography{}',
                '# prohibited_terms = ["CompanyName"]  # terms that must not appear in body',
                "",
            ]
        )

    lines.extend(
        [
            "[llm]",
            'endpoint = "http://localhost:5001/v1"',
            'model = "qwen3"                 # or "gemma3"',
            "",
            "[api]",
            '# polite_email = ""             # recommended for CrossRef/Unpaywall polite pool',
            "# Manage email and API keys interactively: sciwrite-lint config show",
            "# fetch_retry_ttl_days = 30     # cache definitive PDF-not-found results for N days",
            "",
            "[style]",
            "emdash_threshold = 3.0          # max em-dashes per 1000 words",
            "sentence_length_max = 50        # max words per sentence",
            "max_document_chars = 175000     # skip LLM checks for docs larger than this (~50 pages)",
            "",
            "[rules]",
            '# disable = ["style-001"]       # rule IDs to disable',
            "",
            "[vision]",
            '# backend = "vllm"              # "vllm" (8B FP8, default) or "transformers" (2B, no container)',
            '# device = "auto"              # "auto", "cpu", "cuda" (transformers backend only)',
            "",
            "[output]",
            'format = "terminal"             # "terminal" or "json"',
            "color = true",
            "",
            "[logging]",
            'level = "DEBUG"                 # file sink: DEBUG, INFO, WARNING, ERROR',
            "",
        ]
    )

    return "\n".join(lines) + "\n"


def init_project(force: bool = False) -> tuple[bool, str]:
    """Initialize a sciwrite-lint project in the current directory.

    Creates .sciwrite-lint.toml and references/ directory structure.
    Never overwrites existing files or directories.

    Returns (success, message).
    """
    defaults = LintConfig()
    toml_path = Path(".sciwrite-lint.toml")
    refs_dir = defaults.references_dir

    created: list[str] = []
    skipped: list[str] = []

    # Config file
    if toml_path.exists():
        if force:
            # Scan for .tex files to populate papers
            papers = _detect_papers()
            toml_path.write_text(generate_default_toml(papers or None))
            created.append(str(toml_path) + " (overwritten)")
        else:
            skipped.append(str(toml_path) + " (already exists)")
    else:
        papers = _detect_papers()
        toml_path.write_text(generate_default_toml(papers or None))
        created.append(str(toml_path))

    # References root directory (per-paper subdirs created on first pipeline run)
    if refs_dir.exists():
        skipped.append(str(refs_dir) + "/ (already exists)")
    else:
        refs_dir.mkdir()
        created.append(str(refs_dir) + "/")

    # Academic-source drop folder — PDFs you already have (paywalled OR
    # OA pages that need a browser to pass a captcha/JS wall) plus .md summaries.
    local_pdfs = defaults.local_pdfs_dir
    if local_pdfs.exists():
        skipped.append(str(local_pdfs) + "/ (already exists)")
    else:
        local_pdfs.mkdir()
        created.append(str(local_pdfs) + "/")

    # Web-capture drop folder (user-provided .md + .mhtml for JS-rendered pages).
    local_web = defaults.local_web_dir
    if local_web.exists():
        skipped.append(str(local_web) + "/ (already exists)")
    else:
        local_web.mkdir()
        created.append(str(local_web) + "/")

    # Build message
    parts: list[str] = []
    if created:
        parts.append("Created:\n" + "\n".join(f"  {c}" for c in created))
    if skipped:
        parts.append(
            "Skipped (not overwritten):\n" + "\n".join(f"  {s}" for s in skipped)
        )

    if not created and skipped:
        parts.append("\nProject already initialized. Use --force to overwrite config.")

    if created:
        parts.append("\nNext steps:")
        parts.append(
            "  1. Edit .sciwrite-lint.toml — register your papers in [[papers]]"
        )
        parts.append(
            "  2. Set your email (required for full-text download + retraction checks):"
        )
        parts.append("     sciwrite-lint config set-email you@example.com")
        parts.append(
            "  3. Drop pre-downloaded PDFs into local_pdfs/ (paywalled, or OA"
            " pages a browser was needed to reach) and web captures (.md or"
            " .mhtml) into local_web/ (filename ~ reference title)"
        )
        parts.append("  4. Start containers: sciwrite-lint containers start")
        parts.append("  5. Run: sciwrite-lint check")
        parts.append("\nOptional — faster API rate limits: sciwrite-lint config show")

    return bool(created), "\n".join(parts)


def _detect_papers() -> list[dict[str, str]]:
    """Scan current directory for .tex files and build paper entries."""
    papers = []
    for tex in sorted(Path(".").glob("**/*.tex")):
        # Skip common non-paper files
        if any(part.startswith(".") for part in tex.parts):
            continue
        if "references" in tex.parts or "archive" in tex.parts:
            continue

        name = tex.stem
        entry: dict[str, str] = {"name": name, "file_path": str(tex)}

        # Check for matching .bib file
        tex_text = tex.read_text(encoding="utf-8", errors="ignore")
        import re

        bib_match = re.search(r"\\bibliography\{([^}]+)\}", tex_text)
        if bib_match:
            bib_name = bib_match.group(1)
            if not bib_name.endswith(".bib"):
                bib_name += ".bib"
            bib_path = tex.parent / bib_name
            if bib_path.exists():
                entry["bib"] = str(bib_path)

        papers.append(entry)

    return papers
