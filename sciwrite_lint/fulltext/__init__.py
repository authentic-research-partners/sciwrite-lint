"""Full-text acquisition: download PDFs from open-access sources.

Downloads go to references/. Sources are tried in priority order — trusted,
validated sources first; less reliable sources last:

 1. arXiv (direct PDF, most reliable)
 2. Semantic Scholar openAccessPdf (URL from verify stage)
 3. OA URL from OpenAlex (often arXiv or publisher open-access)
 4. PubMed Central (NIH-funded, trusted; DOI→PMCID lookup if needed)
 5. Europe PMC (broader than PMC, includes preprints)
 6. Unpaywall (legal OA copy aggregator)
 7. bioRxiv/medRxiv (preprint servers, DOI prefix 10.1101/)
 8. NBER Working Papers (title search — US economics, JSON API)
 9. RePEc / IDEAS (title search — broader economics, two-hop HTML scrape)
10. HAL (title search — French national open archive, JSON API)
11. ERIC (title search — US Dept of Education, ED* documents, JSON API)
12. NASA ADS (title search — astronomy/astrophysics, JSON API, key required)
13. OSF Preprints (title search — SocArXiv/PsyArXiv/ChemRxiv/…, two-hop JSON)
14. CORE (institutional repos — least reliable, can return corrupt files)

All downloads are validated: minimum 5 KB size + PDF magic header check,
then a multi-signal match check via ``_validation.validate_pdf_match``
(template-pattern rejection, body-length floor for articles, GROBID
header title + pypdf first-page surname / DOI / year signals). Wrong-
paper matches from title-search sources are caught before the bytes are
cached. NASA ADS is skipped with a one-time INFO log when no API key is
set.

This module is a compatibility surface: implementations live in per-source
submodules (``arxiv``, ``s2``, ``pmc``, ...) but every symbol that external
callers and tests import is re-exported here.

Public-API surface note: the public OA-acquisition API at
``sciwrite_lint.oa._pdf`` and ``sciwrite_lint.oa._search`` imports
``AcquisitionResult``, the six ``lookup_*_by_title`` functions,
``lookup_europepmc``, ``lookup_pmc``, ``lookup_unpaywall``, ``search_core``,
``is_valid_arxiv_id``, and the private ``_download_pdf``. Treat those as
part of the effective public contract even when underscore-prefixed — do
not rename or remove them without updating the ``sciwrite_lint/oa/``
imports.
"""

from __future__ import annotations

from sciwrite_lint._network import (
    clean_and_validate_doi as _clean_and_validate_doi,
    is_valid_arxiv_id,
    ssrf_safe_client,
    stream_with_limit,
)
from sciwrite_lint.fulltext._common import (
    AcquisitionResult,
    _classify_acquisition_failure,
    _DEFINITIVE_FAILURE_PHRASES,
    _is_valid_pdf,
    _MAX_HTML_BYTES,
    _MAX_JSON_BYTES,
    _MAX_PDF_SIZE,
    _MIN_PDF_SIZE,
    _core_limiter,
    _eric_limiter,
    _europepmc_limiter,
    _hal_limiter,
    _ideas_limiter,
    _nasa_ads_limiter,
    _nber_limiter,
    _osf_limiter,
    _pmc_limiter,
    _read_core_key,
    _read_key_file,
    _read_nasa_ads_key,
    _read_ncbi_key,
    _read_s2_key,
    _unpaywall_limiter,
)
from sciwrite_lint.fulltext._download import _download_pdf
from sciwrite_lint.fulltext._orchestrator import acquire_fulltext
from sciwrite_lint.fulltext._search import SearchCandidate, rank_candidates
from sciwrite_lint.fulltext._validation import (
    BibEvidence,
    ValidationResult,
    validate_pdf_match,
)
from sciwrite_lint.fulltext.arxiv import download_arxiv_pdf
from sciwrite_lint.fulltext.biorxiv import download_biorxiv_pdf
from sciwrite_lint.fulltext.core import download_core_pdf, search_core
from sciwrite_lint.fulltext.eric import (
    _parse_eric_candidates,
    download_eric_pdf,
    lookup_eric_by_title,
)
from sciwrite_lint.fulltext.europepmc import download_europepmc_pdf, lookup_europepmc
from sciwrite_lint.fulltext.hal import (
    _parse_hal_candidates,
    download_hal_pdf,
    lookup_hal_by_title,
)
from sciwrite_lint.fulltext.ideas import (
    _fetch_ideas_landing,
    _parse_ideas_file_url,
    _parse_ideas_landing_urls,
    download_ideas_pdf,
    lookup_ideas_by_title,
)
from sciwrite_lint.fulltext.nasa_ads import (
    _parse_ads_candidates,
    download_nasa_ads_pdf,
    lookup_nasa_ads_by_title,
)
from sciwrite_lint.fulltext.nber import (
    _parse_nber_candidates,
    download_nber_pdf,
    lookup_nber_by_title,
)
from sciwrite_lint.fulltext.osf import (
    _parse_osf_download_url,
    _parse_osf_preprints,
    download_osf_pdf,
    lookup_osf_by_title,
)
from sciwrite_lint.fulltext.pmc import download_pmc_pdf, lookup_pmc
from sciwrite_lint.fulltext.s2 import download_s2_pdf
from sciwrite_lint.fulltext.unpaywall import lookup_unpaywall

__all__ = [
    # Public symbols
    "AcquisitionResult",
    "BibEvidence",
    "SearchCandidate",
    "ValidationResult",
    "acquire_fulltext",
    "download_arxiv_pdf",
    "download_biorxiv_pdf",
    "download_core_pdf",
    "download_eric_pdf",
    "download_europepmc_pdf",
    "download_hal_pdf",
    "download_ideas_pdf",
    "download_nasa_ads_pdf",
    "download_nber_pdf",
    "download_osf_pdf",
    "download_pmc_pdf",
    "download_s2_pdf",
    "is_valid_arxiv_id",
    "lookup_eric_by_title",
    "lookup_europepmc",
    "lookup_hal_by_title",
    "lookup_ideas_by_title",
    "lookup_nasa_ads_by_title",
    "lookup_nber_by_title",
    "lookup_osf_by_title",
    "lookup_pmc",
    "lookup_unpaywall",
    "rank_candidates",
    "search_core",
    "ssrf_safe_client",
    "stream_with_limit",
    "validate_pdf_match",
    # Re-exported for tests and other internal modules
    "_DEFINITIVE_FAILURE_PHRASES",
    "_MAX_HTML_BYTES",
    "_MAX_JSON_BYTES",
    "_MAX_PDF_SIZE",
    "_MIN_PDF_SIZE",
    "_classify_acquisition_failure",
    "_clean_and_validate_doi",
    "_core_limiter",
    "_download_pdf",
    "_eric_limiter",
    "_europepmc_limiter",
    "_fetch_ideas_landing",
    "_hal_limiter",
    "_ideas_limiter",
    "_is_valid_pdf",
    "_nasa_ads_limiter",
    "_nber_limiter",
    "_osf_limiter",
    "_parse_ads_candidates",
    "_parse_eric_candidates",
    "_parse_hal_candidates",
    "_parse_ideas_file_url",
    "_parse_ideas_landing_urls",
    "_parse_nber_candidates",
    "_parse_osf_download_url",
    "_parse_osf_preprints",
    "_pmc_limiter",
    "_read_core_key",
    "_read_key_file",
    "_read_nasa_ads_key",
    "_read_ncbi_key",
    "_read_s2_key",
    "_unpaywall_limiter",
]
