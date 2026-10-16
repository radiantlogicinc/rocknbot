"""Parse and sanitize techsupport_qa_pairs.md heading blocks.

Kept free of DSPy/LanceDB so unit tests can run without the pipeline venv.
"""

import re
from typing import Any, Dict, List, Optional

# Splits on any line starting with "## " (the title heading marker), so the
# entry text itself must never start a line with "## ".
_TITLE_BLOCK_SPLIT_PATTERN = re.compile(r"(?m)^## ")
_TITLE_SUMMARY_PATTERN = re.compile(r"(.+?)\n\n(.*)", re.DOTALL)
_LEADING_HEADING_MARKERS = re.compile(r"^#+\s*")


def sanitize_techsupport_title(title: str) -> str:
    """Strip markdown heading markers so titles cannot re-split the file."""
    cleaned = _LEADING_HEADING_MARKERS.sub("", (title or "").strip())
    cleaned = cleaned.replace("#", "").strip()
    return cleaned or "Untitled"


def sanitize_techsupport_summary(summary: str) -> str:
    """Indent lines that start with ## so parse_summary_markdown will not split."""
    lines = []
    for line in (summary or "").splitlines():
        if line.startswith("##"):
            lines.append("  " + line.lstrip())
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def uniquify_techsupport_title(title: str, existing_titles: List[str]) -> str:
    """If title already exists, append ' - 2', ' - 3', ... (pr42-mp.1.10)."""
    taken = set(existing_titles)
    if title not in taken:
        return title
    n = 2
    while f"{title} - {n}" in taken:
        n += 1
    return f"{title} - {n}"


def prepare_title_and_summary_for_markdown(title: str, summary: str, existing_titles: List[str]) -> tuple[str, str]:
    """Sanitize heading markers then make the title unique among existing_titles."""
    title = uniquify_techsupport_title(sanitize_techsupport_title(title), existing_titles)
    summary = sanitize_techsupport_summary(summary)
    return title, summary


def _normalize_for_duplicate_check(text: str) -> str:
    """Collapse whitespace and casing so two independently-generated summaries
    of the SAME conversation are recognized as the same content even if
    whitespace or capitalization drifted slightly between generations."""
    return " ".join((text or "").split()).lower()


def find_duplicate_entry_index(summary: str, existing_entries: List[Dict[str, Any]]) -> Optional[int]:
    """Return the index of an existing entry whose summary text is identical
    (after whitespace/case normalization) to `summary`, or None if there is no
    such entry.

    This is a content-based dedup check, independent of
    techsupport_review_state.json's thread_ts -> entry mapping (the normal
    dedup guard in techsupport_qa_ingest.add_verified_qa_pair()). That mapping
    can only prevent a duplicate if it still has this thread's entry recorded;
    if the state file was lost, reset, or never had this thread tracked (e.g.
    added before thread_ts tracking existed), re-ingesting the same
    conversation would otherwise regenerate near-identical prose and get
    appended as a brand-new entry -- usually with a "- 2" title suffix from
    uniquify_techsupport_title(), since the title typically regenerates
    identically too. Matching on title alone is NOT used here: a title
    collision between two genuinely DIFFERENT threads is legitimate and is
    exactly what uniquify_techsupport_title() is for. Identical summary text
    is the strong signal that this is the same content being ingested twice.
    """
    target = _normalize_for_duplicate_check(summary)
    if not target:
        return None
    for index, entry in enumerate(existing_entries):
        if _normalize_for_duplicate_check(entry.get("summary", "")) == target:
            return index
    return None


def parse_summary_markdown(file_content: str) -> List[Dict[str, Any]]:
    """Parse techsupport_qa_pairs.md into an ordered list of
    {"title", "summary"} dicts, one per "## {title}" heading block, in file
    order."""
    entries = []
    for raw_block in _TITLE_BLOCK_SPLIT_PATTERN.split(file_content):
        block = raw_block.strip()
        if not block:
            continue
        match = _TITLE_SUMMARY_PATTERN.match(block)
        if not match:
            continue
        title, summary = match[1].strip(), match[2].strip()
        if title and summary:
            entries.append({"title": title, "summary": summary})
    return entries
