"""
Unit tests for techsupport markdown title/summary sanitizing and unique titles.

Run from LilLisa_Server/cron:
    PYTHONPATH=. python3 tests/test_techsupport_markdown.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from techsupport_markdown import (  # noqa: E402
    find_duplicate_entry_index,
    parse_summary_markdown,
    prepare_title_and_summary_for_markdown,
    sanitize_techsupport_summary,
    sanitize_techsupport_title,
    uniquify_techsupport_title,
)


class SanitizeTechsupportMarkdownTests(unittest.TestCase):
    def test_strips_heading_markers_from_title(self):
        self.assertEqual(sanitize_techsupport_title("## LDAP bind"), "LDAP bind")
        self.assertEqual(sanitize_techsupport_title("Timeout #3"), "Timeout 3")

    def test_indents_hash_hash_in_body(self):
        body = "Intro\n## not a heading\nDone"
        sanitized = sanitize_techsupport_summary(body)
        self.assertTrue(sanitized.splitlines()[1].startswith("  ##"))
        parsed = parse_summary_markdown(f"## Real title\n\n{sanitized}\n\n")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["title"], "Real title")

    def test_uniquify_suffix(self):
        self.assertEqual(uniquify_techsupport_title("LDAP bind", ["other"]), "LDAP bind")
        self.assertEqual(uniquify_techsupport_title("LDAP bind", ["LDAP bind"]), "LDAP bind - 2")
        self.assertEqual(
            uniquify_techsupport_title("LDAP bind", ["LDAP bind", "LDAP bind - 2"]),
            "LDAP bind - 3",
        )

    def test_prepare_sanitizes_then_uniquifies(self):
        title, summary = prepare_title_and_summary_for_markdown(
            "## LDAP bind",
            "See\n## nested",
            ["LDAP bind"],
        )
        self.assertEqual(title, "LDAP bind - 2")
        self.assertIn("  ## nested", summary)


class FindDuplicateEntryIndexTests(unittest.TestCase):
    """Content-based dedup guard: catches a thread being re-ingested as a
    brand-new entry when techsupport_review_state.json's thread_ts -> entry
    mapping is missing or stale (e.g. lost in a state-folder migration), which
    would otherwise regenerate near-identical prose and append a duplicate
    block with a "- 2" title suffix from uniquify_techsupport_title()."""

    def test_no_entries(self):
        self.assertIsNone(find_duplicate_entry_index("Some summary text.", []))

    def test_empty_summary_never_matches(self):
        existing = [{"title": "LDAP bind", "summary": ""}]
        self.assertIsNone(find_duplicate_entry_index("", existing))
        self.assertIsNone(find_duplicate_entry_index("   ", existing))

    def test_exact_match_found(self):
        existing = [
            {"title": "LDAP bind", "summary": "Some resolved content about LDAP binding."},
            {"title": "Timeout tuning", "summary": "Raise the socket timeout to 30s."},
        ]
        self.assertEqual(
            find_duplicate_entry_index("Raise the socket timeout to 30s.", existing),
            1,
        )

    def test_match_ignores_whitespace_and_case_drift(self):
        existing = [{"title": "LDAP bind", "summary": "Some resolved   content.\nAbout LDAP."}]
        self.assertEqual(
            find_duplicate_entry_index("some resolved content. about ldap.", existing),
            0,
        )

    def test_no_match_for_different_content(self):
        existing = [{"title": "LDAP bind", "summary": "Some resolved content about LDAP binding."}]
        self.assertIsNone(find_duplicate_entry_index("Completely unrelated summary text.", existing))

    def test_missing_summary_key_does_not_crash(self):
        existing = [{"title": "LDAP bind"}]
        self.assertIsNone(find_duplicate_entry_index("anything", existing))


if __name__ == "__main__":
    unittest.main()
