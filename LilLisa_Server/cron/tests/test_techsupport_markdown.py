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


if __name__ == "__main__":
    unittest.main()
