"""
Unit tests that DSPy LM setup is lazy (not at import).

Run from LilLisa_Server/cron:
    PYTHONPATH=. python3 tests/test_lazy_dspy_configure.py
"""

import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import techsupport_classifier as classifier  # noqa: E402


def _module_level_configure_calls(source: str) -> int:
    tree = ast.parse(source)
    count = 0
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "configure_dspy_lm":
                count += 1
    return count


class LazyDspyConfigureTests(unittest.TestCase):
    def test_classifier_and_ingest_have_no_module_level_configure(self):
        for name in ("techsupport_classifier.py", "techsupport_qa_ingest.py"):
            source = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
            self.assertEqual(_module_level_configure_calls(source), 0, msg=name)

    def test_classify_thread_configures_before_predict(self):
        useful = MagicMock()
        useful.is_useful = "no"
        with patch.object(classifier, "configure_dspy_lm") as configure, patch.object(
            classifier, "check_useful", return_value=useful
        ):
            result = classifier.classify_thread([{"ts": "1.0", "user": "U1", "text": "hello"}])
        configure.assert_called_once()
        self.assertFalse(result["is_useful"])

    def test_format_thread_does_not_configure(self):
        with patch.object(classifier, "configure_dspy_lm") as configure:
            text = classifier.format_thread_messages([{"ts": "1.0", "user": "U1", "text": "hello"}])
        configure.assert_not_called()
        self.assertIn("hello", text)


if __name__ == "__main__":
    unittest.main()
