"""
Unit tests for src/techsupport_cron.run_startup_bootstrap(): the startup
bridge that hands off to cron/techsupport_bootstrap.run_bootstrap(), which
clones GITHUB_REPO_URL (the source of truth for verified techsupport
content) and rebuilds TECHSUPPORT_QA_PAIRS from it.

src.main is deliberately NOT imported here (it pulls in the whole server) --
see tests/test_techsupport_cron_product_scan.py's docstring for the same
convention. The one-line call site in main.lifespan is therefore not covered
directly; everything it calls into IS.

No git, no network, no LanceDB: run_bootstrap() itself is mocked, since
cron/tests/test_techsupport_bootstrap.py already covers its decision logic.
What matters here is the wrapper's contract -- any failed step raises and
aborts startup (like the golden QA pairs rebuild), rather than letting
run_bootstrap()'s swallowed per-step errors pass silently.

Run from LilLisa_Server:
    PYTHONPATH=. python3 tests/test_techsupport_cron_bootstrap.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import techsupport_cron  # noqa: E402
from src import utils  # noqa: E402


class RunStartupBootstrapTests(unittest.TestCase):
    def test_clean_run_is_logged_at_info_and_returned(self):
        outcome = {
            "markdown_sync": {"direction": "pull", "pull": {"pulled": True, "entry_count": 638}},
            "lancedb_rebuild": {"ran": True, "forced": True},
        }
        fake_bootstrap = MagicMock()
        fake_bootstrap.run_bootstrap.return_value = outcome

        with patch.object(techsupport_cron, "techsupport_bootstrap", fake_bootstrap), patch.object(
            utils.logger, "info"
        ) as info, patch.object(utils.logger, "error") as error:
            result = techsupport_cron.run_startup_bootstrap()

        self.assertEqual(result, outcome)
        info.assert_called_once()
        error.assert_not_called()

    def test_already_in_sync_restart_is_still_a_clean_run(self):
        # The common case: nothing to pull, row count already matches, so the
        # wrapper must stay quiet rather than reporting a problem.
        outcome = {
            "markdown_sync": {"direction": "none", "pull": {"pulled": False, "reason": "unchanged", "entry_count": 638}},
            "lancedb_rebuild": {"ran": False, "reason": "already_in_sync", "entry_count": 638, "row_count": 638},
        }
        fake_bootstrap = MagicMock()
        fake_bootstrap.run_bootstrap.return_value = outcome

        with patch.object(techsupport_cron, "techsupport_bootstrap", fake_bootstrap), patch.object(
            utils.logger, "error"
        ) as error:
            result = techsupport_cron.run_startup_bootstrap()

        self.assertEqual(result, outcome)
        error.assert_not_called()

    def test_failed_step_aborts_startup(self):
        # run_bootstrap() turns its own exceptions into {"error": ...} entries,
        # which makes this wrapper the only place they become a startup abort --
        # the same fail-fast contract as the golden QA pairs rebuild.
        outcome = {
            "markdown_sync": {"direction": "none", "error": "github unreachable"},
            "lancedb_rebuild": {"ran": False, "reason": "markdown_sync_failed"},
        }
        fake_bootstrap = MagicMock()
        fake_bootstrap.run_bootstrap.return_value = outcome

        with patch.object(techsupport_cron, "techsupport_bootstrap", fake_bootstrap), patch.object(
            utils.logger, "critical"
        ) as critical:
            with self.assertRaises(RuntimeError) as ctx:
                techsupport_cron.run_startup_bootstrap()

        self.assertIn("markdown_sync", str(ctx.exception))
        critical.assert_called_once()

    def test_failed_rebuild_alone_also_aborts_startup(self):
        outcome = {
            "markdown_sync": {"direction": "push", "push": {"pushed": True}},
            "lancedb_rebuild": {"ran": False, "error": "voyage unreachable"},
        }
        fake_bootstrap = MagicMock()
        fake_bootstrap.run_bootstrap.return_value = outcome

        with patch.object(techsupport_cron, "techsupport_bootstrap", fake_bootstrap):
            with self.assertRaises(RuntimeError) as ctx:
                techsupport_cron.run_startup_bootstrap()

        self.assertIn("lancedb_rebuild", str(ctx.exception))

    def test_rebuild_reloads_retrievers(self):
        # main.lifespan opened the retrievers before the table was rebuilt, so a
        # rebuild that is not followed by a reload stays invisible to queries.
        outcome = {
            "markdown_sync": {"direction": "pull", "pull": {"pulled": True, "entry_count": 680}},
            "lancedb_rebuild": {"ran": True, "forced": True, "row_count": 680, "qa_pair_count": 680},
        }
        fake_bootstrap = MagicMock()
        fake_bootstrap.run_bootstrap.return_value = outcome
        reload_retrievers = MagicMock()

        with patch.object(techsupport_cron, "techsupport_bootstrap", fake_bootstrap):
            techsupport_cron.run_startup_bootstrap(reload_retrievers=reload_retrievers)

        reload_retrievers.assert_called_once_with()

    def test_no_rebuild_skips_reload(self):
        outcome = {
            "markdown_sync": {"direction": "none", "pull": {"pulled": False, "reason": "unchanged", "entry_count": 680}},
            "lancedb_rebuild": {"ran": False, "reason": "already_in_sync", "entry_count": 680, "row_count": 680},
        }
        fake_bootstrap = MagicMock()
        fake_bootstrap.run_bootstrap.return_value = outcome
        reload_retrievers = MagicMock()

        with patch.object(techsupport_cron, "techsupport_bootstrap", fake_bootstrap):
            techsupport_cron.run_startup_bootstrap(reload_retrievers=reload_retrievers)

        reload_retrievers.assert_not_called()

    def test_failed_bootstrap_aborts_before_reload(self):
        outcome = {
            "markdown_sync": {"direction": "push", "push": {"pushed": True}},
            "lancedb_rebuild": {"ran": True, "error": "row count mismatch"},
        }
        fake_bootstrap = MagicMock()
        fake_bootstrap.run_bootstrap.return_value = outcome
        reload_retrievers = MagicMock()

        with patch.object(techsupport_cron, "techsupport_bootstrap", fake_bootstrap):
            with self.assertRaises(RuntimeError):
                techsupport_cron.run_startup_bootstrap(reload_retrievers=reload_retrievers)

        reload_retrievers.assert_not_called()

    def test_broken_cron_install_is_a_quiet_no_op(self):
        # A missing cron package is a broken IMAGE, not missing config -- and
        # start_scheduler(), called right after in main.lifespan, already logs it
        # at ERROR. Same handling as run_startup_preflight().
        with patch.object(techsupport_cron, "techsupport_bootstrap", None), patch.object(
            techsupport_cron, "IMPORT_ERROR", "ImportError: no module named nightly_pipeline"
        ), patch.object(utils.logger, "error") as error, patch.object(utils.logger, "debug") as debug:
            result = techsupport_cron.run_startup_bootstrap()

        self.assertEqual(result["reason"], "unavailable")
        self.assertFalse(result["ran"])
        error.assert_not_called()
        debug.assert_called_once()


if __name__ == "__main__":
    unittest.main()
