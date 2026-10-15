"""
Unit tests for techsupport_preflight: every BLOCKING / DEGRADED rule, the
"resolved config, not file existence" contract (env injection with no .env
file), aggregation of every problem into one report, the secret-redaction
guarantee, and the TECHSUPPORT_PREFLIGHT_STRICT helper.

No network, no LanceDB, no Slack: env files are passed in as mappings (or
written to a temp dir) and every path lives under tempfile.TemporaryDirectory.

Run from LilLisa_Server/cron:
    PYTHONPATH=. python3 tests/test_techsupport_preflight.py
"""

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import techsupport_preflight as preflight  # noqa: E402
from techsupport_preflight import Severity  # noqa: E402

# test_github_sync.py replaces dotenv.dotenv_values with a MagicMock for the
# rest of a `unittest discover` run, so the one test that parses a real env
# file pins the real parser explicitly.
try:
    from dotenv.main import dotenv_values as REAL_DOTENV_VALUES
except ImportError:  # pragma: no cover
    REAL_DOTENV_VALUES = None

RUNNING_AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0
ROOT_SKIP_REASON = "root ignores directory mode bits, so chmod cannot make a directory unwritable"

SECRET_SLACK = "xoxb-SECRET-DO-NOT-LOG"
SECRET_PAT = "ghp_SECRETPAT"
SECRET_URL = "https://user:pw@github.com/o/r.git"
SECRET_AUTH = "auth-SECRET-KEY-VALUE"


class PreflightTestCase(unittest.TestCase):
    """Builds a fully valid configuration rooted in a temp dir; each test
    breaks exactly what it is about."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cron = self.root / "cron"
        self.cron.mkdir()
        self.missing_dir = self.root / "no_env_files_here"
        self._locked = []
        self.sync = {
            "SLACK_BOT_TOKEN": "xoxb-valid",
            "TECHSUPPORT_CHANNEL_ID": "C_TS",
            "PRODUCT_CHANNEL_ID_IDA": "C_IDA",
            "PRODUCT_CHANNEL_ID_IDDM": "C_IDDM",
            "EXPERT_GROUP_ID_IDA": "S_IDA",
            "EXPERT_GROUP_ID_IDDM": "S_IDDM",
        }
        self.github = {
            "GITHUB_TOKEN": "ghp_valid",
            "GITHUB_REPO_URL": "https://github.com/org/repo.git",
        }
        self.lillisa = {
            "AUTHENTICATION_KEY": "auth",
            "LIL_LISA_SERVER_URL": "http://127.0.0.1:8000",
            "LANCEDB_FOLDERPATH": "./lancedb",
            "TECHSUPPORT_SCAN_PRODUCT_CHANNELS": "true",
        }
        self.environ: Dict[str, str] = {}

    def tearDown(self):
        for path in self._locked:
            os.chmod(path, stat.S_IRWXU)
        self._tmp.cleanup()

    def lock(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o500)
        self._locked.append(path)
        return path

    def run_preflight(self, **overrides) -> "preflight.PreflightResult":
        kwargs = dict(
            environ=self.environ,
            sync_env=self.sync,
            github_env=self.github,
            lillisa_env=self.lillisa,
            sync_env_path=self.root / "cron" / "env" / "techsupport_sync.env",
            github_env_path=self.root / "cron" / "env" / "github_push.env",
            lillisa_env_path=self.root / "env" / "lillisa_server.env",
            # LANCEDB_FOLDERPATH=./lancedb resolves against this, so the state
            # folder (techsupport_state/ beside it) is inside the tempdir too.
            server_root=self.root,
        )
        kwargs.update(overrides)
        return preflight.run_preflight(**kwargs)

    def keys_by_severity(self, result, severity: Severity):
        return {key for p in result.problems if p.severity is severity for key in p.keys}

    def find(self, result, key: str) -> Optional["preflight.PreflightProblem"]:
        return next((p for p in result.problems if key in p.keys), None)


class HappyPathTests(PreflightTestCase):
    def test_fully_configured_has_zero_problems(self):
        result = self.run_preflight()
        self.assertEqual(result.problems, ())
        self.assertTrue(result.ok)
        self.assertIn("passed", preflight.format_preflight_report(result, strict=True))

    def test_first_run_without_qa_table_or_markdown_passes(self):
        # A brand-new deployment has neither lancedb/ (so no
        # TECHSUPPORT_QA_PAIRS table) nor techsupport_state/ yet. Both are
        # created by the first run, so their absence must not be a problem.
        state_dir = self.root / "techsupport_state"
        self.assertFalse((self.root / "lancedb").exists())
        self.assertFalse(state_dir.exists())
        result = self.run_preflight()
        self.assertEqual(result.problems, ())
        # Preflight must not create them either -- it is side-effect free.
        self.assertFalse((self.root / "lancedb").exists())
        self.assertFalse(state_dir.exists())


class BlockingKeyTests(PreflightTestCase):
    # Only the keys without which the pipeline cannot ingest AT ALL. Every
    # product-channel-scan key (PRODUCT_CHANNEL_ID_*, EXPERT_GROUP_ID_*) is
    # deliberately absent: the scan runs after the techsupport ingest loop and
    # its failures are caught per product, so those are DEGRADED. See
    # _check_product_scan()'s docstring and ProductScanTests.
    SYNC_KEYS = (
        "SLACK_BOT_TOKEN",
        "TECHSUPPORT_CHANNEL_ID",
    )

    def test_each_blocking_key_missing_individually(self):
        cases = [("sync", key) for key in self.SYNC_KEYS] + [
            ("lillisa", "AUTHENTICATION_KEY"),
            ("lillisa", "LANCEDB_FOLDERPATH"),
        ]
        for source, key in cases:
            with self.subTest(key=key):
                env = {k: v for k, v in getattr(self, source).items() if k != key}
                result = self.run_preflight(**{f"{source}_env": env})
                self.assertFalse(result.ok)
                self.assertEqual([p.keys for p in result.problems], [(key,)])
                self.assertIs(result.problems[0].severity, Severity.BLOCKING)

    def test_several_missing_are_all_reported(self):
        for key in ("SLACK_BOT_TOKEN", "TECHSUPPORT_CHANNEL_ID", "EXPERT_GROUP_ID_IDA"):
            self.sync.pop(key)
        self.lillisa.pop("AUTHENTICATION_KEY")
        result = self.run_preflight()
        self.assertEqual(
            self.keys_by_severity(result, Severity.BLOCKING),
            {"SLACK_BOT_TOKEN", "TECHSUPPORT_CHANNEL_ID", "AUTHENTICATION_KEY"},
        )
        # The report carries every problem regardless of tier, so one rollout
        # fixes all of them -- including the DEGRADED expert group.
        report = preflight.format_preflight_report(result, strict=True)
        for key in ("SLACK_BOT_TOKEN", "TECHSUPPORT_CHANNEL_ID", "EXPERT_GROUP_ID_IDA", "AUTHENTICATION_KEY"):
            self.assertIn(key, report)
        # BLOCKING is listed first and numbered contiguously, so an operator
        # reads what stops the boot before what merely degrades it.
        for number in range(1, 4):
            self.assertIn(f"  {number}. [BLOCKING]", report)
        self.assertIn("  4. [DEGRADED]", report)
        self.assertIn("4 problems: 3 blocking, 1 degraded", report)

    def test_empty_and_whitespace_values_count_as_missing(self):
        self.sync["SLACK_BOT_TOKEN"] = ""
        self.sync["TECHSUPPORT_CHANNEL_ID"] = "   \t"
        self.lillisa["AUTHENTICATION_KEY"] = " "
        result = self.run_preflight()
        self.assertEqual(
            self.keys_by_severity(result, Severity.BLOCKING),
            {"SLACK_BOT_TOKEN", "TECHSUPPORT_CHANNEL_ID", "AUTHENTICATION_KEY"},
        )

    def test_staging_incident_no_env_files_mounted(self):
        # The real failure: no cron/env/*.env mounted, nothing injected.
        result = self.run_preflight(sync_env=None, github_env=None)
        self.assertFalse(result.ok)
        self.assertTrue(
            {"SLACK_BOT_TOKEN", "TECHSUPPORT_CHANNEL_ID"} <= self.keys_by_severity(result, Severity.BLOCKING)
        )


class ResolutionTests(PreflightTestCase):
    def test_values_only_in_environ_with_env_files_absent(self):
        # Documented injection route: no .env files at all, everything in the
        # container environment.
        self.environ = {**self.sync, **self.github, **self.lillisa}
        result = self.run_preflight(
            sync_env=None,
            github_env=None,
            lillisa_env=None,
            sync_env_path=self.missing_dir / "techsupport_sync.env",
            github_env_path=self.missing_dir / "github_push.env",
            lillisa_env_path=self.missing_dir / "lillisa_server.env",
        )
        self.assertEqual(result.problems, ())

    def test_environ_overrides_blank_file_value(self):
        self.sync["SLACK_BOT_TOKEN"] = ""
        self.environ = {"SLACK_BOT_TOKEN": "xoxb-from-env"}
        self.assertEqual(self.run_preflight().problems, ())

    @unittest.skipIf(REAL_DOTENV_VALUES is None, "python-dotenv not installed")
    def test_reads_real_env_file_from_disk(self):
        env_path = self.root / "cron" / "env" / "techsupport_sync.env"
        env_path.parent.mkdir(parents=True)
        env_path.write_text("".join(f"{k}={v}\n" for k, v in self.sync.items()), encoding="utf-8")
        with patch.object(preflight, "dotenv_values", REAL_DOTENV_VALUES):
            result = self.run_preflight(sync_env=None, sync_env_path=env_path)
        self.assertEqual(result.problems, ())

    def test_default_paths_point_at_cron_env_files(self):
        self.assertEqual(preflight.SYNC_ENV_PATH, preflight.PACKAGE_ROOT / "env" / "techsupport_sync.env")
        self.assertEqual(preflight.GITHUB_ENV_PATH, preflight.PACKAGE_ROOT / "env" / "github_push.env")


class ChannelMismatchTests(PreflightTestCase):
    def test_mismatched_alias_is_blocking(self):
        self.sync["TECHSUPPORT_CHANNEL_ID_IDA"] = "C_TS"
        self.sync["TECHSUPPORT_CHANNEL_ID_IDDM"] = "C_OTHER"
        result = self.run_preflight()
        self.assertEqual(len(result.problems), 1)
        problem = result.problems[0]
        self.assertIs(problem.severity, Severity.BLOCKING)
        self.assertIn("TECHSUPPORT_CHANNEL_ID_IDDM", problem.keys)
        self.assertNotIn("TECHSUPPORT_CHANNEL_ID_IDA", problem.keys)

    def test_matching_aliases_are_fine(self):
        for product in ("IDA", "IDDM", "IDO"):
            self.sync[f"TECHSUPPORT_CHANNEL_ID_{product}"] = "C_TS"
        self.assertEqual(self.run_preflight().problems, ())


class GithubTests(PreflightTestCase):
    def test_non_https_url_is_blocking(self):
        for url in ("git@github.com:org/repo.git", "ssh://git@github.com/org/repo.git", "http://github.com/o/r"):
            with self.subTest(url=url):
                self.github["GITHUB_REPO_URL"] = url
                result = self.run_preflight()
                self.assertEqual(self.keys_by_severity(result, Severity.BLOCKING), {"GITHUB_REPO_URL"})

    def test_url_with_embedded_credentials_is_blocking(self):
        for url in (SECRET_URL, "https://ghp_abc@github.com/org/repo.git"):
            with self.subTest(url=url):
                self.github["GITHUB_REPO_URL"] = url
                result = self.run_preflight()
                self.assertEqual(self.keys_by_severity(result, Severity.BLOCKING), {"GITHUB_REPO_URL"})
                self.assertIn("credentials", self.find(result, "GITHUB_REPO_URL").problem)

    def test_absent_github_settings_are_degraded_only(self):
        self.github = {"GITHUB_TOKEN": "", "GITHUB_REPO_URL": "  "}
        result = self.run_preflight()
        self.assertTrue(result.ok)
        self.assertEqual(result.blocking, ())
        self.assertEqual(self.keys_by_severity(result, Severity.DEGRADED), {"GITHUB_TOKEN", "GITHUB_REPO_URL"})

    def test_absent_server_url_is_degraded(self):
        self.lillisa.pop("LIL_LISA_SERVER_URL")
        result = self.run_preflight()
        self.assertTrue(result.ok)
        self.assertEqual(self.keys_by_severity(result, Severity.DEGRADED), {"LIL_LISA_SERVER_URL"})
        self.assertIn("127.0.0.1", self.find(result, "LIL_LISA_SERVER_URL").problem)

    def test_degraded_only_report_uses_degraded_code_and_starts(self):
        self.github = {}
        report = preflight.format_preflight_report(self.run_preflight(), strict=True)
        self.assertIn(preflight.DEGRADED_CODE, report)
        self.assertNotIn(preflight.FAILURE_CODE, report)
        self.assertNotIn("Refusing to start", report)


class ProductScanTests(PreflightTestCase):
    # Every product-channel-scan key is DEGRADED -- see _check_product_scan()'s
    # docstring. The pass runs after the techsupport ingest loop has already
    # completed and its per-product failures are caught, so none of these keys
    # may ever refuse startup. That matters because the scan defaults to ON:
    # promoting any of them would silently make four extra keys mandatory for
    # every deployment.
    SCAN_KEYS = {"PRODUCT_CHANNEL_ID_IDA", "PRODUCT_CHANNEL_ID_IDDM", "EXPERT_GROUP_ID_IDA", "EXPERT_GROUP_ID_IDDM"}

    def _strip_scan_keys(self):
        for key in self.SCAN_KEYS:
            self.sync.pop(key)

    def test_reported_but_never_blocking_when_scan_enabled(self):
        self._strip_scan_keys()
        result = self.run_preflight()
        self.assertEqual(result.blocking, ())
        self.assertEqual(self.keys_by_severity(result, Severity.DEGRADED), self.SCAN_KEYS)

    def test_missing_product_channel_alone_still_starts(self):
        """Scanning only one product is a supported deployment, so a missing
        channel id must stay out of the blocking tier entirely."""
        channel_keys = {"PRODUCT_CHANNEL_ID_IDA", "PRODUCT_CHANNEL_ID_IDDM"}
        for key in channel_keys:
            self.sync.pop(key)
        result = self.run_preflight()
        self.assertTrue(result.ok)
        self.assertEqual(result.blocking, ())
        self.assertEqual(self.keys_by_severity(result, Severity.DEGRADED), channel_keys)

    def test_unconfigured_scan_does_not_refuse_startup(self):
        """The regression guard for the tier decision: a pod whose techsupport
        ingest is fully configured must boot even though the on-by-default scan
        has nothing configured at all."""
        self._strip_scan_keys()
        result = self.run_preflight()
        self.assertTrue(result.ok)
        report = preflight.format_preflight_report(result, strict=True)
        self.assertNotIn("Refusing to start", report)

    def test_scan_defaults_to_enabled(self):
        self._strip_scan_keys()
        self.lillisa.pop("TECHSUPPORT_SCAN_PRODUCT_CHANNELS")
        result = self.run_preflight()
        self.assertEqual(result.blocking, ())
        self.assertEqual(self.keys_by_severity(result, Severity.DEGRADED), self.SCAN_KEYS)

    def test_not_required_when_scan_disabled(self):
        self._strip_scan_keys()
        for value in ("false", "FALSE", "0", "no", "off"):
            with self.subTest(value=value):
                self.lillisa["TECHSUPPORT_SCAN_PRODUCT_CHANNELS"] = value
                self.assertEqual(self.run_preflight().problems, ())

    def test_scan_disabled_via_environ_overrides_file(self):
        self._strip_scan_keys()
        self.environ = {"TECHSUPPORT_SCAN_PRODUCT_CHANNELS": "false"}
        self.assertEqual(self.run_preflight().problems, ())

    def test_ido_keys_stay_optional(self):
        self.sync.pop("PRODUCT_CHANNEL_ID_IDO", None)
        self.sync.pop("EXPERT_GROUP_ID_IDO", None)
        self.assertEqual(self.run_preflight().problems, ())


@unittest.skipIf(RUNNING_AS_ROOT, ROOT_SKIP_REASON)
class WritabilityTests(PreflightTestCase):
    def test_unwritable_lancedb_folder_is_blocking(self):
        locked = self.lock(self.root / "lancedb_locked")
        self.lillisa["LANCEDB_FOLDERPATH"] = str(locked)
        problem = self.find(self.run_preflight(), "LANCEDB_FOLDERPATH")
        self.assertIsNotNone(problem)
        self.assertIs(problem.severity, Severity.BLOCKING)
        self.assertEqual(problem.path, str(locked))

    def test_uncreatable_relative_lancedb_folder_is_blocking(self):
        self.lock(self.root / "ro")
        self.lillisa["LANCEDB_FOLDERPATH"] = "./ro/lancedb"
        problem = self.find(self.run_preflight(), "LANCEDB_FOLDERPATH")
        self.assertIsNotNone(problem)
        self.assertEqual(problem.path, str(self.root / "ro" / "lancedb"))

    def test_unwritable_verified_folder_is_blocking(self):
        locked = self.lock(self.root / "verified_locked")
        self.environ = {"VERIFIED_TECHSUPPORT_QA_FOLDERPATH": str(locked)}
        result = self.run_preflight()
        self.assertEqual(self.keys_by_severity(result, Severity.BLOCKING), {"VERIFIED_TECHSUPPORT_QA_FOLDERPATH"})

    def test_unwritable_default_verified_folder_is_blocking(self):
        # Default when VERIFIED_TECHSUPPORT_QA_FOLDERPATH is unset:
        # <state dir>/verified_techsupport, the state dir being beside lancedb/.
        verified = self.lock(self.root / "techsupport_state" / "verified_techsupport")
        problem = self.find(self.run_preflight(), "VERIFIED_TECHSUPPORT_QA_FOLDERPATH")
        self.assertIsNotNone(problem)
        self.assertEqual(problem.path, str(verified))

    def test_unwritable_state_dir_is_blocking(self):
        state_dir = self.lock(self.root / "techsupport_state")
        result = self.run_preflight()
        problem = self.find(result, "TECHSUPPORT_STATE_DIR")
        self.assertIsNotNone(problem)
        self.assertIs(problem.severity, Severity.BLOCKING)
        self.assertEqual(problem.path, str(state_dir))
        self.assertIn("techsupport_sync_state.json", problem.remediation)
        self.assertIn("TECHSUPPORT_STATE_DIR", problem.remediation)

    def test_state_dir_follows_lancedb_folder(self):
        """The default lives beside LANCEDB_FOLDERPATH, on the same volume."""
        volume = self.lock(self.root / "volume")
        self.lillisa["LANCEDB_FOLDERPATH"] = str(volume / "lancedb")
        problem = self.find(self.run_preflight(), "TECHSUPPORT_STATE_DIR")
        self.assertIsNotNone(problem)
        self.assertEqual(problem.path, str(volume / "techsupport_state"))

    def test_state_dir_override_is_checked(self):
        locked = self.lock(self.root / "state_locked")
        self.environ = {"TECHSUPPORT_STATE_DIR": str(locked)}
        result = self.run_preflight()
        # The default markdown folder sits inside the state folder, so it is flagged too.
        self.assertEqual(
            self.keys_by_severity(result, Severity.BLOCKING),
            {"TECHSUPPORT_STATE_DIR", "VERIFIED_TECHSUPPORT_QA_FOLDERPATH"},
        )
        self.assertEqual(self.find(result, "TECHSUPPORT_STATE_DIR").path, str(locked))

    def test_uncreatable_default_state_dir_is_blocking(self):
        """The LanceDB folder's parent read-only: the state folder cannot be created on first run."""
        volume = self.lock(self.root / "volume")
        self.lillisa["LANCEDB_FOLDERPATH"] = str(volume / "lancedb")
        result = self.run_preflight()
        self.assertEqual(self.find(result, "TECHSUPPORT_STATE_DIR").path, str(volume / "techsupport_state"))
        self.assertEqual(
            self.find(result, "VERIFIED_TECHSUPPORT_QA_FOLDERPATH").path,
            str(volume / "techsupport_state" / "verified_techsupport"),
        )

    def test_overrides_are_read_from_env_file_and_environ(self):
        """The pipeline resolves both overrides from lillisa_server.env with the
        process environment on top -- preflight must match, or it would check a
        folder the pipeline does not actually use."""
        from_file = self.lock(self.root / "from_file")
        self.lillisa["VERIFIED_TECHSUPPORT_QA_FOLDERPATH"] = str(from_file)
        self.assertEqual(self.find(self.run_preflight(), "VERIFIED_TECHSUPPORT_QA_FOLDERPATH").path, str(from_file))
        from_environ = self.lock(self.root / "from_environ")
        self.environ = {"VERIFIED_TECHSUPPORT_QA_FOLDERPATH": str(from_environ)}
        self.assertEqual(
            self.find(self.run_preflight(), "VERIFIED_TECHSUPPORT_QA_FOLDERPATH").path, str(from_environ)
        )

    def test_all_writability_problems_reported_together(self):
        """All three writable-directory problems in one report, so the operator
        fixes one rollout instead of three."""
        self.lillisa["LANCEDB_FOLDERPATH"] = str(self.lock(self.root / "a"))
        self.environ = {
            "VERIFIED_TECHSUPPORT_QA_FOLDERPATH": str(self.lock(self.root / "b")),
            "TECHSUPPORT_STATE_DIR": str(self.lock(self.root / "c")),
        }
        result = self.run_preflight()
        self.assertEqual(
            self.keys_by_severity(result, Severity.BLOCKING),
            {"LANCEDB_FOLDERPATH", "VERIFIED_TECHSUPPORT_QA_FOLDERPATH", "TECHSUPPORT_STATE_DIR"},
        )
        self.assertEqual(len(result.blocking), 3)


class WritabilityHelperTests(unittest.TestCase):
    def test_path_that_is_a_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "f"
            file_path.write_text("x", encoding="utf-8")
            self.assertIn("not a directory", preflight._writability_problem(file_path))
            self.assertIn("not a directory", preflight._writability_problem(file_path / "child"))

    def test_missing_dir_under_writable_parent_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(preflight._writability_problem(Path(tmp) / "x" / "y"))


class RedactionTests(PreflightTestCase):
    def test_no_secret_values_in_report(self):
        self.sync["SLACK_BOT_TOKEN"] = SECRET_SLACK
        self.sync["TECHSUPPORT_CHANNEL_ID_IDA"] = "C_MISMATCH_SECRETISH"
        self.github["GITHUB_TOKEN"] = SECRET_PAT
        self.github["GITHUB_REPO_URL"] = SECRET_URL
        self.lillisa["AUTHENTICATION_KEY"] = SECRET_AUTH
        self.lillisa.pop("LIL_LISA_SERVER_URL")
        self.sync.pop("EXPERT_GROUP_ID_IDA")
        self.environ = {"SOME_OTHER_SECRET": "env-dump-canary"}
        result = self.run_preflight()
        self.assertGreaterEqual(len(result.problems), 4)
        for strict in (True, False):
            report = preflight.format_preflight_report(result, strict=strict)
            for secret in (
                SECRET_SLACK,
                SECRET_PAT,
                SECRET_URL,
                SECRET_AUTH,
                "user:pw",
                "pw@",
                "xoxb-",
                "ghp_",
                "C_MISMATCH_SECRETISH",
                "env-dump-canary",
                "SOME_OTHER_SECRET",
            ):
                self.assertNotIn(secret, report)
        for problem in result.problems:
            for field in (problem.problem, problem.remediation, problem.path or "", *problem.keys):
                for secret in (SECRET_SLACK, SECRET_PAT, SECRET_URL, SECRET_AUTH):
                    self.assertNotIn(secret, field)


class ReportFormatTests(PreflightTestCase):
    def _broken(self):
        self.sync.pop("SLACK_BOT_TOKEN")
        self.sync.pop("TECHSUPPORT_CHANNEL_ID")
        self.github = {}
        return self.run_preflight()

    def test_report_contents(self):
        result = self._broken()
        report = preflight.format_preflight_report(result, strict=True)
        self.assertTrue(report.startswith(f"{preflight.FAILURE_CODE}:"))
        self.assertIn("4 problems", report)
        self.assertIn("[BLOCKING]", report)
        self.assertIn("[DEGRADED]", report)
        self.assertIn(str(self.root / "cron" / "env" / "techsupport_sync.env"), report)
        self.assertIn("overridden by container environment variables", report)
        self.assertIn("subPath", report)
        self.assertIn("Refusing to start", report)

    def test_blocking_listed_before_degraded(self):
        report = preflight.format_preflight_report(self._broken(), strict=True)
        self.assertLess(report.rindex("[BLOCKING]"), report.index("[DEGRADED]"))

    def test_warn_mode_says_pipeline_stays_broken(self):
        report = preflight.format_preflight_report(self._broken(), strict=False)
        self.assertIn(preflight.FAILURE_CODE, report)
        self.assertNotIn("Refusing to start", report)
        self.assertIn("will stay broken", report)


class StrictModeTests(unittest.TestCase):
    def strict(self, environ=None, file_value=None):
        lillisa_env = {} if file_value is None else {"TECHSUPPORT_PREFLIGHT_STRICT": file_value}
        return preflight.is_strict_mode(environ=environ or {}, lillisa_env=lillisa_env)

    def test_default_is_true(self):
        self.assertTrue(self.strict())
        self.assertTrue(self.strict({"TECHSUPPORT_PREFLIGHT_STRICT": ""}))

    def test_falsey_spellings_turn_it_off(self):
        for value in ("false", "0", "no", "False", "NO", " FALSE "):
            with self.subTest(value=value):
                self.assertFalse(self.strict({"TECHSUPPORT_PREFLIGHT_STRICT": value}))

    def test_truthy_spellings_keep_it_on(self):
        for value in ("true", "1", "yes", "TRUE", "Yes"):
            with self.subTest(value=value):
                self.assertTrue(self.strict({"TECHSUPPORT_PREFLIGHT_STRICT": value}))

    def test_unrecognised_value_fails_closed(self):
        self.assertTrue(self.strict({"TECHSUPPORT_PREFLIGHT_STRICT": "flase"}))

    def test_env_file_opt_out_and_environ_override(self):
        self.assertFalse(self.strict(file_value="false"))
        self.assertTrue(self.strict({"TECHSUPPORT_PREFLIGHT_STRICT": "true"}, file_value="false"))

    def test_reads_os_environ_by_default(self):
        with patch.dict(os.environ, {"TECHSUPPORT_PREFLIGHT_STRICT": "no"}):
            self.assertFalse(preflight.is_strict_mode(lillisa_env={}))


class ReportFramingTests(PreflightTestCase):
    """The closing verdict must match where the report is being logged from.

    The runtime net reuses the same formatter on an already-serving pod, where
    a "Refusing to start" verdict would misdescribe what just happened.
    """

    def failing_result(self):
        self.sync.pop("SLACK_BOT_TOKEN")
        result = self.run_preflight()
        self.assertTrue(result.blocking)
        return result

    def test_startup_strict_refuses(self):
        report = preflight.format_preflight_report(self.failing_result(), strict=True)
        self.assertIn("Refusing to start", report)

    def test_startup_non_strict_starts_anyway(self):
        report = preflight.format_preflight_report(self.failing_result(), strict=False)
        self.assertIn("Starting anyway", report)
        self.assertNotIn("Refusing to start", report)

    def test_runtime_framing_mentions_neither_startup_verdict(self):
        result = self.failing_result()
        for strict in (True, False):
            with self.subTest(strict=strict):
                report = preflight.format_preflight_report(result, strict=strict, startup=False)
                self.assertNotIn("Refusing to start", report)
                self.assertNotIn("Starting anyway", report)
                self.assertIn("This is a running pod", report)
                # The problems and remediation still have to be there: the
                # runtime log is an operator's only prompt to go fix this.
                self.assertIn("SLACK_BOT_TOKEN", report)
                self.assertIn(preflight.FAILURE_CODE, report)


class PublicApiTests(unittest.TestCase):
    def test_failure_code_is_stable(self):
        self.assertEqual(preflight.FAILURE_CODE, "TECHSUPPORT_CONFIG_MISSING")

    def test_no_heavy_imports(self):
        # Fresh interpreter: this test process has already imported whatever
        # earlier test modules pulled in.
        # Stdlib + dotenv only. Nothing from src/ either: preflight must stay
        # importable and cheap at startup, which is why it re-states the
        # GITHUB_REPO_URL and channel-id rules instead of importing the modules
        # that own them (those drag in GitPython and requests).
        # The one exception is the state-path resolver (and the env loader it
        # reads), itself stdlib + dotenv only, so preflight checks the exact
        # folders the pipeline writes to instead of a copy of that logic.
        heavy = ("fastapi", "lancedb", "llama_index", "voyageai", "dspy", "requests", "git", "src")
        allowed = ("src", "src.utils", "src.techsupport_state_paths")
        code = (
            "import sys, techsupport_preflight; "
            f"print(sorted(m for m in sys.modules if m.split('.')[0] in {heavy!r} and m not in {allowed!r}))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=str(SCRIPTS_DIR), capture_output=True, text=True, check=True
        )
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
