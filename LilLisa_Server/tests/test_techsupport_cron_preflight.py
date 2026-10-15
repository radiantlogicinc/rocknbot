"""
Unit tests for pr42-44j.3/.4/.5: the startup preflight decision wrapper, the
health-probe payload builder, and the runtime-net dedup, all added to
src/techsupport_cron.py.

src.main is deliberately NOT imported here (it pulls in the whole server) --
see tests/test_techsupport_cron_product_scan.py's docstring for the same
convention. The thin call sites in src/main.py (the lifespan fail-fast call
and GET /health/techsupport_pipeline) are therefore not covered directly by a
test here; everything they call into IS covered.

No network, no LanceDB, no Slack: most tests mock
techsupport_preflight.run_preflight() with hand-built PreflightResult objects
to isolate the pure decision logic in techsupport_cron.py from
techsupport_preflight's own validation rules (cron/tests/test_techsupport_preflight.py
already covers those exhaustively). One test (test_no_secret_values_appear_anywhere_in_the_payload)
deliberately exercises the REAL run_preflight() with injected secrets, to prove the
no-secrets guarantee survives all the way through preflight_health_payload()'s
serialisation.

Run from LilLisa_Server:
    PYTHONPATH=. python3 tests/test_techsupport_cron_preflight.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import techsupport_cron  # noqa: E402
from src import utils  # noqa: E402

# techsupport_cron inserts CRON_ROOT (LilLisa_Server/cron) onto sys.path as an import-time
# side effect, so techsupport_preflight is importable here too -- and it is the SAME module
# object as techsupport_cron.techsupport_preflight (Python caches modules by name).
import techsupport_preflight  # noqa: E402
from techsupport_preflight import PreflightProblem, PreflightResult, Severity  # noqa: E402


def _rendered(call):
    """The actual text a %-style logging call would render, since
    call.args[0] alone is just the format string when extra positional args
    were passed (e.g. utils.logger.critical("...:\\n%s", report))."""
    args = call.args
    if len(args) > 1:
        return args[0] % tuple(args[1:])
    return args[0]


FAKE_SOURCES = (
    Path("/fake/techsupport_sync.env"),
    Path("/fake/github_push.env"),
    Path("/fake/lillisa_server.env"),
)

BLOCKING_PROBLEM = PreflightProblem(
    severity=Severity.BLOCKING,
    keys=("SLACK_BOT_TOKEN",),
    problem="SLACK_BOT_TOKEN is not set -- nightly_techsupport_sync.load_env() raises on every tick.",
    remediation="Mount Secret key techsupport_sync.env with SLACK_BOT_TOKEN set.",
)
# A second, DIFFERENT blocking problem -- used to prove the runtime-net dedup re-fires on a
# genuinely new problem rather than treating every BLOCKING result as "the same" standing one.
OTHER_BLOCKING_PROBLEM = PreflightProblem(
    severity=Severity.BLOCKING,
    keys=("AUTHENTICATION_KEY",),
    problem="AUTHENTICATION_KEY is not set -- the post-ingest index reload cannot authenticate.",
    remediation="Mount Secret key lillisa_server.env with AUTHENTICATION_KEY set.",
)
DEGRADED_PROBLEM = PreflightProblem(
    severity=Severity.DEGRADED,
    keys=("GITHUB_TOKEN",),
    problem="GITHUB_TOKEN is not set; the GitHub mirror is disabled.",
    remediation="Mount Secret key github_push.env with GITHUB_TOKEN set.",
)

BLOCKING_RESULT = PreflightResult(problems=(BLOCKING_PROBLEM,), sources=FAKE_SOURCES)
OTHER_BLOCKING_RESULT = PreflightResult(problems=(OTHER_BLOCKING_PROBLEM,), sources=FAKE_SOURCES)
DEGRADED_RESULT = PreflightResult(problems=(DEGRADED_PROBLEM,), sources=FAKE_SOURCES)
CLEAN_RESULT = PreflightResult(problems=(), sources=FAKE_SOURCES)


class _ResetStateMixin:
    """run_once()'s log-spam guard and last-run record are module globals (see
    techsupport_cron._last_config_failure_signature / _last_run_outcome); reset/check
    them so tests don't leak state into each other regardless of run order."""

    def setUp(self):
        super().setUp()
        techsupport_cron._last_config_failure_signature = None  # pylint: disable=protected-access
        self.assertFalse(techsupport_cron.is_running(), "a previous test left the run lock held")


class RunStartupPreflightTests(_ResetStateMixin, unittest.TestCase):
    """pr42-44j.3: the fail-fast decision wrapper called from main.lifespan."""

    def test_strict_mode_and_blocking_problems_raises_and_logs_critical(self):
        with (
            patch.object(techsupport_preflight, "run_preflight", return_value=BLOCKING_RESULT),
            patch.object(techsupport_preflight, "is_strict_mode", return_value=True),
            patch.object(utils.logger, "critical") as critical_log,
            patch.object(utils.logger, "warning") as warning_log,
            patch.object(utils.logger, "error") as error_log,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                techsupport_cron.run_startup_preflight()
        self.assertIn(techsupport_preflight.FAILURE_CODE, str(ctx.exception))
        critical_log.assert_called_once()
        self.assertIn(techsupport_preflight.FAILURE_CODE, critical_log.call_args.args[0])
        self.assertIn("SLACK_BOT_TOKEN", critical_log.call_args.args[0])
        warning_log.assert_not_called()
        error_log.assert_not_called()

    def test_non_strict_mode_and_blocking_problems_does_not_raise_but_logs_full_report(self):
        with (
            patch.object(techsupport_preflight, "run_preflight", return_value=BLOCKING_RESULT),
            patch.object(techsupport_preflight, "is_strict_mode", return_value=False),
            patch.object(utils.logger, "error") as error_log,
            patch.object(utils.logger, "critical") as critical_log,
        ):
            techsupport_cron.run_startup_preflight()  # must not raise
        critical_log.assert_not_called()
        error_log.assert_called_once()
        self.assertIn(techsupport_preflight.FAILURE_CODE, error_log.call_args.args[0])
        self.assertIn("SLACK_BOT_TOKEN", error_log.call_args.args[0])

    def test_degraded_only_never_raises_in_strict_mode_and_logs_warning(self):
        with (
            patch.object(techsupport_preflight, "run_preflight", return_value=DEGRADED_RESULT),
            patch.object(techsupport_preflight, "is_strict_mode", return_value=True),
            patch.object(utils.logger, "warning") as warning_log,
            patch.object(utils.logger, "critical") as critical_log,
            patch.object(utils.logger, "error") as error_log,
        ):
            techsupport_cron.run_startup_preflight()  # must not raise
        critical_log.assert_not_called()
        error_log.assert_not_called()
        warning_log.assert_called_once()
        self.assertIn(techsupport_preflight.DEGRADED_CODE, warning_log.call_args.args[0])

    def test_degraded_only_never_raises_in_non_strict_mode_either(self):
        with (
            patch.object(techsupport_preflight, "run_preflight", return_value=DEGRADED_RESULT),
            patch.object(techsupport_preflight, "is_strict_mode", return_value=False),
            patch.object(utils.logger, "warning") as warning_log,
        ):
            techsupport_cron.run_startup_preflight()  # must not raise
        warning_log.assert_called_once()

    def test_clean_config_does_not_raise_and_logs_one_info_line(self):
        with (
            patch.object(techsupport_preflight, "run_preflight", return_value=CLEAN_RESULT),
            patch.object(utils.logger, "info") as info_log,
            patch.object(utils.logger, "warning") as warning_log,
            patch.object(utils.logger, "critical") as critical_log,
        ):
            techsupport_cron.run_startup_preflight()  # must not raise
        info_log.assert_called_once()
        warning_log.assert_not_called()
        critical_log.assert_not_called()

    def test_cron_package_unavailable_does_not_raise_and_does_not_duplicate_start_scheduler_log(self):
        # IMPORT_ERROR set -> run_pipeline AND techsupport_preflight both import to None
        # together in the real failure case (they come from the same try/except). start_
        # scheduler(), called immediately after this in main.lifespan, already logs this
        # exact condition at ERROR -- see start_scheduler()'s own test coverage -- so this
        # must stay quiet.
        with (
            patch.object(techsupport_cron, "run_pipeline", None),
            patch.object(techsupport_cron, "techsupport_preflight", None),
            patch.object(utils.logger, "debug") as debug_log,
            patch.object(utils.logger, "critical") as critical_log,
            patch.object(utils.logger, "error") as error_log,
            patch.object(utils.logger, "warning") as warning_log,
        ):
            techsupport_cron.run_startup_preflight()  # must not raise
        debug_log.assert_called_once()
        critical_log.assert_not_called()
        error_log.assert_not_called()
        warning_log.assert_not_called()


class PreflightHealthPayloadTests(_ResetStateMixin, unittest.TestCase):
    """pr42-44j.4: the GET /health/techsupport_pipeline payload builder."""

    def test_clean_configuration_is_ok(self):
        with (
            patch.object(techsupport_preflight, "run_preflight", return_value=CLEAN_RESULT),
            patch.object(techsupport_preflight, "is_strict_mode", return_value=True),
        ):
            payload = techsupport_cron.preflight_health_payload()
        self.assertTrue(payload["ok"])
        self.assertIsNone(payload["code"])
        self.assertEqual(payload["blocking_count"], 0)
        self.assertEqual(payload["degraded_count"], 0)
        self.assertEqual(payload["problems"], [])
        self.assertTrue(payload["cron_available"])

    def test_degraded_configuration_is_still_ok(self):
        # DEGRADED-only must map to HTTP 200 in main.py's handler (payload["ok"] is what
        # it branches on) -- a degraded pipeline must not fail a rollout.
        with (
            patch.object(techsupport_preflight, "run_preflight", return_value=DEGRADED_RESULT),
            patch.object(techsupport_preflight, "is_strict_mode", return_value=True),
        ):
            payload = techsupport_cron.preflight_health_payload()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["code"], techsupport_preflight.DEGRADED_CODE)
        self.assertEqual(payload["degraded_count"], 1)
        self.assertEqual(payload["blocking_count"], 0)
        self.assertEqual(payload["problems"][0]["severity"], "DEGRADED")
        self.assertEqual(payload["problems"][0]["keys"], ["GITHUB_TOKEN"])

    def test_blocking_configuration_is_not_ok(self):
        # BLOCKING must map to HTTP 503 in main.py's handler.
        with (
            patch.object(techsupport_preflight, "run_preflight", return_value=BLOCKING_RESULT),
            patch.object(techsupport_preflight, "is_strict_mode", return_value=True),
        ):
            payload = techsupport_cron.preflight_health_payload()
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["code"], techsupport_preflight.FAILURE_CODE)
        self.assertEqual(payload["blocking_count"], 1)
        self.assertEqual(payload["problems"][0]["path"], None)
        self.assertEqual(list(payload["sources"]), [str(p) for p in FAKE_SOURCES])

    def test_cron_unavailable_is_not_ok(self):
        # Deliberate decision (see preflight_health_payload()'s docstring): a pod that
        # cannot run the pipeline at all is treated as at least as broken as a BLOCKING
        # config problem, so this also maps to HTTP 503.
        with (
            patch.object(techsupport_cron, "run_pipeline", None),
            patch.object(techsupport_cron, "techsupport_preflight", None),
        ):
            payload = techsupport_cron.preflight_health_payload()
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["cron_available"])
        self.assertIsNone(payload["code"])

    def test_health_status_code_maps_each_state(self):
        """The 200-vs-503 rule the Kubernetes probe is wired to.

        health_status_code() is tested instead of the route because the suite
        never imports src.main; the handler is a one-line call to it.
        """
        for label, result, strict, expected in (
            ("clean", CLEAN_RESULT, True, 200),
            ("degraded only", DEGRADED_RESULT, True, 200),
            ("blocking under strict", BLOCKING_RESULT, True, 503),
            # The escape hatch: the operator set STRICT=false to bring this pod
            # up knowing the pipeline is broken. A 503 here would let a
            # startupProbe restart-loop it, so it must stay 200 even though the
            # body still reports ok=false for alerting.
            ("blocking with strict opted out", BLOCKING_RESULT, False, 200),
        ):
            with self.subTest(label):
                with (
                    patch.object(techsupport_preflight, "run_preflight", return_value=result),
                    patch.object(techsupport_preflight, "is_strict_mode", return_value=strict),
                ):
                    payload = techsupport_cron.preflight_health_payload()
                self.assertEqual(techsupport_cron.health_status_code(payload), expected)
                if expected == 503:
                    self.assertFalse(payload["ok"])

    def test_health_status_code_503_when_cron_unavailable(self):
        """strict is None on this branch, so the opt-out must not swallow it."""
        with (
            patch.object(techsupport_cron, "run_pipeline", None),
            patch.object(techsupport_cron, "techsupport_preflight", None),
        ):
            payload = techsupport_cron.preflight_health_payload()
        self.assertIsNone(payload["strict"])
        self.assertEqual(techsupport_cron.health_status_code(payload), 503)

    def test_last_run_failed_true_and_no_code_for_a_non_config_failure(self):
        with (
            patch.object(techsupport_cron, "run_pipeline", side_effect=RuntimeError("slack timeout")),
            patch.object(techsupport_preflight, "run_preflight", return_value=CLEAN_RESULT),
        ):
            techsupport_cron.run_once()
        status = techsupport_cron.last_run_status()
        self.assertTrue(status["failed"])
        self.assertIsNone(status["code"])

    def test_last_run_failed_true_and_code_set_for_a_config_failure(self):
        with (
            patch.object(techsupport_cron, "run_pipeline", side_effect=RuntimeError("boom")),
            patch.object(techsupport_preflight, "run_preflight", return_value=BLOCKING_RESULT),
            patch.object(techsupport_preflight, "is_strict_mode", return_value=True),
        ):
            techsupport_cron.run_once()
        status = techsupport_cron.last_run_status()
        self.assertTrue(status["failed"])
        self.assertEqual(status["code"], techsupport_preflight.FAILURE_CODE)

    def test_last_run_failed_false_after_a_success(self):
        with patch.object(techsupport_cron, "run_pipeline", return_value={"added": 1}):
            techsupport_cron.run_once()
        status = techsupport_cron.last_run_status()
        self.assertFalse(status["failed"])
        self.assertIsNone(status["code"])

    def test_no_secret_values_appear_anywhere_in_the_payload(self):
        # Exercises the REAL techsupport_preflight.run_preflight() (not mocked), with
        # secrets injected via its dependency-injection kwargs, so this proves the
        # no-secrets guarantee survives all the way through preflight_health_payload()'s
        # serialisation, not just that the serialisation code itself looks safe.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lancedb_dir = root / "lancedb"
            lancedb_dir.mkdir()
            real_result = techsupport_preflight.run_preflight(
                environ={},
                sync_env={
                    "SLACK_BOT_TOKEN": "xoxb-SECRET-DO-NOT-LOG",
                    "TECHSUPPORT_CHANNEL_ID": "C1",
                    "EXPERT_GROUP_ID_IDA": "S1",
                    "EXPERT_GROUP_ID_IDDM": "S2",
                    "PRODUCT_CHANNEL_ID_IDA": "C_IDA",
                    "PRODUCT_CHANNEL_ID_IDDM": "C_IDDM",
                },
                github_env={
                    "GITHUB_TOKEN": "ghp_SECRETPAT",
                    # Deliberately malformed (embedded credentials) so this is the one
                    # BLOCKING problem in this result, and the secret token is embedded IN
                    # the URL -- exercising _repo_url_problem's "value not shown" rule too.
                    "GITHUB_REPO_URL": "https://ghp_SECRETPAT@github.com/o/r.git",
                },
                lillisa_env={
                    "AUTHENTICATION_KEY": "auth-SECRET-KEY-VALUE",
                    "LIL_LISA_SERVER_URL": "http://127.0.0.1:8000",
                    "LANCEDB_FOLDERPATH": str(lancedb_dir),
                },
                sync_env_path=root / "cron" / "env" / "techsupport_sync.env",
                github_env_path=root / "cron" / "env" / "github_push.env",
                lillisa_env_path=root / "env" / "lillisa_server.env",
                server_root=root,
            )
            self.assertTrue(real_result.blocking, "expected the malformed GITHUB_REPO_URL to be BLOCKING")

            with patch.object(techsupport_preflight, "run_preflight", return_value=real_result):
                payload = techsupport_cron.preflight_health_payload()

        serialised = json.dumps(payload)
        for secret in ("xoxb-SECRET-DO-NOT-LOG", "ghp_SECRETPAT", "auth-SECRET-KEY-VALUE"):
            self.assertNotIn(secret, serialised)
        # Sanity: this exercised a non-trivial (not empty) payload.
        self.assertEqual(payload["blocking_count"], 1)
        self.assertEqual(payload["code"], techsupport_preflight.FAILURE_CODE)


class RuntimeNetDedupTests(_ResetStateMixin, unittest.TestCase):
    """pr42-44j.5: the hourly runtime net, and its log-spam guard."""

    def test_manual_force_run_is_forwarded_to_the_pipeline(self):
        with patch.object(techsupport_cron, "run_pipeline", return_value={"checked": 1}) as run_pipeline:
            result = techsupport_cron.run_once(force=True)
        self.assertTrue(result["ran"])
        run_pipeline.assert_called_once_with(force=True)

    def test_run_once_never_raises_even_for_a_config_problem(self):
        with (
            patch.object(techsupport_cron, "run_pipeline", side_effect=RuntimeError("boom")),
            patch.object(techsupport_preflight, "run_preflight", return_value=BLOCKING_RESULT),
            patch.object(techsupport_preflight, "is_strict_mode", return_value=True),
        ):
            result = techsupport_cron.run_once()  # must not raise
        self.assertFalse(result["ran"])
        self.assertFalse(techsupport_cron.is_running(), "the lock must be released")

    def test_first_failure_logs_full_report_a_repeat_does_not(self):
        with (
            patch.object(techsupport_cron, "run_pipeline", side_effect=RuntimeError("boom")),
            patch.object(techsupport_preflight, "run_preflight", return_value=BLOCKING_RESULT),
            patch.object(techsupport_preflight, "is_strict_mode", return_value=True),
            patch.object(utils.logger, "critical") as critical_log,
        ):
            techsupport_cron.run_once()
            techsupport_cron.run_once()
        self.assertEqual(critical_log.call_count, 2)
        first_message = _rendered(critical_log.call_args_list[0])
        second_message = _rendered(critical_log.call_args_list[1])
        # The full report names the specific problem; the repeat reminder does not re-state it.
        self.assertIn("SLACK_BOT_TOKEN", first_message)
        self.assertNotIn("SLACK_BOT_TOKEN", second_message)
        self.assertIn(techsupport_preflight.FAILURE_CODE, second_message)

    def test_a_different_config_problem_logs_full_report_again(self):
        with (
            patch.object(techsupport_cron, "run_pipeline", side_effect=RuntimeError("boom")),
            patch.object(techsupport_preflight, "is_strict_mode", return_value=True),
            patch.object(utils.logger, "critical") as critical_log,
        ):
            with patch.object(techsupport_preflight, "run_preflight", return_value=BLOCKING_RESULT):
                techsupport_cron.run_once()
            with patch.object(techsupport_preflight, "run_preflight", return_value=OTHER_BLOCKING_RESULT):
                techsupport_cron.run_once()
        self.assertEqual(critical_log.call_count, 2)
        self.assertIn("SLACK_BOT_TOKEN", _rendered(critical_log.call_args_list[0]))
        self.assertIn("AUTHENTICATION_KEY", _rendered(critical_log.call_args_list[1]))

    def test_a_success_in_between_resets_the_dedup(self):
        with (
            patch.object(techsupport_preflight, "is_strict_mode", return_value=True),
            patch.object(utils.logger, "critical") as critical_log,
        ):
            with (
                patch.object(techsupport_cron, "run_pipeline", side_effect=RuntimeError("boom")),
                patch.object(techsupport_preflight, "run_preflight", return_value=BLOCKING_RESULT),
            ):
                techsupport_cron.run_once()
            with patch.object(techsupport_cron, "run_pipeline", return_value={"ok": True}):
                techsupport_cron.run_once()
            with (
                patch.object(techsupport_cron, "run_pipeline", side_effect=RuntimeError("boom again")),
                patch.object(techsupport_preflight, "run_preflight", return_value=BLOCKING_RESULT),
            ):
                techsupport_cron.run_once()
        # Both failures got the full report -- the intervening success cleared the signature.
        self.assertEqual(critical_log.call_count, 2)
        self.assertIn("SLACK_BOT_TOKEN", _rendered(critical_log.call_args_list[0]))
        self.assertIn("SLACK_BOT_TOKEN", _rendered(critical_log.call_args_list[1]))

    def test_preflight_itself_raising_falls_back_to_the_original_behaviour(self):
        # The preflight check inside _log_run_failure must never mask the real failure
        # that run_pipeline() raised.
        with (
            patch.object(techsupport_cron, "run_pipeline", side_effect=RuntimeError("boom")),
            patch.object(techsupport_preflight, "run_preflight", side_effect=OSError("disk full")),
            patch.object(utils.logger, "critical") as critical_log,
        ):
            result = techsupport_cron.run_once()
        self.assertFalse(result["ran"])
        self.assertEqual(result["error"], "boom")
        critical_log.assert_called_once()
        args, kwargs = critical_log.call_args
        self.assertIn("Nightly techsupport pipeline failed", args[0])
        self.assertTrue(kwargs.get("exc_info"))

    def test_an_unavailable_preflight_module_also_falls_back_to_the_original_behaviour(self):
        with (
            patch.object(techsupport_cron, "run_pipeline", side_effect=RuntimeError("boom")),
            patch.object(techsupport_cron, "techsupport_preflight", None),
            patch.object(utils.logger, "critical") as critical_log,
        ):
            result = techsupport_cron.run_once()
        self.assertFalse(result["ran"])
        critical_log.assert_called_once()
        self.assertTrue(critical_log.call_args.kwargs.get("exc_info"))


if __name__ == "__main__":
    unittest.main()
