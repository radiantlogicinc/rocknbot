"""
Regression test for pr42-44j.9: nightly_pipeline.reload_techsupport_index()'s
blank-URL bug.

env.get("LIL_LISA_SERVER_URL", DEFAULT_LIL_LISA_SERVER_URL) only falls back to
the loopback default when the key is ABSENT -- a key that is PRESENT but blank
or whitespace-only (easy to produce in a Kubernetes manifest, e.g.
LIL_LISA_SERVER_URL= in lillisa_server.env, or an env var injected as an empty
string) used to yield "", sending the post-ingest index reload POST to a bare
path with no host. This test pins the fix: absent, blank, and whitespace-only
all fall back to DEFAULT_LIL_LISA_SERVER_URL, while a real value is used as-is
(and still has its trailing slash stripped, exactly like before).

No network, no LanceDB: requests.post is mocked and dotenv_values is pinned to
an explicit return value so the test does not depend on run order (under
`unittest discover`, test_github_sync.py globally replaces the `dotenv` module
with a MagicMock -- see test_nightly_pipeline_product_pass.py's comment on the
same issue -- so pinning it explicitly here keeps this test correct whether it
runs standalone or as part of the suite).

Run from LilLisa_Server/cron:
    PYTHONPATH=. python3 tests/test_nightly_pipeline_reload_index.py
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# nightly_pipeline -> techsupport_qa_ingest -> src.utils, whose module-level
# LILLISA_SERVER_ENV_DICT["LOG_LEVEL"] lookup is the one hard env requirement in
# that chain -- same workaround as test_nightly_pipeline_product_pass.py.
os.environ.setdefault("LOG_LEVEL", "INFO")

import nightly_pipeline as pipeline  # noqa: E402

AUTH_KEY = "auth-key-value"
REAL_URL = "http://realhost:9000/"
REAL_URL_STRIPPED = "http://realhost:9000"


class ReloadTechsupportIndexBaseUrlTests(unittest.TestCase):
    """Each test fixes dotenv_values() and os.environ, mocks requests.post, and
    asserts the URL reload_techsupport_index() actually posts to."""

    def _run(self, lillisa_env: dict, environ: dict) -> str:
        """Runs reload_techsupport_index() with the given resolved config and
        returns the base_url it posted to (the host+scheme prefix before
        /reload_techsupport_qa_pairs/)."""
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {"success": True}
        with (
            patch.object(pipeline, "dotenv_values", return_value=lillisa_env),
            patch.dict(os.environ, environ, clear=True),
            patch.object(pipeline, "requests") as mock_requests,
        ):
            mock_requests.post.return_value = fake_response
            result = pipeline.reload_techsupport_index()
            self.assertTrue(result["reloaded"])
            posted_url = mock_requests.post.call_args.args[0]
        self.assertTrue(posted_url.endswith("/reload_techsupport_qa_pairs/"))
        return posted_url[: -len("/reload_techsupport_qa_pairs/")]

    def test_absent_key_falls_back_to_loopback_default(self):
        base_url = self._run(lillisa_env={"AUTHENTICATION_KEY": AUTH_KEY}, environ={})
        self.assertEqual(base_url, pipeline.DEFAULT_LIL_LISA_SERVER_URL)

    def test_blank_value_falls_back_to_loopback_default(self):
        base_url = self._run(lillisa_env={"AUTHENTICATION_KEY": AUTH_KEY, "LIL_LISA_SERVER_URL": ""}, environ={})
        self.assertEqual(base_url, pipeline.DEFAULT_LIL_LISA_SERVER_URL)

    def test_whitespace_only_value_falls_back_to_loopback_default(self):
        base_url = self._run(lillisa_env={"AUTHENTICATION_KEY": AUTH_KEY, "LIL_LISA_SERVER_URL": "   "}, environ={})
        self.assertEqual(base_url, pipeline.DEFAULT_LIL_LISA_SERVER_URL)

    def test_blank_value_injected_via_process_environment_also_falls_back(self):
        # env = {**file_env, **os.environ} -- an env var overriding the file with an
        # empty string must fall back exactly like a blank line in the file itself.
        base_url = self._run(
            lillisa_env={"AUTHENTICATION_KEY": AUTH_KEY, "LIL_LISA_SERVER_URL": REAL_URL},
            environ={"LIL_LISA_SERVER_URL": ""},
        )
        self.assertEqual(base_url, pipeline.DEFAULT_LIL_LISA_SERVER_URL)

    def test_real_value_is_used_as_is_with_trailing_slash_stripped(self):
        base_url = self._run(lillisa_env={"AUTHENTICATION_KEY": AUTH_KEY, "LIL_LISA_SERVER_URL": REAL_URL}, environ={})
        self.assertEqual(base_url, REAL_URL_STRIPPED)

    def test_missing_authentication_key_still_short_circuits_before_any_post(self):
        # Unrelated to the blank-URL bug, but worth pinning: no AUTHENTICATION_KEY means
        # no JWT can be minted, so reload_techsupport_index() must return early.
        with (
            patch.object(pipeline, "dotenv_values", return_value={}),
            patch.dict(os.environ, {}, clear=True),
            patch.object(pipeline, "requests") as mock_requests,
        ):
            result = pipeline.reload_techsupport_index()
        self.assertFalse(result["reloaded"])
        mock_requests.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
