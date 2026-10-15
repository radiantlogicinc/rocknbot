"""In-process driver for the nightly techsupport pipeline.

This process is the scheduler. There is no host crontab: `main.lifespan`
starts the periodic tick below, and POST /run_nightly_pipeline/ forces a run
on demand. The pipeline modules stay in `cron/` rather than under `src/` so
they remain a self-contained package that can still be invoked directly
(`cd cron && make run-nightly`) for one-off ops work.

Runs are serialised by the lock below, so a manual trigger landing during a
tick run (or vice versa) is dropped rather than queued. The manual
product-channel scan (run_product_channel_scan_once, behind POST
/run_product_channel_scan/) takes the same lock: it touches the same state
file and the same verified-entry store as the nightly run, so the two must
never overlap in either direction.
"""

import asyncio
import os
import sys
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

from src import utils

# The pipeline gates itself on TECHSUPPORT_SYNC_INTERVAL_HOURS (see
# nightly_pipeline.is_channel_check_due), so the tick only needs to fire more
# often than that interval -- not at a specific hour. Extra ticks no-op, and a
# container restart self-heals on the next one.
DEFAULT_TICK_MINUTES = 60


def _resolve_cron_root() -> Optional[Path]:
    """Locate the cron package: <server root>/cron in both the repo and the image."""
    server_root = Path(__file__).resolve().parent.parent
    candidate = Path(os.environ.get("TECHSUPPORT_CRON_ROOT") or server_root / "cron")
    return candidate.resolve() if (candidate / "nightly_pipeline.py").is_file() else None


CRON_ROOT = _resolve_cron_root()
if CRON_ROOT and str(CRON_ROOT) not in sys.path:
    sys.path.insert(0, str(CRON_ROOT))

try:
    # Needs CRON_ROOT on sys.path above; the cron package uses flat imports.
    from nightly_pipeline import run_pipeline, run_product_channel_scan  # noqa: E402

    # techsupport_preflight (pr42-44j.1/.2) lives in the same cron package, so it fails
    # to import exactly when nightly_pipeline does -- the "broken cron install" case
    # below already covers it, with no separate detection needed.
    import techsupport_preflight  # noqa: E402

    IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 -- a broken cron install must not stop the API from serving
    run_pipeline = None  # type: ignore[assignment]
    run_product_channel_scan = None  # type: ignore[assignment]
    techsupport_preflight = None  # type: ignore[assignment]
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

_run_lock = threading.Lock()

# The most recent run_once() outcome, read by GET /health/techsupport_pipeline (via
# last_run_status(), see preflight_health_payload()) so the probe can distinguish
# "misconfigured" from "configured but idle". A separate, tiny lock from _run_lock:
# _run_lock is held for the whole run (minutes -- Slack + LLM calls), and a probe must
# never block behind that, it only needs whichever outcome was recorded most recently.
# `code` is None or techsupport_preflight.FAILURE_CODE ONLY -- never the raw exception
# text -- so this record stays safe to serve verbatim from an unauthenticated endpoint.
_last_run_lock = threading.Lock()
_last_run_outcome: Dict[str, Any] = {"failed": None, "code": None}

# Signature of the BLOCKING preflight problems blamed for the most recent run_once()
# failure. Read and written ONLY from inside run_once()'s _run_lock-protected section
# (see _log_run_failure below), so it needs no lock of its own -- that section is
# already exclusive. None means "no standing config failure recorded", so the very next
# failure (of any kind) always logs a full report. See _log_run_failure for why this
# exists: it is the hourly-tick log-spam guard for pr42-44j.5.
_last_config_failure_signature: Optional[str] = None


def is_available() -> bool:
    """False when the cron package is absent or failed to import."""
    return run_pipeline is not None


def is_running() -> bool:
    """True while either the nightly pipeline or a manual product-channel scan
    is in flight -- they share one lock."""
    return _run_lock.locked()


def run_once() -> Dict[str, Any]:
    """Run the pipeline unless one is already in flight.

    Never raises: both callers (a FastAPI BackgroundTask and the tick) would
    otherwise drop the exception. run_pipeline() already posts per-step
    failures to the admin Slack channel; this is the last-resort net.
    """
    global _last_config_failure_signature  # pylint: disable=global-statement

    if run_pipeline is None:
        utils.logger.error("Nightly techsupport pipeline unavailable: %s", IMPORT_ERROR)
        return {"ran": False, "reason": "unavailable", "error": IMPORT_ERROR}

    if not _run_lock.acquire(blocking=False):
        utils.logger.info("Nightly techsupport pipeline already running -- ignoring this trigger")
        return {"ran": False, "reason": "already_running"}

    try:
        utils.logger.info("Nightly techsupport pipeline starting (cron root: %s)", CRON_ROOT)
        summary = run_pipeline()
        utils.logger.info("Nightly techsupport pipeline finished: %s", summary)
        _record_last_run(failed=False)
        # A successful run proves config is fine right now, so the NEXT failure (even an
        # identical-looking one) is a fresh occurrence, not a continuation of a previously
        # reported standing problem -- it deserves the full report again, not a one-liner.
        _last_config_failure_signature = None
        return {"ran": True, "summary": summary}
    except Exception as exc:  # noqa: BLE001
        failure_code = _log_run_failure(exc)
        _record_last_run(failed=True, code=failure_code)
        return {"ran": False, "error": str(exc)}
    finally:
        _run_lock.release()


def _record_last_run(*, failed: bool, code: Optional[str] = None) -> None:
    """Thread-safe write of the most recent run_once() outcome (see
    _last_run_outcome above for why this has its own lock, separate from _run_lock)."""
    with _last_run_lock:
        _last_run_outcome["failed"] = failed
        _last_run_outcome["code"] = code


def last_run_status() -> Dict[str, Any]:
    """Snapshot of the most recent run_once() outcome, for the health probe
    (preflight_health_payload()). `failed` is None until the first run_once()
    in this process completes -- i.e. "no run yet", distinct from "ran and ok"."""
    with _last_run_lock:
        return dict(_last_run_outcome)


def _log_run_failure(exc: Exception) -> Optional[str]:
    """Logs a failed run_once() tick and returns the failure code to record via
    _record_last_run (None, or techsupport_preflight.FAILURE_CODE).

    This is the RUNTIME NET for pr42-44j.5, not the first line of defence --
    that is run_startup_preflight() below, called once at boot before the
    scheduler starts. A tick can still fail here because configuration
    regressed AFTER boot: the common case is a rotated Secret mounted via
    subPath, which Kubernetes does not re-deliver into a running pod's
    filesystem without a restart. Re-running the SAME preflight validator here
    (cheap: stdlib + dotenv only, no network) lets a post-boot config
    regression get the identical prescriptive remediation text and
    TECHSUPPORT_CONFIG_MISSING code, instead of a bare traceback that only
    names whatever call happened to raise first (e.g. a cryptic Slack SDK auth
    error that doesn't say *why* the token is missing).

    Log-spam guard: the tick fires hourly (DEFAULT_TICK_MINUTES), so a standing
    config problem must not re-log the full multi-line report every single
    time forever. The mechanism is a signature of exactly WHICH BLOCKING
    problems are present, kept only as a module global for the lifetime of
    this process (reset on any success or any non-config failure, see
    _last_config_failure_signature above) -- the full report is logged once
    when a given set of problems first appears (or changes), and every repeat
    tick with the SAME set gets a one-line reminder that still names the code
    (so a log-based alert keeps firing on every occurrence) without filling
    the logs with an identical multi-line page every hour.
    """
    global _last_config_failure_signature  # pylint: disable=global-statement

    result = None
    if techsupport_preflight is not None:
        try:
            result = techsupport_preflight.run_preflight()
        except Exception:  # noqa: BLE001 -- the preflight check itself must never mask the real failure below
            result = None

    if result is not None and result.blocking:
        # Signature of WHICH problems, not WHEN -- a repeat tick with the identical
        # set of BLOCKING problems is a "repeat", a different set is a "new" failure.
        signature = repr(
            tuple(sorted((problem.severity.value, problem.keys, problem.problem) for problem in result.blocking))
        )
        code = techsupport_preflight.FAILURE_CODE
        if signature != _last_config_failure_signature:
            _last_config_failure_signature = signature
            # startup=False: this pod is already serving, so the report must not
            # close with "Refusing to start" -- see format_preflight_report().
            report = techsupport_preflight.format_preflight_report(
                result, strict=techsupport_preflight.is_strict_mode(), startup=False
            )
            utils.logger.critical(
                "Nightly techsupport pipeline tick failed because of a configuration problem (possibly "
                "introduced after startup, e.g. a rotated Secret that needs a pod restart to be re-mounted "
                "under a subPath volume):\n%s",
                report,
            )
        else:
            utils.logger.critical(
                "%s: nightly techsupport pipeline tick failed again with the same configuration problem "
                "already reported in full above. See the previous CRITICAL log for the full report, or "
                "GET /health/techsupport_pipeline for the current state. (%s)",
                code,
                exc,
            )
        return code

    # Not a (currently detectable) configuration problem -- or the cron package / the
    # preflight module itself is unavailable -- so there is no more specific guidance to
    # give than the original behaviour: log the full traceback. Also clears the dedup
    # signature, so a LATER config failure (a new, different problem) logs in full again
    # rather than being mistaken for a continuation of a since-resolved one.
    _last_config_failure_signature = None
    utils.logger.critical("Nightly techsupport pipeline failed: %s", exc, exc_info=True)
    return None


def run_startup_preflight() -> None:
    """Called once, synchronously, from main.lifespan immediately BEFORE
    start_scheduler() -- the first and primary line of defence against the
    staging incident described in cron/techsupport_preflight.py's module
    docstring (missing Slack config left the pipeline 100% dead for hours
    while Kubernetes reported a healthy rollout).

    Decision logic lives here, in techsupport_cron.py, rather than in
    src/main.py, for two reasons: (1) this module already owns CRON_ROOT /
    sys.path setup for the cron package techsupport_preflight lives in, and
    (2) keeping it here keeps the logic unit-testable without importing
    src.main, which tests/test_techsupport_cron_product_scan.py's docstring
    explains is deliberately never imported in tests (it pulls in the whole
    server). The call site in main.lifespan is therefore just this one line.

    Raises RuntimeError -- which main.lifespan lets propagate, so uvicorn
    aborts startup and the pod exits non-zero, failing `kubectl rollout
    status` -- only when there is at least one BLOCKING problem AND strict
    mode (TECHSUPPORT_PREFLIGHT_STRICT, default true) is on. Every other
    outcome logs and returns normally, so the server keeps serving:
      * BLOCKING problems with strict mode OFF: an operator explicitly set
        TECHSUPPORT_PREFLIGHT_STRICT=false to start anyway (the documented
        dev-only escape hatch) -- logged at ERROR, not WARNING, because the
        pipeline really is broken; it is just not fatal by operator choice,
        and ERROR keeps that visually distinct from a DEGRADED-only report.
      * DEGRADED-only problems: the techsupport channel can still be ingested
        and served (a product-channel scan is unconfigured, or a setting is
        using its working default), so this never blocks startup regardless
        of strict mode -- logged at WARNING.
      * no problems: a single INFO line confirming preflight passed.
      * the cron package unavailable (IMPORT_ERROR set, techsupport_preflight
        is None): see the comment below -- deliberately NOT a boot failure.
    """
    if techsupport_preflight is None:
        # The cron package (and therefore techsupport_preflight, which lives inside it)
        # failed to import entirely. That is a BROKEN IMAGE, not a missing Secret: the
        # fix is "rebuild/redeploy the image", not "mount a different config file" --
        # refusing to boot here would not even point the operator at the right
        # remediation. start_scheduler(), called immediately after this function returns
        # in main.lifespan, already logs this exact condition at ERROR and disables the
        # scheduler, so this is intentionally quiet (DEBUG only) to avoid duplicating
        # that message a moment before it is logged anyway.
        utils.logger.debug("Skipping techsupport preflight: cron package unavailable (%s)", IMPORT_ERROR)
        return

    result = techsupport_preflight.run_preflight()
    if not result.problems:
        utils.logger.info("techsupport pipeline preflight passed: no configuration problems found.")
        return

    strict = techsupport_preflight.is_strict_mode()
    report = techsupport_preflight.format_preflight_report(result, strict=strict)

    if not result.blocking:
        # DEGRADED only -- ingestion and retrieval still work, so this never blocks
        # startup regardless of strict mode (see Severity.DEGRADED's docstring).
        utils.logger.warning(report)
        return

    if strict:
        utils.logger.critical(report)
        raise RuntimeError(
            f"{techsupport_preflight.FAILURE_CODE}: refusing to start -- see the preceding CRITICAL log for "
            f"the full preflight report. Set {techsupport_preflight.STRICT_ENV_KEY}=false to start anyway "
            "(dev only -- the pipeline will stay broken)."
        )

    # strict mode is off: an operator deliberately chose to start a pipeline known to be
    # broken (the dev-only escape hatch). ERROR, not WARNING -- see the docstring above.
    utils.logger.error(report)


def preflight_health_payload() -> Dict[str, Any]:
    """Builds the response body for GET /health/techsupport_pipeline (see
    src/main.py) -- provably secret-free, because every field here is one of: a
    stable failure/degraded CODE constant, a Severity name, a config KEY NAME,
    a filesystem PATH, a count, or a boolean, and NEVER a value.
    cron/techsupport_preflight.py's PreflightProblem guarantees that by
    construction (its module docstring: "never echoes a value"), so its
    dataclasses are serialised directly below rather than re-deriving the same
    guarantee here.

    Runs run_preflight() fresh on every call rather than caching the startup
    result: the check is cheap (stdlib + dotenv only -- no network, no
    LanceDB), and re-running it is exactly what lets this probe catch a
    post-boot config regression (e.g. a rotated Secret) that
    run_startup_preflight() could not have seen at boot time.

    "ok" (and therefore the HTTP 503-vs-200 split in main.py) is False both
    when there is a BLOCKING problem AND when the cron package itself failed
    to import -- the latter has no PreflightResult to report, but a pod that
    cannot run the pipeline AT ALL is at least as broken as one blocked by
    missing config, and this probe exists precisely to catch states like it.
    """
    last_run = last_run_status()
    payload: Dict[str, Any] = {
        "cron_available": is_available(),
        "running": is_running(),
        "last_run_failed": last_run["failed"],
        "last_run_failure_code": last_run["code"],
    }

    if techsupport_preflight is None:
        payload.update(
            {
                "ok": False,
                # Same keys as the normal path below, so a probe consumer parsing this
                # body never has to special-case the broken-image branch. "strict" is
                # None rather than a bool here because is_strict_mode() lives in the
                # module that failed to import, so the answer is genuinely unknown.
                "ok_reason": "cron package unavailable",
                "strict": None,
                "code": None,
                "blocking_count": 0,
                "degraded_count": 0,
                "problems": [],
                "sources": [],
            }
        )
        return payload

    result = techsupport_preflight.run_preflight()
    blocking = result.blocking
    degraded = result.degraded
    if blocking:
        code = techsupport_preflight.FAILURE_CODE
    elif degraded:
        code = techsupport_preflight.DEGRADED_CODE
    else:
        code = None

    payload.update(
        {
            "ok": result.ok,
            "strict": techsupport_preflight.is_strict_mode(),
            "code": code,
            "blocking_count": len(blocking),
            "degraded_count": len(degraded),
            "problems": [
                {
                    "severity": problem.severity.value,
                    "keys": list(problem.keys),
                    "problem": problem.problem,
                    "remediation": problem.remediation,
                    "path": problem.path,
                }
                for problem in result.problems
            ],
            "sources": [str(source) for source in result.sources],
        }
    )
    return payload


def health_status_code(payload: Dict[str, Any]) -> int:
    """Maps a preflight_health_payload() body to the HTTP status
    GET /health/techsupport_pipeline should answer with.

    Lives here rather than inline in src/main.py's handler so it is testable:
    the test suite deliberately never imports src.main (that would pull in the
    whole server), and this rule has a non-obvious exception worth pinning
    down. See the handler's docstring for the full reasoning; in short, a
    BLOCKING problem means 503 so a readinessProbe catches a post-boot config
    regression, EXCEPT when TECHSUPPORT_PREFLIGHT_STRICT=false, where 503
    would restart-loop a pod the operator deliberately brought up degraded.
    """
    # "strict" is None when the cron package failed to import -- strictness is
    # then unknowable and the pod is broken regardless -- so this tests for
    # False explicitly instead of relying on truthiness.
    opted_out = payload["strict"] is False
    return 200 if (payload["ok"] or opted_out) else 503


def run_product_channel_scan_once(force: bool = False) -> Dict[str, Any]:
    """Run just the product-channel expert-correction pass, unless a run is
    already in flight.

    The companion to run_once() for POST /run_product_channel_scan/: same
    lock, same "drop rather than queue" behaviour, same never-raises contract
    (the caller is a FastAPI BackgroundTask, which would swallow the
    exception). `force` bypasses the per-channel interval gate.
    """
    if run_product_channel_scan is None:
        utils.logger.error("Product-channel scan unavailable: %s", IMPORT_ERROR)
        return {"ran": False, "reason": "unavailable", "error": IMPORT_ERROR}

    if not _run_lock.acquire(blocking=False):
        utils.logger.info("Techsupport pipeline already running -- ignoring this product-channel scan trigger")
        return {"ran": False, "reason": "already_running"}

    try:
        utils.logger.info("Product-channel expert-correction scan starting (force=%s)", force)
        summary = run_product_channel_scan(force=force)
        utils.logger.info("Product-channel expert-correction scan finished: %s", summary)
        return {"ran": True, "summary": summary}
    except Exception as exc:  # noqa: BLE001
        utils.logger.critical("Product-channel expert-correction scan failed: %s", exc, exc_info=True)
        return {"ran": False, "error": str(exc)}
    finally:
        _run_lock.release()


def tick_minutes() -> int:
    raw = (utils.LILLISA_SERVER_ENV_DICT.get("TECHSUPPORT_PIPELINE_TICK_MINUTES") or "").strip()
    if not raw:
        return DEFAULT_TICK_MINUTES
    try:
        minutes = int(raw)
    except ValueError:
        utils.logger.warning(
            "TECHSUPPORT_PIPELINE_TICK_MINUTES=%r is not an integer -- using %s", raw, DEFAULT_TICK_MINUTES
        )
        return DEFAULT_TICK_MINUTES
    if minutes <= 0:
        utils.logger.warning(
            "TECHSUPPORT_PIPELINE_TICK_MINUTES=%s must be positive -- using %s", minutes, DEFAULT_TICK_MINUTES
        )
        return DEFAULT_TICK_MINUTES
    return minutes


async def _scheduler_loop() -> None:
    """Sleep-then-run, so startup (which may rebuild indices) stays clean."""
    interval_seconds = tick_minutes() * 60
    while True:
        await asyncio.sleep(interval_seconds)
        # run_pipeline() is blocking (Slack + LLM calls), so keep it off the event loop.
        await asyncio.to_thread(run_once)


def start_scheduler() -> Optional[asyncio.Task]:
    """Start the periodic tick. This process is the only scheduler."""
    if not is_available():
        utils.logger.error(
            "Nightly techsupport pipeline will not run -- the cron package is unavailable: %s", IMPORT_ERROR
        )
        return None
    utils.logger.info("Nightly techsupport pipeline tick every %s minute(s)", tick_minutes())
    return asyncio.create_task(_scheduler_loop())


async def stop_scheduler(task: Optional[asyncio.Task]) -> None:
    if task is None:
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
