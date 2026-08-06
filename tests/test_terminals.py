# [claude fixed] new file: guards the spinner shutdown fix (TASK.md Phase 6.1)
import io
import time
from contextlib import redirect_stdout

from mlsampler.terminals import spinning


def run(work=0.0):
    buf = io.StringIO()
    started = time.perf_counter()
    with redirect_stdout(buf):
        with spinning():
            if work:
                time.sleep(work)
    return buf.getvalue(), time.perf_counter() - started


def test_shutdown_is_not_blocked_by_the_frame_interval():
    """The worker thread must wake on `set()` rather than sleeping out its
    0.1s frame interval, which used to add ~100ms to every call."""
    _, elapsed = run()
    assert elapsed < 0.05


def test_frame_interval_is_unchanged():
    """Roughly one frame per 0.1s: the display itself must not change."""
    out, _ = run(work=0.25)
    assert out.count("\rsampling... ") == 3


def test_reports_completion():
    out, _ = run()
    assert out.startswith("\rsampling... ")
    assert out.endswith("\rsampling completed    \n")
