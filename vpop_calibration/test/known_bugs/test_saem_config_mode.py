"""Regression test documenting the mode-dependent SAEM config defaults bug.

See BUG_FINDINGS.md #7 and IMPLEMENTATION_REVIEW.md #12.

`SaemConfigDict`'s `live_plot`/`logging`/`progress_bars` fields
(vpop_calibration/saem/config.py) are declared as `NamedTuple` class-body
expressions depending on `mode`, e.g. `live_plot: bool = mode in
["notebook", "debug"]`. Those defaults are evaluated ONCE at class-definition
time using the literal default `mode="notebook"`, not per-instance from
whatever `mode` is actually passed to the constructor. So passing a
different `mode` has no effect on these three fields.

This test asserts that "cli" mode disables plotting/progress bars and that
"debug" mode enables logging, as the field names and the `mode` docstring/
intent imply. It currently FAILS because all three modes produce identical
`live_plot=True, logging=False, progress_bars=True`.
"""

from vpop_calibration.saem.config import SaemConfigDict


def test_cli_mode_disables_plotting_and_progress_bars():
    config = SaemConfigDict(mode="cli")
    assert not config.live_plot, (
        "mode='cli' should disable live plotting for a headless run; "
        f"got live_plot={config.live_plot}."
    )
    assert not config.progress_bars, (
        "mode='cli' should disable progress bars for a headless run; "
        f"got progress_bars={config.progress_bars}."
    )


def test_debug_mode_enables_logging():
    config = SaemConfigDict(mode="debug")
    assert config.logging, (
        "mode='debug' should enable logging; got logging=False."
    )
