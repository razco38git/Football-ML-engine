"""Whether the scrapers drive Chrome visibly.

soccerdata reaches FBref through SeleniumBase with ``uc=True``, and its reader
defaults to ``headless=False`` -- so every scrape opened a Chrome window and
stole focus, several times per run, on a machine somebody is using.

**The cost of silence, stated plainly.** soccerdata only attempts its captcha
and Cloudflare handling when the browser is visible::

    def solve_captcha(self) -> None:
        if self.headless:
            logger.warning("Cannot use GUI captcha solver in headless mode.")
            self._driver.sleep(10)
            return

FBref sits behind Cloudflare, so running headless gives that up. The failure
mode is staleness rather than bad data: a failed scrape falls back to the
cached copy, a match that cannot be merged is parked in `fbref_deferred.csv`
for a later run, and every European step is optional in the weekly. Nothing
half-written reaches the model.

So the default is silence and the escape hatch is one environment variable::

    FOOTBALLML_BROWSER_HEADLESS=0   # windows come back, captcha handling back

Read per call rather than captured at import, so a single run can be changed
without restarting anything that imported this first.
"""

from __future__ import annotations

import os

ENV_VAR = "FOOTBALLML_BROWSER_HEADLESS"

#: Silent unless asked otherwise. The windows were the reported problem.
DEFAULT = True

#: What counts as "no". Anything else -- including an empty string, which is
#: what `set VAR=` leaves behind on Windows -- keeps the default.
_FALSEY = frozenset({"0", "false", "no", "off"})


def headless() -> bool:
    """Whether to drive the browser without a window.

    Returns:
        ``True`` unless :data:`ENV_VAR` is set to one of :data:`_FALSEY`.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return DEFAULT
    return raw.strip().lower() not in _FALSEY
