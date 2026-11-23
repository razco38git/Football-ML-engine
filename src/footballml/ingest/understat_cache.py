"""Keep Understat's stored payloads from being overwritten with a maintenance page.

On 2026-11-02 understat.com began answering every request with HTTP 200 and a
26,719-byte page whose visible text is "We will come back soon". soccerdata's
`Understat._request_api` writes whatever body arrives to the cache path *before*
anyone parses it, so a failed run did not merely fail -- it replaced the stored
data with that page. `league_3_season_2026.json`, the Bundesliga payload the
shipped `player_ratings.csv` was built from, was last rewritten that way at
10:21 on 2026-11-23, and it was the only raw copy.

Two properties of that write are worth naming, because the fix has to answer
both:

- **It trusts the body.** A 200 is not data. Nothing between the socket and
  the cache file asks whether the bytes parse.
- **It truncates first.** ``open(path, "wb")`` empties the file before the
  first byte is written, so a run killed part-way through a *good* download
  damages the stored copy as surely as a holding page does.

The guard is therefore on the *write*, and it is two things: validate the body
before it is stored, and store it with `footballml.atomic` so a reader sees the
old file or the new one and never half of either. A body that fails validation
raises :class:`UnusablePayloadError` and the stored file is left exactly as it
was. Reads are not touched at all, which is why finished seasons -- served from
cache, never re-fetched -- were never at risk from this and are not slowed by
it.

Everything that talks to Understat must build its reader here. A test enforces
that, because the one call site that forgets is the one that overwrites a good
payload on the next outage.
"""

from __future__ import annotations

import functools
import io
import json
from pathlib import Path
from typing import IO, Any

from footballml.atomic import write_bytes

#: The league-season files, which are the ones the ratings are built from.
LEAGUE_PAYLOADS = "league_*_season_*.json"

#: What soccerdata indexes straight after loading a league-season file
#: (`_read_league_season`). A file without them is not a league-season payload
#: however cleanly it parses: a rate-limit reply such as ``{"error": ...}`` is
#: valid JSON and would otherwise be stored in place of the data.
_LEAGUE_KEYS = ("dates", "players", "teams")


class UnusablePayloadError(RuntimeError):
    """Understat answered, but not with the data that was asked for."""


def _is_league_file(path: Path) -> bool:
    return path.name.startswith("league_") and "_season_" in path.name


def check_payload(path: Path, payload: bytes) -> None:
    """Raise :class:`UnusablePayloadError` unless ``payload`` may be stored at ``path``.

    Only ``.json`` files are judged: a cache path with any other suffix is not
    one this guards, and refusing it would be a different bug.
    """
    if path.suffix != ".json":
        return
    head = payload[:60].decode("utf-8", "replace").replace("\n", " ").strip()
    try:
        data = json.loads(payload)
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError are both ValueErrors
        raise UnusablePayloadError(
            f"{len(payload):,} bytes that are not JSON, starting {head!r}"
        ) from exc
    if _is_league_file(path) and (
        not isinstance(data, dict) or any(key not in data for key in _LEAGUE_KEYS)
    ):
        raise UnusablePayloadError(
            f"JSON that is not a league-season payload (expected {', '.join(_LEAGUE_KEYS)}), "
            f"starting {head!r}"
        )


def unusable_payloads(data_dir: Path, pattern: str = "*.json") -> list[Path]:
    """Files in the cache that are not data -- for a check, or a restore."""
    bad: list[Path] = []
    for path in sorted(Path(data_dir).glob(pattern)):
        try:
            check_payload(path, path.read_bytes())
        except UnusablePayloadError:
            bad.append(path)
    return bad


@functools.cache
def _guarded_class() -> type:
    import soccerdata as sd  # imported lazily: pulls a heavy browser stack

    # Named `Understat` on purpose. soccerdata resolves the league ids a reader
    # accepts through `cls.__name__`, so a subclass called anything else finds
    # no leagues at all and fails with an error that points nowhere near here.
    class Understat(sd.Understat):
        def _request_api(
            self, url: str, filepath: Path | None = None, no_cache: bool = False
        ) -> IO[bytes]:
            # soccerdata's own condition for serving the stored file. Repeated
            # rather than delegated because the fetch below has to be told there
            # is no file to write, and that would also tell it there is none to
            # read -- a finished season would then go to the network. The test
            # that a cached file never touches the session is what notices if
            # soccerdata ever changes this predicate.
            cached = (
                filepath is not None and filepath.exists() and not no_cache and not self.no_cache
            )
            if cached:
                return super()._request_api(url, filepath, no_cache)

            payload = super()._request_api(url, None, no_cache).read()
            if filepath is not None and not self.no_store:
                try:
                    check_payload(filepath, payload)
                except UnusablePayloadError as exc:
                    kept = (
                        "the stored copy was left untouched"
                        if filepath.exists()
                        else "nothing was stored"
                    )
                    raise UnusablePayloadError(
                        f"Understat answered {url} with {exc}; {kept} ({filepath.name})."
                    ) from exc
                write_bytes(payload, filepath)
            return io.BytesIO(payload)

    return Understat


def understat_reader(leagues: list[str], seasons: list[str] | None, **kwargs: Any) -> Any:
    """A ``soccerdata.Understat`` that will not store a page that is not data.

    Same arguments, same methods, same cache directory: the only difference is
    what happens to a response on its way to disk.
    """
    return _guarded_class()(leagues=leagues, seasons=seasons, **kwargs)
