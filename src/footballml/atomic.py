"""Publish a file in one step, or not at all.

Every derived file here is read by something that cannot tell a truncated copy
from a complete one. A CSV cut off part-way still parses -- it just has fewer
rows -- and a JSON object cut off part-way either fails to parse or, worse,
parses with a key missing. Neither failure announces itself.

It is not hypothetical. A timeout once left `team_strength.csv` holding an
experiment's numbers, 1,160 of 1,170 rows wrong, while the API served them.
`build_players` grew a private `_write_csv` for that; this is the same idea,
shared, because the files it did not cover are written by jobs that can run
within seconds of each other.

The mechanism is the obvious one and the details are what make it work:

- the temporary file sits **in the same directory** as its target, so the
  replace never crosses a filesystem and never falls back to a copy;
- the handle is flushed and `fsync`'d before the replace, so a power loss
  cannot leave the directory entry pointing at unwritten blocks;
- `Path.replace` is atomic on POSIX and on Windows, which is what lets a
  reader see either the old file or the new one and never a half of either;
- a failure **removes the temporary file and leaves the target untouched**,
  which is the property the callers actually depend on.

The temporary name is unique per call, so two writers publishing the same path
at once cannot scribble on each other's half-written copy. They still race for
the final replace -- one wins and the other's content is lost -- but that is
what `pipelines/_lock.py` is for, and a lost write is a different failure from
a torn one.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any
from uuid import uuid4

import pandas as pd


@contextlib.contextmanager
def atomic_path(path: Path, suffix: str = "") -> Iterator[Path]:
    """Yield a temporary path to write, then move it into place.

    The target is replaced only if the block completes. On any exception the
    temporary file is removed and the target is left exactly as it was.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}{suffix}.tmp")
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(f"nothing was written to {tmp}")
        tmp.replace(path)
    finally:
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()


@contextlib.contextmanager
def atomic_open(path: Path, encoding: str = "utf-8") -> Iterator[IO[str]]:
    """`open` for writing, published atomically, flushed and fsync'd."""
    with atomic_path(path) as tmp, tmp.open("w", encoding=encoding, newline="") as handle:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(frame: pd.DataFrame, path: Path, **kwargs: Any) -> Path:
    """`DataFrame.to_csv`, published atomically. ``index=False`` by default."""
    kwargs.setdefault("index", False)
    with atomic_path(path) as tmp:
        frame.to_csv(tmp, **kwargs)
    return Path(path)


def write_json(payload: Any, path: Path, **kwargs: Any) -> Path:
    """`json.dump`, published atomically. ``indent=2`` by default."""
    kwargs.setdefault("indent", 2)
    with atomic_open(path) as handle:
        json.dump(payload, handle, **kwargs)
    return Path(path)


def write_bytes(data: bytes, path: Path) -> Path:
    """Bytes, published atomically."""
    with atomic_path(path) as tmp:
        tmp.write_bytes(data)
    return Path(path)


def unchanged(frame: pd.DataFrame, path: Path, **kwargs: Any) -> bool:
    """True when writing ``frame`` to ``path`` would reproduce it byte for byte.

    Offered because several of these files are rewritten weekly with identical
    contents, and a rewrite is a window in which a reader can be unlucky.
    Callers use it to skip the write, not to decide correctness -- a False from
    a read error means "write it", which is the safe direction.
    """
    path = Path(path)
    if not path.exists():
        return False
    kwargs.setdefault("index", False)
    try:
        return path.read_bytes() == frame.to_csv(**kwargs).encode()
    except (OSError, UnicodeEncodeError):  # pragma: no cover - unreadable target
        return False
