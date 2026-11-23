"""Copy Understat's stored payloads somewhere a cache wipe cannot reach.

Understat has been down since 2 November 2026 with no recovery date. Its
finished seasons, 2014/15 to 2025/26, now exist on this machine **only** as
soccerdata's cache, `~/soccerdata/data/Understat`. Nothing re-fetches a finished
season -- which is what keeps them safe, and also what makes them
irreplaceable: they are the training history behind every player rating, and
while the site is down a lost or "cleared" cache cannot be rebuilt by anyone.

This is the second line of defence. `footballml.ingest.understat_cache` stops a
maintenance page being written over a good payload; this keeps a copy for the
day something else goes wrong.

Only files that are *data* are copied. A file that parses as a maintenance page
is reported and left out, so a backup can never carry the damage forward -- and
the manifest says which files were refused and why.

Run with::

    python -m pipelines.backup_understat
    python -m pipelines.backup_understat --dest D:/backups/understat

To restore, copy the files back into the cache directory. A directory without
a `manifest.json` is an incomplete backup: the manifest is written last.

Writes nothing under `data/processed`, and never modifies the source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.ingest.understat_cache import (  # noqa: E402
    UnusablePayloadError,
    check_payload,
    understat_reader,
)

ROOT = Path(__file__).resolve().parents[1]
BACKUPS = ROOT / "data" / "backups"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backup(source: Path, dest: Path) -> dict[str, Any]:
    """Copy every valid ``*.json`` in ``source`` to ``dest`` and describe it.

    Each copy is read back and hashed against the source before it counts, so a
    short write cannot pass for a backup. Returns the manifest, which is also
    written to ``dest/manifest.json`` -- last, so its presence means the copy
    finished.
    """
    source, dest = Path(source), Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} already holds files; refusing to mix two backups")
    dest.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, Any]] = []
    refused: list[dict[str, str]] = []
    for path in sorted(source.glob("*.json")):
        payload = path.read_bytes()
        try:
            check_payload(path, payload)
        except UnusablePayloadError as exc:
            refused.append({"name": path.name, "reason": str(exc)})
            continue
        target = dest / path.name
        shutil.copy2(path, target)
        digest = _sha256(path)
        if _sha256(target) != digest:
            raise OSError(f"copy of {path.name} does not match its source")
        copied.append(
            {
                "name": path.name,
                "bytes": len(payload),
                "sha256": digest,
                "source_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            }
        )

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source": str(source),
        "files": copied,
        "refused": refused,
        "total_bytes": sum(f["bytes"] for f in copied),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Understat cache directory (soccerdata's).")
    parser.add_argument("--dest", type=Path, help="Where to put the copy.")
    args = parser.parse_args()

    source = args.source or Path(understat_reader(["ENG-Premier League"], None).data_dir)
    dest = args.dest or BACKUPS / f"understat-{datetime.now(UTC):%Y%m%d-%H%M}"

    manifest = backup(source, dest)
    print(f"Copied {len(manifest['files'])} file(s), {manifest['total_bytes']:,} bytes -> {dest}")
    for item in manifest["refused"]:
        print(f"  NOT copied: {item['name']} -- {item['reason']}")
    return 0 if manifest["files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
