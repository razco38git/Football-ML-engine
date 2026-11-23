"""The backup must carry data, never damage, and must be provably complete.

Understat's finished seasons exist on this machine only as soccerdata's cache,
and nothing re-fetches them. A backup that quietly copied a maintenance page
alongside the real files would be worse than none, because it would be trusted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.backup_understat import backup

HOLDING_PAGE = b"<!DOCTYPE html><html><body>We will come back soon.</body></html>"


def league(*ids: str) -> bytes:
    return json.dumps({"dates": [], "players": [{"id": i} for i in ids], "teams": {}}).encode()


@pytest.fixture
def source(tmp_path: Path) -> Path:
    src = tmp_path / "cache"
    src.mkdir()
    (src / "league_1_season_2025.json").write_bytes(league("1", "2"))
    (src / "league_2_season_2025.json").write_bytes(league("3"))
    (src / "league_3_season_2026.json").write_bytes(HOLDING_PAGE)
    (src / "match_31948.json").write_bytes(b'{"shots": []}')
    return src


def test_good_files_are_copied_byte_for_byte(source: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out"
    manifest = backup(source, dest)

    copied = {f["name"] for f in manifest["files"]}
    assert copied == {"league_1_season_2025.json", "league_2_season_2025.json", "match_31948.json"}
    for name in copied:
        assert (dest / name).read_bytes() == (source / name).read_bytes()


def test_a_maintenance_page_is_refused_and_named(source: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out"
    manifest = backup(source, dest)

    assert not (dest / "league_3_season_2026.json").exists()
    assert [r["name"] for r in manifest["refused"]] == ["league_3_season_2026.json"]
    assert "not JSON" in manifest["refused"][0]["reason"]


def test_the_manifest_hashes_match_what_was_written(source: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out"
    backup(source, dest)

    on_disk = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    for entry in on_disk["files"]:
        digest = hashlib.sha256((dest / entry["name"]).read_bytes()).hexdigest()
        assert digest == entry["sha256"]
    assert on_disk["total_bytes"] == sum(f["bytes"] for f in on_disk["files"])


def test_the_source_is_never_modified(source: Path, tmp_path: Path) -> None:
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    backup(source, tmp_path / "out")
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_two_backups_are_never_mixed(source: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out"
    backup(source, dest)
    with pytest.raises(FileExistsError):
        backup(source, dest)
