"""Publishing a file in one step, or not at all.

The property every caller depends on is the failure path: if a write dies
part-way, the previous file must still be there and still be whole. A CSV cut
short still parses, just with fewer rows, and nothing downstream can tell that
from a complete one -- a timeout once left `team_strength.csv` holding an
experiment's numbers, 1,160 of 1,170 rows wrong, while the API served them.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from footballml import atomic


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


# --- the happy path -----------------------------------------------------------


def test_a_csv_is_written_and_readable(tmp_path, frame) -> None:
    path = tmp_path / "out.csv"
    atomic.write_csv(frame, path)
    pd.testing.assert_frame_equal(pd.read_csv(path), frame)


def test_the_csv_matches_what_to_csv_would_have_written(tmp_path, frame) -> None:
    """Format is preserved: this replaces the write, not the serialisation."""
    path = tmp_path / "out.csv"
    atomic.write_csv(frame, path)
    assert path.read_bytes() == frame.to_csv(index=False).encode()


def test_json_round_trips(tmp_path) -> None:
    path = tmp_path / "out.json"
    atomic.write_json({"offsets": {"E0": 0.1}, "model_version": "v1"}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["model_version"] == "v1"


def test_no_temporary_file_is_left_behind(tmp_path, frame) -> None:
    atomic.write_csv(frame, tmp_path / "out.csv")
    assert [p.name for p in tmp_path.iterdir()] == ["out.csv"]


def test_a_missing_parent_directory_is_created(tmp_path, frame) -> None:
    path = tmp_path / "nested" / "deep" / "out.csv"
    atomic.write_csv(frame, path)
    assert path.exists()


# --- the failure path, which is the point -------------------------------------


def test_a_failed_write_leaves_the_previous_file_whole(tmp_path, frame) -> None:
    path = tmp_path / "out.csv"
    atomic.write_csv(frame, path)
    before = path.read_bytes()

    with pytest.raises(RuntimeError), atomic.atomic_path(path) as tmp:
        tmp.write_bytes(b"half a f")
        raise RuntimeError("killed mid-write")

    assert path.read_bytes() == before, "the previous file was damaged"


def test_a_failed_write_removes_its_temporary_file(tmp_path, frame) -> None:
    path = tmp_path / "out.csv"
    atomic.write_csv(frame, path)
    with pytest.raises(RuntimeError), atomic.atomic_path(path) as tmp:
        tmp.write_bytes(b"partial")
        raise RuntimeError("boom")
    assert [p.name for p in tmp_path.iterdir()] == ["out.csv"]


def test_a_failed_first_write_creates_no_target(tmp_path) -> None:
    path = tmp_path / "out.csv"
    with pytest.raises(RuntimeError), atomic.atomic_path(path) as tmp:
        tmp.write_bytes(b"partial")
        raise RuntimeError("boom")
    assert not path.exists(), "a half-written file was published"


def test_writing_nothing_is_an_error_not_an_empty_file(tmp_path, frame) -> None:
    """A block that silently produced no file must not truncate the target."""
    path = tmp_path / "out.csv"
    atomic.write_csv(frame, path)
    before = path.read_bytes()
    with pytest.raises(FileNotFoundError), atomic.atomic_path(path):
        pass
    assert path.read_bytes() == before


def test_a_serialisation_failure_leaves_the_target_intact(tmp_path, frame) -> None:
    """`to_csv` raising part-way is the realistic version of this."""
    path = tmp_path / "out.csv"
    atomic.write_csv(frame, path)
    before = path.read_bytes()

    class Explodes:
        def __str__(self) -> str:
            raise ValueError("cannot serialise")

        __repr__ = __str__

    bad = pd.DataFrame({"a": [Explodes()]})
    with pytest.raises(ValueError):
        atomic.write_csv(bad, path)
    assert path.read_bytes() == before


# --- concurrency --------------------------------------------------------------


def test_two_writers_do_not_share_a_temporary_file(tmp_path, frame) -> None:
    """Two jobs publishing the same path must not scribble on each other's
    half-written copy. They still race for the replace, which the refresh lock
    is for, but a lost write is a different failure from a torn one."""
    path = tmp_path / "out.csv"
    with atomic.atomic_path(path) as first:
        first.write_bytes(b"a")
        with atomic.atomic_path(path) as second:
            second.write_bytes(b"b")
            assert first != second, "two writers shared one temporary file"


def test_a_reader_never_sees_a_partial_file(tmp_path, frame) -> None:
    """Whatever is at the path parses, at every point during a write."""
    path = tmp_path / "out.csv"
    atomic.write_csv(frame, path)
    with atomic.atomic_path(path) as tmp:
        tmp.write_bytes(b"Div,Date\nincomplete")
        # mid-write: the target is still the old, complete file
        pd.testing.assert_frame_equal(pd.read_csv(path), frame)


# --- skipping an unnecessary rewrite ------------------------------------------


def test_unchanged_is_true_for_an_identical_frame(tmp_path, frame) -> None:
    path = tmp_path / "out.csv"
    atomic.write_csv(frame, path)
    assert atomic.unchanged(frame, path)


def test_unchanged_is_false_when_a_value_moved(tmp_path, frame) -> None:
    path = tmp_path / "out.csv"
    atomic.write_csv(frame, path)
    assert not atomic.unchanged(frame.assign(a=[1, 2, 4]), path)


def test_unchanged_is_false_when_the_target_is_absent(tmp_path, frame) -> None:
    """Absent means write it, which is the safe direction."""
    assert not atomic.unchanged(frame, tmp_path / "missing.csv")
