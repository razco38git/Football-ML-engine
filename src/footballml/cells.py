"""Reading one cell out of a pandas object, with a type a checker can use.

`pandas-stubs` types a single cell -- `series.at[i]`, `frame.at[i, col]`, an
attribute off an `itertuples` row -- as `Scalar`: a thirteen-member union of
everything a cell could conceivably hold, `str` and `bytes` and `date` and
`timedelta` and the numpy scalar types among them. That is honest about pandas
and useless at a call site, because `float()` and `int()` do not accept `date`
or `timedelta`, so `float(row.GoalsFor)` is an error even where the column is a
float column and has been for its whole life.

These three say "this cell holds a number, or a date" once, rather than a
`cast` at each of the dozen-odd places that read one. At runtime they *are* the
builtins -- `as_float(value)` is `float(value)` -- so nothing is skipped,
guessed or coerced differently; the only thing that changes is that mypy stops
objecting to the column's dtype being unprovable from the stubs.

Reach for them on a cell read and nowhere else. Where the type is actually
known, plain `float(x)` says more about the value than `as_float(x)` does.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def as_float(value: Any) -> float:
    """``float(value)``, for a cell the stubs type as any scalar at all."""
    return float(value)


def as_int(value: Any) -> int:
    """``int(value)``, for a cell the stubs type as any scalar at all."""
    return int(value)


def as_timestamp(value: Any) -> pd.Timestamp:
    """``pd.Timestamp(value)``, for a cell the stubs type as any scalar at all."""
    return pd.Timestamp(value)
