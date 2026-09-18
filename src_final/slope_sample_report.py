"""Compact counts of matched rainfall scenarios per slope group."""

from collections import Counter
import math

import pandas as pd

from config import (
    ACCUMULATED_RAIN_COLUMN,
    RAIN_DURATION_COLUMN,
    RETURN_PERIOD_COLUMN,
)


FOUR_EVENTS = ((-1.0, 0.0), (30.0, 100.0), (200.0, 100.0), (500.0, 100.0))
SEVEN_EVENTS = (
    (-1.0, 0.0), (30.0, 15.0), (30.0, 30.0),
    (200.0, 15.0), (200.0, 30.0), (500.0, 15.0), (500.0, 30.0),
)


def print_slope_sample_breakdown(df, slope_groups, label):
    """Describe sample counts and exact Tr/duration patterns for existing groups."""
    groups = pd.Series(slope_groups).reset_index(drop=True)
    if len(df) != len(groups):
        raise ValueError("The slope groups must contain one ID per dataframe row")

    events = df[[RETURN_PERIOD_COLUMN, RAIN_DURATION_COLUMN]].reset_index(drop=True)
    events = events.apply(pd.to_numeric, errors="coerce")
    rain = pd.to_numeric(df[ACCUMULATED_RAIN_COLUMN], errors="coerce").reset_index(drop=True)
    events.loc[rain.eq(0), [RETURN_PERIOD_COLUMN, RAIN_DURATION_COLUMN]] = (-1, 0)

    counts = Counter()
    for indices in groups.groupby(groups, sort=False, dropna=False).indices.values():
        pattern = tuple(sorted(
            (
                tuple(value if pd.notna(value) else None for value in event)
                for event in events.iloc[indices].itertuples(index=False, name=None)
            ),
            key=lambda event: tuple(
                value if value is not None else math.inf for value in event
            ),
        ))
        counts[(len(indices), pattern)] += 1

    print(f"  {label}:")
    if not counts:
        print("    # slopes with matched samples: 0")
        return
    for (sample_count, pattern), slope_count in sorted(
        counts.items(), key=lambda item: (item[0][0], repr(item[0][1]))
    ):
        if pattern == FOUR_EVENTS:
            description = "Tr -, 30, 200, 500; duration 100 h"
        elif pattern == SEVEN_EVENTS:
            description = "Tr -, 30, 200, 500; duration 15, 30 h"
        else:
            description = "Tr/duration [h]: " + ", ".join(
                "-" if tr == -1 and duration == 0
                else f"{tr:g}/{duration:g}" if tr is not None and duration is not None
                else "?/?"
                for tr, duration in pattern
            )
        print(f"    # slopes with {sample_count} samples: {slope_count} ({description})")
