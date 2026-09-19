# -*- coding: utf-8 -*-
"""Match and preprocess D.1, D.2 and UN using one selectable strategy.

Set MODE to one of:
- canonical_freq
- canonical_near
- real_freq
- real_near

Outputs are written to DATASET_DIR / MODE.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict, deque
import argparse
import math
import numbers
import os
import re

import numpy as np
import pandas as pd

from s0_build_combinations import (
    HYDRAULIC,
    get_d2_geometries,
    get_combinations_1a_geometries,
)

from config import BASE_DIR, DATASET_DIR


# ============================================================================
# GLOBAL STRATEGY
# ============================================================================
MODE = "real_near"
VALID_MODES = {"canonical_freq", "canonical_near", "real_freq", "real_near"}

if MODE not in VALID_MODES:
    raise ValueError(f"Invalid MODE={MODE!r}. Expected one of {sorted(VALID_MODES)}")

USE_CANONICAL_MATCHING = MODE.startswith("canonical_")
VALUE_SELECTION = MODE.rsplit("_", 1)[1]  # "freq" or "near"
OUTPUT_DIR = os.path.join(DATASET_DIR, MODE)

# Only the source column names and their order are used; display aliases are not.
COLUMNS = (
    "Unit weight [kN/m3]",
    "Effective cohesion [kPa]",
    "Effective friction angle [°]",
    "Undrained Shear Strength [kPa]",
    "Saturated permeability [m/s]",
    "Soil Type [-]",
    "Slope angle [°]",
    "Slope length [m]",
    "Total length [m]",
    "Slope height [m]",
    "Total height Upstream [m]",
    "Total height Downstream [m]",
    "Soil depth upstream [m]",
    "Soil depth downstream [m]",
    "Bedrock depth upstream [m]",
    "Bedrock depth downstream [m]",
    "Initial piezometric surface depth - upstream [m]",
    "Initial piezometric surface depth - downstream [m]",
    "Return period of precipitation [years]",
    "Accumulated precipitation [mm]",
    "Precipitation duration [hrs]",
    "Factor of safety [-]",
    "Depth of slip surface [m]",
    "Final Piezometric surface depth - upstream [m]",
    "Final Piezometric surface depth - downstream [m]",
)

SOURCE_METADATA_COLUMNS = ["Source file", "Sheet", "Excel row"]

GEOMETRY_COLUMNS = [
    "Slope angle [°]",
    "Slope length [m]",
    "Total length [m]",
    "Slope height [m]",
    "Total height Upstream [m]",
    "Total height Downstream [m]",
    "Soil depth upstream [m]",
    "Soil depth downstream [m]",
    "Bedrock depth upstream [m]",
    "Bedrock depth downstream [m]",
]

GEOMETRY_ROUNDING_ATOL = 0.11

GEOMETRY_MATCH_ATOL = {
    "Slope height [m]": GEOMETRY_ROUNDING_ATOL,
    "Total height Downstream [m]": GEOMETRY_ROUNDING_ATOL,
    "Soil depth upstream [m]": GEOMETRY_ROUNDING_ATOL,
    "Soil depth downstream [m]": GEOMETRY_ROUNDING_ATOL,
    "Bedrock depth upstream [m]": GEOMETRY_ROUNDING_ATOL,
    "Bedrock depth downstream [m]": GEOMETRY_ROUNDING_ATOL,
    "Initial piezometric surface depth - upstream [m]": GEOMETRY_ROUNDING_ATOL,
    "Initial piezometric surface depth - downstream [m]": GEOMETRY_ROUNDING_ATOL,
}

TR = "Return period of precipitation [years]"

RAIN = "Accumulated precipitation [mm]"

DURATION = "Precipitation duration [hrs]"

FOS = "Factor of safety [-]"

SLIP_DEPTH = "Depth of slip surface [m]"

FINAL_WATER_UP = "Final Piezometric surface depth - upstream [m]"

FINAL_WATER_DOWN = "Final Piezometric surface depth - downstream [m]"

EFFECTIVE_COHESION = "Effective cohesion [kPa]"
EFFECTIVE_ANGLE = "Effective friction angle [°]"
UNDRAINED_STRENGTH = "Undrained Shear Strength [kPa]"

TARGET_COLUMNS = (FOS, SLIP_DEPTH, FINAL_WATER_UP, FINAL_WATER_DOWN)

FOS_CORRECTIONS = {
    1725.0: 1.725,
    1634.0: 1.634,
    156.0: 1.560,
    1492.0: 1.492,
    1460.0: 1.460,
}

MISSING_TEXT = {"", "nan", "none", "na", "<na>", "null"}

AUDIT_COLUMNS = [
    "Condition",
    "Source file",
    "CSV row",
    "Column",
    "Action",
    "Old value",
    "New value",
    "Previous CSV row",
    "Previous Tr",
    "Previous duration",
    "Next CSV row",
    "Next Tr",
    "Next duration",
    "Support count",
    "Support CSV rows",
]

def get_condition_from_filename(excel_path):
    """Return the combinations-table condition for a source file."""
    file_name = os.path.basename(excel_path).upper()
    return next((condition for prefix, condition in (("D.1", "D.1"), ("D.2", "D.2"), ("UN", "UN")) if file_name.startswith(prefix)), None)

def get_expected_geometries(excel_path):
    """Return the canonical geometries for one source file."""
    condition = get_condition_from_filename(excel_path)
    if condition == "D.2":
        return get_d2_geometries()
    return get_combinations_1a_geometries(condition) if condition in {"D.1", "UN"} else []

def parse_sheet_geometry_identity(sheet_name):
    """Extract the reliable slope angle and length from a sheet name."""
    slope_match = re.search(r"slope\s*(\d+)", sheet_name, flags=re.I)
    length_match = re.search(r"_l\s*(\d+)", sheet_name, flags=re.I)
    if not slope_match or not length_match:
        return None
    return int(slope_match.group(1)), int(length_match.group(1))

def normalize_sheet_geometry(sheet_data, excel_path, sheet_name):
    """
    Correct source geometry using the sheet identity and combinations table.

    The sheet name supplies slope angle and slope length. Within that
    geometry family, hSu identifies the 90%, 25%, H, or 2 m configuration.
    This copy is used for matching; compatible source precision is retained
    in the output dataset.
    """
    identity = parse_sheet_geometry_identity(sheet_name)
    expected_geometries = get_expected_geometries(excel_path)

    if identity is None or not expected_geometries:
        return sheet_data, {}

    slope, length = identity
    # Restrict the canonical geometries to the slope family identified by the sheet name.
    candidates = [geometry for geometry in expected_geometries if geometry["Slope angle [°]"] == slope and geometry["Slope length [m]"] == length]

    if not candidates:
        raise ValueError(f"{os.path.basename(excel_path)} - sheet '{sheet_name}': no generated geometry for slope={slope}, L={length}.")

    result = sheet_data.copy()
    source_hsu = pd.to_numeric(result["Soil depth upstream [m]"], errors="coerce")
    correction_counts = defaultdict(int)

    for row_idx in result.index[source_hsu.notna()]:
        # hSu uniquely identifies the geometry variant within the current slope family.
        expected = min(candidates, key=lambda geometry: abs(float(source_hsu.at[row_idx]) - float(geometry["Soil depth upstream [m]"])))
        hsu_distance = abs(float(source_hsu.at[row_idx]) - float(expected["Soil depth upstream [m]"]))

        if hsu_distance > GEOMETRY_ROUNDING_ATOL:
            raise ValueError(f"{os.path.basename(excel_path)} - sheet '{sheet_name}', row {row_idx + 2}: hSu={source_hsu.at[row_idx]} is not compatible with any generated geometry.")

        for column in GEOMETRY_COLUMNS:
            old_simulations = result.at[row_idx, column]
            new_simulations = expected[column]
            old_number = pd.to_numeric(pd.Series([old_simulations]), errors="coerce").iloc[0]
            if pd.isna(old_number) or not math.isclose(float(old_number), float(new_simulations), rel_tol=0.0, abs_tol=1e-12):
                correction_counts[column] += 1
            result.at[row_idx, column] = new_simulations

    return result, dict(correction_counts)

def normalize_hydraulic_parameters(sheet_data, excel_path):
    """Correct Soil Type from the unambiguous permeability mapping."""
    condition = get_condition_from_filename(excel_path)
    if condition is None:
        return sheet_data, 0

    result = sheet_data.copy()
    permeability = pd.to_numeric(result["Saturated permeability [m/s]"], errors="coerce")
    soil_type = pd.to_numeric(result["Soil Type [-]"], errors="coerce")
    corrected = 0

    # Soil type is deterministic once ksat and the analysis condition are known.
    for hydraulic_case in HYDRAULIC[condition]:
        expected_ksat = hydraulic_case["Saturated permeability [m/s]"]
        expected_soil_type = hydraulic_case["Soil Type [-]"]
        mask = permeability.notna() & np.isclose(permeability, expected_ksat, rtol=1e-9, atol=0.0)
        corrected += int((mask & (soil_type.isna() | ~np.isclose(soil_type, expected_soil_type, rtol=0.0, atol=0.0))).sum())
        result.loc[mask, "Soil Type [-]"] = expected_soil_type

    return result, corrected

def normalize_structured_water_tables(sheet_data, excel_path):
    """
    Repair high/low water-table blocks when the file has all 3 states.

    D.1 stores 60 rows per state and soil type.  UN stores 48 rows per
    state and geometry.  In complete three-state groups the order is
    always high, low, intermediate.  Only high and low are deterministic;
    intermediate values are intentionally preserved.
    """
    condition = get_condition_from_filename(excel_path)
    result = sheet_data.copy()
    zwu_column = "Initial piezometric surface depth - upstream [m]"
    zwd_column = "Initial piezometric surface depth - downstream [m]"
    corrected = defaultdict(int)

    if condition == "D.1":
        permeability = pd.to_numeric(result["Saturated permeability [m/s]"], errors="coerce")
        groups = [list(indices) for _, indices in result.groupby(permeability, sort=False, dropna=False).groups.items()]
        block_size = 60
    elif condition == "UN":
        soil_depth = pd.to_numeric(result["Soil depth upstream [m]"], errors="coerce")
        groups = [list(indices) for _, indices in result.groupby(soil_depth, sort=False, dropna=False).groups.items()]
        block_size = 48
    else:
        return result, {}

    # Only complete high/low/intermediate blocks are safe to reconstruct.
    for indices in groups:
        if len(indices) != 3 * block_size:
            continue
        # Source ordering is fixed: high first, low second, intermediate third.
        high_indices = indices[:block_size]
        low_indices = indices[block_size:2 * block_size]

        expected_blocks = (
            (high_indices, {zwu_column: pd.Series(0, index=high_indices, dtype=float), zwd_column: pd.Series(0, index=high_indices, dtype=float)}),
            (low_indices, {zwu_column: pd.to_numeric(result.loc[low_indices, "Soil depth upstream [m]"], errors="coerce"), zwd_column: pd.to_numeric(result.loc[low_indices, "Soil depth downstream [m]"], errors="coerce")}),
        )
        for row_indices, expected_values in expected_blocks:
            for column, expected in expected_values.items():
                current = pd.to_numeric(result.loc[row_indices, column], errors="coerce")
                changed = current.isna() | ~np.isclose(current, expected, rtol=0.0, atol=1e-12)
                corrected[column] += int(changed.sum())
                result.loc[row_indices, column] = expected.to_numpy()

    return result, dict(corrected)

def align_d1_rainfall_water_level(sheet_data, excel_path, sheet_name):
    """Align the three rainy events with their base water level in the affected D.1 block."""
    if get_condition_from_filename(excel_path) != "D.1" or sheet_name != "Slope40_l40_b200_hm1_90":
        return sheet_data, 0

    result = sheet_data.copy()
    zwu_column = "Initial piezometric surface depth - upstream [m]"
    zwu = pd.to_numeric(result[zwu_column], errors="coerce")
    zwd = pd.to_numeric(result["Initial piezometric surface depth - downstream [m]"], errors="coerce")
    ksat = pd.to_numeric(result["Saturated permeability [m/s]"], errors="coerce")
    tr = pd.to_numeric(result["Return period of precipitation [years]"], errors="coerce")
    rain = pd.to_numeric(result["Accumulated precipitation [mm]"], errors="coerce")

    # Isolate the known D.1 source block containing the upstream water-level inconsistency.
    block = np.isclose(ksat, 1e-8, rtol=0.0, atol=1e-15) & zwd.eq(0)
    base = block & tr.eq(-1) & rain.eq(0) & np.isclose(zwu, 22.402, rtol=0.0, atol=1e-9)
    to_correct = block & tr.isin([30, 200, 500]) & rain.gt(0) & np.isclose(zwu, 22.392, rtol=0.0, atol=1e-9)
    if not to_correct.any():
        return result, 0

    mechanics = ["Unit weight [kN/m3]", "Effective cohesion [kPa]", "Effective friction angle [°]"]
    base_cases = result.loc[base].groupby(mechanics, dropna=False).size().sort_index()
    if base_cases.empty:
        raise ValueError(f"{sheet_name}: missing base cases for rainy water-table correction")
    base_depths = zwu.loc[base].unique()
    if len(base_depths) != 1:
        raise ValueError(f"{sheet_name}: base cases have inconsistent initial upstream water levels")
    # Verify that every rainy event contains exactly the same mechanical cases as the base event.
    for return_period in (30, 200, 500):
        event = block & tr.eq(return_period) & rain.gt(0) & (
            np.isclose(zwu, 22.392, rtol=0.0, atol=1e-9) |
            np.isclose(zwu, 22.402, rtol=0.0, atol=1e-9)
        )
        event_cases = result.loc[event].groupby(mechanics, dropna=False).size().sort_index()
        if not event_cases.equals(base_cases):
            raise ValueError(f"{sheet_name}: TR={return_period} does not match the base mechanical cases")

    result.loc[to_correct, zwu_column] = float(base_depths[0])
    return result, int(to_correct.sum())

def load_source_and_matching_excel(excel_path):
    """
    Load source simulations and build a parallel normalized copy for matching.

    The returned dataframes always have the same rows and indices:
    - source_data keeps the values exactly as stored in the Excel sheets;
    - matching_data applies the existing geometry, hydraulic, water-table,
      and no-rain normalizations used by the original matching procedure.

    This separation preserves matching behavior while keeping original
    simulation precision in the output when the source value is valid.
    """
    xls = pd.ExcelFile(excel_path)
    source_sheets = []
    matching_sheets = []
    geometry_corrections = defaultdict(int)
    hydraulic_corrections = 0
    water_combinations_corrections = defaultdict(int)
    rain_water_corrections = 0

    print(f"\nReading: {excel_path}")

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
        df.columns = df.columns.astype(str).str.strip()

        # Build the source row exactly as in the original file structure.
        sheet_source = pd.concat(
            [df.iloc[1:, 1:-4].reset_index(drop=True), df.iloc[1:, -4:].reset_index(drop=True)],
            axis=1,
        )
        if sheet_source.shape[1] != len(COLUMNS):
            raise ValueError(
                f"{os.path.basename(excel_path)} - sheet '{sheet_name}': "
                f"found {sheet_source.shape[1]} columns, expected {len(COLUMNS)}."
            )
        sheet_source.columns = list(COLUMNS)

        # Keep untouched source targets and a normalized copy for matching and
        # feature-value candidates.
        sheet_match = sheet_source.copy()
        sheet_match, sheet_corrections = normalize_sheet_geometry(
            sheet_match, excel_path, sheet_name
        )
        for column, count in sheet_corrections.items():
            geometry_corrections[column] += count

        sheet_match, corrected_hydraulic_rows = normalize_hydraulic_parameters(
            sheet_match, excel_path
        )
        hydraulic_corrections += corrected_hydraulic_rows

        sheet_match, sheet_water_corrections = normalize_structured_water_tables(
            sheet_match, excel_path
        )
        for column, count in sheet_water_corrections.items():
            water_combinations_corrections[column] += count

        sheet_match = normalize_no_rain_event(sheet_match)
        sheet_match, corrected_rain_rows = align_d1_rainfall_water_level(
            sheet_match, excel_path, sheet_name
        )
        rain_water_corrections += corrected_rain_rows

        metadata = {
            "Source file": os.path.basename(excel_path),
            "Sheet": sheet_name,
            "Excel row": np.arange(3, len(sheet_source) + 3),
        }
        for column, values in reversed(list(metadata.items())):
            sheet_source.insert(0, column, values)
            sheet_match.insert(0, column, values)

        source_sheets.append(sheet_source)
        matching_sheets.append(sheet_match)
        print(f"  {sheet_name}: {len(sheet_source)} rows")

    if not source_sheets:
        raise ValueError(f"No sheets found in {excel_path}")

    source_data = pd.concat(source_sheets, ignore_index=True)
    matching_data = pd.concat(matching_sheets, ignore_index=True)

    if geometry_corrections:
        print("Geometry cells normalized while loading:")
        for column in GEOMETRY_COLUMNS:
            count = geometry_corrections.get(column, 0)
            if count:
                print(f"  {column}: {count}")
    if hydraulic_corrections:
        print(f"Soil Type values corrected from permeability: {hydraulic_corrections}")
    if water_combinations_corrections:
        print("Water-table cells corrected from block structure:")
        for column, count in water_combinations_corrections.items():
            if count:
                print(f"  {column}: {count}")
    if rain_water_corrections:
        print(
            "Rainy initial upstream water levels aligned with base case: "
            f"{rain_water_corrections}"
        )

    return source_data, matching_data


def normalize_no_rain_event(df):
    """
    Return a copy with one representation for no rain.

    The source files use different encodings (blank, '-', or a
    return period attached to hw=0).  They all represent the same base
    event and therefore match Tr=-1, hw=0, duration=0.
    """
    result = df.copy()
    rain = pd.to_numeric(result[RAIN], errors="coerce")
    # Any row with zero accumulated rainfall represents the same base event.
    no_rain = rain.notna() & np.isclose(rain, 0.0, atol=1e-12)
    result.loc[no_rain, TR] = -1
    result.loc[no_rain, RAIN] = 0
    result.loc[no_rain, DURATION] = 0
    return result

def normalize_value_for_matching(value):
    """
    Normalize values ONLY for comparison.

    This comparison function does not modify its input value.

    Examples:
        20 == 20.0
        1e-4 == 0.0001
        NaN == NaN

    Text that cannot be interpreted as a number is preserved.
    """
    if pd.isna(value):
        return ("NA",)
    # Python / NumPy numeric value
    if isinstance(value, numbers.Number):
        return ("NUM", round(float(value), 10))
    value_str = str(value).strip()
    if value_str == "*":
        return ("WILDCARD",)
    # Allow numeric strings such as "20", "20.0", "1e-4"
    try:
        return ("NUM", round(float(value_str), 10))
    except ValueError:
        return ("TXT", value_str)

def build_match_key(row, columns):
    """Build the comparison key for a row."""
    return tuple(normalize_value_for_matching(row[col]) for col in columns)

def to_numeric_match_value(value):
    """Convert a matching value to float, or return None."""
    if pd.isna(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None

def fos_match_priority(value):
    """Rank source rows by the FoS that numeric preprocessing will produce."""
    if pd.isna(value):
        return math.inf
    cleaned = str(value).strip().replace("..", ".")
    cleaned = re.sub(r"^([+-]?\d+(?:\.\d+)?)\.$", r"\1", cleaned)
    number = to_numeric_match_value(cleaned)
    if number is None or not math.isfinite(number):
        return math.inf
    return FOS_CORRECTIONS.get(number, number)

def values_match_with_tolerance(expected, actual, column):
    """Compare one of the explicitly tolerant geometry columns."""
    if pd.isna(expected) or pd.isna(actual):
        return pd.isna(expected) and pd.isna(actual)
    expected_number = to_numeric_match_value(expected)
    actual_number = to_numeric_match_value(actual)
    if expected_number is None or actual_number is None:
        return normalize_value_for_matching(expected) == normalize_value_for_matching(actual)
    return math.isclose(expected_number, actual_number, rel_tol=0.0, abs_tol=GEOMETRY_MATCH_ATOL[column])

def compute_match_distance(expected_row, actual_row, columns):
    """Score valid candidates so that the closest geometry is used."""
    distance = 0.0
    for column in columns:
        expected = to_numeric_match_value(expected_row[column])
        actual = to_numeric_match_value(actual_row[column])
        if expected is not None and actual is not None:
            distance += abs(expected - actual) / GEOMETRY_MATCH_ATOL[column]
    return distance

def build_real_value_candidates(source_df, columns):
    """Index observed numeric simulation values for nearest-value lookup."""
    candidates = {}
    for column in columns:
        first_by_number = {}
        for source_order, value in enumerate(source_df[column]):
            if pd.isna(value):
                continue
            number = to_numeric_match_value(value)
            if number is not None and math.isfinite(number) and number not in first_by_number:
                first_by_number[number] = (source_order, value)
        numbers = sorted(first_by_number)
        candidates[column] = (
            numbers,
            [first_by_number[number][0] for number in numbers],
            [first_by_number[number][1] for number in numbers],
        )
    return candidates


def closest_real_value(candidates, column, canonical_value):
    """Return the observed real value closest to a canonical numeric value."""
    canonical_number = to_numeric_match_value(canonical_value)
    if canonical_number is None or not math.isfinite(canonical_number):
        return None

    numbers, orders, values = candidates.get(column, ((), (), ()))
    if not numbers:
        return None

    position = bisect_left(numbers, canonical_number)
    nearby = (index for index in (position - 1, position) if 0 <= index < len(numbers))
    best = min(
        nearby,
        key=lambda index: (abs(numbers[index] - canonical_number), orders[index]),
    )
    return values[best]


def closest_real_value_in_same_slope(
    source_df, real_match_df, sheet_indices, table_match_row,
    real_idx, column, canonical_value, feature_columns,
):
    """Find a missing input only among simulations of the same sheet and slope."""
    canonical_number = to_numeric_match_value(canonical_value)
    if canonical_number is None or not math.isfinite(canonical_number):
        return None

    # Identify the slope from the generated combination. A wildcard must be
    # resolved by the selected normalized simulation; otherwise the slope is
    # ambiguous and borrowing a value from another row would be unsafe.
    slope_values = {}
    for feature in feature_columns:
        if feature in (TR, RAIN, DURATION):
            continue
        expected = table_match_row[feature]
        if pd.isna(expected) or str(expected).strip() == "*":
            expected = real_match_df.at[real_idx, feature]
        if pd.isna(expected) or str(expected).strip() == "*":
            return None
        slope_values[feature] = expected

    best_score = None
    best_value = None
    for source_order, candidate_idx in enumerate(sheet_indices):
        value = source_df.at[candidate_idx, column]
        number = to_numeric_match_value(value)
        if number is None or not math.isfinite(number):
            continue
        if not all(
            values_match_with_tolerance(expected, real_match_df.at[candidate_idx, feature], feature)
            if feature in GEOMETRY_MATCH_ATOL
            else normalize_value_for_matching(expected)
            == normalize_value_for_matching(real_match_df.at[candidate_idx, feature])
            for feature, expected in slope_values.items()
        ):
            continue
        score = (abs(number - canonical_number), source_order)
        if best_score is None or score < best_score:
            best_score = score
            best_value = value
    return best_value


def map_simulations_to_canonical(table_match_df, real_match_df, feature_columns):
    """Map each simulation input to the nearest combination value in its column.

    This copy is used only for matching. Exported inputs are instead selected
    independently from the observed real simulation values in each column.
    """
    canonicalized = real_match_df.copy()
    for column in feature_columns:
        canonical_numbers = sorted({
            number
            for value in table_match_df[column].unique()
            if (number := to_numeric_match_value(value)) is not None
            and math.isfinite(number)
        })
        if not canonical_numbers:
            continue

        mapped_values = {}
        for value in real_match_df[column].unique():
            number = to_numeric_match_value(value)
            if number is None or not math.isfinite(number):
                continue
            position = bisect_left(canonical_numbers, number)
            nearby = (index for index in (position - 1, position)
                      if 0 <= index < len(canonical_numbers))
            mapped_values[number] = canonical_numbers[min(
                nearby,
                key=lambda index: (abs(canonical_numbers[index] - number), index),
            )]

        canonicalized[column] = real_match_df[column].map(
            lambda value: mapped_values.get(to_numeric_match_value(value), value)
        )
    return canonicalized


def build_modal_real_value_map(source_inputs, canonicalized_real, feature_columns):
    """Find the most frequent observed value for each column/canonical value.

    Count simulation rows, not just distinct values. Only real values mapped to
    that canonical value compete; ties prefer proximity to the canonical value,
    then the earliest source row.
    """
    modal = {}
    for column in feature_columns:
        counts = defaultdict(lambda: defaultdict(int))
        first_rows = {}
        original_values = {}
        for order, (canonical, observed) in enumerate(
            zip(canonicalized_real[column], source_inputs[column])
        ):
            canonical_number = to_numeric_match_value(canonical)
            observed_number = to_numeric_match_value(observed)
            if (
                canonical_number is None or not math.isfinite(canonical_number)
                or observed_number is None or not math.isfinite(observed_number)
            ):
                continue
            pair = (canonical_number, observed_number)
            counts[canonical_number][observed_number] += 1
            first_rows.setdefault(pair, order)
            original_values.setdefault(pair, observed)

        modal[column] = {}
        for canonical_number, frequencies in counts.items():
            chosen_number = min(
                frequencies,
                key=lambda number: (
                    -frequencies[number],
                    abs(number - canonical_number),
                    first_rows[(canonical_number, number)],
                ),
            )
            modal[column][canonical_number] = original_values[
                (canonical_number, chosen_number)
            ]
    return modal


def infer_sheet_value_from_event_history(
    source_real, real_match_df, real_idx, column, feature_columns, sheet_indices
):
    """Resolve an unanchored wildcard only if one sheet value lacks this event.

    Other features identify the slope family. A candidate must occur at least
    twice at earlier rainy events and not at the current event. Ambiguous
    candidates deliberately remain missing.
    """
    event_columns = (TR, RAIN, DURATION)
    identifying_columns = [
        feature for feature in feature_columns
        if feature != column and feature not in event_columns
    ]
    reference = real_match_df.loc[real_idx]
    reference_key = build_match_key(reference, identifying_columns)
    current_event = (
        to_numeric_match_value(reference[TR]),
        to_numeric_match_value(reference[DURATION]),
    )
    if None in current_event:
        return None

    candidate_events = defaultdict(set)
    candidate_values = {}
    for other_idx in sheet_indices:
        if other_idx == real_idx:
            continue
        other = real_match_df.loc[other_idx]
        if build_match_key(other, identifying_columns) != reference_key:
            continue
        value = source_real.at[other_idx, column]
        number = to_numeric_match_value(value)
        event = (to_numeric_match_value(other[TR]),
                 to_numeric_match_value(other[DURATION]))
        if number is None or not math.isfinite(number) or None in event:
            continue
        candidate_values[number] = value
        candidate_events[number].add(event)

    eligible = [
        candidate_values[number]
        for number, events in candidate_events.items()
        if current_event not in events
        and sum(event[0] > 0 and event < current_event for event in events) >= 2
    ]
    return eligible[0] if len(eligible) == 1 else None


def report_matched_slope_samples(matching_groups):
    """Count slopes by sample count and exact Tr/duration sequence."""
    slope_columns = list(COLUMNS[:COLUMNS.index(TR)])
    slope_inputs = matching_groups[slope_columns].apply(pd.to_numeric, errors="coerce")
    slope_indices = slope_inputs.groupby(
        slope_columns, dropna=False, sort=False
    ).indices
    pattern_counts = defaultdict(int)

    for indices in slope_indices.values():
        events = tuple(sorted(
            (
                (to_numeric_match_value(tr), to_numeric_match_value(duration))
                for tr, duration in matching_groups.iloc[indices][[TR, DURATION]]
                .itertuples(index=False, name=None)
            ),
            key=lambda event: tuple(
                value if value is not None and math.isfinite(value) else math.inf
                for value in event
            ),
        ))
        pattern_counts[(len(indices), events)] += 1

    if not pattern_counts:
        print("# slopes with matched samples: 0")
        return

    def number_label(value):
        return f"{value:g}" if value is not None and math.isfinite(value) else "?"

    for (sample_count, events), slope_count in sorted(
        pattern_counts.items(), key=lambda item: (item[0][0], repr(item[0][1]))
    ):
        event_labels = ", ".join(
            "-" if tr == -1 and duration == 0
            else f"{number_label(tr)}/{number_label(duration)}"
            for tr, duration in events
        )
        print(
            f"# slopes with {sample_count} samples: {slope_count} "
            f"(Tr/duration [h]: {event_labels})"
        )


def match_generated_combinations_to_simulations(table_path, source_real, matching_real):
    """Match one generated combinations table to the available simulations.

    MODE controls two independent choices:
    - canonical_*: map source inputs to their nearest canonical values before matching;
      real_*: match directly against the normalized real inputs.
    - *_freq / *_near: choose the representative real input using frequency or
      nearest-value logic, respectively.
    """
    table_df = pd.read_csv(table_path, dtype=object, keep_default_na=True)
    expected_columns = list(COLUMNS)
    missing_combinations = [col for col in expected_columns if col not in table_df.columns]
    if missing_combinations:
        raise ValueError(f"Missing columns in {table_path}:\n{missing_combinations}")

    table_df = table_df[expected_columns].copy()
    source_df = source_real[expected_columns].copy()
    real_df = matching_real[expected_columns].copy()

    table_match_df = normalize_no_rain_event(table_df)
    normalized_real_df = normalize_no_rain_event(real_df)
    feature_columns = expected_columns[:-4]

    if USE_CANONICAL_MATCHING:
        # Matching uses canonicalized simulations; exported inputs remain real values.
        real_match_df = map_simulations_to_canonical(
            table_match_df, normalized_real_df, feature_columns
        )
        source_inputs = source_df[feature_columns].copy()
        source_inputs[[TR, RAIN, DURATION]] = normalized_real_df[[TR, RAIN, DURATION]]

        if VALUE_SELECTION == "freq":
            modal_real_values = build_modal_real_value_map(
                source_inputs, real_match_df, feature_columns
            )
        else:
            real_candidates = build_real_value_candidates(source_inputs, feature_columns)
    else:
        # Match directly using the normalized real values.
        real_match_df = normalized_real_df

    rows_by_wildcards = defaultdict(list)
    for idx, row in table_df.iterrows():
        wildcard_columns = tuple(
            col for col in feature_columns if str(row[col]).strip() == "*"
        )
        rows_by_wildcards[wildcard_columns].append(idx)
    wildcard_groups = sorted(rows_by_wildcards.keys(), key=lambda x: len(x))

    available_real_indices = set(real_df.index)
    fos_values = real_df[FOS].map(fos_match_priority)

    def fos_priority(real_idx):
        return fos_values.at[real_idx]

    consumed_real_feature_keys = set()
    matched_rows = {}
    matched_group_rows = {}
    unmatched_combinations_indices = []

    if USE_CANONICAL_MATCHING:
        history_indices_by_column = {}

        def matching_sheet_history(real_idx, column, sheet):
            """Find same-slope sheet rows without rescanning a whole sheet per NaN."""
            if column not in history_indices_by_column:
                identifying_columns = [
                    feature for feature in feature_columns
                    if feature != column and feature not in (TR, RAIN, DURATION)
                ]
                grouped = defaultdict(list)
                for other_idx, other_row in real_match_df.iterrows():
                    key = build_match_key(other_row, identifying_columns)
                    grouped[(source_real.at[other_idx, "Sheet"], key)].append(other_idx)
                history_indices_by_column[column] = (identifying_columns, grouped)
            identifying_columns, grouped = history_indices_by_column[column]
            key = build_match_key(real_match_df.loc[real_idx], identifying_columns)
            return grouped.get((sheet, key), ())
    else:
        sheet_indices = {
            sheet: list(indices)
            for sheet, indices in source_real.groupby("Sheet", sort=False).groups.items()
        }

    for wildcard_columns in wildcard_groups:
        matching_columns = [col for col in feature_columns if col not in wildcard_columns]
        exact_matching_columns = [
            col for col in matching_columns if col not in GEOMETRY_MATCH_ATOL
        ]
        approximate_matching_columns = [
            col for col in matching_columns if col in GEOMETRY_MATCH_ATOL
        ]

        print(
            f"\nMatching rows with wildcards: "
            f"{list(wildcard_columns) if wildcard_columns else 'none'}"
        )
        print(f"Using {len(matching_columns)} feature columns.")

        real_index = defaultdict(deque)
        indexed_real_feature_keys = set()

        for real_idx in sorted(available_real_indices, key=lambda idx: (fos_priority(idx), idx)):
            real_row = real_match_df.loc[real_idx]
            full_feature_key = build_match_key(real_row, feature_columns)
            if (
                full_feature_key in consumed_real_feature_keys
                or full_feature_key in indexed_real_feature_keys
            ):
                continue
            indexed_real_feature_keys.add(full_feature_key)
            key = build_match_key(real_row, exact_matching_columns)
            real_index[key].append(real_idx)

        for table_idx in rows_by_wildcards[wildcard_columns]:
            table_match_row = table_match_df.loc[table_idx]
            key = build_match_key(table_match_row, exact_matching_columns)
            candidates = real_index.get(key)

            if not candidates:
                unmatched_combinations_indices.append(table_idx)
                continue

            valid_candidates = [
                real_idx
                for real_idx in candidates
                if all(
                    values_match_with_tolerance(
                        table_match_row[col], real_match_df.at[real_idx, col], col
                    )
                    for col in approximate_matching_columns
                )
            ]

            if not valid_candidates:
                unmatched_combinations_indices.append(table_idx)
                continue

            real_idx = min(
                valid_candidates,
                key=lambda idx: (
                    fos_priority(idx),
                    compute_match_distance(
                        table_match_row,
                        real_match_df.loc[idx],
                        approximate_matching_columns,
                    ),
                    idx,
                ),
            )
            candidates.remove(real_idx)
            if not candidates:
                real_index.pop(key, None)

            available_real_indices.remove(real_idx)
            consumed_real_feature_keys.add(
                build_match_key(real_match_df.loc[real_idx], feature_columns)
            )

            output_row = source_df.loc[real_idx].copy()
            sheet = source_real.at[real_idx, "Sheet"]

            if USE_CANONICAL_MATCHING:
                # Targets stay from the selected simulation. Inputs are reconstructed
                # from real values according to the selected canonical strategy.
                for column in feature_columns:
                    canonical_value = table_match_row[column]
                    if str(canonical_value).strip() == "*" or pd.isna(canonical_value):
                        canonical_value = real_match_df.at[real_idx, column]

                    if VALUE_SELECTION == "freq":
                        canonical_number = to_numeric_match_value(canonical_value)
                        replacement = modal_real_values[column].get(canonical_number)
                        if replacement is None and canonical_number is None:
                            replacement = infer_sheet_value_from_event_history(
                                source_real, real_match_df, real_idx, column,
                                feature_columns,
                                matching_sheet_history(real_idx, column, sheet),
                            )
                    else:
                        replacement = closest_real_value(
                            real_candidates, column, canonical_value
                        )
                        if (
                            replacement is None
                            and to_numeric_match_value(canonical_value) is None
                        ):
                            inferred = infer_sheet_value_from_event_history(
                                source_real, real_match_df, real_idx, column,
                                feature_columns,
                                matching_sheet_history(real_idx, column, sheet),
                            )
                            replacement = closest_real_value(
                                real_candidates, column, inferred
                            )

                    if replacement is not None:
                        output_row[column] = replacement
            else:
                # Preserve valid source inputs. A missing input may only borrow
                # a real value from the same sheet and the same identified slope.
                for column in feature_columns:
                    source_value = to_numeric_match_value(output_row[column])
                    if source_value is not None and math.isfinite(source_value):
                        continue
                    canonical_value = table_match_row[column]
                    if str(canonical_value).strip() == "*" or pd.isna(canonical_value):
                        canonical_value = real_match_df.at[real_idx, column]
                    replacement = closest_real_value_in_same_slope(
                        source_df, real_match_df, sheet_indices[sheet],
                        table_match_row, real_idx, column, canonical_value,
                        feature_columns,
                    )
                    if replacement is not None:
                        output_row[column] = replacement

            # All no-rain encodings mean the same base event: Tr=-1, hw=0, tw=0.
            if to_numeric_match_value(real_match_df.at[real_idx, RAIN]) == 0:
                output_row[[TR, RAIN, DURATION]] = real_match_df.loc[
                    real_idx, [TR, RAIN, DURATION]
                ]

            matched_rows[table_idx] = output_row
            group_row = table_match_row[expected_columns].copy()
            for column in wildcard_columns:
                resolved = real_match_df.at[real_idx, column]
                group_row[column] = output_row[column] if pd.isna(resolved) else resolved
            matched_group_rows[table_idx] = group_row

    matched_indices = sorted(matched_rows.keys())
    output_df = (
        pd.DataFrame([matched_rows[idx] for idx in matched_indices])[expected_columns]
        if matched_indices
        else pd.DataFrame(columns=expected_columns)
    )
    matching_groups = (
        pd.DataFrame([matched_group_rows[idx] for idx in matched_indices])[expected_columns]
        .reset_index(drop=True)
        if matched_indices else pd.DataFrame(columns=expected_columns)
    )

    mismatch_combinations_df = table_df.loc[
        sorted(unmatched_combinations_indices)
    ].copy()
    mismatch_simulations_df = source_real.loc[
        sorted(available_real_indices), SOURCE_METADATA_COLUMNS + expected_columns
    ].copy()

    print("\n" + "=" * 70)
    print(f"MODE: {MODE}")
    print(f"TABLE: {table_path}")
    print("-" * 70)
    print(f"Generated table rows : {len(table_df)}")
    print(f"Real rows            : {len(real_df)}")
    print(f"Matched rows         : {len(output_df)}")
    print(f"Missing table rows   : {len(mismatch_combinations_df)}")
    print(f"Unused real rows     : {len(mismatch_simulations_df)}")
    print("=" * 70)
    report_matched_slope_samples(matching_groups)
    return output_df, mismatch_combinations_df, mismatch_simulations_df, matching_groups


def classify_water_table_state(row):
    """Label a generated row as high, low, or intermediate water table."""
    zwu_column = "Initial piezometric surface depth - upstream [m]"
    zwd_column = "Initial piezometric surface depth - downstream [m]"

    if "*" in {str(row[zwu_column]).strip(), str(row[zwd_column]).strip()}:
        return "intermediate"

    zwu = to_numeric_match_value(row[zwu_column])
    zwd = to_numeric_match_value(row[zwd_column])
    hsu = to_numeric_match_value(row["Soil depth upstream [m]"])
    hsd = to_numeric_match_value(row["Soil depth downstream [m]"])

    if None not in (zwu, zwd) and math.isclose(zwu, 0.0) and math.isclose(zwd, 0.0):
        return "high"

    if None not in (zwu, zwd, hsu, hsd) and math.isclose(zwu, hsu, abs_tol=GEOMETRY_ROUNDING_ATOL) and math.isclose(zwd, hsd, abs_tol=GEOMETRY_ROUNDING_ATOL):
        return "low"

    return "intermediate"

def annotate_missing_combinations(missing_df, condition):
    """Add condition, water-table state, and diagnostic reason to missing combinations."""
    result = missing_df.copy()
    result.insert(0, "Condition", condition)
    result.insert(1, "Water table state", result.apply(classify_water_table_state, axis=1))
    result.insert(2, "Missing source reason", "No corresponding simulation in the corrected source file")
    return result

def save_mismatch_reports(results, output_dir):
    """Save aggregate mismatch reports and the inferred missing D.1 candidate block."""
    report_dir = os.fspath(output_dir)
    labeled = {condition: annotate_missing_combinations(missing_df, condition) for condition, (_, missing_df, _) in results.items()}

    all_missing = pd.concat(labeled.values(), ignore_index=True)

    # Requested aggregate dataframe: generated combinations for which no
    # corrected source simulation exists.
    mismatch_combinations_path = os.path.join(report_dir, "mismatch_combinations.csv")
    all_missing.to_csv(mismatch_combinations_path, index=False)

    unused_simulations = []
    for condition, (_, _, unused_df) in results.items():
        condition_unused = unused_df.copy()
        condition_unused.insert(0, "Condition", condition)
        provenance_columns = ["Condition", "Sheet", "Source file", "Excel row"]
        condition_unused = condition_unused[provenance_columns + [column for column in condition_unused.columns if column not in provenance_columns]]
        unused_simulations.append(condition_unused)

    # Requested aggregate dataframe: real source simulations which were not
    # consumed by any generated combinationsination.
    mismatch_simulations = pd.concat(unused_simulations, ignore_index=True)
    mismatch_simulations_path = os.path.join(report_dir, "mismatch_simulations.csv")
    mismatch_simulations.to_csv(mismatch_simulations_path, index=False)

    # The only D.1 geometry absent altogether is alpha=50, L=80,
    # hSu=72.5 m.  The three adjacent source sheets in this geometry family
    # contain only the low water-table state.  This is therefore the
    # 156-combinationsination candidate that would reconcile 15,444 with the
    # manuscript's 15,600, but it is not a simulated output.
    d1_missing_sheet = labeled["D.1"].loc[pd.to_numeric(labeled["D.1"]["Slope angle [°]"], errors="coerce").eq(50) & pd.to_numeric(labeled["D.1"]["Slope length [m]"], errors="coerce").eq(80) & pd.to_numeric(labeled["D.1"]["Soil depth upstream [m]"], errors="coerce").eq(72.5) & labeled["D.1"]["Water table state"].eq("low")].copy()
    d1_missing_sheet.insert(1, "Inferred missing sheet", "Slope50_l80_b400_hm1_25")
    d1_missing_sheet.insert(2, "Inference status", "Candidate inferred from adjacent one-state sheets; no simulation")
    d1_missing_sheet_path = os.path.join(report_dir, "D.1_missing_156_candidate.csv")

    assert len(d1_missing_sheet) == 156
    assert pd.to_numeric(d1_missing_sheet[TR], errors="raise").value_counts().to_dict() == {-1: 39, 30: 39, 200: 39, 500: 39}
    assert pd.to_numeric(d1_missing_sheet["Soil Type [-]"], errors="raise").value_counts().to_dict() == {1: 52, 2: 52, 3: 52}
    d1_missing_sheet.to_csv(d1_missing_sheet_path, index=False)

    print(f"Saved: {mismatch_combinations_path} ({len(all_missing)} rows)")
    print(f"Saved: {mismatch_simulations_path} ({len(mismatch_simulations)} rows)")
    print(f"Saved: {d1_missing_sheet_path} ({len(d1_missing_sheet)} candidate combinations)")

def find_source_excel(prefix):
    """Return the unique source Excel file whose name starts with the requested prefix."""
    data_dir = os.path.join(BASE_DIR, "data")
    matches = [
        file_name for file_name in os.listdir(data_dir)
        if file_name.lower().endswith((".xlsx", ".xls")) and file_name.startswith(prefix)
    ]
    if len(matches) == 0:
        raise FileNotFoundError(f"No Excel file starting with '{prefix}' found in {data_dir}")
    if len(matches) > 1:
        raise ValueError(f"More than one Excel file starts with '{prefix}':\n{matches}")
    return os.path.join(data_dir, matches[0])

def parse_numeric_dataframe(
    frame: pd.DataFrame, condition: str, source_file: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Convert a matched DataFrame to numeric values and record parsing corrections."""
    # Preserve the original text representation for audit messages before numeric conversion.
    raw = frame.reset_index(drop=True).astype("string").fillna("")
    if TR not in raw or DURATION not in raw or FOS not in raw:
        raise ValueError(f"Missing required columns in {source_file}")

    numeric = pd.DataFrame(index=raw.index)
    audit = []
    for column in raw:
        stripped = raw[column].str.strip()
        # Repair unambiguous punctuation: "1..81" -> "1.81" and
        # "17.5." -> "17.5". Other malformed text remains NaN.
        cleaned = stripped.str.replace("..", ".", regex=False).str.replace(
            r"^([+-]?\d+(?:\.\d+)?)\.$", r"\1", regex=True
        )
        for row in raw.index[cleaned.ne(stripped)]:
            audit.append({
                "Condition": condition, "Source file": source_file, "CSV row": int(row) + 2,
                "Column": column, "Action": "numeric_punctuation_correction",
                "Old value": raw.at[row, column], "New value": cleaned.at[row],
            })
        # Parse each value without changing the last digit of a source float:
        # pd.to_numeric's vectorized parser can shift it by one ULP.
        numeric[column] = cleaned.map(to_numeric_match_value)
        invalid = numeric[column].isna() & ~cleaned.str.lower().isin(MISSING_TEXT)
        for row in raw.index[invalid]:
            audit.append({
                "Condition": condition, "Source file": source_file, "CSV row": int(row) + 2,
                "Column": column, "Action": "invalid_text_to_nan",
                "Old value": raw.at[row, column], "New value": "",
            })

    # Apply the explicit FoS decimal corrections defined for known source-entry errors.
    numeric[FOS] = numeric[FOS].astype("Float64")
    for old_value, new_value in FOS_CORRECTIONS.items():
        changed = numeric[FOS].eq(old_value)
        for row in numeric.index[changed]:
            audit.append({
                "Condition": condition, "Source file": source_file, "CSV row": int(row) + 2,
                "Column": FOS, "Action": "fos_decimal_correction",
                "Old value": old_value, "New value": new_value,
            })
        numeric.loc[changed, FOS] = new_value

    return raw, numeric, audit

def fill_missing_from_equal_neighbors(
    raw: pd.DataFrame,
    numeric: pd.DataFrame,
    condition: str,
    source_file: str,
    grouping: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fill an interior gap only when adjacent values of the same slope agree."""
    identity_columns = list(grouping.columns[: grouping.columns.get_loc(TR)])
    sort_columns = [TR, DURATION]
    result = numeric.copy()
    audit = []

    columns_to_fill = (
        [column for column in TARGET_COLUMNS if column in numeric]
        if USE_CANONICAL_MATCHING
        else [column for column in numeric.columns if column not in (TR, RAIN, DURATION)]
    )
    for column in columns_to_fill:
        if not numeric[column].isna().any():
            continue
        keys = [key for key in identity_columns if key != column]
        if not keys:
            continue
        for indices in grouping.groupby(keys, dropna=False, sort=False).indices.values():
            if len(indices) < 3:
                continue
            group = grouping.loc[indices]
            if group[sort_columns].isna().any().any() or group.duplicated(subset=sort_columns).any():
                continue

            ordered = group.sort_values(sort_columns, kind="stable")
            rows = ordered.index.to_list()
            values = numeric.loc[rows, column].to_list()
            for position in range(1, len(rows) - 1):
                if pd.notna(values[position]):
                    continue
                before, after = values[position - 1], values[position + 1]
                if pd.isna(before) or pd.isna(after) or before != after:
                    continue
                row = rows[position]
                previous_row = rows[position - 1]
                next_row = rows[position + 1]
                result.at[row, column] = before
                audit.append({
                    "Condition": condition, "Source file": source_file, "CSV row": int(row) + 2,
                    "Column": column, "Action": "equal_neighbors_fill", "Old value": raw.at[row, column], "New value": before,
                    "Previous CSV row": int(previous_row) + 2,
                    "Previous Tr": grouping.at[previous_row, TR],
                    "Previous duration": grouping.at[previous_row, DURATION],
                    "Next CSV row": int(next_row) + 2,
                    "Next Tr": grouping.at[next_row, TR], "Next duration": grouping.at[next_row, DURATION],
                })

    return result, audit

def fill_terminal_missing_targets(
    raw: pd.DataFrame,
    neighbor_filled: pd.DataFrame,
    condition: str,
    source_file: str,
    grouping: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fill a final gap when every earlier rainy value agrees (not the base)."""
    identity_columns = list(grouping.columns[: grouping.columns.get_loc(TR)])
    sort_columns = [TR, DURATION]
    result = neighbor_filled.copy()
    audit = []

    columns_to_fill = (
        [column for column in TARGET_COLUMNS if column in result]
        if USE_CANONICAL_MATCHING
        else [column for column in result.columns if column not in (TR, RAIN, DURATION)]
    )
    for column in columns_to_fill:
        if not result[column].isna().any():
            continue
        keys = [key for key in identity_columns if key != column]
        if not keys:
            continue
        for indices in grouping.groupby(keys, dropna=False, sort=False).indices.values():
            group = grouping.loc[indices]
            if len(group) < 3 or group[sort_columns].isna().any().any():
                continue

            if group.duplicated(subset=sort_columns).any():
                # Several states at the same event cannot identify one slope.
                continue

            rows = group.sort_values(sort_columns, kind="stable").index.to_list()
            row = rows[-1]
            if pd.notna(result.at[row, column]):
                continue
            support_rows = [previous_row for previous_row in rows[:-1]
                            if grouping.at[previous_row, TR] > 0]
            if len(support_rows) < 2:
                continue
            preceding = neighbor_filled.loc[support_rows, column]
            if not preceding.notna().all():
                continue
            candidate = preceding.iloc[0]
            if not preceding.eq(candidate).all():
                continue

            result.at[row, column] = candidate
            previous_row = support_rows[-1]
            audit.append({
                "Condition": condition, "Source file": source_file, "CSV row": int(row) + 2,
                "Column": column, "Action": "last_row_constant_fill",
                "Old value": raw.at[row, column], "New value": candidate,
                "Previous CSV row": int(previous_row) + 2,
                "Previous Tr": grouping.at[previous_row, TR],
                "Previous duration": grouping.at[previous_row, DURATION],
                "Support count": len(support_rows),
                "Support CSV rows": ";".join(str(int(support_row) + 2) for support_row in support_rows),
            })

    return result, audit

def replace_varying_slope_inputs_with_mode(
    numeric: pd.DataFrame,
    grouping: pd.DataFrame,
    condition: str,
    source_file: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """Give every event of a canonical slope the modal real input value.

    Rainfall fields and targets are not intrinsic slope inputs. If modes tie,
    prefer the observed value closest to the canonical one, then the first
    observed value. Missing inputs do not vote but receive the chosen mode.
    """
    input_columns = list(grouping.columns[:grouping.columns.get_loc(TR)])
    result = numeric.copy()
    audit = []

    for positions in grouping.groupby(input_columns, dropna=False, sort=False).indices.values():
        if len(positions) < 2:
            continue
        rows = grouping.index.take(positions).to_list()
        for column in input_columns:
            observed = result.loc[rows, column]
            nonmissing = observed.dropna()
            if nonmissing.empty or observed.nunique(dropna=False) == 1:
                continue

            counts = nonmissing.value_counts(sort=False)
            most_frequent = counts[counts.eq(counts.max())].index.to_list()
            observed_values = nonmissing.to_list()
            first_position = {
                value: observed_values.index(value) for value in most_frequent
            }
            canonical = grouping.at[rows[0], column]
            if pd.notna(canonical):
                chosen = min(
                    most_frequent,
                    key=lambda value: (abs(float(value) - float(canonical)), first_position[value]),
                )
            else:
                chosen = min(most_frequent, key=lambda value: first_position[value])

            support_rows = [row for row in rows if pd.notna(observed.at[row]) and observed.at[row] == chosen]
            support_csv_rows = ";".join(str(int(row) + 2) for row in support_rows)
            for row in rows:
                old = observed.at[row]
                if pd.notna(old) and old == chosen:
                    continue
                result.at[row, column] = chosen
                audit.append({
                    "Condition": condition, "Source file": source_file,
                    "CSV row": int(row) + 2, "Column": column,
                    "Action": "slope_input_mode",
                    "Old value": "" if pd.isna(old) else old,
                    "New value": chosen,
                    "Support count": len(support_rows),
                    "Support CSV rows": support_csv_rows,
                })

    return result, audit


def replace_varying_slope_inputs_with_nearest(
    numeric: pd.DataFrame,
    grouping: pd.DataFrame,
    condition: str,
    source_file: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """Give every event of a canonical slope its nearest observed input.

    Only values observed within this slope compete. Rainfall fields and targets
    are untouched. Equal-distance values use frequency, then source-row order.
    A missing canonical value provides no anchor and is left unchanged.
    """
    input_columns = list(grouping.columns[:grouping.columns.get_loc(TR)])
    result = numeric.copy()
    audit = []

    for positions in grouping.groupby(input_columns, dropna=False, sort=False).indices.values():
        if len(positions) < 2:
            continue
        rows = grouping.index.take(positions).to_list()
        for column in input_columns:
            observed = result.loc[rows, column]
            nonmissing = observed.dropna()
            if nonmissing.empty or observed.nunique(dropna=False) == 1:
                continue

            canonical = grouping.at[rows[0], column]
            if pd.isna(canonical):
                continue
            counts = nonmissing.value_counts(sort=False)
            observed_values = nonmissing.to_list()
            candidates = counts.index.to_list()
            first_position = {
                value: observed_values.index(value) for value in candidates
            }
            chosen = min(
                candidates,
                key=lambda value: (
                    abs(float(value) - float(canonical)),
                    -int(counts.loc[value]),
                    first_position[value],
                ),
            )

            support_rows = [row for row in rows if pd.notna(observed.at[row]) and observed.at[row] == chosen]
            support_csv_rows = ";".join(str(int(row) + 2) for row in support_rows)
            for row in rows:
                old = observed.at[row]
                if pd.notna(old) and old == chosen:
                    continue
                result.at[row, column] = chosen
                audit.append({
                    "Condition": condition, "Source file": source_file,
                    "CSV row": int(row) + 2, "Column": column,
                    "Action": "slope_input_nearest",
                    "Old value": "" if pd.isna(old) else old,
                    "New value": chosen,
                    "Support count": len(support_rows),
                    "Support CSV rows": support_csv_rows,
                })

    return result, audit


def collect_remaining_missing_cells(
    raw: pd.DataFrame,
    result: pd.DataFrame,
    condition: str,
    source_file: str,
    grouping: pd.DataFrame,
) -> list[dict]:
    """Report unresolved cells, omitting columns absent in every source row."""
    rows = []
    for column in result:
        if result[column].isna().all():
            continue
        for row in result.index[result[column].isna()]:
            rows.append({
                "Condition": condition, "Source file": source_file, "CSV row": int(row) + 2,
                "Column": column, "Original value": raw.at[row, column],
                "Tr": grouping.at[row, TR], "Duration": grouping.at[row, DURATION],
            })
    return rows

def preprocess_matched_dataframe(
    frame: pd.DataFrame, condition: str, source_file: str,
    matching_groups: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Parse matched simulations, fill supported gaps and finalize inputs."""
    raw, numeric, parsing_audit = parse_numeric_dataframe(frame, condition, source_file)
    grouping = matching_groups.reset_index(drop=True).apply(
        pd.to_numeric, errors="coerce"
    )

    print(f"\nPreprocessing {condition} ({MODE})")
    print(f"  Rows: {len(numeric)}")
    missing_before = int(numeric.isna().sum().sum())
    print(f"  Missing cells before fill: {missing_before}")

    neighbor_filled, neighbor_audit = fill_missing_from_equal_neighbors(
        raw, numeric, condition, source_file, grouping
    )
    missing_after_neighbors = int(neighbor_filled.isna().sum().sum())
    print(
        f"  Equal-neighbor fill: {missing_before - missing_after_neighbors} "
        f"cells ({len(neighbor_audit)} audit records)"
    )

    filled, terminal_audit = fill_terminal_missing_targets(
        raw, neighbor_filled, condition, source_file, grouping
    )
    missing_after_terminal = int(filled.isna().sum().sum())
    print(
        f"  Terminal fill: {missing_after_neighbors - missing_after_terminal} "
        f"cells ({len(terminal_audit)} audit records)"
    )

    alignment_audit = []
    if not USE_CANONICAL_MATCHING:
        if VALUE_SELECTION == "freq":
            filled, alignment_audit = replace_varying_slope_inputs_with_mode(
                filled, grouping, condition, source_file
            )
        else:
            filled, alignment_audit = replace_varying_slope_inputs_with_nearest(
                filled, grouping, condition, source_file
            )
        print(f"  Aligned slope inputs ({VALUE_SELECTION}): {len(alignment_audit)} cells")

    if condition in {"D.1", "D.2"}:
        filled = filled.drop(columns=UNDRAINED_STRENGTH, errors="ignore")
    elif condition == "UN":
        filled = filled.drop(columns=[EFFECTIVE_COHESION, EFFECTIVE_ANGLE], errors="ignore")

    unresolved = collect_remaining_missing_cells(
        raw, filled, condition, source_file, grouping
    )
    audit = parsing_audit + neighbor_audit + terminal_audit + alignment_audit
    return filled, audit, unresolved


def report_slope_group_counts(
    processed: pd.DataFrame, matching_groups: pd.DataFrame, condition: str
) -> None:
    """Report canonical slopes and distinct real-valued variants."""
    intrinsic_columns = [
        column for column in list(COLUMNS)[:list(COLUMNS).index(TR)]
        if column in processed.columns
    ]
    canonical = matching_groups[intrinsic_columns].reset_index(drop=True).apply(
        pd.to_numeric, errors="coerce"
    )
    actual = processed[intrinsic_columns].reset_index(drop=True)
    canonical_ids = canonical.groupby(
        intrinsic_columns, dropna=False, sort=False
    ).ngroup()
    actual_ids = actual.groupby(
        intrinsic_columns, dropna=False, sort=False
    ).ngroup()
    distinct_outputs = pd.DataFrame({
        "canonical_id": canonical_ids, "actual_id": actual_ids,
    }).groupby("canonical_id")["actual_id"].nunique()
    real_count = int(actual_ids.nunique())
    split_count = int(distinct_outputs.gt(1).sum())
    print(
        f"{condition}: {len(distinct_outputs)} canonical slope groups, "
        f"{real_count} real-valued slope groups, "
        f"{split_count} canonical groups with multiple real variants"
    )

CONDITIONS = (
    ("D.1", "D.1", "D.1_combinations.csv", "D.1_drained.csv", "D.1"),
    ("D.2", "D.2", "D.2_combinations.csv", "D.2_drained.csv", "D.2"),
    ("UN", "U", "U_combinations.csv", "U_undrained.csv", "U"),
)

REMAINING_COLUMNS = ("Condition", "Source file", "CSV row", "Column", "Original value", "Tr", "Duration")

def build_processed_datasets(table_dir: str):
    """Build all processed datasets and diagnostics entirely in memory."""
    results = {}
    audit_rows = []
    unresolved_rows = []

    for condition, excel_prefix, table_name, dataset_name, _ in CONDITIONS:
        table_path = os.path.join(table_dir, table_name)
        if not os.path.isfile(table_path):
            raise FileNotFoundError(
                f"Missing combinations table for {condition}: {table_path}"
            )
        # Load/correct the source simulations, match them to the generated table, then preprocess the matches.
        source_real, matching_real = load_source_and_matching_excel(find_source_excel(excel_prefix))
        matched, missing, unused, matching_groups = match_generated_combinations_to_simulations(
            table_path, source_real, matching_real
        )
        processed, audit, unresolved = preprocess_matched_dataframe(
            matched, condition, dataset_name, matching_groups
        )
        if USE_CANONICAL_MATCHING:
            report_slope_group_counts(processed, matching_groups, condition)
        # Guard against accidental row creation or loss anywhere in the pipeline.
        if len(processed) != len(matched) or len(matched) + len(unused) != len(source_real):
            raise ValueError(f"{condition}: row counts changed unexpectedly")

        results[condition] = (processed, missing, unused)
        audit_rows.extend(audit)
        unresolved_rows.extend(unresolved)
        print(
            f"{condition}: {len(processed)} preprocessed simulations, "
            f"{len(missing)} unmatched combinations, "
            f"{len(unresolved)} unresolved cells"
        )

    return results, audit_rows, unresolved_rows

def save_processed_outputs(results, audit_rows, unresolved_rows, output_dir: str):
    """Save the final processed datasets and diagnostic reports."""
    os.makedirs(output_dir, exist_ok=True)
    for condition, _, _, dataset_name, report_prefix in CONDITIONS:
        processed, missing, unused = results[condition]
        dataset_path = os.path.join(output_dir, dataset_name)
        processed.to_csv(dataset_path, index=False)
        missing.to_csv(os.path.join(output_dir, f"{report_prefix}_mismatch_combinations.csv"), index=False)
        unused.to_csv(os.path.join(output_dir, f"{report_prefix}_mismatch_simulations.csv"), index=False)
        print(f"Saved: {dataset_path}")

    # Store preprocessing actions separately from unresolved missing values.
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    for column in (
        "CSV row", "Previous CSV row", "Previous Tr", "Previous duration",
        "Next CSV row", "Next Tr", "Next duration", "Support count",
    ):
        audit[column] = pd.to_numeric(audit[column], errors="coerce").astype("Int64")
    audit.to_csv(os.path.join(output_dir, "preprocessing_audit.csv"), index=False)
    pd.DataFrame(unresolved_rows, columns=REMAINING_COLUMNS).to_csv(
        os.path.join(output_dir, "remaining_nans.csv"), index=False
    )

    save_mismatch_reports(results, output_dir=output_dir)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", default=DATASET_DIR)
    args = parser.parse_args()

    print(f"Selected MODE: {MODE}")
    print(f"Output directory: {OUTPUT_DIR}")
    results, audit_rows, unresolved_rows = build_processed_datasets(args.table_dir)
    save_processed_outputs(results, audit_rows, unresolved_rows, OUTPUT_DIR)

if __name__ == "__main__":
    main()
