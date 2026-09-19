from itertools import product
import os
import numpy as np
import pandas as pd

from config import BASE_DIR, DATASET_DIR

# ============================================================
# COLUMNS
# ============================================================
COLUMNS = {
    "Unit weight [kN/m3]": "γ",
    "Effective cohesion [kPa]": "c'",
    "Effective friction angle [°]": "φ'",
    "Undrained Shear Strength [kPa]": "Cu",
    "Saturated permeability [m/s]": "ksat",
    "Soil Type [-]": "ST",
    "Slope angle [°]": "α",
    "Slope length [m]": "L",
    "Total length [m]": "B",
    "Slope height [m]": "H",
    "Total height Upstream [m]": "hm",
    "Total height Downstream [m]": "hd",
    "Soil depth upstream [m]": "hSu",
    "Soil depth downstream [m]": "hSd",
    "Bedrock depth upstream [m]": "hBu",
    "Bedrock depth downstream [m]": "hBd",
    "Initial piezometric surface depth - upstream [m]": "zwuinit",
    "Initial piezometric surface depth - downstream [m]": "zwdinit",
    "Return period of precipitation [years]": "Tr",
    "Accumulated precipitation [mm]": "hw",
    "Precipitation duration [hrs]": "tw",
    "Factor of safety [-]": "FoS",
    "Depth of slip surface [m]": "zs",
    "Final Piezometric surface depth - upstream [m]": "zwufinal",
    "Final Piezometric surface depth - downstream [m]": "zwdfinal",
}

# ============================================================
# TABLE 1a
# D.1 and UN geometric parameters
# ============================================================
TABLE_1A = [
    # α, L, B, H, hu, hd, hSu values, hSd values
    (20, 20, 100, 7.3, 25, 17.7, [22.5, 6.3, 7.3, 2], [15.2, 4.4, 0, 2]),
    (20, 40, 200, 14.6, 45, 30.4, [40.5, 11.3, 14.6, 2], [25.9, 7.6, 0, 2]),
    (20, 80, 400, 29.1, 90, 60.9, [81, 22.5, 29.1, 2], [51.9, 15.2, 0, 2]),
    (30, 20, 100, 11.5, 35, 23.5, [31.5, 8.8, 11.5, 2], [20, 5.9, 0, 2]),
    (30, 40, 200, 23.1, 70, 46.9, [63, 17.5, 23.1, 2], [39.9, 11.7, 0, 2]),
    (30, 80, 400, 46.2, 140, 93.8, [126, 35, 46.2, 2], [79.8, 23.5, 0, 2]),
    (40, 20, 100, 16.8, 50, 33.2, [45, 12.5, 16.8, 2], [28.2, 8.3, 0, 2]),
    (40, 40, 200, 33.6, 100, 66.4, [90, 25, 33.6, 2], [56.4, 16.6, 0, 2]),
    (40, 80, 400, 67.1, 200, 132.9, [180, 50, 67.1, 2], [112.9, 33.2, 0, 2]),
    (50, 20, 100, 23.8, 70, 46.2, [63, 17.5, 23.8, 2], [39.2, 11.5, 0, 2]),
    (50, 40, 200, 47.7, 145, 97.3, [130.5, 36.3, 47.7, 2], [82.8, 24.3, 0, 2]),
    (50, 80, 400, 95.3, 290, 194.7, [261, 72.5, 95.3, 2], [165.7, 48.7, 0, 2]),
]

def get_combinations_1a_geometries(condition):
    geometries = []
    # Table description:
    # D.1 -> 48 combinations = 12 rows x 4 configurations
    # UN  -> 36 combinations = 12 rows x 3 configurations
    n_configs = 4 if condition == "D.1" else 3

    for alpha, L, B, H, hm, hd, hsu_values, hsd_values in TABLE_1A:
        for i in range(n_configs):
            hSu = hsu_values[i]
            hSd = hsd_values[i]
            geometries.append({
                "Slope angle [°]": alpha,
                "Slope length [m]": L,
                "Total length [m]": B,
                "Slope height [m]": H,
                "Total height Upstream [m]": hm,
                "Total height Downstream [m]": hd,
                "Soil depth upstream [m]": hSu,
                "Soil depth downstream [m]": hSd,
                # Derived from the geometry
                "Bedrock depth upstream [m]": round(hm - hSu, 1),
                "Bedrock depth downstream [m]": round(hd - hSd, 1),
            })
    return geometries

# ============================================================
# TABLE 1b
# D.2 geometric parameters
# ============================================================
TABLE_1B = [
    # α, L, B, H, hBu, hBd, hSu, hSd
    (30, 40, 100, 23.1, 10, 10, 35, 11.9),
    (30, 40, 100, 23.1, 10, 10, 45, 21.9),
    (30, 40, 100, 23.1, 10, 10, 55, 31.9),
    (30, 80, 160, 46.2, 10, 10, 60, 13.8),
    (30, 80, 160, 46.2, 10, 10, 70, 23.8),
    (40, 20, 60, 16.8, 10, 10, 25, 8.2),
    (40, 20, 60, 16.8, 10, 10, 30, 13.2),
    (40, 20, 60, 16.8, 10, 10, 35, 18.2),
    # The D.2 workbook stores these three cases with H=16.78 m
    # and the corresponding downstream soil depths.  Keep the
    # one-decimal table precision here; matching handles the 0.02 m
    # difference from the workbook values.
    (45, 20, 60, 16.8, 10, 10, 30, 13.2),
    (45, 20, 60, 16.8, 10, 10, 35, 18.2),
    (45, 20, 60, 16.8, 10, 10, 40, 23.2),
]

def get_d2_geometries():
    geometries = []
    for alpha, L, B, H, hBu, hBd, hSu, hSd in TABLE_1B:
        geometries.append({
            "Slope angle [°]": alpha,
            "Slope length [m]": L,
            "Total length [m]": B,
            "Slope height [m]": H,
            "Total height Upstream [m]": round(hSu + hBu, 1),
             "Total height Downstream [m]": round(hSd + hBd, 1),
            "Soil depth upstream [m]": hSu,
            "Soil depth downstream [m]": hSd,
            "Bedrock depth upstream [m]": hBu,
            "Bedrock depth downstream [m]": hBd,
        })
    return geometries

# ============================================================
# TABLE 2a
# Mechanical parameters
#
# The number of combinations reported in the paper is obtained
# by changing one parameter at a time around the mean values.
# ============================================================
def one_factor_at_a_time(means, values):
    combinations = [means.copy()]
    for parameter, parameter_values in values.items():
        for value in parameter_values:
            # Mean combination already inserted
            if value == means[parameter]:
                continue
            row = means.copy()
            row[parameter] = value
            combinations.append(row)
    return combinations

def get_mechanical_combinations(condition):
    if condition == "D.1":
        means = {"Unit weight [kN/m3]": 18, 
                 "Effective cohesion [kPa]": 20,
                 "Effective friction angle [°]": 25,
                 "Undrained Shear Strength [kPa]": np.nan}
        values = {"Effective cohesion [kPa]": [0, 10, 20, 30, 40],
                  "Effective friction angle [°]": [5, 15, 25, 35, 45],
                  "Unit weight [kN/m3]": [12, 15, 18, 21, 24]}
    elif condition == "D.2":
        means = {"Unit weight [kN/m3]": 15,
                 "Effective cohesion [kPa]": 20,
                 "Effective friction angle [°]": 25,
                 "Undrained Shear Strength [kPa]": np.nan}
        values = {"Effective cohesion [kPa]": list(range(0, 41, 5)),
                  "Effective friction angle [°]": list(range(0, 51, 5)),
                  "Unit weight [kN/m3]": list(range(9, 22, 2))}
    elif condition == "UN":
        means = {"Unit weight [kN/m3]": 18,
                 "Effective cohesion [kPa]": np.nan,
                 "Effective friction angle [°]": np.nan,
                 "Undrained Shear Strength [kPa]": 175}
        values = {"Undrained Shear Strength [kPa]": [25, 75, 125, 175, 225, 275, 325],
                  "Unit weight [kN/m3]": [12, 15, 18, 21, 24]}
    else:
        raise ValueError(condition)
    return one_factor_at_a_time(means, values)

# ============================================================
# TABLE 2b
# Hydraulic parameters
# ============================================================
HYDRAULIC = {
    "D.1": [{"Soil Type [-]": 1, "Saturated permeability [m/s]": 1e-4},
            {"Soil Type [-]": 2, "Saturated permeability [m/s]": 1e-6},
            {"Soil Type [-]": 3, "Saturated permeability [m/s]": 1e-8}
    ],
    "D.2": [{"Soil Type [-]": 1, "Saturated permeability [m/s]": 1e-3},
            {"Soil Type [-]": 2, "Saturated permeability [m/s]": 1e-5},
            {"Soil Type [-]": 3, "Saturated permeability [m/s]": 1e-7}
    ],
    "UN": [{"Soil Type [-]": 3, "Saturated permeability [m/s]": 1e-8}],
}

# ============================================================
# TABLE 3
# Rainfall events
# ============================================================
NO_RAIN_EVENT = {"Return period of precipitation [years]": -1,
                 "Accumulated precipitation [mm]": 0,
                 "Precipitation duration [hrs]": 0}

RAINFALL = {
    "D.1": [
        NO_RAIN_EVENT.copy(),
        {"Return period of precipitation [years]": 30,
         "Accumulated precipitation [mm]": 315.90,
         "Precipitation duration [hrs]": 100
        },
        {"Return period of precipitation [years]": 200,
         "Accumulated precipitation [mm]": 480.50,
         "Precipitation duration [hrs]": 100
        },
        {"Return period of precipitation [years]": 500,
         "Accumulated precipitation [mm]": 586.00,
         "Precipitation duration [hrs]": 100
        },
    ],
    "UN": [
        NO_RAIN_EVENT.copy(),
        {"Return period of precipitation [years]": 30,
         "Accumulated precipitation [mm]": 315.90,
         "Precipitation duration [hrs]": 100
        },
        {"Return period of precipitation [years]": 200,
         "Accumulated precipitation [mm]": 480.50,
         "Precipitation duration [hrs]": 100
        },
        {"Return period of precipitation [years]": 500,
         "Accumulated precipitation [mm]": 586.00,
         "Precipitation duration [hrs]": 100
        },
    ],
    "D.2": [
        # Reference condition without precipitation
        NO_RAIN_EVENT.copy(),
        # 15 h
        {"Return period of precipitation [years]": 30,
         "Accumulated precipitation [mm]": 125.18,
         "Precipitation duration [hrs]": 15
        },
        {"Return period of precipitation [years]": 200,
         "Accumulated precipitation [mm]": 191.19,
         "Precipitation duration [hrs]": 15
        },
        {"Return period of precipitation [years]": 500,
         "Accumulated precipitation [mm]": 232.16,
         "Precipitation duration [hrs]": 15
        },
        # 30 h
        {"Return period of precipitation [years]": 30,
         "Accumulated precipitation [mm]": 200.29,
         "Precipitation duration [hrs]": 30
        },
        {"Return period of precipitation [years]": 200,
         "Accumulated precipitation [mm]": 305.90,
         "Precipitation duration [hrs]": 30
        },
        {"Return period of precipitation [years]": 500,
         "Accumulated precipitation [mm]": 371.45,
         "Precipitation duration [hrs]": 30
        },
    ],
}

# ============================================================
# INITIAL WATER TABLE
#
# high:
#   zwuinit = 0
#   zwdinit = 0
#
# low:
#   zwuinit = hSu
#   zwdinit = hSd
#
# intermediate:
#   unknown -> "*"
# ============================================================
def get_water_combinations(geometry, condition):
    high = {"Initial piezometric surface depth - upstream [m]": 0,
            "Initial piezometric surface depth - downstream [m]": 0}
    low = {"Initial piezometric surface depth - upstream [m]": geometry["Soil depth upstream [m]"],
           "Initial piezometric surface depth - downstream [m]": geometry["Soil depth downstream [m]"]}
    intermediate = {"Initial piezometric surface depth - upstream [m]": "*",
                    "Initial piezometric surface depth - downstream [m]": "*"}
    
    # D.1 and UN use high, low and intermediate
    if condition in ["D.1", "UN"]:
        return [high, low, intermediate]
    
    # D.2 only uses zw3init = intermediate
    if condition == "D.2":
        return [intermediate]
    raise ValueError(condition)

# ============================================================
# GENERATE TABLE
# ============================================================
def generate_combinations(condition):
    if condition == "D.1":
        geometries = get_combinations_1a_geometries("D.1")
    elif condition == "D.2":
        geometries = get_d2_geometries()
    elif condition == "UN":
        geometries = get_combinations_1a_geometries("UN")
    else:
        raise ValueError(condition)

    mechanical = get_mechanical_combinations(condition)
    hydraulic = HYDRAULIC[condition]
    rainfall = RAINFALL[condition]
    rows = []

    for geometry in geometries:
        water_combinations = get_water_combinations(geometry, condition)
        for soil, mech, water, rain in product(hydraulic, mechanical, water_combinations, rainfall):
            row = {column: np.nan for column in COLUMNS.keys()}
            row.update(geometry)
            row.update(mech)
            row.update(soil)
            row.update(water)
            row.update(rain)
            rows.append(row)

    return pd.DataFrame(rows, columns=COLUMNS.keys())

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)
    D1 = generate_combinations("D.1")
    D2 = generate_combinations("D.2")
    U = generate_combinations("UN")
    D1_rain_only = D1[D1["Return period of precipitation [years]"].isin([30, 200, 500])].copy()

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------
    assert len(get_combinations_1a_geometries("D.1")) == 48
    assert len(get_combinations_1a_geometries("UN")) == 36
    assert len(get_d2_geometries()) == 11

    assert len(get_mechanical_combinations("D.1")) == 13
    assert len(get_mechanical_combinations("D.2")) == 25
    assert len(get_mechanical_combinations("UN")) == 11
    # assert len(D1_rain_only) == 16848

    assert set(D1["Return period of precipitation [years]"]) == {-1, 30, 200, 500}
    assert set(D2["Return period of precipitation [years]"]) == {-1, 30, 200, 500}
    assert set(U["Return period of precipitation [years]"]) == {-1, 30, 200, 500}

    for table in (D1, D2, U):
        no_rain = table[table["Return period of precipitation [years]"] == -1]
        assert (no_rain["Accumulated precipitation [mm]"] == 0).all()
        assert (no_rain["Precipitation duration [hrs]"] == 0).all()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    D1.to_csv(os.path.join(DATASET_DIR, "D.1_combinations.csv"), index=False)
    # D1_rain_only.to_csv(os.path.join(DATASET_DIR, "D.1_combinations_TR_30_200_500.csv"), index=False)
    D2.to_csv(os.path.join(DATASET_DIR, "D.2_combinations.csv"), index=False)
    U.to_csv(os.path.join(DATASET_DIR, "U_combinations.csv"), index=False)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("Tables generated:")
    print(f"D.1_combinations.csv : {len(D1)} rows")
    # print(f"D.1_combinations_TR_30_200_500.csv : {len(D1_rain_only)} rows")
    print(f"D.2_combinations.csv : {len(D2)} rows")
    print(f"U_combinations.csv   : {len(U)} rows")