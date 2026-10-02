from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE / "Mavoglurant_Dataset.csv"
OUTPUT_PATH = HERE / "Mavoglurant_Jinko_DataTable.csv"


data = pd.read_csv(INPUT_PATH)
observations = data.loc[(data["EVID"] == 0) & (data["MDV"] == 0)]
observations = observations.assign(logC15=np.log(observations["DV"]))

# Give each patient equal weight if a patient has duplicate values at a time point.
patient_values = observations.groupby(["TIME", "ID"], as_index=False)[
    "logC15"
].median()

jinko_table = (
    patient_values.groupby("TIME")["logC15"]
    .agg(
        value="median",
        wideRangeLowBound=lambda values: values.quantile(0.1),
        wideRangeHighBound=lambda values: values.quantile(0.9),
    )
    .reset_index()
    .rename(columns={"TIME": "time"})
)

# Jinko rejects collapsed ranges, which occur when a time point has only one
# distinct patient value. Widen those ranges by 10% around the median.
collapsed_ranges = (
    (jinko_table["wideRangeLowBound"] == jinko_table["value"])
    & (jinko_table["value"] == jinko_table["wideRangeHighBound"])
)
margin = jinko_table.loc[collapsed_ranges, "value"].abs() * 0.1
jinko_table.loc[collapsed_ranges, "wideRangeLowBound"] -= margin
jinko_table.loc[collapsed_ranges, "wideRangeHighBound"] += margin

jinko_table["time"] = jinko_table["time"].map(
    lambda hours: pd.to_timedelta(
        round(float(hours) * 3_600_000), unit="ms"
    ).isoformat()
)
jinko_table.insert(0, "obsId", "logC15")
jinko_table["armScope"] = "identity"
jinko_table = jinko_table[
    [
        "obsId",
        "time",
        "value",
        "armScope",
        "wideRangeLowBound",
        "wideRangeHighBound",
    ]
]

jinko_table.to_csv(OUTPUT_PATH, index=False)
print(f"Wrote {len(jinko_table)} rows to {OUTPUT_PATH}")
