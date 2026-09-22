"""
load_nhs_diagnosis_data.py

Parses the official NHS Digital "Hospital Admitted Patient Care Activity"
diagnosis summary file (hosp-epis-stat-admi-diag-<year>.csv) and extracts
just the eating disorder (ICD-10 F50.x) rows into a clean, wide table.

Source file structure (long format):
  UID, Code, Category, Attribute, Value
  - Code: ICD-10 code (e.g. "F50", "F50.0")
  - Category: DIAG_3_01 (3-character code) or DIAG_4_01 (4-character code)
  - Attribute: the metric name (FCE_SUM, Age_10_14_Sum, FCE_Male_Sum, etc.)
  - Value: the count

Eating disorder ICD-10 codes:
  F50   - Eating disorders (all, 3-char total)
  F50.0 - Anorexia Nervosa
  F50.1 - Atypical Anorexia Nervosa
  F50.2 - Bulimia Nervosa
  F50.3 - Atypical Bulimia Nervosa
  F50.4 - Overeating associated with other psychological disturbances
  F50.5 - Vomiting associated with other psychological disturbances
  F50.8 - Other eating disorders (includes Binge Eating Disorder in practice)
  F50.9 - Eating disorder, unspecified
"""

import pandas as pd

ED_CODE_LABELS = {
    "F50": "All Eating Disorders (total)",
    "F50.0": "Anorexia Nervosa",
    "F50.1": "Atypical Anorexia Nervosa",
    "F50.2": "Bulimia Nervosa",
    "F50.3": "Atypical Bulimia Nervosa",
    "F50.4": "Overeating (psychological)",
    "F50.5": "Vomiting (psychological)",
    "F50.8": "Other Eating Disorders (incl. Binge Eating Disorder)",
    "F50.9": "Eating Disorder, Unspecified",
}


def load_raw(path):
    df = pd.read_csv(path, low_memory=False)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df


def extract_eating_disorder_table(df, year_label="2024-25"):
    """Returns a wide table: one row per F50.x code, one column per attribute."""
    ed_codes = list(ED_CODE_LABELS.keys())
    ed_df = df[df["Code"].isin(ed_codes)].copy()

    wide = ed_df.pivot_table(
        index="Code", columns="Attribute", values="Value", aggfunc="first"
    ).reset_index()

    wide["diagnosis_label"] = wide["Code"].map(ED_CODE_LABELS)
    wide["financial_year"] = year_label

    # put label columns first
    cols = ["financial_year", "Code", "diagnosis_label"] + [
        c for c in wide.columns if c not in ("financial_year", "Code", "diagnosis_label")
    ]
    return wide[cols]


def extract_age_breakdown(wide_df):
    """Long-format age breakdown for the F50.x subtype rows (excludes the F50 total)."""
    age_cols = [c for c in wide_df.columns if c.startswith("Age_")]
    subtypes = wide_df[wide_df["Code"] != "F50"]
    long = subtypes.melt(
        id_vars=["Code", "diagnosis_label", "financial_year"],
        value_vars=age_cols,
        var_name="age_band",
        value_name="count",
    )
    long["age_band"] = (
        long["age_band"].str.replace("Age_", "", regex=False)
        .str.replace("_Sum", "", regex=False)
        .str.replace("_", "-", regex=False)
    )
    return long


def main():
    df = load_raw("data/raw/hosp-epis-stat-admi-diag-2024-25.csv")
    wide = extract_eating_disorder_table(df, year_label="2024-25")

    wide.to_csv("data/processed/nhs_ed_subtypes_2024-25.csv", index=False)
    print(f"Saved eating disorder subtype table -> data/processed/nhs_ed_subtypes_2024-25.csv")
    print(wide[["Code", "diagnosis_label", "FCE_SUM", "FAE_SUM", "FCE_Male_Sum", "FCE_Female_Sum"]])

    age_long = extract_age_breakdown(wide)
    age_long.to_csv("data/processed/nhs_ed_age_breakdown_2024-25.csv", index=False)
    print(f"\nSaved age breakdown -> data/processed/nhs_ed_age_breakdown_2024-25.csv")


if __name__ == "__main__":
    main()
