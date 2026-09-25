"""
load_to_mysql.py

Loads the processed CSVs into the MySQL database created by schema.sql.

Requires: pip install mysql-connector-python pandas

Set your connection details via environment variables (don't hardcode
credentials in code):
    MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE (defaults to uk_ed_analytics)

Usage:
    export MYSQL_HOST=localhost
    export MYSQL_USER=root
    export MYSQL_PASSWORD=yourpassword
    python load_to_mysql.py
"""

import os
import pandas as pd
import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", ""),
    "database": os.environ.get("MYSQL_DATABASE", "uk_ed_analytics"),
}

DATA_DIR = "data/processed"


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def load_eat26(cursor):
    df = pd.read_csv(f"{DATA_DIR}/eat26_synthetic.csv")
    rows = df[[
        "age", "gender", "bmi_category", "dieting_score",
        "bulimia_food_preoccupation_score", "oral_control_score",
        "total_eat26_score", "risk_label"
    ]].values.tolist()

    cursor.executemany(
        """INSERT INTO eat26_synthetic
           (age, gender, bmi_category, dieting_score,
            bulimia_food_preoccupation_score, oral_control_score,
            total_eat26_score, risk_label)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        rows,
    )
    print(f"Loaded {len(rows)} rows -> eat26_synthetic")


def load_nhs_subtypes(cursor):
    df = pd.read_csv(f"{DATA_DIR}/nhs_ed_subtypes_all_years.csv")
    df = df.where(pd.notnull(df), None)
    rows = df[[
        "financial_year", "Code", "diagnosis_label",
        "FCE_SUM", "FAE_SUM", "FCE_Male_Sum", "FCE_Female_Sum"
    ]].values.tolist()

    cursor.executemany(
        """INSERT INTO nhs_ed_subtypes
           (financial_year, diag_code, diagnosis_label,
            fce_sum, fae_sum, fce_male_sum, fce_female_sum)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        rows,
    )
    print(f"Loaded {len(rows)} rows -> nhs_ed_subtypes")


def load_age_breakdown(cursor):
    df = pd.read_csv(f"{DATA_DIR}/nhs_ed_age_breakdown_2024-25.csv")
    df = df.where(pd.notnull(df), None)
    rows = df[["Code", "diagnosis_label", "financial_year", "age_band", "count"]].values.tolist()

    cursor.executemany(
        """INSERT INTO nhs_ed_age_breakdown
           (diag_code, diagnosis_label, financial_year, age_band, episode_count)
           VALUES (%s, %s, %s, %s, %s)""",
        rows,
    )
    print(f"Loaded {len(rows)} rows -> nhs_ed_age_breakdown")


def load_yearly_series(cursor):
    df = pd.read_csv(f"{DATA_DIR}/f50_yearly_series.csv")
    rows = df[["financial_year", "year_end", "admissions"]].values.tolist()

    cursor.executemany(
        """INSERT INTO ed_yearly_admissions (financial_year, year_end, admissions)
           VALUES (%s, %s, %s)""",
        rows,
    )
    print(f"Loaded {len(rows)} rows -> ed_yearly_admissions")


def load_mortality(cursor):
    df = pd.read_csv(f"{DATA_DIR}/ons_ed_mortality_2024.csv")
    df = df[df["region"] != "Column Total"]
    rows = df[["region", "all_cause_deaths", "ed_deaths", "ed_death_rate_per_100k_allcause"]].values.tolist()

    cursor.executemany(
        """INSERT INTO ons_ed_mortality
           (region, all_cause_deaths, ed_deaths, ed_death_rate_per_100k_allcause)
           VALUES (%s, %s, %s, %s)""",
        rows,
    )
    print(f"Loaded {len(rows)} rows -> ons_ed_mortality")


def main():
    try:
        conn = get_connection()
    except Error as e:
        print(f"Connection failed: {e}")
        print("Check MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD env vars, and that schema.sql has been run.")
        return

    cursor = conn.cursor()

    load_eat26(cursor)
    load_nhs_subtypes(cursor)
    load_age_breakdown(cursor)
    load_yearly_series(cursor)
    load_mortality(cursor)

    conn.commit()
    cursor.close()
    conn.close()
    print("\nAll data loaded and committed.")


if __name__ == "__main__":
    main()
