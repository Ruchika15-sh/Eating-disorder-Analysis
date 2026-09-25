import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = "reports/figures"


def load():
    df = pd.read_excel(
        "data/raw/NOMIS-newfile.xlsx",
        sheet_name="Sheet1",
        skiprows=8,
        header=0,
    )
    df.columns = ["region", "all_cause_deaths", "ed_deaths"]
    df = df[df["region"] != "Column Total"].dropna(subset=["region"])
    df["ed_death_rate_per_100k_allcause"] = df["ed_deaths"] / df["all_cause_deaths"] * 100000
    return df


def plot_by_region(df):
    d = df[df["region"] != "Unknown or Abroad"].sort_values("ed_deaths", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(d["region"], d["ed_deaths"], color=sns.color_palette("viridis", len(d)))
    plt.title("England & Wales: Eating Disorder Deaths (F50) by Region, 2024")
    plt.xlabel("Deaths")
    for i, v in enumerate(d["ed_deaths"]):
        plt.text(v + 0.1, i, str(int(v)), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/ons_mortality_by_region.png", dpi=150)
    plt.close()
    print("Saved ons_mortality_by_region.png")


def main():
    df = load()
    df.to_csv("data/processed/ons_ed_mortality_2024.csv", index=False)
    print(df.to_string(index=False))
    plot_by_region(df)


if __name__ == "__main__":
    main()
