import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = "reports/figures"
sns.set_style("whitegrid")


def load():
    subtypes = pd.read_csv("data/processed/nhs_ed_subtypes_2024-25.csv")
    age = pd.read_csv("data/processed/nhs_ed_age_breakdown_2024-25.csv")
    return subtypes, age


def plot_subtype_admissions(subtypes):
    d = subtypes[subtypes["Code"] != "F50"].sort_values("FCE_SUM", ascending=True)

    plt.figure(figsize=(9, 5))
    bars = plt.barh(d["diagnosis_label"], d["FCE_SUM"], color=sns.color_palette("viridis", len(d)))
    plt.title("England: Hospital Episodes by Eating Disorder Subtype, 2024-25")
    plt.xlabel("Finished Consultant Episodes (FCEs)")
    for i, v in enumerate(d["FCE_SUM"]):
        plt.text(v + 30, i, f"{int(v):,}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/nhs_subtype_admissions.png", dpi=150)
    plt.close()
    print("Saved nhs_subtype_admissions.png")


def plot_gender_split(subtypes):
    d = subtypes[subtypes["Code"] != "F50"].copy()
    d = d.sort_values("FCE_SUM", ascending=False)

    male_pct = d["FCE_Male_Sum"] / d["FCE_SUM"] * 100
    female_pct = d["FCE_Female_Sum"] / d["FCE_SUM"] * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(d["diagnosis_label"], female_pct, label="Female", color="#c2185b")
    ax.barh(d["diagnosis_label"], male_pct, left=female_pct, label="Male", color="#1976d2")
    ax.set_xlabel("% of episodes")
    ax.set_title("England: Gender Split by Eating Disorder Subtype, 2024-25")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/nhs_gender_split.png", dpi=150)
    plt.close()
    print("Saved nhs_gender_split.png")


def plot_age_pattern(age_long):
    # order age bands sensibly
    order = [
        "0", "1-4", "5-9", "10-14", "15", "16", "17", "18", "19",
        "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
        "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-120"
    ]
    age_long["age_band"] = pd.Categorical(age_long["age_band"], categories=order, ordered=True)

    # focus on the 3 largest subtypes for readability
    top_codes = ["F50.0", "F50.9", "F50.8"]
    d = age_long[age_long["Code"].isin(top_codes)]

    plt.figure(figsize=(12, 5.5))
    sns.lineplot(data=d, x="age_band", y="count", hue="diagnosis_label", marker="o")
    plt.xticks(rotation=60)
    plt.title("England: Age Distribution by Eating Disorder Subtype, 2024-25\n(Anorexia Nervosa, Other, Unspecified)")
    plt.ylabel("Episodes")
    plt.xlabel("Age band")
    plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/nhs_age_pattern.png", dpi=150)
    plt.close()
    print("Saved nhs_age_pattern.png")


def summary_stats(subtypes):
    d = subtypes[subtypes["Code"] != "F50"].copy()
    total = subtypes.loc[subtypes["Code"] == "F50", "FCE_SUM"].values[0]

    print("\n=== England Eating Disorder Hospital Episodes, 2024-25 ===")
    print(f"Total episodes (all subtypes): {total:,.0f}\n")

    d = d.sort_values("FCE_SUM", ascending=False)
    for _, row in d.iterrows():
        pct = row["FCE_SUM"] / total * 100
        pct_female = row["FCE_Female_Sum"] / row["FCE_SUM"] * 100
        print(f"  {row['diagnosis_label']:<52} {row['FCE_SUM']:>7,.0f}  ({pct:4.1f}% of total, {pct_female:4.1f}% female)")

    print(f"\nDay case rate overall: {d['FCE_DAY_CASES'].sum() / d['FCE_SUM'].sum() * 100:.1f}%")
    print(f"Emergency admission share: {d['FAE_EMERGENCY_Sum'].sum() / d['FAE_SUM'].sum() * 100:.1f}%")


def main():
    subtypes, age = load()
    summary_stats(subtypes)
    plot_subtype_admissions(subtypes)
    plot_gender_split(subtypes)
    plot_age_pattern(age)


if __name__ == "__main__":
    main()
