import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_PER_CLASS = {
    "Low Risk": 1400,
    "Restrictive (AN-type)": 500,
    "Bulimic (BN-type)": 450,
    "Binge (BED-type)": 650,
}

# subscale item counts (approximate EAT-26 structure)
DIETING_MAX = 39       # 13 items x max 3
BULIMIA_MAX = 18       # 6 items x max 3
ORAL_CONTROL_MAX = 21  # 7 items x max 3


def clip(arr, lo, hi):
    return np.clip(arr, lo, hi).round().astype(int)


def generate_class(label, n):
    if label == "Low Risk":
        dieting = clip(RNG.normal(5, 3, n), 0, DIETING_MAX)
        bulimia = clip(RNG.normal(1.5, 1.5, n), 0, BULIMIA_MAX)
        oral_control = clip(RNG.normal(4, 3, n), 0, ORAL_CONTROL_MAX)
        age = clip(RNG.normal(22, 5, n), 13, 45)
        bmi = RNG.choice(
            ["Underweight", "Normal", "Overweight", "Obese"],
            n, p=[0.05, 0.70, 0.18, 0.07]
        )
        gender = RNG.choice(["Female", "Male"], n, p=[0.65, 0.35])

    elif label == "Restrictive (AN-type)":
        dieting = clip(RNG.normal(27, 6, n), 5, DIETING_MAX)
        bulimia = clip(RNG.normal(4, 3, n), 0, BULIMIA_MAX)
        oral_control = clip(RNG.normal(11, 4, n), 0, ORAL_CONTROL_MAX)
        age = clip(RNG.normal(19, 4, n), 13, 40)
        bmi = RNG.choice(
            ["Underweight", "Normal", "Overweight", "Obese"],
            n, p=[0.55, 0.40, 0.04, 0.01]
        )
        gender = RNG.choice(["Female", "Male"], n, p=[0.88, 0.12])

    elif label == "Bulimic (BN-type)":
        dieting = clip(RNG.normal(20, 6, n), 3, DIETING_MAX)
        bulimia = clip(RNG.normal(12, 3, n), 3, BULIMIA_MAX)
        oral_control = clip(RNG.normal(6, 3, n), 0, ORAL_CONTROL_MAX)
        age = clip(RNG.normal(23, 5, n), 14, 45)
        bmi = RNG.choice(
            ["Underweight", "Normal", "Overweight", "Obese"],
            n, p=[0.10, 0.60, 0.22, 0.08]
        )
        gender = RNG.choice(["Female", "Male"], n, p=[0.85, 0.15])

    elif label == "Binge (BED-type)":
        dieting = clip(RNG.normal(12, 5, n), 0, DIETING_MAX)
        bulimia = clip(RNG.normal(9, 3, n), 2, BULIMIA_MAX)
        oral_control = clip(RNG.normal(3, 2, n), 0, ORAL_CONTROL_MAX)
        age = clip(RNG.normal(28, 7, n), 16, 50)
        bmi = RNG.choice(
            ["Underweight", "Normal", "Overweight", "Obese"],
            n, p=[0.01, 0.24, 0.35, 0.40]
        )
        gender = RNG.choice(["Female", "Male"], n, p=[0.62, 0.38])

    total = dieting + bulimia + oral_control

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "bmi_category": bmi,
        "dieting_score": dieting,
        "bulimia_food_preoccupation_score": bulimia,
        "oral_control_score": oral_control,
        "total_eat26_score": total,
        "risk_label": label,
    })


def main():
    frames = [generate_class(label, n) for label, n in N_PER_CLASS.items()]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    out_path = "data/processed/eat26_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic rows -> {out_path}")
    print(df["risk_label"].value_counts())
    print(df.head())


if __name__ == "__main__":
    main()
