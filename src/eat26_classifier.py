"""
eat26_classifier.py

Trains a multi-class classifier to predict a "risk pattern" label from
EAT-26 subscale scores + basic demographics, on the synthetic dataset
produced by generate_eat26_dataset.py.

IMPORTANT: this is a screening-pattern demo, not a diagnostic tool. The
labels represent broad symptom *patterns* documented in EAT-26 literature,
not clinical diagnoses, and the data itself is synthetic.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score
)
import joblib

DATA_PATH = "data/processed/eat26_synthetic.csv"
FIG_DIR = "reports/figures"


def load_data():
    return pd.read_csv(DATA_PATH)


def eda(df):
    sns.set_style("whitegrid")

    # 1. Class distribution
    plt.figure(figsize=(7, 4))
    order = df["risk_label"].value_counts().index
    sns.countplot(data=df, y="risk_label", order=order, hue="risk_label",
                   palette="viridis", legend=False)
    plt.title("Risk Pattern Class Distribution (Synthetic EAT-26 Dataset)")
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/class_distribution.png", dpi=150)
    plt.close()

    # 2. Subscale score distributions by class
    subscales = ["dieting_score", "bulimia_food_preoccupation_score", "oral_control_score"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, subscales):
        sns.boxplot(data=df, x="risk_label", y=col, ax=ax, hue="risk_label",
                    palette="viridis", legend=False)
        ax.set_title(col.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/subscale_distributions.png", dpi=150)
    plt.close()

    # 3. BMI category vs risk label
    plt.figure(figsize=(8, 4.5))
    ct = pd.crosstab(df["bmi_category"], df["risk_label"], normalize="index")
    ct = ct[df["risk_label"].value_counts().index]
    ct.plot(kind="bar", stacked=True, colormap="viridis", ax=plt.gca())
    plt.title("Risk Pattern Composition by BMI Category")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/bmi_vs_risklabel.png", dpi=150)
    plt.close()

    print(f"Saved 3 EDA figures to {FIG_DIR}/")


def build_pipeline(model_name="random_forest"):
    numeric_features = [
        "age", "dieting_score", "bulimia_food_preoccupation_score",
        "oral_control_score", "total_eat26_score"
    ]
    categorical_features = ["gender", "bmi_category"]

    if model_name == "random_forest":
        preprocessor = ColumnTransformer(transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ])
        model = RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=42,
            class_weight="balanced"
        )
    else:
        # Logistic Regression benefits from scaled numeric features
        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ])
        model = LogisticRegression(max_iter=2000, class_weight="balanced")

    return Pipeline([
        ("preprocess", preprocessor),
        ("classifier", model),
    ])


def train_and_evaluate(df, model_name="random_forest"):
    feature_cols = [
        "age", "gender", "bmi_category", "dieting_score",
        "bulimia_food_preoccupation_score", "oral_control_score",
        "total_eat26_score"
    ]
    X = df[feature_cols]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(model_name)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {f1_macro:.3f}\n")
    print(classification_report(y_test, y_pred))

    # Confusion matrix plot
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrix_{model_name}.png", dpi=150)
    plt.close()

    return pipe, acc, f1_macro


def feature_importance(pipe, df):
    """Only meaningful for the random forest model."""
    model = pipe.named_steps["classifier"]
    if not hasattr(model, "feature_importances_"):
        return

    ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(["gender", "bmi_category"])
    num_names = ["age", "dieting_score", "bulimia_food_preoccupation_score",
                 "oral_control_score", "total_eat26_score"]
    all_names = list(num_names) + list(cat_names)

    importances = pd.Series(model.feature_importances_, index=all_names)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(7, 6))
    importances.plot(kind="barh", color="teal")
    plt.title("Feature Importance — Random Forest")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {FIG_DIR}/feature_importance.png")


def main():
    df = load_data()
    eda(df)

    rf_pipe, rf_acc, rf_f1 = train_and_evaluate(df, "random_forest")
    lr_pipe, lr_acc, lr_f1 = train_and_evaluate(df, "logistic_regression")

    feature_importance(rf_pipe, df)

    # Save the better-performing model
    best_pipe, best_name = (rf_pipe, "random_forest") if rf_f1 >= lr_f1 else (lr_pipe, "logistic_regression")
    joblib.dump(best_pipe, "src/eat26_model.joblib")
    print(f"\nBest model ({best_name}) saved to src/eat26_model.joblib")

    print("\nModel comparison:")
    print(f"  Random Forest       -> Accuracy: {rf_acc:.3f} | Macro F1: {rf_f1:.3f}")
    print(f"  Logistic Regression -> Accuracy: {lr_acc:.3f} | Macro F1: {lr_f1:.3f}")


if __name__ == "__main__":
    main()
