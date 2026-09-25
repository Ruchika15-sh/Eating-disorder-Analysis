import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import Holt

DATA_PATH = "data/processed/f50_yearly_series.csv"
FIG_DIR = "reports/figures"
FORECAST_YEARS = 3


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("year_end").reset_index(drop=True)
    return df


def linear_trend_forecast(df, n_future):
    X = df[["year_end"]].values
    y = df["admissions"].values

    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(df["year_end"].max() + 1, df["year_end"].max() + 1 + n_future)
    future_X = future_years.reshape(-1, 1)
    forecast = model.predict(future_X)

    return future_years, forecast, model


def holt_forecast(df, n_future):
    series = pd.Series(df["admissions"].values)
    model = Holt(series, initialization_method="estimated").fit(
        optimized=True
    )
    forecast = model.forecast(n_future)
    future_years = np.arange(df["year_end"].max() + 1, df["year_end"].max() + 1 + n_future)
    return future_years, forecast.values, model


def leave_last_out_eval(df):
    """Quick honesty-check: how well would each method have predicted
    the most recent known year, using only the years before it?"""
    train = df.iloc[:-1]
    actual_last = df.iloc[-1]["admissions"]

    # linear trend
    _, lt_pred, _ = linear_trend_forecast(train, 1)
    lt_error = abs(lt_pred[0] - actual_last)

    # holt
    _, holt_pred, _ = holt_forecast(train, 1)
    holt_error = abs(holt_pred[0] - actual_last)

    print("=== Leave-last-year-out check (predicting last known year from prior years) ===")
    print(f"Actual {df.iloc[-1]['financial_year']}: {actual_last:,.0f}")
    print(f"Linear trend predicted: {lt_pred[0]:,.0f}  (abs error: {lt_error:,.0f})")
    print(f"Holt's linear predicted: {holt_pred[0]:,.0f}  (abs error: {holt_error:,.0f})")
    print()


def plot_forecast(df, lt_years, lt_forecast, holt_years, holt_forecast_vals):
    plt.figure(figsize=(9, 5.5))
    plt.plot(df["year_end"], df["admissions"], marker="o", label="Actual (NHS Digital, reported)", color="#1f4e79", linewidth=2)
    plt.plot(lt_years, lt_forecast, marker="s", linestyle="--", label="Linear trend forecast", color="#d97706")
    plt.plot(holt_years, holt_forecast_vals, marker="^", linestyle="--", label="Holt's linear forecast", color="#059669")

    # connect last actual point to forecasts
    last_year, last_val = df["year_end"].iloc[-1], df["admissions"].iloc[-1]
    plt.plot([last_year, lt_years[0]], [last_val, lt_forecast[0]], linestyle=":", color="#d97706", alpha=0.6)
    plt.plot([last_year, holt_years[0]], [last_val, holt_forecast_vals[0]], linestyle=":", color="#059669", alpha=0.6)

    plt.title("England Eating Disorder Hospital Admissions — Actual & Forecast")
    plt.xlabel("Year (FY ending)")
    plt.ylabel("Admissions (primary/secondary diagnosis)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/admissions_forecast.png", dpi=150)
    plt.close()
    print(f"Saved forecast plot to {FIG_DIR}/admissions_forecast.png")


def main():
    df = load_data()
    print("Loaded admissions series:")
    print(df[["financial_year", "admissions"]].to_string(index=False))
    print()

    leave_last_out_eval(df)

    lt_years, lt_forecast, _ = linear_trend_forecast(df, FORECAST_YEARS)
    holt_years, holt_forecast_vals, _ = holt_forecast(df, FORECAST_YEARS)

    print("=== Forecast (next 3 years) ===")
    for y, lt, h in zip(lt_years, lt_forecast, holt_forecast_vals):
        print(f"FY ending {y}: linear trend = {lt:,.0f} | Holt's linear = {h:,.0f}")

    plot_forecast(df, lt_years, lt_forecast, holt_years, holt_forecast_vals)


if __name__ == "__main__":
    main()
