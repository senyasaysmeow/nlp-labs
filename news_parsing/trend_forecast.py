import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import os
from pathlib import Path

script_dir = Path(__file__).parent
dir_path = script_dir / "analysis/"
os.makedirs(dir_path, exist_ok=True)


def MNK(S0, poly_order=2):
    n = len(S0)
    x = np.arange(n, dtype=float)
    # np.polyfit повертає коефіцієнти від старшого до молодшого
    coeffs_desc = np.polyfit(x, S0, poly_order)
    Yout = np.polyval(coeffs_desc, x).reshape(-1, 1)
    C = coeffs_desc[::-1].reshape(-1, 1)

    terms = [f"{C[0, 0]:.6f}"]
    for p in range(1, poly_order + 1):
        terms.append(f"{C[p, 0]:.6e} * t^{p}")
    print(f"  Регресійна модель: y(t) = {' + '.join(terms)}")
    return Yout, C


def MNK_Extrapol(S0, koef, poly_order=2):
    n = len(S0)
    x = np.arange(n, dtype=float)
    coeffs_desc = np.polyfit(x, S0, poly_order)
    # Обчислення на розширеному діапазоні
    x_ext = np.arange(n + koef, dtype=float)
    Yout_ext = np.polyval(coeffs_desc, x_ext).reshape(-1, 1)
    C = coeffs_desc[::-1].reshape(-1, 1)

    terms = [f"{C[0, 0]:.6f}"]
    for p in range(1, poly_order + 1):
        terms.append(f"{C[p, 0]:.6e} * t^{p}")
    print(f"  Регресійна модель (екстраполяція): y(t) = {' + '.join(terms)}")
    return Yout_ext, C


def stat_characteristics(S, label=""):
    n = len(S)
    mean_val = np.mean(S)
    median_val = np.median(S)
    var_val = np.var(S)
    std_val = np.std(S)
    min_val = np.min(S)
    max_val = np.max(S)
    print(f"\n{'=' * 60}")
    print(f"  Статистичні характеристики: {label}")
    print(f"{'=' * 60}")
    print(f"  Кількість елементів вибірки  : {n}")
    print(f"  Середнє арифметичне          : {mean_val:.2f}")
    print(f"  Медіана                      : {median_val:.2f}")
    print(f"  Дисперсія                    : {var_val:.2f}")
    print(f"  СКВ                          : {std_val:.2f}")
    print(f"  Мінімум                      : {min_val:.2f}")
    print(f"  Максимум                     : {max_val:.2f}")
    print(f"{'=' * 60}")
    return {
        "n": n,
        "mean": mean_val,
        "median": median_val,
        "var": var_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
    }


def r2_score(S_real, S_model, label=""):
    ss_res = np.sum((S_real - S_model) ** 2)
    ss_tot = np.sum((S_real - np.mean(S_real)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    print(f"\n  R^2 ({label}): {r2:.6f}")
    return r2


FORECAST_DAYS = 7
# З 7 точок квадратична парабола перегинається за межами спостережень —
# використовуємо лінійний тренд (poly_order=1), який коректно екстраполює
POLY_ORDER = 1  # лінійний тренд (МНК)
COLORS = ["#e63946", "#2a9d8f", "#f4a261"]
MARKERS = ["o", "s", "^"]


df = pd.read_csv(f"{dir_path}top3_timeseries.csv", encoding="utf-8-sig")
dates_str = df["Дата"].tolist()
terms = [c for c in df.columns if c != "Дата"]

print("=" * 70)
print("  ТРЕНДОВИЙ АНАЛІЗ ТА ПРОГНОЗУВАННЯ ЧАСТОТИ ТЕРМІНІВ (МНК)")
print("=" * 70)
print(f"  Термінів для аналізу : {terms}")
print(
    f"  Діапазон даних       : {dates_str[0]} — {dates_str[-1]}  ({len(dates_str)} днів)"
)
print(f"  Горизонт прогнозу    : {FORECAST_DAYS} днів")
print(f"  Степінь полінома МНК : {POLY_ORDER}")

n = len(dates_str)
x_obs = np.arange(n, dtype=float)
x_ext = np.arange(n + FORECAST_DAYS, dtype=float)

last_date = datetime.strptime(dates_str[-1], "%d.%m.%Y")
future_dates = [
    (last_date + timedelta(days=i + 1)).strftime("%d.%m.%Y")
    for i in range(FORECAST_DAYS)
]
all_dates = dates_str + future_dates


results = []


for term in terms:
    S = df[term].values.astype(float)

    print(f"\n{'═' * 70}")
    print(f"  ТЕРМІН: «{term}»")
    print(f"{'═' * 70}")

    stat_characteristics(S, f"«{term}» — вхідний ряд")

    print("\n  [МНК оцінювання]")
    Yout, _ = MNK(S, poly_order=POLY_ORDER)
    trend = Yout.flatten()

    r2 = r2_score(S, trend, f"«{term}» тренд МНК")

    residuals = S - trend
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

    std_resid = np.std(residuals)
    print(f"  СКВ залишків = {std_resid:.4f}")
    print(f"  Довірчий інтервал 95% (±2σ) = ±{2 * std_resid:.4f}")

    print(f"\n  [МНК прогноз на {FORECAST_DAYS} днів]")
    Yout_ext, _ = MNK_Extrapol(S, FORECAST_DAYS, poly_order=POLY_ORDER)
    trend_ext = Yout_ext.flatten()
    forecast = trend_ext[n:]

    stat_characteristics(forecast, f"«{term}» — прогнозні значення")

    results.append(
        {
            "term": term,
            "S": S,
            "trend": trend,
            "trend_ext": trend_ext,
            "residuals": residuals,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "forecast": forecast,
        }
    )


print(f"\n{'═' * 70}")
print("  ФОРМУВАННЯ ПІДСУМКОВОГО CSV")
print(f"{'═' * 70}")

rows = []
for i, d in enumerate(all_dates):
    row = {"Дата": d}
    for r in results:
        if i < n:
            row[f"{r['term']}_факт"] = r["S"][i]
            row[f"{r['term']}_тренд"] = round(r["trend"][i], 2)
            row[f"{r['term']}_прогноз"] = ""
        else:
            row[f"{r['term']}_факт"] = ""
            row[f"{r['term']}_тренд"] = ""
            row[f"{r['term']}_прогноз"] = round(r["trend_ext"][i], 2)
    rows.append(row)

df_out = pd.DataFrame(rows)
df_out.to_csv(f"{dir_path}forecast_timeseries.csv", index=False, encoding="utf-8-sig")
print(f"  Збережено: {dir_path}forecast_timeseries.csv")
print(df_out.to_string(index=False))


# ── Граф 1: дані + тренд МНК (3 підграфіки) ──────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
fig.suptitle(
    "Часові ряди топ-3 термінів та тренд МНК (оцінювання)",
    fontsize=14,
    fontweight="bold",
)

for ax, r, color in zip(axes, results, COLORS):
    ax.plot(
        x_obs,
        r["S"],
        color=color,
        alpha=0.5,
        linewidth=1.2,
        marker="o",
        markersize=5,
        label="Факт",
    )
    ax.plot(
        x_obs, r["trend"], color="black", linewidth=2, linestyle="--", label="Тренд МНК"
    )
    ax.set_xticks(x_obs)
    ax.set_xticklabels(dates_str, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Кількість згадувань", fontsize=9)
    ax.set_title(
        f"«{r['term']}»   R²={r['r2']:.4f}   RMSE={r['rmse']:.1f}   MAE={r['mae']:.1f}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(f"{dir_path}01_trends.png", dpi=150, bbox_inches="tight")
print(f"\n  Збережено: {dir_path}01_trends.png")


# ── Граф 2: екстраполяція (прогноз) ──────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
fig.suptitle(
    f"МНК-прогноз на {FORECAST_DAYS} днів для топ-3 термінів",
    fontsize=14,
    fontweight="bold",
)

for ax, r, color in zip(axes, results, COLORS):
    std_r = np.std(r["residuals"])
    ci_upper = r["trend_ext"][n:] + 2 * std_r
    ci_lower = r["trend_ext"][n:] - 2 * std_r

    ax.plot(
        x_obs,
        r["S"],
        color=color,
        alpha=0.55,
        linewidth=1.2,
        marker="o",
        markersize=5,
        label="Факт (очищений)",
    )
    ax.plot(
        x_ext[:n],
        r["trend"],
        color="black",
        linewidth=2,
        linestyle="--",
        label="Тренд МНК",
    )
    ax.plot(
        x_ext[n:],
        r["forecast"],
        color="red",
        linewidth=2,
        linestyle="-",
        marker="^",
        markersize=6,
        label="Прогноз МНК",
    )
    ax.fill_between(
        x_ext[n:], ci_lower, ci_upper, color="red", alpha=0.15, label="Довірч. інт. ±2σ"
    )
    ax.axvline(
        x=n - 0.5, color="orange", linestyle=":", linewidth=1.8, label="Межа прогнозу"
    )

    tick_x = list(x_ext)
    tick_labels = all_dates
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Кількість згадувань", fontsize=9)
    ax.set_title(
        f"«{r['term']}»   прогноз: {r['forecast'][0]:.0f}→{r['forecast'][-1]:.0f}   ±2σ={2 * std_r:.1f}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(f"{dir_path}02_forecast.png", dpi=150, bbox_inches="tight")
print(f"  Збережено: {dir_path}02_forecast.png")


print(f"\n{'═' * 70}")
print("  ПІДСУМКОВА ТАБЛИЦЯ ЯКОСТІ МОДЕЛЕЙ")
print(f"{'═' * 70}")
print(
    f"  {'Термін':<12} {'R²':>8} {'RMSE':>8} {'MAE':>8}  {'Тренд (початок→кінець)':>26}  {'Прогноз (d+1→d+7)'}"
)
print(f"  {'-' * 90}")
for r in results:
    direction = "зростання" if r["trend"][-1] > r["trend"][0] else "спадання"
    print(
        f"  «{r['term']:<10}»  {r['r2']:>8.4f}  {r['rmse']:>8.2f}  {r['mae']:>8.2f}"
        f"  {r['trend'][0]:>6.1f} → {r['trend'][-1]:>6.1f} ({direction})"
        f"  {r['forecast'][0]:.0f} → {r['forecast'][-1]:.0f}"
    )
print(f"  {'-' * 90}")
