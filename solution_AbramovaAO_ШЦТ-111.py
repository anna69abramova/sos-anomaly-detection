# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ======================== ПУТИ ========================
# Укажи путь к папке с данными за нужный месяц
DATA_DIR = Path(r"C:\Users\aniaa\OneDrive\Рабочий стол\polusem\data_train\month=2025-06-01")

SCRIPT_DIR = Path.cwd()
OUTPUT_DIR = SCRIPT_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR.mkdir(exist_ok=True)

print(f"Данные: {DATA_DIR}")
print(f"Результаты: {OUTPUT_DIR}")

# ======================== ПАРАМЕТРЫ ========================
SHARE_THRESHOLD = 0.5
OTS_QUANTILE = 0.999
PEAK_QUANTILE = 0.999

# ======================== ЗАГРУЗКА ========================
def load_parquet(data_dir):
    files = list(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Нет .parquet файлов в {data_dir}")
    print(f"Найдено {len(files)} файлов")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def preprocess(df):
    df = df[(df["BrandinDelivery"] == 1.0) & (df["CategoryNameDelivery"].notna())].copy()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df[df["Weight"] > 0]
    df["researchdate"] = pd.to_datetime(df["researchdate"]).dt.date
    df = df.rename(columns={"CategoryNameDelivery": "CategoryDelivery"})
    return df

def compute_daily_ots(df):
    agg = df.groupby(["SubjectID", "researchdate", "CategoryDelivery", "BrandID"], as_index=False).agg(
        count_rows=("QueryText", "size"),
        Weight=("Weight", "first"),
        Brand=("Brand", "first")
    )
    agg["daily_ots"] = agg["Weight"] * agg["count_rows"]
    return agg

# ======================== ОБНАРУЖЕНИЕ АНОМАЛИЙ ========================
def detect_anomalies(agg_df):
    ots_threshold = agg_df["daily_ots"].quantile(OTS_QUANTILE)
    print(f"Порог daily_ots (99.9%): {ots_threshold:.1f}")

    group_cols = ["researchdate", "CategoryDelivery", "BrandID"]
    agg_df["brand_day_total"] = agg_df.groupby(group_cols, observed=True)["daily_ots"].transform("sum")
    agg_df["share"] = agg_df["daily_ots"] / agg_df["brand_day_total"]

    bd_threshold = agg_df["brand_day_total"].quantile(PEAK_QUANTILE)
    print(f"Порог пика бренд-дня (99.9%): {bd_threshold:.1f}")

    big_ots = agg_df["daily_ots"] >= ots_threshold
    rule_A = big_ots & (agg_df["share"] >= SHARE_THRESHOLD)
    rule_B = big_ots & (agg_df["brand_day_total"] >= bd_threshold)
    is_anomaly = rule_A | rule_B

    anomalies = agg_df[is_anomaly].copy()
    anomalies["rule"] = np.where(rule_A[is_anomaly], "доминирование", "участие в экстремальном пике")
    anomalies["score"] = anomalies["daily_ots"]
    anomalies["threshold"] = ots_threshold

    def make_reason(row):
        if row["rule"] == "доминирование":
            return (f"Респондент удалён по правилу доминирования: бренд '{row['Brand']}' "
                    f"(категория '{row['CategoryDelivery']}') за {row['researchdate']}. "
                    f"Его daily_ots = {row['daily_ots']:,.0f} ({int(row['count_rows'])} запросов) "
                    f"составляет {row['share']*100:.1f}% от суммарного OTS бренда за день ({row['brand_day_total']:,.0f}). "
                    f"Порог daily_ots = {row['threshold']:,.0f}.")
        else:
            return (f"Респондент удалён как участник экстремального пика: бренд '{row['Brand']}' "
                    f"(категория '{row['CategoryDelivery']}') за {row['researchdate']}. "
                    f"Его daily_ots = {row['daily_ots']:,.0f} ({int(row['count_rows'])} запросов). "
                    f"Суммарный OTS бренда за день ({row['brand_day_total']:,.0f}) превышает "
                    f"99.9% перцентиль ({bd_threshold:.0f}).")
    anomalies["reason"] = anomalies.apply(make_reason, axis=1)
    return anomalies

# ======================== ТРИ ГРАФИКА ========================
def save_outputs(anomalies_df, agg_df, df_clean):
    to_remove = anomalies_df[["SubjectID", "researchdate"]].drop_duplicates()
    to_remove.to_csv(OUTPUT_DIR / "anomalies.csv", index=False)

    reasons = anomalies_df[["SubjectID","researchdate","BrandID","Brand","CategoryDelivery",
                            "daily_ots","score","threshold","reason"]]
    reasons.to_csv(OUTPUT_DIR / "anomaly_reasons.csv", index=False, encoding='utf-8-sig')

    remove_set = set(to_remove.itertuples(index=False, name=None))
    mask_keep = ~agg_df.apply(lambda r: (r["SubjectID"], r["researchdate"]) in remove_set, axis=1)
    agg_after = agg_df[mask_keep]
    
    # Фильтруем исходные данные для аналитических графиков
    mask_keep_orig = ~df_clean.apply(lambda r: (r["SubjectID"], r["researchdate"]) in remove_set, axis=1)
    df_clean_after = df_clean[mask_keep_orig]

    # Диапазон дат из данных
    all_dates = pd.date_range(
        start=agg_df["researchdate"].min(),
        end=agg_df["researchdate"].max(),
        freq='D'
    ).date
    date_labels = [d.strftime('%m-%d') for d in all_dates]

    # График 1
    daily_before = agg_df.groupby("researchdate")["daily_ots"].sum().reindex(all_dates, fill_value=0)
    daily_after = agg_after.groupby("researchdate")["daily_ots"].sum().reindex(all_dates, fill_value=0)
    
    plt.figure(figsize=(16, 6))
    plt.plot(range(len(all_dates)), daily_before.values, marker='o', linestyle='-', linewidth=1.5, 
             color='red', label='До удаления', markersize=4)
    plt.plot(range(len(all_dates)), daily_after.values, marker='s', linestyle='-', linewidth=1.5, 
             color='green', label='После удаления', markersize=4)
    plt.xlabel("Дата")
    plt.ylabel("Суммарный OTS")
    plt.title("Общий OTS по дням до и после очистки")
    plt.legend()
    plt.grid(True, alpha=0.3)
    step = max(1, len(all_dates)//15)
    plt.xticks(range(0, len(all_dates), step), [date_labels[i] for i in range(0, len(all_dates), step)], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "total_ots_before_after.png", dpi=150)
    plt.close()

    # График 2
    cat_before = agg_df.groupby("CategoryDelivery")["daily_ots"].sum()
    cat_after = agg_after.groupby("CategoryDelivery")["daily_ots"].sum()
    cat_removed_pct = (cat_before - cat_after) / cat_before * 100
    cat_removed_pct = cat_removed_pct.sort_values()
    
    plt.figure(figsize=(12,6))
    bars = plt.bar(cat_removed_pct.index, -cat_removed_pct.values, color='steelblue')
    plt.axhline(y=0, color='black', linewidth=0.8)
    plt.xlabel("Категория")
    plt.ylabel("Удалено OTS (%)")
    plt.title("Процент удалённого OTS по категориям")
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, cat_removed_pct.values):
        plt.text(bar.get_x() + bar.get_width()/2, -val - 0.3, f'{val:.1f}%', ha='center', va='top', fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "category_ots_change.png", dpi=150)
    plt.close()

    # График 3
    anom_by_day_full = to_remove.groupby("researchdate").size().reindex(all_dates, fill_value=0)
    
    plt.figure(figsize=(16, 6))
    plt.bar(range(len(all_dates)), anom_by_day_full.values, color='coral', width=0.8)
    plt.xlabel("Дата")
    plt.ylabel("Количество аномальных респондентов")
    plt.title("Ежедневное количество респондентов, подлежащих удалению")
    plt.xticks(range(0, len(all_dates), step), [date_labels[i] for i in range(0, len(all_dates), step)], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "daily_anomaly_count.png", dpi=150)
    plt.close()

    return agg_after, to_remove, df_clean_after, remove_set

# ======================== ДОПОЛНИТЕЛЬНЫЕ АНАЛИТИЧЕСКИЕ ФУНКЦИИ ========================
def plot_before_after_by_column(df_before, df_after, column):
    before = df_before.groupby(column)["Weight"].sum()
    after = df_after.groupby(column)["Weight"].sum()
    compare = pd.DataFrame({"до": before, "после": after}).fillna(0).sort_values("до")
    compare.plot.barh(figsize=(10, max(4, len(compare)*0.4)))
    plt.title(f"OTS до/после очистки — {column}")
    plt.xlabel("Суммарный Weight")
    plt.tight_layout()
    safe_name = str(column).replace('/', '_').replace(' ', '_')
    plt.savefig(PLOTS_DIR / f"before_after_{safe_name}.png", dpi=150)
    plt.close()

def show_queries_for_anomaly(subject_id, date, df_original):
    mask = (df_original["SubjectID"] == subject_id) & (df_original["researchdate"] == pd.to_datetime(date).date())
    queries = df_original.loc[mask, "QueryText"].tolist()
    print(f"\nЗапросы для SubjectID={subject_id} на дату {date}:")
    for q in queries[:15]:
        print(f"  - {q}")

def brand_timeline(brand_id, category, agg_before, agg_after):
    mask = (agg_before["BrandID"] == brand_id) & (agg_before["CategoryDelivery"] == category)
    before = agg_before[mask].groupby("researchdate")["daily_ots"].sum()
    after = agg_after[mask].groupby("researchdate")["daily_ots"].sum()
    plt.figure(figsize=(12,4))
    plt.plot(before.index, before.values, 'o-', label='До')
    plt.plot(after.index, after.values, 's-', label='После', linestyle='--')
    plt.title(f"OTS бренда {brand_id} ({category})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"brand_{brand_id}_trend.png", dpi=150)
    plt.close()

# ======================== ЗАПУСК ========================
def main():
    if not DATA_DIR.exists():
        print(f"Ошибка: папка {DATA_DIR} не найдена")
        return

    print("Загрузка данных...")
    df_raw = load_parquet(DATA_DIR)
    print(f"Сырых строк: {len(df_raw)}")

    print("Предобработка...")
    df_clean = preprocess(df_raw)
    print(f"После фильтрации: {len(df_clean)}")

    print("Расчёт daily_ots...")
    daily = compute_daily_ots(df_clean)
    print(f"Уникальных комбинаций: {len(daily)}")

    print("Поиск аномалий...")
    anomalies = detect_anomalies(daily)
    if anomalies.empty:
        print("Аномалий не найдено.")
        return

    print("Сохранение результатов...")
    agg_after, to_remove, df_clean_after, remove_set = save_outputs(anomalies, daily, df_clean)

    unique_pairs = to_remove
    print(f"Аномальных триггеров: {len(anomalies)}")
    print(f"Уникальных респондент-дней для удаления: {len(unique_pairs)}")
    print(f"Доля аномальных респондентов: {100 * len(unique_pairs) / daily['SubjectID'].nunique():.2f}%")

    total_before = daily["daily_ots"].sum()
    total_after = agg_after["daily_ots"].sum()
    print(f"Общий OTS до: {total_before:,.0f}, после: {total_after:,.0f} ({100*total_after/total_before:.2f}%)")
    print(f"Результаты в {OUTPUT_DIR}")

    # ========== ДОПОЛНИТЕЛЬНЫЕ АНАЛИТИЧЕСКИЕ ВОЗМОЖНОСТИ ==========
    print("\n=== Дополнительные аналитические графики ===")
    
    for col in ['Пол', 'Возраст', 'Регион', 'Федеральный_округ', 'Занятость', 'Доход', 'Количество_детей',
                'ResourceName', 'ResourceType', 'Platform', 'UseType',
                'CategoryDelivery', 'Category1', 'Category2', 'Category3']:
        if col in df_clean.columns:
            plot_before_after_by_column(df_clean, df_clean_after, col)
            print(f"  Построен график для {col}")

    if not unique_pairs.empty:
        first = unique_pairs.iloc[0]
        show_queries_for_anomaly(first['SubjectID'], first['researchdate'], df_clean)

    if not anomalies.empty:
        top_brand = anomalies.groupby('BrandID').size().idxmax()
        top_cat = anomalies[anomalies['BrandID'] == top_brand]['CategoryDelivery'].iloc[0]
        brand_timeline(top_brand, top_cat, daily, agg_after)
        print(f"  Построен график тренда для бренда {top_brand} (категория {top_cat})")

    print(f"\nВсе результаты сохранены в {OUTPUT_DIR}")

if __name__ == "__main__":
    main()


