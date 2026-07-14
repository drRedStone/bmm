"""
estimate_lambda.py
===================
Оценка интенсивности λ(t) потока клиентов по часам/дням недели/отделениям.

Источник: dataset/hourly_load.csv (уже агрегированные по часам данные,
кросс-валидируется с client_arrivals.csv — расхождений не найдено).

Ключевая идея: интенсивность лямбда оценивается как среднее число клиентов,
пришедших в данный час данного дня недели, усреднённое по 4 наблюдаемым неделям.
Это несмещённая MLE-оценка параметра пуассоновского распределения (лямбда = среднее).

Дополнительно проверяем степень отклонения от чистого пуассоновского процесса:
если var/mean > 1 (overdispersion), это согласуется с описанием генератора
в README/брифе — интенсивность самого дня "гуляет" на ±15% (случайный шум),
то есть реальный процесс — Poisson-Gamma mixture (по сути Negative Binomial)
на уровне "день-час", а не чистый однородный Пуассон с фиксированной лямбдой.
Внутри одного часа в один конкретный день условно (при фиксированной лямбда
этого дня) распределение всё ещё пуассоновское — это стандартное и корректное
допущение для очередной модели M/M/c на часовых интервалах (используется в
required_windows.py).

Этот скрипт — самостоятельная утилита для первичного EDA и выгрузки CSV;
required_windows.required_windows_table() делает аналогичную оценку λ
заново из client_arrivals.csv напрямую (без промежуточного hourly_load.csv),
т.к. ей дополнительно нужна разбивка по типу операции (basic/credit/mortgage),
которой в hourly_load.csv нет.
"""
import pandas as pd
import numpy as np

WEEKDAY_ORDER = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб']
WD_MAP = {0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт', 4: 'Пт', 5: 'Сб', 6: 'Вс'}


def load() -> pd.DataFrame:
    """
    Загружает dataset/hourly_load.csv и добавляет столбец weekday (день
    недели по-русски, выведенный из даты).

    Returns:
        DataFrame со столбцами исходного hourly_load.csv (date, branch_id,
        hour, n_clients, total_service_min) плюс weekday.
    """
    df = pd.read_csv('dataset/hourly_load.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday.map(WD_MAP)
    return df


def lambda_table(df: pd.DataFrame, branch_id: str = None) -> pd.DataFrame:
    """
    MLE-оценка λ(hour, weekday) — среднее число клиентов по наблюдаемым
    неделям для каждого часового слота, плюс дисперсия и индекс дисперсии
    (var/λ) для проверки допущения о пуассоновости.

    Args:
        df: датафрейм из load() (или совместимый по структуре).
        branch_id: если задан — оценка только по этому отделению; если
            None — по всем отделениям, СМЕШАННЫМ вместе построчно (не
            просуммированным по часам — для суммарной оценки по всем
            отделениям сразу см. блок __main__ ниже, где потоки сначала
            суммируются по дате/часу, а потом усредняются по неделям).

    Returns:
        DataFrame со столбцами weekday, hour, lambda_hat, var_hat, n_weeks,
        dispersion_index — отсортированный по (weekday, hour).
    """
    d = df if branch_id is None else df[df['branch_id'] == branch_id]
    grp = d.groupby(['weekday', 'hour'])['n_clients']
    lam = grp.mean().rename('lambda_hat')
    var = grp.var(ddof=1).rename('var_hat')
    n_obs = grp.count().rename('n_weeks')
    out = pd.concat([lam, var, n_obs], axis=1).reset_index()
    out['dispersion_index'] = out['var_hat'] / out['lambda_hat']
    out['weekday'] = pd.Categorical(out['weekday'], categories=WEEKDAY_ORDER, ordered=True)
    return out.sort_values(['weekday', 'hour'])

if __name__ == '__main__':
    df = load()

    # 1) Общая (по всем 3 отделениям суммарно) таблица λ
    total_by_hour_wd = df.groupby(['date', 'weekday', 'hour'])['n_clients'].sum().reset_index()
    grp = total_by_hour_wd.groupby(['weekday', 'hour'])['n_clients']
    lam_total = grp.mean().rename('lambda_hat').reset_index()
    var_total = grp.var(ddof=1).rename('var_hat').reset_index()
    lam_total = lam_total.merge(var_total, on=['weekday', 'hour'])
    lam_total['dispersion_index'] = lam_total['var_hat'] / lam_total['lambda_hat']
    lam_total['weekday'] = pd.Categorical(lam_total['weekday'], categories=WEEKDAY_ORDER, ordered=True)
    lam_total = lam_total.sort_values(['weekday', 'hour'])
    lam_total.to_csv('lambda_estimates_all_branches.csv', index=False)

    print("=== λ(hour, weekday), суммарно по 3 отделениям (клиентов/час) ===")
    pivot = lam_total.pivot(index='weekday', columns='hour', values='lambda_hat').reindex(WEEKDAY_ORDER)
    print(pivot.round(1).to_string())

    print("\n=== Индекс дисперсии var/λ (>1 = overdispersion относительно Пуассона) ===")
    disp_pivot = lam_total.pivot(index='weekday', columns='hour', values='dispersion_index').reindex(WEEKDAY_ORDER)
    print(disp_pivot.round(2).to_string())

    mean_disp = lam_total['dispersion_index'].mean()
    print(f"\nСредний индекс дисперсии по всем слотам: {mean_disp:.2f}")
    print("(=1 -> чистый Пуассон; >1 -> избыточная дисперсия, день-к-дню шум в λ)")

    # per-branch tables
    for b in sorted(df['branch_id'].unique()):
        t = lambda_table(df, b)
        t.to_csv(f'lambda_estimates_{b}.csv', index=False)
        print(f"\nСохранено: lambda_estimates_{b}.csv")

    # Пиковые и минимальные часы (по суммарной нагрузке)
    peak = lam_total.loc[lam_total['lambda_hat'].idxmax()]
    low = lam_total.loc[lam_total[lam_total['lambda_hat'] > 0]['lambda_hat'].idxmin()]
    print(f"\nПик нагрузки: {peak['weekday']} {int(peak['hour'])}:00, λ≈{peak['lambda_hat']:.1f} клиентов/час (все 3 отделения)")
    print(f"Минимум (не считая закрытых часов): {low['weekday']} {int(low['hour'])}:00, λ≈{low['lambda_hat']:.1f}")
