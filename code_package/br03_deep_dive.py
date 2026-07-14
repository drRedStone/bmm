"""
br03_deep_dive.py
==================
Диагностика BR03 — отделения с наименьшей кадровой гибкостью (по актуальным
данным employees.csv — 1 senior на 6 человек, было 0 в более ранней версии
датасета). Разбирает ДВЕ РАЗНЫЕ проблемы, которые легко перепутать, но
которые лечатся совершенно по-разному:

1. Дефицит навыка (структурный, почти постоянный): мало/нет senior ->
   R_mortgage не полностью покрывается штатом ни при каком расписании —
   это вопрос состава персонала, а не алгоритма.
2. Физическая нехватка окон (редко, но критично): в отдельные часы
   предложенная нагрузка a = λ/μ превышает даже все физически открытые
   окна (rho >= 1) — очередь математически нестабильна, никакое
   расписание тут не поможет, нужны либо доп. окна, либо сглаживание
   спроса (запись на приём и т.п.).

Проблема 2 (rho, offered load) зависит только от потока клиентов (λ) и
числа физических окон (branches.csv) — НЕ от состава сотрудников, поэтому
считается напрямую из required_windows_table. Проблема 1 (реальное
покрытие mortgage-спроса) зависит от состава сотрудников, поэтому требует
employees.csv и полноценного ILP-расписания на неделю (schedule_week_ilp)
— опциональный, более дорогой расчёт (см. Args diagnose()).
"""
import pandas as pd
import numpy as np
from required_windows import required_windows_table

from config import WEEKDAYS



def diagnose(ca: pd.DataFrame, ops: pd.DataFrame, br: pd.DataFrame,
             branch_id: str = 'BR03', emp_raw: pd.DataFrame = None) -> pd.DataFrame:
    """
    Считает по всем часам недели предложенную нагрузку и загрузку при
    максимуме физически открытых окон отделения (Проблема 2 — не зависит
    от штата), и печатает сводку. Если передан emp_raw — дополнительно
    считает РЕАЛЬНОЕ покрытие mortgage-спроса штатом за неделю через
    schedule_week_ilp (Проблема 1 — честный расчёт вместо предположений
    о составе персонала).

    Args:
        ca: датафрейм client_arrivals.csv.
        ops: датафрейм operations.csv.
        br: датафрейм branches.csv, индексированный по branch_id.
        branch_id: идентификатор отделения для диагностики.
        emp_raw: (опционально) полный датафрейм employees.csv, skills уже
            распарсены в list. Если передан — считается реальный процент
            покрытия mortgage-спроса штатом за неделю (Проблема 1). Если не
            передан — печатается только суммарный спрос без утверждений о
            проценте покрытия (чтобы не полагаться на устаревшие
            предположения о составе штата).

    Returns:
        DataFrame по часам недели: weekday, hour, lambda_total,
        lambda_mortgage, offered_load_a, rho_at_max_windows,
        threshold_reachable.
    """
    n_win = int(br.loc[branch_id, 'n_windows'])
    rows = []
    for wd in WEEKDAYS:
        req = required_windows_table(ca, ops, branch_id, wd, n_win)
        for _, r in req.iterrows():
            mu = 60 / r['avg_service_min'] if r['avg_service_min'] > 0 else np.nan
            a = r['lambda_total'] / mu                # предложенная нагрузка, Эрланг
            rho = r['lambda_total'] / (n_win * mu)     # загрузка при ВСЕХ физических окнах открытыми
            rows.append(dict(weekday=wd, hour=int(r['hour']), lambda_total=r['lambda_total'],
                              lambda_mortgage=r['lambda_mortgage'], offered_load_a=round(a, 2),
                              rho_at_max_windows=round(rho, 2), threshold_reachable=r['threshold_reachable']))
    diag = pd.DataFrame(rows)

    n_unstable = (diag['rho_at_max_windows'] >= 1).sum()
    n_unreachable = (~diag['threshold_reachable']).sum()
    weekly_mortgage_demand = diag['lambda_mortgage'].sum()  # клиентов/неделю, требующих mortgage-навык

    print(f"=== {branch_id}: диагностика ===")
    print(f"\n--- Проблема 2: физическая нестабильность (не зависит от штата) ---")
    print(f"Часов с rho>=1 (очередь физически нестабильна даже на {n_win} окнах): {n_unstable} из {len(diag)}")
    if n_unstable:
        print(diag[diag['rho_at_max_windows'] >= 1][['weekday', 'hour', 'offered_load_a', 'rho_at_max_windows']]
              .to_string(index=False))
    print(f"Часов, где порог 10 мин недостижим даже на максимуме окон: {n_unreachable} из {len(diag)}")

    print(f"\n--- Проблема 1: дефицит навыка (зависит от штата) ---")
    print(f"Суммарный поток mortgage-клиентов за неделю: {weekly_mortgage_demand:.1f} чел.")
    if emp_raw is not None:
        # честный расчёт реального покрытия — прогоняем ILP на всю неделю и
        # сравниваем достигнутое покрытие mortgage-окон с требуемым
        from schedule_ilp import schedule_week_ilp
        branch_emp = emp_raw[emp_raw['branch_id'] == branch_id]
        n_senior = int((branch_emp['grade'] == 'senior').sum())
        results, _ = schedule_week_ilp(ca, ops, emp_raw, br, branch_id)
        total_r_mortgage = sum(r['coverage']['R_mortgage'].sum() for r in results.values())
        total_cov_mortgage = sum(r['coverage']['cov_mortgage'].sum() for r in results.values())
        pct = total_cov_mortgage / total_r_mortgage * 100 if total_r_mortgage > 0 else 100
        print(f"Сотрудников уровня senior в штате: {n_senior}")
        print(f"Требуется mortgage-окно-часов за неделю: {total_r_mortgage:.0f}, "
              f"покрывается (ILP-оптимум): {total_cov_mortgage:.0f} ({pct:.0f}%)")
    else:
        print("(передайте emp_raw в diagnose(), чтобы посчитать реальный % покрытия штатом)")

    return diag


if __name__ == '__main__':
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
    emp = pd.read_csv('dataset/employees.csv')
    emp['skills'] = emp['skills'].str.split(',')
    diagnose(ca, ops, br, 'BR03', emp_raw=emp)
