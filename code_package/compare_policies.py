"""
compare_policies.py
====================
Честное сравнение трёх политик расписания на один день/отделение по
РЕАЛЬНОМУ среднему времени ожидания — не по целевому R(t), а по формуле
Эрланга C, применённой к ФАКТИЧЕСКИ достигнутому числу окон каждой
политики. Так наивная/жадная/ILP схемы сравниваются на равных: у каждой
может быть свой недобор относительно R(t), и важно именно то, во что этот
недобор выливается в реальном ожидании клиента.

Три политики:
    1. naive  — «все выходят в 9:00» (текущий ручной способ из брифа: столько
       сотрудников, сколько окон, все с одной смены, обед по очереди).
    2. greedy — жадный алгоритм (schedule_greedy.schedule_day).
    3. ilp    — точный ILP с мягкими штрафами за недобор (schedule_ilp.schedule_day_ilp_soft).
"""
import pandas as pd
import numpy as np
from required_windows import required_windows_table, erlang_c_wait_minutes
from schedule_greedy import schedule_day, generate_patterns
from schedule_ilp import schedule_day_ilp_soft

from config import WEEKDAY_HOURS, OPEN_HOUR, MAX_WORK_TIME

def naive_schedule(employees: pd.DataFrame, n_windows_max: int, open_hour: int = OPEN_HOUR,
                    lunch_start: int = 12):
    """
    Строит "наивное" расписание — имитацию текущего ручного процесса из
    брифа ("все выходят на 9:00, обед по очереди"): берутся первые
    n_windows_max сотрудников (без учёта квалификации или стоимости — именно
    так поступает менеджер "на глаз"), все получают одинаковую 9-часовую
    смену с open_hour, обед распределяется по очереди небольшими сдвигами.

    Специально НЕ учитывает форму спроса по часам и квалификацию — в этом
    и состоит "наивность" данной политики, она служит базовой линией
    сравнения (baseline) для greedy и ilp.

    Args:
        employees: датафрейм сотрудников отделения.
        n_windows_max: физическое число окон отделения (= число сотрудников,
            которых выберет наивная схема).
        open_hour: час начала смены (по умолчанию 9 — открытие отделения).
        lunch_start: с какого часа начинается очередь обедов.

    Returns:
        Список dict-назначений (тот же формат, что у schedule_greedy.schedule_day).
    """
    chosen = employees.head(n_windows_max).reset_index(drop=True)
    assignments = []
    for i, e in chosen.iterrows():
        lunch_hour = lunch_start + (i % 4)  # обед по очереди, максимум растягиваем на 4 часа
        start, end = open_hour, open_hour + MAX_WORK_TIME
        serving = tuple(h for h in range(start, end) if h != lunch_hour)
        assignments.append(dict(employee_id=e['employee_id'], grade=e['grade'], start=start, end=end,
                                 lunch_hour=lunch_hour, serving_hours=serving, paid_hours=len(serving),
                                 hourly_cost=e['hourly_cost_rub'], shift_cost=len(serving) * e['hourly_cost_rub']))
    return assignments


def coverage_from_assignments(assignments: list, hours: list) -> dict:
    """
    Считает фактическое число открытых окон по часам для заданного списка
    назначений смен — сколько сотрудников реально обслуживают окно в каждый
    час (не в отпуске/на обеде).

    Args:
        assignments: список dict-назначений (как возвращает schedule_day,
            schedule_day_ilp_soft или naive_schedule) — используются только
            поля employee_id и serving_hours.
        hours: список часов, для которых нужно посчитать покрытие (обычно
            все часы работы отделения в этот день).

    Returns:
        dict {hour: число сотрудников, обслуживающих окно в этот час}.
    """
    cov = {h: 0 for h in hours}
    for a in assignments:
        for h in a['serving_hours']:
            if h in cov:
                cov[h] += 1
    return cov


def avg_wait_for_coverage(req: pd.DataFrame, cov: dict, cap_minutes: float = 90.0):
    """
    Переводит фактическое почасовое покрытие (число открытых окон) в
    реальное среднее время ожидания через формулу Эрланга C — это и есть
    "честная" метрика качества расписания, независимая от того, как именно
    расписание было построено.

    Args:
        req: датафрейм из required_windows_table (нужны столбцы hour,
            lambda_total, avg_service_min).
        cov: dict {hour: число открытых окон} — обычно результат
            coverage_from_assignments().
        cap_minutes: потолок для отображения на графиках — если Wq
            бесконечно (система нестабильна, окон не хватает даже
            теоретически) или очень велико, значение обрезается до
            cap_minutes, чтобы графики оставались читаемыми.

    Returns:
        Кортеж (waits, weights) — оба numpy-массивы одинаковой длины
        (по одному элементу на час из req): waits — среднее ожидание в
        минутах в этот час, weights — интенсивность потока в этот час
        (используется как вес при подсчёте средневзвешенного показателя за
        день: np.average(waits, weights=weights), чтобы часы с большим
        потоком клиентов вносили больший вклад в итоговую оценку).
    """
    waits, weights = [], []
    for _, row in req.iterrows():
        h = row['hour']
        lam = row['lambda_total']
        mu = 60.0 / row['avg_service_min'] if row['avg_service_min'] > 0 else np.nan
        c = cov.get(h, 0)
        wq = erlang_c_wait_minutes(lam, mu, c) if lam > 0 else 0.0
        wq = min(wq, cap_minutes) if np.isfinite(wq) else cap_minutes  # для читаемости графика ограничиваем "потолком"
        waits.append(wq)
        weights.append(lam)
    return np.array(waits), np.array(weights)


if __name__ == '__main__':
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
    emp_raw = pd.read_csv('dataset/employees.csv')
    emp_raw['skills'] = emp_raw['skills'].str.split(',')

    BRANCH, WEEKDAY_EN = 'BR01', 'Monday' # Это для таблицы сравнения графиков по отделниям на 1 день
    n_win = int(br.loc[BRANCH, 'n_windows'])
    req = required_windows_table(ca, ops, BRANCH, WEEKDAY_EN, n_win)
    branch_emp = emp_raw[emp_raw['branch_id'] == BRANCH].reset_index(drop=True)
    hours = list(req['hour'])

    a_naive = naive_schedule(branch_emp, n_win)
    a_greedy, cov_greedy_df, unresolved, _ = schedule_day(branch_emp, req, *WEEKDAY_HOURS[WEEKDAY_EN], n_windows_max=n_win)
    a_ilp, cov_ilp_df, status, _, shortfall = schedule_day_ilp_soft(branch_emp, req, *WEEKDAY_HOURS[WEEKDAY_EN], n_win)

    cov_naive = coverage_from_assignments(a_naive, hours)
    cov_greedy = coverage_from_assignments(a_greedy, hours)
    cov_ilp = coverage_from_assignments(a_ilp, hours)

    results = {}
    for name, cov, assigns in [('naive', cov_naive, a_naive), ('greedy', cov_greedy, a_greedy), ('ilp', cov_ilp, a_ilp)]:
        waits, weights = avg_wait_for_coverage(req, cov)
        avg_wait = np.average(waits, weights=weights)
        cost = sum(a['shift_cost'] for a in assigns)
        n_staff = len(assigns)
        results[name] = dict(cov=cov, waits=waits, avg_wait=avg_wait, cost=cost, n_staff=n_staff)
        print(f"{name:8s}: сотрудников={n_staff}, ФОТ={cost:5.0f} руб., "
              f"средневзв. ожидание={avg_wait:5.1f} мин, покрытие по часам={list(cov.values())}")

    import json
    with open('policy_comparison.json', 'w') as f:
        json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items() if kk != 'cov'}
                   for k, v in results.items()}, f, indent=2, default=str)
