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
from binascii import Error
from pathlib import Path

import pandas as pd
import numpy as np
import os
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from required_windows import required_windows_table, erlang_c_wait_minutes
from schedule_greedy import schedule_day, generate_patterns, schedule_week
from schedule_ilp import schedule_day_ilp_soft, schedule_week_ilp

from config import WEEKDAY_HOURS, OPEN_HOUR, MAX_WORK_TIME, MAX_HOURS_WEEK, REPORT_SAVE_DIRECTORY, GRADE_COLORS


if __name__ == '__main__':
    if os.path.exists(REPORT_SAVE_DIRECTORY) and not os.path.isdir(REPORT_SAVE_DIRECTORY):
        raise Error('failed to create save dir')
    elif not os.path.exists(REPORT_SAVE_DIRECTORY):
        os.mkdir(REPORT_SAVE_DIRECTORY)


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


# --------------- Новые функции анализа ---------------

def get_weekly_metrics(
    results: dict,
    summary: pd.DataFrame,
    client_arrivals: pd.DataFrame,
    operations: pd.DataFrame,
    branch_id: str,
    branches: pd.DataFrame
) -> dict:
    """Вычисляет ключевые метрики по недельному расписанию отделения.

    Для каждого дня недели:
    - рассчитывает фактическое покрытие окон,
    - средневзвешенное время ожидания (через формулу Эрланга C),
    - ФОТ и суммарный недобор total-окон.

    Args:
        results: результат работы schedule_week_ilp – словарь
            {weekday: {'assignments': ..., 'coverage': ..., 'status': ..., 'shortfall': ...}}.
        summary: недельная загрузка сотрудников (DataFrame из schedule_week_ilp).
        client_arrivals: сырой DataFrame визитов.
        operations: DataFrame операций.
        branch_id: идентификатор отделения.
        branches: DataFrame отделений, индекс branch_id.

    Returns:
        Словарь с ключами:
        - total_cost: общий ФОТ за неделю,
        - avg_wait_weighted: средневзвешенное ожидание за неделю (мин),
        - total_shortfall: суммарный недобор total-окон за неделю (окно‑часов),
        - daily_metrics: список словарей для каждого дня с полями:
            weekday, cost, avg_wait, shortfall, weight (интенсивность потока),
        - employee_summary: DataFrame загрузки сотрудников (hours_this_week, utilization).
    """
    n_win = int(branches.loc[branch_id, 'n_windows'])
    daily_metrics = []
    total_cost = 0.0
    total_shortfall = 0.0
    total_weight = 0.0
    weighted_wait_sum = 0.0

    for weekday, data in results.items():
        assignments = data['assignments']
        coverage = data['coverage']  # DataFrame
        req = required_windows_table(client_arrivals, operations, branch_id, weekday, n_win)
        if req.empty:
            continue

        day_cost = sum(a['shift_cost'] for a in assignments)
        total_cost += day_cost

        day_shortfall = (coverage['R_total'] - coverage['cov_total']).clip(lower=0).sum()
        total_shortfall += day_shortfall

        cov = coverage_from_assignments(assignments, list(req['hour']))
        waits, weights = avg_wait_for_coverage(req, cov)
        day_weight = weights.sum()
        day_avg_wait = np.average(waits, weights=weights) if day_weight > 0 else 0.0
        weighted_wait_sum += day_avg_wait * day_weight
        total_weight += day_weight

        daily_metrics.append({
            'weekday': weekday,
            'cost': day_cost,
            'avg_wait': round(day_avg_wait, 2),
            'shortfall': day_shortfall,
            'weight': day_weight
        })

    avg_wait_weighted = weighted_wait_sum / total_weight if total_weight > 0 else 0.0

    return {
        'total_cost': total_cost,
        'avg_wait_weighted': round(avg_wait_weighted, 2),
        'total_shortfall': total_shortfall,
        'daily_metrics': daily_metrics,
        'employee_summary': summary
    }


def plot_weekly_schedule(
    results: dict,
    summary: pd.DataFrame,
    client_arrivals: pd.DataFrame,
    operations: pd.DataFrame,
    branch_id: str,
    branches: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Визуализирует недельное расписание: ожидание, покрытие, загрузка.

    Строит три графика:
    1. Среднее ожидание по дням недели (столбцы).
    2. Покрытие окон (R_total vs cov_total) по часам для первого дня.
    3. Загрузка сотрудников (utilization, горизонтальные столбцы).

    Args:
        results: результат schedule_week_ilp (словарь дней).
        summary: недельная загрузка сотрудников.
        client_arrivals, operations, branch_id, branches: данные для расчёта метрик.
        save_path: если указан, сохраняет график в файл; иначе показывает на экране.
    """
    metrics = get_weekly_metrics(results, summary, client_arrivals, operations, branch_id, branches)

    days = [m['weekday'] for m in metrics['daily_metrics']]
    waits = [m['avg_wait'] for m in metrics['daily_metrics']]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.bar(days, waits, color='skyblue')
    plt.title('Среднее ожидание по дням')
    plt.ylabel('Минут')
    plt.xticks(rotation=45)

    first_day = list(results.keys())[0]
    cov_df = results[first_day]['coverage']
    hours = cov_df['hour']
    plt.subplot(1, 3, 2)
    # Увеличенные маркеры, чтобы точки не терялись
    plt.plot(hours, cov_df['R_total'], 'o-', markersize=8, label='Требуется')
    plt.plot(hours, cov_df['cov_total'], 's-', markersize=8, label='Открыто')
    plt.title(f'Покрытие ({first_day})')
    plt.xlabel('Час')
    plt.ylabel('Окна')
    plt.legend()
    plt.grid(True)

    emp_summary = metrics['employee_summary']
    employees = emp_summary['employee_id'].tolist()
    utilization = (emp_summary['utilization'] * 100).tolist()
    plt.subplot(1, 3, 3)
    plt.barh(employees, utilization, color='salmon')
    plt.title('Загрузка сотрудников (%)')
    plt.xlabel('% от 40ч')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График сохранён: {save_path}")
    else:
        plt.show()


def weekly_schedule_to_json(
    results: dict,
    summary: pd.DataFrame,
    client_arrivals: pd.DataFrame,
    operations: pd.DataFrame,
    branch_id: str,
    branches: pd.DataFrame
) -> dict:
    """Формирует JSON-совместимый словарь с метриками и данными для клиентского рендеринга.

    Возвращает структуру, которую можно напрямую передать в json.dumps.
    Все числовые значения приведены к стандартным типам Python.

    Args:
        Аргументы те же, что и у get_weekly_metrics.

    Returns:
        Словарь с ключами:
        - total_cost, avg_wait_weighted, total_shortfall,
        - daily: список по дням, где каждый день содержит:
            weekday, cost, avg_wait, shortfall,
            hourly: список {hour, R_total, cov_total, wait_min}.
        - employees: список {employee_id, hours, utilization}.
    """
    metrics = get_weekly_metrics(results, summary, client_arrivals, operations, branch_id, branches)
    n_win = int(branches.loc[branch_id, 'n_windows'])

    daily_json = []
    for day_metric in metrics['daily_metrics']:
        weekday = day_metric['weekday']
        data = results[weekday]
        assignments = data['assignments']
        coverage_df = data['coverage']

        req = required_windows_table(client_arrivals, operations, branch_id, weekday, n_win)
        cov = coverage_from_assignments(assignments, list(req['hour']))
        waits_hourly, _ = avg_wait_for_coverage(req, cov)

        hourly_data = []
        for _, row in coverage_df.iterrows():
            h = int(row['hour'])
            w = waits_hourly[list(req['hour']).index(h)]
            hourly_data.append({
                'hour': h,
                'R_total': int(row['R_total']),
                'cov_total': int(row['cov_total']),
                'wait_min': float(round(w, 2))
            })

        daily_json.append({
            'weekday': weekday,
            'cost': float(day_metric['cost']),
            'avg_wait': float(day_metric['avg_wait']),
            'shortfall': int(day_metric['shortfall']),
            'hourly': hourly_data
        })

    employees_json = []
    for _, emp_row in metrics['employee_summary'].iterrows():
        employees_json.append({
            'employee_id': emp_row['employee_id'],
            'hours': int(emp_row['hours_this_week']),
            'utilization': float(emp_row['utilization'])
        })

    return {
        'total_cost': float(metrics['total_cost']),
        'avg_wait_weighted': float(metrics['avg_wait_weighted']),
        'total_shortfall': int(metrics['total_shortfall']),
        'daily': daily_json,
        'employees': employees_json
    }


def plot_daily_schedule(assignments: list, title: str = "Расписание смен", save_path: Optional[str] = None):
    """Гант-диаграмма смен на один день.

    Args:
        assignments: список назначений (тот же формат, что возвращает schedule_day).
        title: заголовок графика.
        save_path: если задан, сохраняет график в файл.
    """
    if not assignments:
        print("Нет назначений для отображения.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(assignments) * 0.5)))
    employees = sorted(assignments, key=lambda a: (a['start'], a['employee_id']))
    y_labels = []
    y_pos = 0
    for a in employees:
        eid = a['employee_id']
        grade = a.get('grade', '')
        label = f"{eid} ({grade})" if grade else eid
        y_labels.append(label)
        work_ranges = []
        lunch_range = None
        if a['lunch_hour'] is not None:
            if a['start'] < a['lunch_hour']:
                work_ranges.append((a['start'], a['lunch_hour'] - a['start']))
            if a['lunch_hour'] + 1 < a['end']:
                work_ranges.append((a['lunch_hour'] + 1, a['end'] - (a['lunch_hour'] + 1)))
            lunch_range = (a['lunch_hour'], 1)
        else:
            work_ranges.append((a['start'], a['end'] - a['start']))

        for start, duration in work_ranges:
            ax.broken_barh([(start, duration)], (y_pos - 0.4, 0.8), facecolors=GRADE_COLORS.get(grade, 'tab:blue'), edgecolor='black')
        if lunch_range:
            ax.broken_barh([lunch_range], (y_pos - 0.4, 0.8), facecolors='lightgray', edgecolor='black', hatch='//')

        y_pos += 1

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Час дня')
    ax.set_title(title)
    min_hour = min(a['start'] for a in employees)
    max_hour = max(a['end'] for a in employees)
    ax.set_xlim(min_hour - 0.5, max_hour + 0.5)
    ax.set_xticks(range(min_hour, max_hour + 1))
    ax.grid(axis='x', linestyle='-', alpha=0.3)
    legend_elements = [
    Patch(facecolor=color, label=str(grade)) 
    for grade, color in GRADE_COLORS.items()
    ] + [Patch(facecolor='lightgray', hatch='//', label='Обед')]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График смен сохранён: {save_path}")
    else:
        plt.show()


def main():
    """Сравнение naive, greedy и ILP за один день (BR01, Monday)."""
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


def main2():
    """Недельное планирование для BR01: ILP и Greedy, вывод метрик, графиков и JSON."""
    # (Оставлена для совместимости – теперь можно вызывать generate_all_reports)
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
    emp = pd.read_csv('dataset/employees.csv')
    emp['skills'] = emp['skills'].str.split(',')

    BRANCH = 'BR01'
    n_win = int(br.loc[BRANCH, 'n_windows'])

    # === ILP ===
    print("\n" + "="*50)
    print("НЕДЕЛЬНОЕ ПЛАНИРОВАНИЕ (ILP)")
    results_ilp, summary_ilp = schedule_week_ilp(ca, ops, emp, br, BRANCH)
    metrics_ilp = get_weekly_metrics(results_ilp, summary_ilp, ca, ops, BRANCH, br)
    print(f"[ILP] Общий ФОТ за неделю: {metrics_ilp['total_cost']:.0f} руб.")
    print(f"[ILP] Средневзвешенное ожидание за неделю: {metrics_ilp['avg_wait_weighted']} мин")
    print("[ILP] Недельная загрузка сотрудников:")
    print(summary_ilp.to_string(index=False))

    if 'Monday' in results_ilp:
        plot_daily_schedule(results_ilp['Monday']['assignments'],
                            title="ILP – BR01 Monday", save_path=str(REPORT_SAVE_DIRECTORY / "gantt_ilp_monday.png"))

    plot_weekly_schedule(results_ilp, summary_ilp, ca, ops, BRANCH, br, save_path=str(REPORT_SAVE_DIRECTORY / "weekly_schedule_ilp.png"))
    json_ilp = weekly_schedule_to_json(results_ilp, summary_ilp, ca, ops, BRANCH, br)
    import json
    with open(str(REPORT_SAVE_DIRECTORY / 'weekly_schedule_ilp.json'), 'w') as f:
        json.dump(json_ilp, f, indent=2, ensure_ascii=False)

    # === Greedy ===
    print("\n" + "="*50)
    print("НЕДЕЛЬНОЕ ПЛАНИРОВАНИЕ (Greedy)")
    results_greedy, summary_greedy = schedule_week(ca, ops, emp, br, BRANCH)
    metrics_greedy = get_weekly_metrics(results_greedy, summary_greedy, ca, ops, BRANCH, br)
    print(f"[Greedy] Общий ФОТ за неделю: {metrics_greedy['total_cost']:.0f} руб.")
    print(f"[Greedy] Средневзвешенное ожидание за неделю: {metrics_greedy['avg_wait_weighted']} мин")
    print("[Greedy] Недельная загрузка сотрудников:")
    print(summary_greedy.to_string(index=False))

    if 'Monday' in results_greedy:
        plot_daily_schedule(results_greedy['Monday']['assignments'],
                            title="Greedy – BR01 Monday", save_path=str(REPORT_SAVE_DIRECTORY / "gantt_greedy_monday.png"))

    plot_weekly_schedule(results_greedy, summary_greedy, ca, ops, BRANCH, br, save_path=str(REPORT_SAVE_DIRECTORY / "weekly_schedule_greedy.png"))
    json_greedy = weekly_schedule_to_json(results_greedy, summary_greedy, ca, ops, BRANCH, br)
    with open(str(REPORT_SAVE_DIRECTORY / 'weekly_schedule_greedy.json'), 'w') as f:
        json.dump(json_greedy, f, indent=2, ensure_ascii=False)

    print("\nВсе графики и JSON сохранены.")


# ----------------------------------------------------------------------
# Основная функция генерации всех отчётов
# ----------------------------------------------------------------------
def generate_all_reports():
    """
    Генерирует для каждого отделения и каждого рабочего дня:
      - гант-диаграммы расписаний (ILP и Greedy)
      - недельные сводные графики (ожидание, покрытие, загрузка)
      - JSON с метриками.
    Все файлы сохраняются в report/<branch>/<algorithm>/
    """
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
    emp = pd.read_csv('dataset/employees.csv')
    emp['skills'] = emp['skills'].str.split(',')

    branches = ['BR01', 'BR02', 'BR03']
    weekdays = list(WEEKDAY_HOURS.keys())  # Monday..Saturday

    for branch in branches:
        print(f"\n===== Отделение {branch} =====")
        branch_emp = emp[emp['branch_id'] == branch]
        n_win = int(br.loc[branch, 'n_windows'])

        # Папки для отделения
        base_dir = REPORT_SAVE_DIRECTORY / branch
        for alg in ['ilp', 'greedy']:
            (base_dir / alg).mkdir(parents=True, exist_ok=True)

        # Недельное планирование (ILP и Greedy) для получения расписаний
        results_ilp, summary_ilp = schedule_week_ilp(ca, ops, emp, br, branch)
        results_greedy, summary_greedy = schedule_week(ca, ops, emp, br, branch)

        # Гант-диаграммы для каждого дня
        for day in weekdays:
            if day not in results_ilp:
                continue
            # ILP
            plot_daily_schedule(
                results_ilp[day]['assignments'],
                title=f"ILP {branch} {day}",
                save_path=str(base_dir / 'ilp' / f'gantt_{day}.png')
            )
            # Greedy
            plot_daily_schedule(
                results_greedy[day]['assignments'],
                title=f"Greedy {branch} {day}",
                save_path=str(base_dir / 'greedy' / f'gantt_{day}.png')
            )

        # Недельные сводные графики
        plot_weekly_schedule(
            results_ilp, summary_ilp, ca, ops, branch, br,
            save_path=str(base_dir / 'ilp' / 'weekly_summary.png')
        )
        plot_weekly_schedule(
            results_greedy, summary_greedy, ca, ops, branch, br,
            save_path=str(base_dir / 'greedy' / 'weekly_summary.png')
        )

        # Сохранение JSON с метриками
        import json
        json_ilp = weekly_schedule_to_json(results_ilp, summary_ilp, ca, ops, branch, br)
        json_greedy = weekly_schedule_to_json(results_greedy, summary_greedy, ca, ops, branch, br)
        with open(str(base_dir / 'ilp' / 'metrics.json'), 'w') as f:
            json.dump(json_ilp, f, indent=2, ensure_ascii=False)
        with open(str(base_dir / 'greedy' / 'metrics.json'), 'w') as f:
            json.dump(json_greedy, f, indent=2, ensure_ascii=False)

        print(f"  Готово: {branch}")

    print("\nВсе отчёты сохранены в", REPORT_SAVE_DIRECTORY)


if __name__ == '__main__':
    generate_all_reports()
