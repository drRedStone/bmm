"""
week_pipeline.py
=================
Масштабирование сравнения naive/greedy/ILP с одного дня на всю рабочую
неделю (Пн-Сб, Вс закрыто), для всех трёх отделений. Недельный лимит 40ч
соблюдается сквозно для всех трёх политик — в том числе для "наивной", что
само по себе даёт интересную находку: даже имитация текущего процесса
вынуждена ротировать сотрудников по неделе, иначе нарушит трудовое
законодательство (см. naive_schedule_week).
"""
import pandas as pd
import numpy as np
from required_windows import required_windows_table, erlang_c_wait_minutes
from schedule_greedy import schedule_week
from schedule_ilp import schedule_week_ilp
from compare_policies import coverage_from_assignments, avg_wait_for_coverage

from config import MAX_WORK_TIME, MAX_WORK_TIME_BEFORE_LAUNCH, WEEKDAY_HOURS, MIN_WORK_TIME, MAX_HOURS_WEEK

def naive_schedule_week(employees: pd.DataFrame, n_windows_max: int, max_hours_week: int = MAX_HOURS_WEEK):
    """
    Недельная версия наивного расписания ("все выходят на 9:00, обед по
    очереди"), но с поправкой на трудовое законодательство: если КАЖДЫЙ
    день брать одних и тех же n_windows_max сотрудников на полную 8-часовую
    смену, за 6 рабочих дней недели они превысят 40-часовой лимит. Поэтому
    каждый день выбираются сотрудники с НАИБОЛЬШИМ остатком недельных часов
    (простая ротация "по остатку") — это уже не совсем то, что "менеджер на
    глаз" делает в реальности, но необходимый компромисс, чтобы наивная
    политика вообще была допустимым (легальным) расписанием для честного
    сравнения с greedy/ILP.

    Специально НЕ учитывает форму спроса по часам и квалификацию сотрудников
    — в этом и состоит суть "наивности" данной политики как baseline.

    Args:
        employees: датафрейм сотрудников ОДНОГО отделения.
        n_windows_max: физическое число окон отделения (= число сотрудников
            в смене каждый день).
        max_hours_week: недельный лимит часов на сотрудника.

    Returns:
        Кортеж (results, weekly_hours_used):
        - results: dict {weekday: список dict-назначений} (тот же формат,
          что у schedule_greedy.schedule_day, но без coverage_df/unresolved
          — они не нужны наивной схеме, coverage считается отдельно в
          week_summary через coverage_from_assignments);
        - weekly_hours_used: {employee_id: итоговых часов за неделю}.
    """
    weekly_hours_used = {eid: 0 for eid in employees['employee_id']}
    results = {}
    emp_records = employees.to_dict('records')

    for weekday, (open_h, close_h) in WEEKDAY_HOURS.items():
        span = min(MAX_WORK_TIME, close_h - open_h)  # суббота короче будней (7ч), полная 9ч смена не влезает
        lunch_hour = open_h + span // 2 if span >= MAX_WORK_TIME_BEFORE_LAUNCH else None
        end_h = open_h + span
        serving_all = tuple(h for h in range(open_h, end_h) if h != lunch_hour)
        paid_hours = len(serving_all)

        # сортируем по остатку часов (кому больше осталось — тот в приоритете), берём первых n_windows_max с достаточным остатком
        ranked = sorted(emp_records, key=lambda e: -(max_hours_week - weekly_hours_used[e['employee_id']]))
        chosen = [e for e in ranked if max_hours_week - weekly_hours_used[e['employee_id']] >= paid_hours][:n_windows_max]

        assignments = []
        for i, e in enumerate(chosen):  #что за х? v         v   откуда 1 и 3???
            lh = open_h + MIN_WORK_TIME + (i % max(1, span - 3)) if span >= MAX_WORK_TIME_BEFORE_LAUNCH else None  # лёгкий разброс обеда по очереди
            serving = tuple(h for h in range(open_h, end_h) if h != lh) if lh is not None else serving_all
            weekly_hours_used[e['employee_id']] += len(serving)
            assignments.append(dict(employee_id=e['employee_id'], grade=e['grade'], start=open_h, end=end_h,
                                     lunch_hour=lh, serving_hours=serving, paid_hours=len(serving),
                                     hourly_cost=e['hourly_cost_rub'], shift_cost=len(serving) * e['hourly_cost_rub']))
        results[weekday] = assignments
    return results, weekly_hours_used


def week_summary(ca: pd.DataFrame, ops: pd.DataFrame, br: pd.DataFrame,
                  emp_raw: pd.DataFrame, branch_id: str):
    """
    Строит расписание на всю неделю тремя политиками (naive/greedy/ilp) для
    одного отделения и агрегирует результат — недельный ФОТ, средневзвешенное
    (по лямбда) ожидание за неделю, суммарный недобор, плюс подневную
    разбивку (daily) для графиков по дням недели (см. например
    br02_weekday_breakdown в отчёте).

    Args:
        ca: датафрейм client_arrivals.csv.
        ops: датафрейм operations.csv.
        br: датафрейм branches.csv, индексированный по branch_id.
        emp_raw: полный датафрейм employees.csv (skills уже распарсены в list).
        branch_id: идентификатор отделения.

    Returns:
        Кортеж (out, max_util):
        - out: dict {'naive'/'greedy'/'ilp': {total_cost, weighted_avg_wait,
          total_shortfall, daily}}, где daily — dict {weekday: {avg_wait,
          cost, shortfall, weight}} для построения графиков по дням недели;
        - max_util: dict {'naive'/'greedy'/'ilp': максимальная недельная
          загрузка любого сотрудника, доля от 1.0} — показатель "выгорания"
          (близко к 1.0 значит кто-то из сотрудников работает на пределе
          40-часового лимита).
    """
    n_win = int(br.loc[branch_id, 'n_windows'])
    branch_emp = emp_raw[emp_raw['branch_id'] == branch_id].reset_index(drop=True)

    naive_week, naive_hours = naive_schedule_week(branch_emp, n_win)
    greedy_week, greedy_summary = schedule_week(ca, ops, emp_raw, br, branch_id)
    ilp_week, ilp_summary = schedule_week_ilp(ca, ops, emp_raw, br, branch_id)

    out = {'naive': {}, 'greedy': {}, 'ilp': {}}
    daily = {'naive': {}, 'greedy': {}, 'ilp': {}}

    for weekday in WEEKDAY_HOURS:
        req = required_windows_table(ca, ops, branch_id, weekday, n_win)
        hours = list(req['hour'])

        for pol, week_data in [('naive', naive_week), ('greedy', greedy_week), ('ilp', ilp_week)]:
            # naive_schedule_week и schedule_week/schedule_week_ilp возвращают
            # результат в чуть разной структуре (naive — сразу список
            # назначений, остальные — dict с ключом 'assignments'), отсюда
            # разветвление ниже
            if pol == 'naive':
                assigns = week_data.get(weekday, [])
            else:
                assigns = week_data.get(weekday, {}).get('assignments', [])
            cov = coverage_from_assignments(assigns, hours)
            waits, weights = avg_wait_for_coverage(req, cov)
            cost = sum(a['shift_cost'] for a in assigns)
            shortfall = sum(max(0, r - cov[h]) for h, r in zip(hours, req['R_total']))
            daily[pol][weekday] = dict(avg_wait=np.average(waits, weights=weights) if weights.sum() > 0 else 0,
                                        cost=cost, shortfall=shortfall, weight=weights.sum())

    for pol in out:
        total_cost = sum(d['cost'] for d in daily[pol].values())
        total_weight = sum(d['weight'] for d in daily[pol].values())
        # средневзвешенное ожидание за неделю: каждый день весится по своей
        # суммарной интенсивности потока (weight), чтобы дни с большим
        # потоком клиентов вносили больший вклад в итоговую недельную оценку
        weighted_wait = sum(d['avg_wait'] * d['weight'] for d in daily[pol].values()) / total_weight
        total_shortfall = sum(d['shortfall'] for d in daily[pol].values())
        out[pol] = dict(total_cost=total_cost, weighted_avg_wait=weighted_wait, total_shortfall=total_shortfall,
                         daily=daily[pol])

    max_util = dict(naive=max(naive_hours.values()) / MAX_HOURS_WEEK,
                     greedy=greedy_summary['utilization'].max() if len(greedy_summary) else 0,
                     ilp=ilp_summary['utilization'].max() if len(ilp_summary) else 0)
    return out, max_util


if __name__ == '__main__':
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
    emp_raw = pd.read_csv('dataset/employees.csv')
    emp_raw['skills'] = emp_raw['skills'].str.split(',')

    import json
    all_out = {}
    for bid in ['BR01', 'BR02', 'BR03']:
        out, max_util = week_summary(ca, ops, br, emp_raw, bid)
        all_out[bid] = {'summary': out, 'max_util': max_util}
        print(f"\n=== {bid}, неделя (Пн-Сб) ===")
        for pol in ['naive', 'greedy', 'ilp']:
            r = out[pol]
            print(f"  {pol:7s}: ФОТ/нед={r['total_cost']:7.0f} руб., ожидание(взв.)={r['weighted_avg_wait']:5.1f} мин, "
                  f"суммарный недобор={r['total_shortfall']:.0f} окно-час, макс.загрузка сотрудника={max_util[pol]*100:.0f}%")

    with open('week_comparison.json', 'w') as f:
        json.dump({b: {'summary': {p: {k: v for k, v in d.items() if k != 'daily'} for p, d in v['summary'].items()},
                        'max_util': v['max_util']} for b, v in all_out.items()}, f, indent=2, default=str)
