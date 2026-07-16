"""
compare_branches.py
====================
То же сравнение naive/greedy/ILP, что и в compare_policies.py, но по всем
трём отделениям сразу (один день — понедельник) — чтобы увидеть, как разрыв
между жадным и ILP меняется в зависимости от состава персонала отделения
(особенно BR03, где меньше всего гибкости по кадрам).
"""
import pandas as pd
import numpy as np
from required_windows import required_windows_table
from schedule_greedy import schedule_day
from schedule_ilp import schedule_day_ilp_soft
from compare_policies import naive_schedule, coverage_from_assignments, avg_wait_for_coverage

from config import WEEKDAY_HOURS


WEEKDAY_EN = 'Monday' # Это для таблицы сравнения графиков по отделниям на 1 день (на понедельник)


def compare_branch(ca: pd.DataFrame, ops: pd.DataFrame, br: pd.DataFrame,
                    emp_raw: pd.DataFrame, branch_id: str) -> dict:
    """
    Считает наивную/жадную/ILP политику для ОДНОГО отделения в понедельник
    и собирает сводку по каждой (среднее ожидание, ФОТ, число сотрудников,
    недобор total-уровня), плюс справочные метрики состава штата отделения.

    Args:
        ca: датафрейм client_arrivals.csv.
        ops: датафрейм operations.csv.
        br: датафрейм branches.csv, индексированный по branch_id.
        emp_raw: полный датафрейм employees.csv (столбец skills уже должен
            быть распарсен в list через .str.split(',')).
        branch_id: идентификатор отделения.

    Returns:
        dict с ключами 'naive', 'greedy', 'ilp' (каждый — dict с avg_wait,
        cost, n_staff, shortfall_total) и справочными ключами
        n_employees_total, n_credit_plus, n_senior (состав штата отделения —
        удобно для подписей на графиках и объяснения различий между
        отделениями).
    """
    n_win = int(br.loc[branch_id, 'n_windows'])
    req = required_windows_table(ca, ops, branch_id, WEEKDAY_EN, n_win)
    branch_emp = emp_raw[emp_raw['branch_id'] == branch_id].reset_index(drop=True)
    hours = list(req['hour'])

    a_naive = naive_schedule(branch_emp, n_win)
    a_greedy, cov_greedy_df, unresolved, _ = schedule_day(branch_emp, req, *WEEKDAY_HOURS[WEEKDAY_EN], n_windows_max=n_win)
    a_ilp, cov_ilp_df, status, _, shortfall = schedule_day_ilp_soft(branch_emp, req, *WEEKDAY_HOURS[WEEKDAY_EN], n_win)

    out = {}
    for name, assigns in [('naive', a_naive), ('greedy', a_greedy), ('ilp', a_ilp)]:
        cov = coverage_from_assignments(assigns, hours)
        waits, weights = avg_wait_for_coverage(req, cov)
        out[name] = dict(avg_wait=np.average(waits, weights=weights),
                          cost=sum(a['shift_cost'] for a in assigns),
                          n_staff=len(assigns),
                          shortfall_total=sum(max(0, r - cov[h]) for h, r in zip(hours, req['R_total'])))
    out['n_employees_total'] = len(branch_emp)
    out['n_credit_plus'] = int(branch_emp['skills'].apply(lambda s: 'credit' in s or 'mortgage' in s).sum())
    out['n_senior'] = int((branch_emp['grade'] == 'senior').sum())
    return out


if __name__ == '__main__':
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
    emp_raw = pd.read_csv('dataset/employees.csv')
    emp_raw['skills'] = emp_raw['skills'].str.split(',')

    all_results = {}
    for bid in ['BR01', 'BR02', 'BR03']:
        res = compare_branch(ca, ops, br, emp_raw, bid)
        all_results[bid] = res
        print(f"\n=== {bid} (senior={res['n_senior']}, credit+={res['n_credit_plus']}, всего={res['n_employees_total']}) ===")
        for pol in ['naive', 'greedy', 'ilp']:
            r = res[pol]
            print(f"  {pol:7s}: сотрудников={r['n_staff']}, ФОТ={r['cost']:6.0f} руб., "
                  f"ожидание={r['avg_wait']:5.1f} мин, недобор total-окно-час={r['shortfall_total']}")
        gap = res['greedy']['avg_wait'] - res['ilp']['avg_wait']
        print(f"  >> Разрыв greedy-ILP по ожиданию: {gap:.1f} мин")

    import json
    with open('branch_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
