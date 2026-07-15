import pandas as pd
import numpy as np
from week_pipeline import schedule_week_ilp
from config import *


def create_staff(n_junior: int, n_middle: int, n_senior: int, branch_id: str) -> pd.DataFrame:
    """
    Создаёт DataFrame сотрудников для заданного отделения и количества грейдов.
    Возвращает колонки: employee_id, branch_id, name, grade, skills,
    max_hours_week, max_hours_day, lunch_min, hourly_cost_rub.
    """
    employees = []
    emp_id = 1

    for grade, count in [('junior', n_junior), ('middle', n_middle), ('senior', n_senior)]:
        for _ in range(count):
            employees.append({
                'employee_id': f'synth_{emp_id}',
                'branch_id': branch_id,
                'name': None,
                'grade': grade,
                'skills': SKILLS[grade],
                'max_hours_week': 40,
                'max_hours_day': 9,
                'lunch_min': 60,
                'hourly_cost_rub': WERGES[grade]
            })
            emp_id += 1

    return pd.DataFrame(employees)

def opt_fot(branch_id: str,
            client_arrivals: pd.DataFrame,
            opertions: pd.DataFrame,
            branches: pd.DataFrame,
            employees: pd.DataFrame = None,
            max_emps_rel_wins: int = 2):
    
    n = branches.loc(branch_id, "n_winodws")

    
    for emp_count in range(3, n*max_emps_rel_wins):
        for jun in range(3, emp_count):
            for mid in range(emp_count-jun):
                sen = emp_count-jun-mid
                if sen == 0: continue

                sched = schedule_week_ilp(
                    client_arrivals,
                    opertions,
                    create_staff(jun, mid, sen, branch_id),
                    branches,
                    branch_id
                )





def optimize_staff_by_budget(budget, branch_id, ca, ops, br, emp_raw=None,
                             max_junior=8, max_middle=8, max_senior=8,
                             penalty_mortgage=10000):
    """
    Подбирает состав штата (junior, middle, senior) так, чтобы:
    - недельный ФОТ не превышал budget
    - минимизировать взвешенный недобор (total + credit + mortgage*penalty_mortgage)
    """
    # Определяем средние ставки по грейдам
    if emp_raw is not None:
        branch_emp = emp_raw[emp_raw['branch_id'] == branch_id]
        avg_costs = {
            'junior': branch_emp[branch_emp['grade'] == 'junior']['hourly_cost_rub'].mean(),
            'middle': branch_emp[branch_emp['grade'] == 'middle']['hourly_cost_rub'].mean(),
            'senior': branch_emp[branch_emp['grade'] == 'senior']['hourly_cost_rub'].mean()
        }
        # если каких-то грейдов нет – берём общие средние
        for g in ['junior', 'middle', 'senior']:
            if pd.isna(avg_costs[g]):
                avg_costs[g] = emp_raw[emp_raw['grade'] == g]['hourly_cost_rub'].mean()
    else:
        avg_costs = {'junior': 350, 'middle': 500, 'senior': 700}

    n_windows = int(br.loc[branch_id, 'n_windows'])

    best_shortfall = float('inf')
    best_composition = None
    best_cost = 0
    best_results = None

    total_combinations = (max_junior+1)*(max_middle+1)*(max_senior+1)
    print(f"Начинаем перебор {total_combinations} комбинаций для {branch_id}...")

    for j in range(max_junior+1):
        for m in range(max_middle+1):
            for s in range(max_senior+1):
                # Ограничим общее количество сотрудников, чтобы не перебирать лишнее
                if j + m + s > 2 * n_windows:
                    continue

                # Создаём синтетический штат
                employees = []
                emp_id = 1
                for _ in range(j):
                    employees.append({
                        'employee_id': f'synth_{emp_id}',
                        'branch_id': branch_id,
                        'grade': 'junior',
                        'skills': ['basic'],
                        'hourly_cost_rub': avg_costs['junior']
                    })
                    emp_id += 1
                for _ in range(m):
                    employees.append({
                        'employee_id': f'synth_{emp_id}',
                        'branch_id': branch_id,
                        'grade': 'middle',
                        'skills': ['basic', 'credit'],
                        'hourly_cost_rub': avg_costs['middle']
                    })
                    emp_id += 1
                for _ in range(s):
                    employees.append({
                        'employee_id': f'synth_{emp_id}',
                        'branch_id': branch_id,
                        'grade': 'senior',
                        'skills': ['basic', 'credit', 'mortgage'],
                        'hourly_cost_rub': avg_costs['senior']
                    })
                    emp_id += 1

                # Вот где создаётся emp_df! (раньше было пропущено)
                emp_df = pd.DataFrame(employees)

                # Запускаем ILP на неделю
                try:
                    results, summary = schedule_week_ilp(ca, ops, emp_df, br, branch_id)
                except Exception as e:
                    continue

                # Считаем ФОТ и взвешенный недобор
                total_cost = 0
                weighted_shortfall = 0
                for day_data in results.values():
                    if 'assignments' in day_data:
                        total_cost += sum(a['shift_cost'] for a in day_data['assignments'])
                    if 'shortfall' in day_data and day_data['shortfall'] is not None:
                        sf = day_data['shortfall']
                        weighted_shortfall += sf.get('total', 0) + sf.get('credit', 0) + sf.get('mortgage', 0) * penalty_mortgage

                if total_cost > budget:
                    continue

                if weighted_shortfall < best_shortfall:
                    best_shortfall = weighted_shortfall
                    best_composition = (j, m, s)
                    best_cost = total_cost
                    best_results = results

    return {
        'best_composition': best_composition,
        'best_shortfall': best_shortfall,
        'best_cost': best_cost,
        'results': best_results
    }


ca = pd.read_csv('dataset/client_arrivals.csv')
ops = pd.read_csv('dataset/operations.csv')
br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
emp = pd.read_csv('dataset/employees.csv')
emp['skills'] = emp['skills'].str.split(',')

budget = 10*10000  # недельный бюджет

result = optimize_staff_by_budget(
    budget=budget,
    branch_id='BR01',
    ca=ca, ops=ops, br=br,
    emp_raw=emp,
    max_junior=5, max_middle=5, max_senior=5,
    penalty_mortgage=10000   # вес mortgage-недобора (можно менять)
)

if result['best_composition'] is not None:
    j, m, s = result['best_composition']
    print(f"Оптимальный состав: junior={j}, middle={m}, senior={s}")
    print(f"ФОТ за неделю: {result['best_cost']}, взвешенный недобор: {result['best_shortfall']}")
else:
    print("Ни одна комбинация не вписалась в бюджет. Увеличьте бюджет или расширьте диапазон.")