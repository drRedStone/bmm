"""
schedule_greedy.py
Жадный алгоритм: по требуемому числу окон R_total/R_credit/R_mortgage(hour)
(вход из required_windows.py) распределяет сотрудников отделения по сменам
на один день, соблюдая: иерархию навыков, обеденный перерыв, дневной и
недельный лимит часов.

Параметризован по branch_id/weekday -> тривиально масштабируется на другие
отделения и дни (см. schedule_week() в конце файла).
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from required_windows import required_windows_table

from config import MAX_WORK_TIME, MAX_WORK_TIME_BEFORE_LAUNCH, MIN_WORK_TIME, WEEKDAY_HOURS, MAX_HOURS_WEEK, MIN_TIME_BETWEEN_BREAK_AND_END

SKILL_RANK = {'basic': 0, 'credit': 1, 'mortgage': 2}

@dataclass
class Pattern:
    start: int
    end: int                      # открытый интервал [start, end)
    lunch_hour: Optional[int]     # час обеда (исключается из serving_hours), None если смена короткая
    serving_hours: tuple          # часы, в которые сотрудник реально обслуживает окно

    @property
    def span(self):
        return self.end - self.start

    @property
    def paid_hours(self):
        return len(self.serving_hours)


def generate_patterns(open_hour: int, close_hour: int, max_span: int = MAX_WORK_TIME,
                       min_span: int = MIN_WORK_TIME, lunch_required_from_span: int = MAX_WORK_TIME_BEFORE_LAUNCH):
    """Генерирует непересекающиеся по смыслу варианты смен для отделения с часами [open_hour, close_hour)."""
    patterns = []
    for start in range(open_hour, close_hour):
        for span in range(min_span, max_span + 1):
            end = start + span
            if end > close_hour:
                continue
            if span == lunch_required_from_span + 1:
                continue
            if span > lunch_required_from_span:
                # обед где-то в средней трети смены, минимум 2ч работы до и 1ч после 
                for lunch_hour in range(start + MIN_WORK_TIME, min(end - MIN_TIME_BETWEEN_BREAK_AND_END,start + lunch_required_from_span)):
                    serving = tuple(h for h in range(start, end) if h != lunch_hour)
                    patterns.append(Pattern(start, end, lunch_hour, serving))
            else:
                patterns.append(Pattern(start, end, None, tuple(range(start, end))))
    return patterns

def schedule_day(employees: pd.DataFrame, req: pd.DataFrame, open_hour: int, close_hour: int,
                  n_windows_max: int, weekly_hours_used: dict = None,
                  max_hours_week: int = MAX_HOURS_WEEK, max_hours_day: int = MAX_WORK_TIME):
    """
    employees: employee_id, grade, skills(list), hourly_cost, branch_id
    req: DataFrame из required_windows_table (hour, R_total, R_credit, R_mortgage)
    weekly_hours_used: {employee_id: часов уже отработано на этой неделе} — для многодневного режима
    Возвращает: assignments (list of dict), coverage_df (итоговое покрытие по часам), unresolved (list warnings)
    """
    weekly_hours_used = dict(weekly_hours_used or {})
    hours = list(req['hour'])
    R_total = dict(zip(req['hour'], req['R_total']))
    R_credit = dict(zip(req['hour'], req['R_credit']))
    R_mortgage = dict(zip(req['hour'], req['R_mortgage']))

    cov_total = {h: 0 for h in hours}
    cov_credit = {h: 0 for h in hours}
    cov_mortgage = {h: 0 for h in hours}

    patterns = generate_patterns(open_hour, close_hour)
    assigned_today = set()
    assignments = []
    unresolved = []

    def eligible(emp_skills, tier):
        if tier == 'mortgage':
            return 'mortgage' in emp_skills
        if tier == 'credit':
            return 'credit' in emp_skills or 'mortgage' in emp_skills
        return True  # total tier — любой грейд подходит

    given_up = set()  # (tier, hour) — признанные нерешаемыми, чтобы не зацикливаться; R_* при этом не портим

    def deficit(tier, h):
        if (tier, h) in given_up:
            return 0
        if tier == 'mortgage':
            return max(0, R_mortgage[h] - cov_mortgage[h])
        if tier == 'credit':
            return max(0, R_credit[h] - cov_credit[h])
        return max(0, R_total[h] - cov_total[h])

    for tier in ['mortgage', 'credit', 'total']:
        while True:
            deficits = {h: deficit(tier, h) for h in hours}
            if max(deficits.values(), default=0) == 0:
                break
            # час с максимальным дефицитом (при равенстве — самый ранний)
            t_star = max(hours, key=lambda h: (deficits[h], -h))
            if deficits[t_star] == 0:
                break

            candidates = employees[~employees['employee_id'].isin(assigned_today)].copy()
            candidates = candidates[candidates['skills'].apply(lambda s: eligible(s, tier))]

            best = None  # (score, -cost, pattern, employee_id)
            for _, e in candidates.iterrows():
                remaining_week = max_hours_week - weekly_hours_used.get(e['employee_id'], 0)
                if remaining_week <= 0:
                    continue
                for p in patterns:
                    if t_star not in p.serving_hours:
                        continue
                    if p.paid_hours > remaining_week:
                        continue
                    # физический потолок: нельзя занять окно, если все n_windows_max уже заняты в этот час
                    if any(cov_total[h] >= n_windows_max for h in p.serving_hours):
                        continue
                    # сколько текущих дефицитов (по всем уровням) закроет эта смена
                    score = 0.0
                    for h in p.serving_hours:
                        score += deficit('total', h) * 1.0
                        score += deficit('credit', h) * 0.5 if eligible(e['skills'], 'credit') else 0
                        score += deficit('mortgage', h) * 0.5 if eligible(e['skills'], 'mortgage') else 0
                    key = (score, -e['hourly_cost_rub'], -p.paid_hours)
                    if best is None or key > best[0]:
                        best = (key, p, e)

            if best is None:
                unresolved.append(f"Час {t_star}: не нашлось сотрудника/смены, чтобы закрыть дефицит "
                                   f"({tier}, дефицит={deficits[t_star]}) — не хватает людей нужной квалификации/часов "
                                   f"или это столкнётся с лимитом окон в соседние часы смены")
                given_up.add((tier, t_star))  # не зацикливаемся, но R_total/R_credit/R_mortgage остаются как есть
                continue

            _, p, e = best
            assigned_today.add(e['employee_id'])
            weekly_hours_used[e['employee_id']] = weekly_hours_used.get(e['employee_id'], 0) + p.paid_hours
            for h in p.serving_hours:
                cov_total[h] += 1
                if eligible(e['skills'], 'credit'):
                    cov_credit[h] += 1
                if eligible(e['skills'], 'mortgage'):
                    cov_mortgage[h] += 1
            assignments.append(dict(employee_id=e['employee_id'], name=e.get('name', ''), grade=e['grade'],
                                     start=p.start, end=p.end, lunch_hour=p.lunch_hour,
                                     serving_hours=p.serving_hours, paid_hours=p.paid_hours,
                                     hourly_cost=e['hourly_cost_rub'], shift_cost=p.paid_hours * e['hourly_cost_rub']))

    coverage_df = pd.DataFrame({
        'hour': hours,
        'R_total': [R_total[h] for h in hours], 'cov_total': [cov_total[h] for h in hours],
        'R_credit': [R_credit[h] for h in hours], 'cov_credit': [cov_credit[h] for h in hours],
        'R_mortgage': [R_mortgage[h] for h in hours], 'cov_mortgage': [cov_mortgage[h] for h in hours],
    })
    return assignments, coverage_df, unresolved, weekly_hours_used

def schedule_week(client_arrivals: pd.DataFrame, operations: pd.DataFrame, employees: pd.DataFrame,
                   branches: pd.DataFrame, branch_id: str):
    """
    Масштабирование schedule_day на всю рабочую неделю одного отделения.
    Часы, использованные сотрудником, переносятся день в день (недельный лимит 40ч соблюдается сквозно).
    Возвращает: dict{weekday: (assignments, coverage, unresolved)}, итоговую сводку по сотрудникам.
    """
    n_win = int(branches.loc[branch_id, 'n_windows'])
    branch_emp = employees[employees['branch_id'] == branch_id]
    weekly_hours_used = {}
    results = {}
    for weekday, (open_h, close_h) in WEEKDAY_HOURS.items():
        req = required_windows_table(client_arrivals, operations, branch_id, weekday, n_win)
        if req.empty:
            continue
        assignments, coverage, unresolved, weekly_hours_used = schedule_day(
            branch_emp, req, open_h, close_h, n_windows_max=n_win,
            weekly_hours_used=weekly_hours_used)  # <- перенос часов между днями недели
        results[weekday] = dict(assignments=assignments, coverage=coverage, unresolved=unresolved)

    summary = pd.DataFrame([
        dict(employee_id=eid, hours_this_week=h, max_hours_week=MAX_HOURS_WEEK, utilization=round(h/MAX_HOURS_WEEK, 2))
        for eid, h in weekly_hours_used.items()
    ]).sort_values('hours_this_week', ascending=False)
    return results, summary


if __name__ == '__main__':
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
    emp = pd.read_csv('dataset/employees.csv')
    emp['skills'] = emp['skills'].str.split(',')

    BRANCH, WEEKDAY_EN = 'BR01', 'Monday'  #да бл откуда берется эта хрень?
    n_win = int(br.loc[BRANCH, 'n_windows'])
    open_h, close_h = WEEKDAY_HOURS[WEEKDAY_EN] # из hours_weekday «09:00-19:00»

    req = required_windows_table(ca, ops, BRANCH, WEEKDAY_EN, n_win)
    branch_emp = emp[emp['branch_id'] == BRANCH]

    assignments, coverage, unresolved, hours_used = schedule_day(
        branch_emp, req, open_h, close_h, n_windows_max=n_win)

    print(f"=== Расписание {BRANCH}, понедельник ===\n")
    sched_df = pd.DataFrame(assignments)
    print(sched_df[['employee_id', 'grade', 'start', 'end', 'lunch_hour', 'paid_hours', 'shift_cost']]
          .sort_values('start').to_string(index=False))

    print(f"\nСотрудников задействовано: {len(assignments)} из {len(branch_emp)}")
    print(f"Суммарный ФОТ за день: {sched_df['shift_cost'].sum():.0f} руб.")

    print("\n=== Покрытие по часам (требуется / открыто) ===")
    print(coverage.to_string(index=False))

    if unresolved:
        print("\n=== Нерешённые дефициты ===")
        for u in unresolved:
            print(" -", u)
    else:
        print("\nВсе требования по окнам закрыты штатом отделения.")
