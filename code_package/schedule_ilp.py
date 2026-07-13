"""
schedule_ilp.py
================
Точная (не эвристическая) версия планировщика расписания через целочисленное
линейное программирование (ILP).

Формальная постановка — задача set-covering по паттернам смен (тот же набор
паттернов, что генерирует schedule_greedy.generate_patterns):
    Переменные:   z(e, p) in {0, 1} — сотрудник e работает по паттерну смены p.
    Минимизация:  суммарный ФОТ = sum(z(e,p) * hourly_cost(e) * paid_hours(p)).
    Ограничения:  покрытие требуемых окон (R_total/R_credit/R_mortgage) по
                  каждому часу, не более одной смены на сотрудника в день,
                  недельный лимит часов, физический потолок окон отделения.

Решается через scipy.optimize.milp (солвер HiGHS, входит в scipy "из
коробки"). Даёт ДВЕ версии:
    - schedule_day_ilp       — жёсткая: покрытие обязательно (constraint),
      может оказаться infeasible, если ресурсов физически не хватает даже
      теоретически (см. пример использования в конце файла — на BR01 эта
      версия доказуемо infeasible).
    - schedule_day_ilp_soft  — мягкая: недобор допускается через штрафные
      переменные (используется во всех сравнениях compare_*.py, т.к. всегда
      разрешима).

Тот же вход/выход, что у schedule_greedy.schedule_day — их результаты
напрямую сравнимы (см. compare_policies.py).
"""
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from schedule_greedy import generate_patterns, Pattern
from required_windows import required_windows_table


def schedule_day_ilp(employees: pd.DataFrame, req: pd.DataFrame, open_hour: int, close_hour: int,
                      n_windows_max: int, max_hours_week: int = 40, weekly_hours_used: dict = None):
    """
    Жёсткая версия ILP: находит расписание МИНИМАЛЬНОЙ стоимости, которое
    ОБЯЗАТЕЛЬНО закрывает весь требуемый спрос (R_total/R_credit/R_mortgage)
    по каждому часу. Если это физически невозможно при данном штате и
    правилах смен — solver вернёт infeasible (это не баг, а содержательный
    результат: доказательство нехватки ресурса, см. schedule_ilp.py __main__
    и раздел 4.2 отчёта — на BR01 инфизибилити доказана строго).

    Переменные: одна бинарная переменная z(e,p) на каждую физически
    допустимую пару (сотрудник e, паттерн смены p) — "физически допустимую"
    значит, что у сотрудника хватает оставшихся недельных часов на этот
    паттерн (remaining_week >= p.paid_hours).

    Args:
        employees: датафрейм сотрудников ОДНОГО отделения (employee_id,
            grade, skills, hourly_cost_rub).
        req: датафрейм из required_windows_table (hour, R_total, R_credit,
            R_mortgage) для этого отделения/дня недели.
        open_hour: час открытия отделения.
        close_hour: час закрытия отделения.
        n_windows_max: физическое число окон отделения — верхняя граница
            суммарного покрытия total в любой час.
        max_hours_week: недельный лимит часов на сотрудника.
        weekly_hours_used: {employee_id: часов уже отработано на этой
            неделе} — для многодневного режима (перенос между днями).

    Returns:
        Если решение найдено: кортеж (assignments, coverage_df, 'optimal',
        new_weekly_hours_used) — тот же формат, что у schedule_greedy.schedule_day.
        Если задача infeasible: ([], None, сообщение_солвера,
        weekly_hours_used) — сообщение солвера объясняет причину.
    """
    weekly_hours_used = weekly_hours_used or {}
    hours = list(req['hour'])
    R_total = dict(zip(req['hour'], req['R_total']))
    R_credit = dict(zip(req['hour'], req['R_credit']))
    R_mortgage = dict(zip(req['hour'], req['R_mortgage']))

    patterns = generate_patterns(open_hour, close_hour)
    emp_list = employees.to_dict('records')

    # переменные: все пары (сотрудник, паттерн), которые физически возможны
    # (у сотрудника хватает недельных часов на этот паттерн)
    var_emp_idx, var_pat_idx = [], []
    for ei, e in enumerate(emp_list):
        remaining_week = max_hours_week - weekly_hours_used.get(e['employee_id'], 0)
        for pi, p in enumerate(patterns):
            if p.paid_hours <= remaining_week:
                var_emp_idx.append(ei)
                var_pat_idx.append(pi)
    n_vars = len(var_emp_idx)

    # целевая функция: стоимость смены = ставка сотрудника * оплачиваемые часы паттерна
    cost = np.array([emp_list[var_emp_idx[i]]['hourly_cost_rub'] * patterns[var_pat_idx[i]].paid_hours
                      for i in range(n_vars)])

    def can(skills, tier):
        """Может ли сотрудник с данными навыками закрывать окно уровня tier."""
        if tier == 'mortgage':
            return 'mortgage' in skills
        if tier == 'credit':
            return 'credit' in skills or 'mortgage' in skills
        return True

    constraints = []
    rows, lb, ub = [], [], []

    def add_row(coeffs_idx, lo, hi):
        """Добавляет одну строку линейного ограничения lo <= sum(x[coeffs_idx]) <= hi."""
        row = np.zeros(n_vars)
        row[coeffs_idx] = 1.0
        rows.append(row); lb.append(lo); ub.append(hi)

    for h in hours:
        idx_total = [i for i in range(n_vars) if h in patterns[var_pat_idx[i]].serving_hours]
        idx_credit = [i for i in idx_total if can(emp_list[var_emp_idx[i]]['skills'], 'credit')]
        idx_mortgage = [i for i in idx_total if can(emp_list[var_emp_idx[i]]['skills'], 'mortgage')]
        # R_total(h) <= sum <= n_windows_max   (спрос снизу, физический потолок сверху)
        add_row(idx_total, R_total[h], n_windows_max)
        add_row(idx_credit, R_credit[h], n_vars)      # верхнюю границу не ограничиваем отдельно — её держит total
        add_row(idx_mortgage, R_mortgage[h], n_vars)

    # один паттерн на сотрудника в день (сотрудник либо работает одну смену, либо не работает)
    for ei in range(len(emp_list)):
        idx = [i for i in range(n_vars) if var_emp_idx[i] == ei]
        if idx:
            add_row(idx, 0, 1)

    A = np.array(rows)
    constraints = [LinearConstraint(A, lb, ub)]
    bounds = Bounds(0, 1)
    integrality = np.ones(n_vars)  # все переменные бинарные (0/1)

    res = milp(cost, constraints=constraints, bounds=bounds, integrality=integrality)

    if not res.success:
        return [], None, res.message, weekly_hours_used

    # восстанавливаем расписание из вектора решения x (округляем — HiGHS иногда
    # возвращает 0.999999 вместо 1.0 из-за численной точности солвера)
    x = np.round(res.x).astype(int)
    assignments = []
    new_weekly = dict(weekly_hours_used)
    cov_total = {h: 0 for h in hours}
    cov_credit = {h: 0 for h in hours}
    cov_mortgage = {h: 0 for h in hours}

    for i in range(n_vars):
        if x[i] == 1:
            e = emp_list[var_emp_idx[i]]
            p = patterns[var_pat_idx[i]]
            new_weekly[e['employee_id']] = new_weekly.get(e['employee_id'], 0) + p.paid_hours
            for h in p.serving_hours:
                cov_total[h] += 1
                if can(e['skills'], 'credit'):
                    cov_credit[h] += 1
                if can(e['skills'], 'mortgage'):
                    cov_mortgage[h] += 1
            assignments.append(dict(employee_id=e['employee_id'], grade=e['grade'],
                                     start=p.start, end=p.end, lunch_hour=p.lunch_hour,
                                     serving_hours=p.serving_hours, paid_hours=p.paid_hours,
                                     hourly_cost=e['hourly_cost_rub'], shift_cost=p.paid_hours * e['hourly_cost_rub']))

    coverage_df = pd.DataFrame({
        'hour': hours,
        'R_total': [R_total[h] for h in hours], 'cov_total': [cov_total[h] for h in hours],
        'R_credit': [R_credit[h] for h in hours], 'cov_credit': [cov_credit[h] for h in hours],
        'R_mortgage': [R_mortgage[h] for h in hours], 'cov_mortgage': [cov_mortgage[h] for h in hours],
    })
    return assignments, coverage_df, 'optimal', new_weekly


def schedule_day_ilp_soft(employees: pd.DataFrame, req: pd.DataFrame, open_hour: int, close_hour: int,
                           n_windows_max: int, max_hours_week: int = 40, weekly_hours_used: dict = None,
                           penalty_total: float = 8000, penalty_credit: float = 6000, penalty_mortgage: float = 6000):
    """
    Мягкая версия ILP: физический потолок окон (n_windows_max) остаётся
    жёстким ограничением (его нарушить нельзя — это реальная стена, не
    больше окон, чем есть), а недобор по R_total/R_credit/R_mortgage
    допускается через неотрицательные штрафные переменные-слэки
    u_total(h), u_credit(h), u_mortgage(h) — после того, как было строго
    доказано (schedule_day_ilp на BR01), что жёсткая версия иногда попросту
    infeasible при разумном штате и правилах смен.

    Целевая функция = ФОТ + штраф*недобор. Штрафы (penalty_*) на порядок
    больше типичной стоимости смены (350-700 руб./час), поэтому солвер
    сначала минимизирует недобор (закрывает спрос максимально полно), и
    только при равном недоборе — экономит на ФОТ. Это соответствует
    приоритету брифа: "минимизировать ожидание клиента при бюджетных
    ограничениях", где недобор напрямую увеличивает реальное ожидание.

    Переменные вектора решения: сначала n_z бинарных z(e,p) (как в
    schedule_day_ilp), затем 3*n_h непрерывных слэк-переменных (по одной
    на каждый час для каждого из 3 уровней требования).

    Args:
        employees, req, open_hour, close_hour, n_windows_max,
            max_hours_week, weekly_hours_used: см. schedule_day_ilp.
        penalty_total: штраф за 1 час недобора total-уровня.
        penalty_credit: штраф за 1 час недобора credit-уровня.
        penalty_mortgage: штраф за 1 час недобора mortgage-уровня.

    Returns:
        Если решение найдено: кортеж (assignments, coverage_df, 'optimal',
        new_weekly_hours_used, shortfall_summary), где:
        - coverage_df дополнительно содержит столбцы shortfall_total/
          shortfall_credit/shortfall_mortgage по часам;
        - shortfall_summary: dict(total=…, credit=…, mortgage=…) — суммарный
          недобор за день по каждому уровню (в окно-часах).
        Если infeasible (не должно происходить при разумных параметрах,
        т.к. недобор всегда можно "закрыть" штрафной переменной):
        ([], None, сообщение, weekly_hours_used, None).
    """
    weekly_hours_used = weekly_hours_used or {}
    hours = list(req['hour'])
    R_total = dict(zip(req['hour'], req['R_total']))
    R_credit = dict(zip(req['hour'], req['R_credit']))
    R_mortgage = dict(zip(req['hour'], req['R_mortgage']))

    patterns = generate_patterns(open_hour, close_hour)
    emp_list = employees.to_dict('records')

    var_emp_idx, var_pat_idx = [], []
    for ei, e in enumerate(emp_list):
        remaining_week = max_hours_week - weekly_hours_used.get(e['employee_id'], 0)
        for pi, p in enumerate(patterns):
            if p.paid_hours <= remaining_week:
                var_emp_idx.append(ei)
                var_pat_idx.append(pi)
    n_z = len(var_emp_idx)  # число "рабочих" (сотрудник, паттерн) бинарных переменных
    n_h = len(hours)
    # порядок доп. переменных в векторе решения: u_total(h...), u_credit(h...), u_mortgage(h...)
    n_vars = n_z + 3 * n_h

    def can(skills, tier):
        """Может ли сотрудник с данными навыками закрывать окно уровня tier."""
        if tier == 'mortgage':
            return 'mortgage' in skills
        if tier == 'credit':
            return 'credit' in skills or 'mortgage' in skills
        return True

    # целевая функция: ФОТ по бинарным переменным + штраф по слэк-переменным
    cost = np.zeros(n_vars)
    cost[:n_z] = [emp_list[var_emp_idx[i]]['hourly_cost_rub'] * patterns[var_pat_idx[i]].paid_hours
                  for i in range(n_z)]
    cost[n_z:n_z + n_h] = penalty_total
    cost[n_z + n_h:n_z + 2 * n_h] = penalty_credit
    cost[n_z + 2 * n_h:n_z + 3 * n_h] = penalty_mortgage

    rows, lb, ub = [], [], []

    def add_row(z_idx, extra_idx, extra_coef, lo, hi):
        """
        Добавляет строку линейного ограничения lo <= sum(z[z_idx]) + extra_coef*x[extra_idx] <= hi.
        extra_idx=None означает ограничение без слэк-переменной (например, физический потолок).
        """
        row = np.zeros(n_vars)
        row[z_idx] = 1.0
        if extra_idx is not None:
            row[extra_idx] = extra_coef
        rows.append(row); lb.append(lo); ub.append(hi)

    for hi_, h in enumerate(hours):
        idx_total = [i for i in range(n_z) if h in patterns[var_pat_idx[i]].serving_hours]
        idx_credit = [i for i in idx_total if can(emp_list[var_emp_idx[i]]['skills'], 'credit')]
        idx_mortgage = [i for i in idx_total if can(emp_list[var_emp_idx[i]]['skills'], 'mortgage')]
        u_t, u_c, u_m = n_z + hi_, n_z + n_h + hi_, n_z + 2 * n_h + hi_
        add_row(idx_total, u_t, 1.0, R_total[h], np.inf)   # sum + u_total >= R_total  (нижняя граница по спросу, со слэком)
        add_row(idx_total, None, 0, 0, n_windows_max)        # sum <= n_windows_max     (физика — жёстко, без слэка)
        add_row(idx_credit, u_c, 1.0, R_credit[h], np.inf)
        add_row(idx_mortgage, u_m, 1.0, R_mortgage[h], np.inf)

    # один паттерн на сотрудника в день
    for ei in range(len(emp_list)):
        idx = [i for i in range(n_z) if var_emp_idx[i] == ei]
        if idx:
            add_row(idx, None, 0, 0, 1)

    A = np.array(rows)
    constraints = [LinearConstraint(A, lb, ub)]
    # бинарные z в [0,1], слэк-переменные u в [0, n_windows_max] (недобор физически
    # не может превышать сам физический потолок окон)
    bounds = Bounds(np.concatenate([np.zeros(n_z), np.zeros(3 * n_h)]),
                     np.concatenate([np.ones(n_z), np.full(3 * n_h, n_windows_max, dtype=float)]))
    integrality = np.concatenate([np.ones(n_z), np.zeros(3 * n_h)])  # z бинарные, u непрерывные

    res = milp(cost, constraints=constraints, bounds=bounds, integrality=integrality)
    if not res.success:
        return [], None, res.message, weekly_hours_used, None

    x = res.x
    z = np.round(x[:n_z]).astype(int)  # округляем бинарную часть (численная точность солвера)
    u_total = x[n_z:n_z + n_h]
    u_credit = x[n_z + n_h:n_z + 2 * n_h]
    u_mortgage = x[n_z + 2 * n_h:n_z + 3 * n_h]

    assignments = []
    new_weekly = dict(weekly_hours_used)
    cov_total = {h: 0 for h in hours}
    cov_credit = {h: 0 for h in hours}
    cov_mortgage = {h: 0 for h in hours}
    for i in range(n_z):
        if z[i] == 1:
            e = emp_list[var_emp_idx[i]]
            p = patterns[var_pat_idx[i]]
            new_weekly[e['employee_id']] = new_weekly.get(e['employee_id'], 0) + p.paid_hours
            for h in p.serving_hours:
                cov_total[h] += 1
                if can(e['skills'], 'credit'):
                    cov_credit[h] += 1
                if can(e['skills'], 'mortgage'):
                    cov_mortgage[h] += 1
            assignments.append(dict(employee_id=e['employee_id'], grade=e['grade'],
                                     start=p.start, end=p.end, lunch_hour=p.lunch_hour,
                                     serving_hours=p.serving_hours, paid_hours=p.paid_hours,
                                     hourly_cost=e['hourly_cost_rub'], shift_cost=p.paid_hours * e['hourly_cost_rub']))

    coverage_df = pd.DataFrame({
        'hour': hours,
        'R_total': [R_total[h] for h in hours], 'cov_total': [cov_total[h] for h in hours],
        'R_credit': [R_credit[h] for h in hours], 'cov_credit': [cov_credit[h] for h in hours],
        'R_mortgage': [R_mortgage[h] for h in hours], 'cov_mortgage': [cov_mortgage[h] for h in hours],
        'shortfall_total': np.round(u_total, 2), 'shortfall_credit': np.round(u_credit, 2),
        'shortfall_mortgage': np.round(u_mortgage, 2),
    })
    shortfall_summary = dict(total=u_total.sum(), credit=u_credit.sum(), mortgage=u_mortgage.sum())
    return assignments, coverage_df, 'optimal', new_weekly, shortfall_summary


WEEKDAY_HOURS = {  # будни 09-19, суббота 09-16, воскресенье закрыто (branches.csv одинаков для всех отделений)
    'Monday': (9, 19), 'Tuesday': (9, 19), 'Wednesday': (9, 19),
    'Thursday': (9, 19), 'Friday': (9, 19), 'Saturday': (9, 16),
}


def schedule_week_ilp(client_arrivals: pd.DataFrame, operations: pd.DataFrame, employees: pd.DataFrame,
                       branches: pd.DataFrame, branch_id: str):
    """
    ILP-аналог schedule_greedy.schedule_week: тот же перенос отработанных
    часов между днями недели (недельный лимит 40ч соблюдается сквозно), но
    каждый день решается через schedule_day_ilp_soft (глобальный оптимум на
    день, а не жадный локальный выбор) — отсюда систематическое преимущество
    над schedule_week в сравнениях week_pipeline.py.

    Args:
        client_arrivals, operations: сырые датафреймы (client_arrivals.csv,
            operations.csv).
        employees: полный датафрейм сотрудников (все отделения).
        branches: датафрейм branches.csv, индексированный по branch_id.
        branch_id: идентификатор отделения.

    Returns:
        Кортеж (results, summary):
        - results: dict{weekday: {'assignments':…, 'coverage':…,
          'status':…, 'shortfall':…}};
        - summary: DataFrame по сотрудникам с недельной загрузкой
          (hours_this_week, utilization).
    """
    n_win = int(branches.loc[branch_id, 'n_windows'])
    branch_emp = employees[employees['branch_id'] == branch_id]
    weekly_hours_used = {}
    results = {}
    for weekday, (open_h, close_h) in WEEKDAY_HOURS.items():
        req = required_windows_table(client_arrivals, operations, branch_id, weekday, n_win)
        if req.empty:
            continue
        assignments, coverage, status, weekly_hours_used, shortfall = schedule_day_ilp_soft(
            branch_emp, req, open_h, close_h, n_win, weekly_hours_used=weekly_hours_used)
        results[weekday] = dict(assignments=assignments, coverage=coverage, status=status, shortfall=shortfall)

    summary = pd.DataFrame([
        dict(employee_id=eid, hours_this_week=h, utilization=round(h/40, 2))
        for eid, h in weekly_hours_used.items()
    ]).sort_values('hours_this_week', ascending=False)
    return results, summary


if __name__ == '__main__':
    from schedule_greedy import schedule_day

    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
    emp = pd.read_csv('dataset/employees.csv')
    emp['skills'] = emp['skills'].str.split(',')

    BRANCH, WEEKDAY_EN = 'BR01', 'Monday'
    n_win = int(br.loc[BRANCH, 'n_windows'])
    req = required_windows_table(ca, ops, BRANCH, WEEKDAY_EN, n_win)
    branch_emp = emp[emp['branch_id'] == BRANCH]

    print("=== ЖЁСТКАЯ версия (доказывает infeasibility) ===")
    _, _, status_hard, _ = schedule_day_ilp(branch_emp, req, 9, 19, n_win)
    print("Статус:", status_hard)

    print("\n=== МЯГКАЯ версия (ILP, глобальный оптимум) ===")
    assignments_ilp, coverage_ilp, status, hu, shortfall = schedule_day_ilp_soft(branch_emp, req, 9, 19, n_win)
    print("Статус:", status, "| Недобор (total/credit/mortgage):", shortfall)
    sched_ilp = pd.DataFrame(assignments_ilp)
    print(sched_ilp[['employee_id', 'grade', 'start', 'end', 'lunch_hour', 'paid_hours', 'shift_cost']]
          .sort_values('start').to_string(index=False))
    print(f"Сотрудников: {len(assignments_ilp)}, ФОТ за день: {sched_ilp['shift_cost'].sum():.0f} руб.")

    print("\n=== ЖАДНЫЙ (для сравнения) ===")
    assignments_greedy, coverage_greedy, unresolved, _ = schedule_day(branch_emp, req, 9, 19, n_windows_max=n_win)
    sched_greedy = pd.DataFrame(assignments_greedy)
    print(f"Сотрудников: {len(assignments_greedy)}, ФОТ за день: {sched_greedy['shift_cost'].sum():.0f} руб.")
    greedy_shortfall_total = (coverage_greedy['R_total'] - coverage_greedy['cov_total']).clip(lower=0).sum()
    print(f"Недобор total-окно-часов (жадный): {greedy_shortfall_total}")
    ilp_shortfall_total = (coverage_ilp['R_total'] - coverage_ilp['cov_total']).clip(lower=0).sum()
    print(f"Недобор total-окно-часов (ILP): {ilp_shortfall_total}")
