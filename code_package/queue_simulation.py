"""
queue_simulation.py
====================
Дискретно-событийная симуляция системы массового обслуживания (M/M/c) на
SimPy — независимая альтернатива аналитической формуле Эрланга C
(required_windows.erlang_c_wait_minutes), реализующая must-have пункт
брифа: "Реализовать симуляцию... клиенты приходят -> встают в очередь ->
обслуживаются у свободного окна -> уходят".

Три сценария использования:
    1. simulate_mmc — базовая валидация: постоянные λ/μ/c, сравнение
    среднего ожидания в симуляции с формулой Эрланга C (должны совпадать
    с точностью до статистического шума — см. __main__ ниже).
    2. simulate_day_continuous — реалистичная симуляция целого рабочего
    дня с меняющейся по часам нагрузкой; очередь НЕ сбрасывается на
    границе часа (в отличие от почасовой формулы Эрланга C, которая
    предполагает, что нагрузка каждого часа держится бесконечно).
    Показывает, что почасовая формула может как завышать, так и
    занижать реальное ожидание — см. раздел 3.3 отчёта.
    3. empirical_service_sampler — проверка допущения об экспоненциальном
    времени обслуживания: позволяет прогнать симуляцию с РЕАЛЬНЫМ
    эмпирическим распределением длительности операций вместо экспоненты.
"""
from typing import Callable, Optional
import simpy
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from compare_policies import naive_schedule

from required_windows import erlang_c_wait_minutes, required_windows_table
from schedule_ilp import schedule_day_ilp_soft, schedule_week_ilp
from schedule_greedy import schedule_day as schedule_day_greedy, schedule_week as schedule_week_greedy
from compare_policies import (
    coverage_from_assignments,
    avg_wait_for_coverage,
    get_weekly_metrics,
    plot_daily_schedule,
    plot_weekly_schedule,
    weekly_schedule_to_json,
)
from config import WEEKDAY_HOURS, SIM_REPORT_SAVE_DIRECTORY


# ----------------------------------------------------------------------
# 1. Базовые симуляции
# ----------------------------------------------------------------------
def simulate_mmc(lam_per_hour: float, mu_per_hour: float, c: int, sim_hours: float = 200,
                warmup_hours: float = 20, seed: Optional[int] = None,
                service_sampler: Optional[Callable] = None) -> np.ndarray:
    """
    Прогоняет одну непрерывную симуляцию M/M/c с ПОСТОЯННЫМИ λ/μ/c в
    течение sim_hours часов и возвращает времена ожидания клиентов.

    Используется как базовая валидация: при достаточно большом sim_hours
    среднее время ожидания в выборке должно сходиться к аналитическому
    значению erlang_c_wait_minutes(lam_per_hour, mu_per_hour, c) — если нет,
    в реализации (симуляции или формулы) есть ошибка.

    Args:
        lam_per_hour: интенсивность входящего потока, клиентов/час.
        mu_per_hour: интенсивность обслуживания одним окном, клиентов/час.
        c: число обслуживающих окон.
        sim_hours: длительность симуляции в модельных часах. Чем ближе
            ρ=λ/(c·μ) к 1, тем больше нужно sim_hours для сходимости
            (дисперсия оценки резко растёт при ρ→1).
        warmup_hours: сколько начальных часов отбросить перед сбором
            статистики — убирает смещение от пустого старта очереди
            (при t=0 система гарантированно недогружена относительно
            стационарного режима).
        seed: сид генератора случайных чисел, для воспроизводимости.
        service_sampler: функция(rng) -> время обслуживания В ЧАСАХ.
            По умолчанию — экспонента со средним 1/mu_per_hour (стандартное
            допущение M/M/c). Можно передать иной семплер (см.
            empirical_service_sampler) для проверки чувствительности к
            форме распределения времени обслуживания.

    Returns:
        1D numpy-массив времён ожидания в МИНУТАХ, по одному элементу на
        каждого клиента, пришедшего после warmup_hours.
    """
    rng = np.random.default_rng(seed)
    if service_sampler is None:
        service_sampler = lambda rng: rng.exponential(1.0 / mu_per_hour)

    env = simpy.Environment()
    windows = simpy.Resource(env, capacity=c)
    wait_times = []

    def customer(env, arrival_time):
        """Один клиент: встаёт в очередь на окно, ждёт, обслуживается, уходит."""
        with windows.request() as req:
            t0 = env.now
            yield req  # блокируется здесь, пока не освободится окно
            wait = env.now - t0
            if arrival_time >= warmup_hours:
                wait_times.append(wait * 60)  # переводим в минуты для итогового результата
            yield env.timeout(service_sampler(rng))

    def arrival_process(env):
        """Генерирует пуассоновский поток клиентов через экспоненциальные интервалы между приходами."""
        while True:
            interarrival = rng.exponential(1.0 / lam_per_hour)
            yield env.timeout(interarrival)
            if env.now >= sim_hours:
                break
            env.process(customer(env, env.now))

    env.process(arrival_process(env))
    env.run(until=sim_hours)
    return np.array(wait_times)


def simulate_day_continuous(hour_profile: dict, mu_profile: dict, c_profile: dict,
                            sim_days: int = 40, warmup_days: int = 5,
                            seed: Optional[int] = None,
                            service_sampler: Optional[Callable] = None) -> pd.DataFrame:
    """
    Симулирует МНОГО повторов одного рабочего дня с меняющейся по часам
    нагрузкой λ(t)/μ(t)/c(t) — очередь НЕ сбрасывается на границе часа
    внутри дня (перенос переполнения из часа в час учитывается), но
    сбрасывается между "днями" (каждый день — новая simpy.Environment,
    т.к. отделение реально закрывается на ночь и открывается заново с
    пустой очередью).

    Это ключевое отличие от почасовой формулы Эрланга C, которая
    предполагает, что нагрузка каждого часа держится БЕСКОНЕЧНО (стационарный
    режим) — здесь же час длится ровно 60 минут, и то, успела ли очередь
    "разогнаться" или "рассосаться" за это время, зависит от того, что было
    в предыдущие часы того же дня. См. раздел 3.3 отчёта: почасовая формула
    может как завышать реальное ожидание (короткий всплеск ρ>=1 не успевает
    достичь теоретического стационара), так и занижать его (очередь,
    скопившаяся в пиковый час, перетекает в следующий, более спокойный).

    Техническая деталь реализации: SimPy 4 не позволяет менять
    Resource.capacity на лету (это свойство только для чтения), поэтому
    вместимость окон варьируется по часам через приём "блокировки лишних
    слотов" — Resource создаётся с capacity=max(c_profile), а разница
    (max_c - c_profile[h]) удерживается фиктивными "блокирующими" заявками
    в течение соответствующего часа (см. blocker_proc внутри).

    Args:
        hour_profile: dict {hour: lambda_per_hour} — интенсивность потока
            по часам (обычно из required_windows_table).
        mu_profile: dict {hour: mu_per_hour} — интенсивность обслуживания
            по часам (может отличаться по часам, если микс типов операций
            меняется в течение дня).
        c_profile: dict {hour: c} — фактическое число открытых окон по
            часам (например, достигнутое покрытие из ILP/жадного расписания).
            Все три словаря должны иметь одинаковый набор ключей (часов).
        sim_days: сколько "дней" симулировать — чем больше, тем точнее
            оценка среднего (особенно для часов с высокой загрузкой, где
            дисперсия ожидания велика).
        warmup_days: сколько первых дней отбросить (на случай, если внутри
            дня тоже есть небольшой разгон в первые "дни" сходимости —
            обычно не критично при дневном сбросе очереди, но оставлено для
            симметрии с warmup_hours в simulate_mmc).
        seed: сид генератора случайных чисел (базовый — для дня i
            используется seed*10_000 + i, чтобы дни были независимы, но
            воспроизводимы).
        service_sampler: функция(rng, mu) -> время обслуживания В ЧАСАХ.
            По умолчанию — экспонента со средним 1/mu.

    Returns:
        DataFrame со столбцами day, hour, wait_min — по одной строке на
        каждого обслуженного клиента (после warmup_days). Группировка
        df.groupby('hour')['wait_min'].mean() даёт среднее реальное
        ожидание по каждому часу дня.
    """
    hours_sorted = sorted(hour_profile.keys())
    n_hours = len(hours_sorted)
    if service_sampler is None:
        service_sampler = lambda rng, mu: rng.exponential(1.0 / mu)

    records = []
    eps = 1e-6  # малый технический сдвиг для избежания гонки событий на границах часа (см. blocker_proc)

    for day in range(sim_days):
        rng = np.random.default_rng(None if seed is None else seed * 10_000 + day)
        env = simpy.Environment()
        max_c = max(c_profile.values())
        windows = simpy.Resource(env, capacity=max_c)

        # Список ожидающих (ещё не получивших окно) — для учёта при закрытии
        unfinished = []

        def blocker_proc(env, start_t, end_t, n_block):
            """
            Держит n_block "фиктивных" слотов занятыми в течение
            [start_t, end_t], чтобы эффективная (доступная клиентам)
            вместимость окон в этот час была меньше max_c — стандартный
            приём для эмуляции time-varying capacity в SimPy (Resource.capacity
            в SimPy 4 доступен только на чтение, изменить его напрямую нельзя).
            Небольшие сдвиги на eps в начале/конце нужны, чтобы блокер
            текущего часа гарантированно занял слоты ПОСЛЕ того, как блокер
            предыдущего часа их освободил (иначе возможна гонка событий
            ровно в момент границы часа).
            """
            if n_block <= 0:
                return
            yield env.timeout(start_t + eps)
            reqs = [windows.request() for _ in range(n_block)]
            for r in reqs:
                yield r
            yield env.timeout((end_t - eps) - (start_t + eps))
            for r in reqs:
                windows.release(r)

        # заранее планируем блокирующие процессы на весь день — каждый "просыпается"
        # в свой час и держит нужное число слотов занятым до конца этого часа
        for i, h in enumerate(hours_sorted):
            n_block = max_c - c_profile[h]
            env.process(blocker_proc(env, i, i + 1, n_block))

        def customer(env, hour_idx, mu):
            """Один клиент: встаёт в очередь на окно, ждёт, обслуживается, уходит."""
            arrival = env.now
            info = {'arrival': arrival, 'hour_idx': hour_idx}
            unfinished.append(info)               # добавили в список ожидающих
            with windows.request() as req:
                t0 = env.now
                yield req
                # Окно получено — удаляемся из ожидающих
                if info in unfinished:
                    unfinished.remove(info)
                wait = env.now - t0
                if day >= warmup_days:
                    records.append((day, hours_sorted[hour_idx], wait * 60))
                yield env.timeout(service_sampler(rng, mu))

        def arrivals(env):
            """
            Генерирует приходы клиентов час за часом. Внутри часа число
            приходов ~ Poisson(λ), а их позиции равномерно разбросаны по
            часу — это стандартный и точный способ смоделировать процесс с
            кусочно-постоянной интенсивностью (эквивалентно пуассоновскому
            процессу, условленному на число точек в интервале), без
            смещения, которое возникло бы при наивном "дотягивании"
            экспоненциальных интервалов через границу часа.
            """
            for i, h in enumerate(hours_sorted):
                lam = hour_profile[h]
                mu = mu_profile[h]
                if lam > 0:
                    n = rng.poisson(lam)
                    offsets = np.sort(rng.uniform(2 * eps, 1 - 2 * eps, size=n))
                    prev = 0.0
                    for off in offsets:
                        yield env.timeout(off - prev)
                        prev = off
                        env.process(customer(env, i, mu))
                    yield env.timeout(1.0 - prev)
                else:
                    yield env.timeout(1.0)

        env.process(arrivals(env))
        env.run(until=n_hours)

        # Учитываем клиентов, которые не успели обслужиться до закрытия
        for info in unfinished:
            wait = n_hours - info['arrival']
            if day >= warmup_days:
                records.append((day, hours_sorted[info['hour_idx']], wait * 60))

    return pd.DataFrame(records, columns=['day', 'hour', 'wait_min'])


def empirical_service_sampler(operation_mix: list, rng_seed_offset: int = 0) -> Callable:
    """
    Строит семплер времени обслуживания на основе РЕАЛЬНЫХ эмпирических
    данных вместо экспоненциального распределения — для проверки, насколько
    чувствителен результат к допущению об экспоненциальности времени
    обслуживания (одно из двух ключевых допущений M/M/c наряду с
    пуассоновским входом).

    С вероятностью, пропорциональной весу типа операции, выбирается пул
    (basic/credit/mortgage), а затем время обслуживания берётся случайным
    бутстрэп-сэмплом из РЕАЛЬНО НАБЛЮДАВШИХСЯ значений этого типа (не из
    сглаженного распределения) — максимально честная проверка.

    Args:
        operation_mix: список пар (вес, массив_наблюдаемых_времён_в_минутах)
            — например [(0.69, basic_times), (0.26, credit_times),
            (0.05, mortgage_times)], где веса — доли каждого типа операции
            в общем потоке (нормализуются внутри функции, суммировать в 1
            не обязательно).
        rng_seed_offset: зарезервировано для будущего использования
            (независимая случайность семплера относительно потока приходов);
            сейчас сэмплер использует тот же rng, что передаётся при вызове.

    Returns:
        Функция sampler(rng, mu_unused=None) -> время обслуживания В ЧАСАХ,
        совместимая по сигнатуре с параметром service_sampler в
        simulate_mmc / simulate_day_continuous (аргумент mu_unused
        игнорируется — оставлен только для единообразия сигнатуры с
        экспоненциальным семплером по умолчанию, который использует mu).
    """
    weights = np.array([w for w, _ in operation_mix])
    weights = weights / weights.sum()
    pools = [arr for _, arr in operation_mix]

    def sampler(rng, mu_unused=None):
        idx = rng.choice(len(pools), p=weights)
        val_min = rng.choice(pools[idx])
        return val_min / 60.0

    return sampler


# ----------------------------------------------------------------------
# 2. Сводные графики (трёхпанельные)
# ----------------------------------------------------------------------
def _plot_daily_validation_summary(assignments: list, comparison: pd.DataFrame,
                                   title: str, save_path: str) -> None:
    """
    Три панели на одном рисунке:
    (1) аналитика (линии с маркерами): Целевой, Старое, Новое,
    (2) загрузка сотрудников,
    (3) симуляция: Старое vs Новое.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cap = 90.0

    # ---------- Панель 1: Аналитика (линии с маркерами) ----------
    ax = axes[0]
    hours = comparison['hour'].values
    target = np.minimum(comparison['erlang_required_wait_min'].values, cap)
    old_erl = np.minimum(comparison['old_erlang_wait_min'].values, cap)
    new_erl = np.minimum(comparison['erlang_wait_min'].values, cap)

    ax.plot(hours, target, '^-', color='green', markersize=8, label='Целевой (R_total)')
    ax.plot(hours, old_erl, 's-', color='gray', markersize=8, label='Старое расписание (9ч)')
    ax.plot(hours, new_erl, 'o-', color='steelblue', markersize=8, label='Новое расписание')
    ax.set_title('Аналитика (Эрланг C)')
    ax.set_xlabel('Час')
    ax.set_ylabel('Ожидание, мин')
    ax.set_xticks(hours)
    ax.set_yticks(np.arange(0, cap+10, 10))
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # ---------- Панель 2: Загрузка сотрудников ----------
    ax = axes[1]
    if assignments:
        df = pd.DataFrame(assignments)
        emp_hours = df.groupby('employee_id')['paid_hours'].sum().sort_values(ascending=False)
        default_hours = pd.Series(9.0, index=emp_hours.index)
        x = np.arange(len(emp_hours))
        width = 0.35
        ax.bar(x - width/2, default_hours, width, color='lightgray', label='По умолчанию (9 ч)')
        ax.bar(x + width/2, emp_hours, width, color='skyblue', label='По расписанию')
        ax.set_xticks(x)
        ax.set_xticklabels(emp_hours.index, rotation=45, ha='right')
        ax.set_title('Отработано часов')
        ax.set_ylabel('Часы')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'Нет данных', transform=ax.transAxes, ha='center')

    # ---------- Панель 3: Симуляция (столбцы) ----------
    ax = axes[2]
    hours = comparison['hour'].values
    x = np.arange(len(hours))
    width = 0.35
    old_sim = np.minimum(comparison['old_sim_wait_min'].values, cap)
    new_sim = np.minimum(comparison['sim_wait_min'].values, cap)

    ax.bar(x - width/2, old_sim, width, color='gray', label='Старое расписание (симуляция)')
    ax.bar(x + width/2, new_sim, width, color='salmon', label='Новое расписание (симуляция)')
    ax.set_title('Симуляция очереди')
    ax.set_xlabel('Час')
    ax.set_ylabel('Ожидание, мин')
    ax.set_xticks(x)
    ax.set_xticklabels(hours)
    ax.set_yticks(np.arange(0, cap+10, 10))
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сводный дневной отчёт сохранён: {save_path}")




def _plot_weekly_validation_summary(comparison_week: dict, results_week: dict,
                                    summary_week: pd.DataFrame,
                                    title: str, save_path: str) -> None:
    """
    Три панели:
    (1) ожидание по дням (аналитика vs симуляция),
    (2) покрытие первого дня,
    (3) загрузка сотрудников за неделю.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (1) Сравнение ожидания по дням
    ax = axes[0]
    weekdays = comparison_week['weekdays']
    x = np.arange(len(weekdays))
    width = 0.35
    ax.bar(x - width/2, comparison_week['daily_analytic'], width, label='Аналитика')
    ax.bar(x + width/2, comparison_week['daily_sim'], width, label='Симуляция')
    ax.set_xticks(x)
    ax.set_xticklabels(weekdays)
    ax.set_ylabel('Среднее ожидание, мин')
    ax.set_title('Ожидание по дням')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # (2) Покрытие первого дня
    ax = axes[1]
    first_day = list(results_week.keys())[0]
    cov_df = results_week[first_day]['coverage']
    hours = cov_df['hour']
    ax.plot(hours, cov_df['R_total'], 'o-', markersize=8, label='Требуется')
    ax.plot(hours, cov_df['cov_total'], 's-', markersize=8, label='Открыто')
    ax.set_title(f'Покрытие ({first_day})')
    ax.set_xlabel('Час')
    ax.set_ylabel('Окна')
    ax.legend()
    ax.grid(True)

    # (3) Загрузка сотрудников
    ax = axes[2]
    employees = summary_week['employee_id'].tolist()
    utilization = (summary_week['utilization'] * 100).tolist()
    ax.barh(employees, utilization, color='salmon')
    ax.set_title('Загрузка сотрудников (%)')
    ax.set_xlabel('% от 40ч')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Сводный недельный отчёт сохранён: {save_path}")


# ----------------------------------------------------------------------
# 3. Валидация одного дня с сохранением артефактов
# ----------------------------------------------------------------------
def validate_schedule_with_simulation(
    client_arrivals: pd.DataFrame,
    operations: pd.DataFrame,
    branches: pd.DataFrame,
    employees: pd.DataFrame,
    branch_id: str,
    weekday_en: str,
    n_windows_max: int = None,
    sim_days: int = 100,
    warmup_days: int = 10,
    seed: int = 42,
    service_sampler: Optional[Callable] = None,
    algorithm: str = 'ilp'   # 'ilp' или 'greedy'
) -> dict:
    """
    Строит расписание (ILP или Greedy), сохраняет профили, метрики и
    графики в папку SIM_REPORT_SAVE_DIRECTORY. Возвращает словарь с результатами.

    Args:
        ...
        algorithm: 'ilp' или 'greedy'

    Returns:
        Словарь с ключами:
        - 'comparison': DataFrame сравнения аналитики и симуляции по часам.
        - 'assignments': список назначений.
        - 'report_dir': путь к папке с сохранёнными файлами.
        - 'files': словарь с путями к отдельным файлам.
    """
    if n_windows_max is None:
        n_windows_max = int(branches.loc[branch_id, 'n_windows'])

    req = required_windows_table(client_arrivals, operations, branch_id, weekday_en, n_windows_max)
    branch_emp = employees[employees['branch_id'] == branch_id]
    open_h, close_h = WEEKDAY_HOURS[weekday_en]

    if algorithm == 'ilp':
        assignments, coverage_df, _, _, shortfall = schedule_day_ilp_soft(branch_emp, req, open_h, close_h, n_windows_max)
    else:
        assignments, coverage_df, _, _ = schedule_day_greedy(branch_emp, req, open_h, close_h, n_windows_max=n_windows_max)
        shortfall = None

    hours = list(req['hour'])
    lam_profile = dict(zip(req['hour'], req['lambda_total']))
    mu_profile = {h: 60.0 / req.set_index('hour').loc[h, 'avg_service_min'] for h in hours}
    c_profile = dict(zip(coverage_df['hour'], coverage_df['cov_total']))

    sim_df = simulate_day_continuous(lam_profile, mu_profile, c_profile,
                                    sim_days=sim_days, warmup_days=warmup_days, seed=seed,
                                    service_sampler=service_sampler)
    sim_by_hour = sim_df.groupby('hour')['wait_min'].mean()

    analytic_waits = {}
    for h in hours:
        wq = erlang_c_wait_minutes(lam_profile[h], mu_profile[h], c_profile[h])
        analytic_waits[h] = wq if np.isfinite(wq) else np.inf

    comparison = pd.DataFrame({
        'hour': hours,
        'lambda': [lam_profile[h] for h in hours],
        'mu': [mu_profile[h] for h in hours],
        'c': [c_profile[h] for h in hours],
        'erlang_wait_min': [analytic_waits[h] for h in hours],
        'sim_wait_min': [sim_by_hour.get(h, 0) for h in hours]
    })
    req_r_total = dict(zip(req['hour'], req['R_total']))
    erlang_required = {}
    for h in hours:
        wq = erlang_c_wait_minutes(lam_profile[h], mu_profile[h], req_r_total[h])
        erlang_required[h] = wq if np.isfinite(wq) else np.inf
    comparison['erlang_required_wait_min'] = [erlang_required[h] for h in hours]
    # ---------- Старое (9‑часовое) расписание ----------
    old_assignments = naive_schedule(branch_emp, n_windows_max, open_h)
    # покрытие старого расписания
    old_c_profile = coverage_from_assignments(old_assignments, hours)
    old_c_profile = {h: old_c_profile[h] for h in hours}

    # Аналитика по старому расписанию
    old_analytic = {}
    for h in hours:
        wq = erlang_c_wait_minutes(lam_profile[h], mu_profile[h], old_c_profile[h])
        old_analytic[h] = wq if np.isfinite(wq) else np.inf

    # Симуляция для старого расписания
    old_sim_df = simulate_day_continuous(lam_profile, mu_profile, old_c_profile,
                                        sim_days=sim_days, warmup_days=warmup_days, seed=seed,
                                        service_sampler=service_sampler)
    old_sim_by_hour = old_sim_df.groupby('hour')['wait_min'].mean()

    # Добавляем в comparison
    comparison['old_erlang_wait_min'] = [old_analytic[h] for h in hours]
    comparison['old_sim_wait_min'] = [old_sim_by_hour.get(h, 0) for h in hours]


    report_dir = SIM_REPORT_SAVE_DIRECTORY / f"{branch_id}_{weekday_en}_{algorithm}"
    report_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    # Профили
    for name, data in [('lambda', lam_profile), ('mu', mu_profile), ('c', c_profile)]:
        fpath = report_dir / f"{name}_profile.json"
        with open(fpath, 'w') as f:
            json.dump(data, f, indent=2)
        files[name] = str(fpath)

    # Сводный трёхпанельный отчёт (аналитика/симуляция + загрузка + ожидание)
    summary_path = report_dir / 'daily_summary.png'
    _plot_daily_validation_summary(assignments, comparison,
                                title=f"{algorithm.upper()} {branch_id} {weekday_en}",
                                save_path=str(summary_path))
    files['daily_summary'] = str(summary_path)

    # Файл расписания (ранее gantt.png)
    schedule_path = report_dir / 'schedule.png'
    plot_daily_schedule(assignments, title=f"{algorithm.upper()} {branch_id} {weekday_en}",
                        save_path=str(schedule_path))
    files['schedule'] = str(schedule_path)

    # JSON-отчёт
    report_json = {
        'branch': branch_id,
        'weekday': weekday_en,
        'algorithm': algorithm,
        'sim_days': sim_days,
        'warmup_days': warmup_days,
        'shortfall': shortfall,
        'mean_erlang_wait': comparison['erlang_wait_min'].replace(np.inf, np.nan).mean(),
        'mean_sim_wait': comparison['sim_wait_min'].mean(),
        'hourly_data': comparison.to_dict(orient='records')
    }
    json_path = report_dir / 'report.json'
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2, default=str)
    files['report_json'] = str(json_path)

    return {
        'comparison': comparison,
        'assignments': assignments,
        'report_dir': str(report_dir),
        'files': files
    }


# ----------------------------------------------------------------------
# 4. Недельная валидация
# ----------------------------------------------------------------------
def validate_week_schedule(
    client_arrivals: pd.DataFrame,
    operations: pd.DataFrame,
    branches: pd.DataFrame,
    employees: pd.DataFrame,
    branch_id: str,
    algorithm: str = 'ilp',
    sim_days: int = 100,
    warmup_days: int = 10,
    seed: int = 42,
    service_sampler: Optional[Callable] = None
) -> dict:
    """
    Для каждого дня недели симулирует работу, собирает взвешенное ожидание,
    сравнивает с аналитикой и строит графики + загрузку сотрудников.
    Сохраняет детальный и сводный отчёты.
    """
    n_win = int(branches.loc[branch_id, 'n_windows'])
    if algorithm == 'ilp':
        results_week, summary_week = schedule_week_ilp(client_arrivals, operations, employees, branches, branch_id)
    else:
        results_week, summary_week = schedule_week_greedy(client_arrivals, operations, employees, branches, branch_id)

    metrics_analytic = get_weekly_metrics(results_week, summary_week, client_arrivals, operations, branch_id, branches)

    total_weight = 0.0
    weighted_sim_wait = 0.0
    daily_sim_waits = {}
    for weekday, data in results_week.items():
        assignments = data['assignments']
        coverage_df = data['coverage']
        req = required_windows_table(client_arrivals, operations, branch_id, weekday, n_win)
        hours = list(req['hour'])
        lam_profile = dict(zip(req['hour'], req['lambda_total']))
        mu_profile = {h: 60.0 / req.set_index('hour').loc[h, 'avg_service_min'] for h in hours}
        c_profile = dict(zip(coverage_df['hour'], coverage_df['cov_total']))

        sim_df = simulate_day_continuous(lam_profile, mu_profile, c_profile,
                                        sim_days=sim_days, warmup_days=warmup_days, seed=seed,
                                        service_sampler=service_sampler)
        sim_by_hour = sim_df.groupby('hour')['wait_min'].mean()
        day_weight = req['lambda_total'].sum()
        day_avg_wait = np.average([sim_by_hour.get(h, 0) for h in hours],
                                weights=req['lambda_total'].values) if day_weight > 0 else 0.0
        daily_sim_waits[weekday] = day_avg_wait
        total_weight += day_weight
        weighted_sim_wait += day_avg_wait * day_weight

    week_sim_wait = weighted_sim_wait / total_weight if total_weight > 0 else 0.0

    comparison_week = {
        'branch': branch_id,
        'algorithm': algorithm,
        'analytic_weekly_wait': metrics_analytic['avg_wait_weighted'],
        'sim_weekly_wait': round(week_sim_wait, 2),
        'daily_analytic': [m['avg_wait'] for m in metrics_analytic['daily_metrics']],
        'daily_sim': [daily_sim_waits[m['weekday']] for m in metrics_analytic['daily_metrics']],
        'weekdays': [m['weekday'] for m in metrics_analytic['daily_metrics']]
    }

    report_dir = SIM_REPORT_SAVE_DIRECTORY / f"{branch_id}_week_{algorithm}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Детальный недельный отчёт (plot_weekly_schedule из compare_policies)
    plot_weekly_schedule(results_week, summary_week, client_arrivals, operations, branch_id, branches,
                        save_path=str(report_dir / 'weekly_summary.png'))

    # Сводный трёхпанельный отчёт по неделе
    _plot_weekly_validation_summary(comparison_week, results_week, summary_week,
                                    title=f"Week {branch_id} {algorithm.upper()}",
                                    save_path=str(report_dir / 'weekly_validation_summary.png'))

    # JSON-отчёт
    json_path = report_dir / 'week_validation.json'
    with open(json_path, 'w') as f:
        json.dump(comparison_week, f, indent=2, default=str)

    print(f"Недельная валидация сохранена в {report_dir}")
    return {
        'comparison': comparison_week,
        'report_dir': str(report_dir)
    }


# ----------------------------------------------------------------------
# 5. Главный запуск – полная валидация всех отделений, дней, алгоритмов
# ----------------------------------------------------------------------
if __name__ == '__main__':
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
    emp = pd.read_csv('dataset/employees.csv')
    emp['skills'] = emp['skills'].str.split(',')

    # Для каждого отделения и каждого алгоритма
    for branch in ['BR01', 'BR02', 'BR03']:
        for alg in ['ilp', 'greedy']:
            # Симуляция всех рабочих дней
            for day in WEEKDAY_HOURS:
                print(f"=== {branch} {day} {alg} ===")
                validate_schedule_with_simulation(ca, ops, br, emp, branch, day,
                                                algorithm=alg, sim_days=80, warmup_days=10)
            # Симуляция недели
            print(f"=== Неделя {branch} {alg} ===")
            validate_week_schedule(ca, ops, br, emp, branch, algorithm=alg, sim_days=80, warmup_days=10)

    # Базовая валидация формулы Эрланга
    print("\n=== Базовая валидация Эрланга C ===")
    c, mu = 4, 6.0
    for rho in [0.3, 0.5, 0.7, 0.85, 0.95]:
        lam = rho * c * mu
        analytic = erlang_c_wait_minutes(lam, mu, c)
        sim = simulate_mmc(lam, mu, c, sim_hours=1500, warmup_hours=100, seed=42).mean()
        print(f"  rho={rho:.2f}  Эрланг C={analytic:7.2f} мин  Симуляция={sim:7.2f} мин")