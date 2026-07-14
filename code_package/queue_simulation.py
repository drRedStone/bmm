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
            with windows.request() as req:
                t0 = env.now
                yield req
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


if __name__ == '__main__':
    import pandas as pd
    from required_windows import erlang_c_wait_minutes, required_windows_table
    from schedule_ilp import schedule_day_ilp_soft

    print("=== 1. Базовая валидация: симуляция vs формула Эрланга C ===")
    c, mu = 4, 6.0
    for rho in [0.3, 0.5, 0.7, 0.85, 0.95]:
        lam = rho * c * mu
        analytic = erlang_c_wait_minutes(lam, mu, c)
        sim = simulate_mmc(lam, mu, c, sim_hours=1500, warmup_hours=100, seed=42).mean()
        print(f"  rho={rho:.2f}  Эрланг C={analytic:7.2f} мин  Симуляция={sim:7.2f} мин")

    print("\n=== 2. BR01/понедельник: изолированный час vs непрерывный день ===")
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
    emp = pd.read_csv('dataset/employees.csv')
    emp['skills'] = emp['skills'].str.split(',')

    n_win = int(br.loc['BR01', 'n_windows'])
    req = required_windows_table(ca, ops, 'BR01', 'Monday', n_win)
    branch_emp = emp[emp['branch_id'] == 'BR01']
    _, coverage, _, _, _ = schedule_day_ilp_soft(branch_emp, req, 9, 19, n_win)

    hours = list(req['hour'])
    lam_profile = dict(zip(req['hour'], req['lambda_total']))
    mu_profile = {h: 60.0 / req.set_index('hour').loc[h, 'avg_service_min'] for h in hours}
    c_profile = dict(zip(coverage['hour'], coverage['cov_total']))

    df = simulate_day_continuous(lam_profile, mu_profile, c_profile, sim_days=150, warmup_days=10, seed=7)
    sim_by_hour = df.groupby('hour')['wait_min'].mean()

    for h in hours:
        a = erlang_c_wait_minutes(lam_profile[h], mu_profile[h], c_profile[h])
        a_str = f"{a:6.1f}" if a != float('inf') else "  inf"
        print(f"  {h:>2}:00  Эрланг C(изолированно)={a_str} мин   Симуляция(непрерывно)={sim_by_hour.loc[h]:6.1f} мин")
