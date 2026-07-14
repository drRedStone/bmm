"""
required_windows.py
====================
Оценка требуемого числа открытых окон обслуживания по часам.

Пайплайн модуля:
    1. Оценить интенсивность потока λ(час, день недели) по историческим данным
       (MLE-оценка параметра пуассоновского процесса — среднее число клиентов
       в данном часовом слоте по наблюдаемым неделям).
    2. Через формулу Эрланга C (аналитическая модель очереди M/M/c) найти
       минимальное число окон c, при котором среднее время ожидания Wq не
       превышает целевой порог (THRESHOLD_MIN).
    3. Учесть иерархию навыков сотрудников (senior ⊃ credit ⊃ basic из
       employees.csv): помимо общего требования R_total считаются под-
       требования R_credit и R_mortgage — сколько из открытых окон обязаны
       обслуживать сотрудники соответствующей квалификации.
    4. Ограничить всё физическим числом окон отделения (n_windows_max из
       branches.csv) и явно пометить часы, где порог недостижим даже на
       максимуме окон (threshold_reachable=False) — это уже не проблема
       расписания, а инфраструктурное ограничение отделения.

Результат (required_windows_table) — вход для планировщиков расписания:
    schedule_greedy.py (жадный алгоритм) и schedule_ilp.py (точный ILP).

Независимая проверка модели очереди — дискретно-событийная симуляция в
queue_simulation.py (сравнение с erlang_c_wait_minutes ниже).
"""
import pandas as pd
import numpy as np
import math

THRESHOLD_MIN = 10          # целевой порог среднего времени ожидания, минут
SERVICE_MIN = {              # средняя длительность обслуживания по уровню навыка, минут
    'basic': 8.98,            # эмпирическое среднее по client_arrivals.csv (операции OP01-OP03)
    'credit': 30.55,          # операции OP04-OP05 (кредит/рефинансирование)
    'mortgage': 50.0,         # операция OP06 (ипотека, только senior)
}


def erlang_c_wait_minutes(lam_per_hour: float, mu_per_hour: float, c: int) -> float:
    """
    Среднее время ожидания в очереди Wq для системы массового обслуживания
    M/M/c (формула Эрланга C), в минутах.

    Модель: c идентичных обслуживающих окон, пуассоновский вход с
    интенсивностью lam_per_hour, экспоненциальное время обслуживания со
    средним 1/mu_per_hour. Формула предполагает СТАЦИОНАРНЫЙ режим — то
    есть параметры считаются неизменными сколь угодно долго. Для часового
    слота с быстро меняющейся нагрузкой это даёт консервативную (worst-case)
    оценку — см. queue_simulation.py для проверки через реалистичную
    непрерывную симуляцию дня, где нагрузка меняется по часам.

    Args:
        lam_per_hour: интенсивность входящего потока, клиентов/час.
        mu_per_hour: интенсивность обслуживания одним окном, клиентов/час
            (= 60 / среднее_время_обслуживания_в_минутах).
        c: число открытых обслуживающих окон.

    Returns:
        Wq в минутах. math.inf, если система нестабильна (предложенная
        нагрузка a = lam/mu >= c, т.е. окна физически не справляются с
        потоком даже в теории).
    """
    if c <= 0:
        return math.inf
    a = lam_per_hour / mu_per_hour  # предлагаемая нагрузка в Эрлангах (offered load)
    if c <= a:
        return math.inf  # ρ = a/c >= 1 -> очередь растёт неограниченно
    # Классическая формула Эрланга C: сначала считаем P0 (вероятность пустой
    # системы), затем P_wait (вероятность, что клиент встанет в очередь, а не
    # будет обслужен немедленно), затем Wq через среднее число ожидающих.
    sum_terms = sum(a ** k / math.factorial(k) for k in range(c))
    last_term = (a ** c / math.factorial(c)) * (c / (c - a))
    p0 = 1.0 / (sum_terms + last_term)
    p_wait = last_term * p0  # вероятность ожидания (собственно "формула Эрланга C")
    wq_hours = p_wait / (c * mu_per_hour - lam_per_hour)
    return wq_hours * 60


def min_windows_erlang(lam_per_hour: float, avg_service_min: float,
                        threshold_min: float = THRESHOLD_MIN, max_c: int = 20) -> int:
    """
    Подбирает минимальное число окон c, при котором Wq (формула Эрланга C)
    не превышает threshold_min. Простой линейный перебор от минимально
    стабильного c вверх — пространство поиска маленькое (обычно c < 10),
    так что перебор быстрее и надёжнее любой аналитической инверсии формулы.

    Args:
        lam_per_hour: интенсивность входящего потока, клиентов/час.
        avg_service_min: среднее время обслуживания, минут.
        threshold_min: целевой порог среднего ожидания, минут.
        max_c: верхний предел перебора (защита от бесконечного цикла и
            от нереалистично больших ответов, если threshold_min слишком мал).

    Returns:
        Минимальное c, удовлетворяющее порогу. Если лямбда=0 — возвращает 0
        (окна не нужны). Если порог недостижим даже при max_c окнах —
        возвращает max_c (вызывающий код сам решает, что с этим делать;
        required_windows_table дополнительно помечает такие часы флагом
        threshold_reachable=False после сравнения с физическим потолком окон).
    """
    if lam_per_hour <= 0:
        return 0
    mu_per_hour = 60.0 / avg_service_min
    a = lam_per_hour / mu_per_hour
    c = max(1, math.ceil(a))  # стартуем с минимально стабильного c (иначе Wq=inf)
    while c <= max_c:
        wq = erlang_c_wait_minutes(lam_per_hour, mu_per_hour, c)
        if wq <= threshold_min:
            return c
        c += 1
    return max_c  # порог физически недостижим даже при максимуме окон


def min_skill_servers(lam_skill_per_hour: float, avg_service_min: float,
                       safety_buffer: int = 1, safety_trashold: float = 0.3) -> int:
    """
    Оценивает минимальное число серверов конкретной квалификации (credit
    или mortgage), необходимое для под-потока клиентов этого типа.

    Полноценная формула Эрланга C здесь плохо применима: лямбда под-потоков
    часто маленькая (< 2 клиента/час), и Erlang C при таких значениях даёт
    очень шумные, скачкообразные оценки. Вместо этого используем более
    грубое, но устойчивое правило: минимум серверов, чтобы загрузка была
    < 1 (иначе очередь по этому под-потоку растёт неограниченно), плюс
    запас в safety_buffer окно(а) на пиковую изменчивость, если нагрузка (safety_trashold)
    заметная (a > 0.3 Эрланга) — иначе буфер не добавляем, чтобы не
    завышать требование на почти пустых часах.

    Args:
        lam_skill_per_hour: интенсивность под-потока (клиентов данного
            типа операции), клиентов/час.
        avg_service_min: среднее время обслуживания ЭТОГО типа операции,
            минут (не общее среднее по отделению).
        safety_buffer: сколько дополнительных окон закладывать сверх
            минимально стабильного количества, если нагрузка заметная.

    Returns:
        Минимальное число окон нужной квалификации. 0, если под-потока нет.
    """
    if lam_skill_per_hour <= 0:
        return 0
    mu = 60.0 / avg_service_min
    a = lam_skill_per_hour / mu
    return math.ceil(a) + (safety_buffer if a > safety_trashold else 0)


def required_windows_table(client_arrivals: pd.DataFrame, operations: pd.DataFrame,
                            branch_id: str, weekday_en: str, n_windows_max: int) -> pd.DataFrame:
    """
    Строит по часам таблицу требуемого числа открытых окон для одного
    отделения и одного дня недели — основной вход для планировщиков
    расписания (schedule_greedy.schedule_day, schedule_ilp.schedule_day_ilp*).

    Для каждого часа считает:
      - lambda_total / lambda_credit / lambda_mortgage — интенсивность
        общего потока и под-потоков, требующих credit+ и mortgage-навык
        соответственно (credit+ включает mortgage — вложенность навыков
        senior ⊃ credit ⊃ basic из employees.csv).
      - R_total / R_credit / R_mortgage — минимальное число окон каждой
        категории, с гарантией вложенности R_mortgage <= R_credit <= R_total
        и жёстким потолком R_total <= n_windows_max (физическое число окон
        отделения из branches.csv).
      - Wq_at_cap_min / threshold_reachable — реальное ожидание и флаг
        достижимости порога THRESHOLD_MIN, если открыть ВСЕ физически
        доступные окна. False означает инфраструктурное ограничение
        (не хватает окон), которое никаким расписанием не решить.

    Args:
        client_arrivals: датафрейм client_arrivals.csv (или эквивалент) со
            столбцами arrival_datetime, branch_id, operation_id,
            service_time_min.
        operations: датафрейм operations.csv со столбцами operation_id,
            required_skill.
        branch_id: идентификатор отделения (например, 'BR01').
        weekday_en: день недели на английском в формате pandas.day_name()
            (например, 'Monday').
        n_windows_max: физическое число окон отделения (branches.csv,
            столбец n_windows).

    Returns:
        DataFrame, отсортированный по часу, со столбцами: hour, lambda_total,
        lambda_credit, lambda_mortgage, avg_service_min, R_total, R_credit,
        R_mortgage, Wq_at_cap_min, threshold_reachable.
    """
    ops = operations.set_index('operation_id')['required_skill']
    ca = client_arrivals.copy()
    ca['arrival_datetime'] = pd.to_datetime(ca['arrival_datetime'])
    ca['weekday_en'] = ca['arrival_datetime'].dt.day_name()
    ca['hour'] = ca['arrival_datetime'].dt.hour
    ca['skill'] = ca['operation_id'].map(ops)

    sub = ca[(ca['branch_id'] == branch_id) & (ca['weekday_en'] == weekday_en)]
    n_weeks = sub['arrival_datetime'].dt.date.nunique()  # число наблюдаемых недель для усреднения (MLE лямбды)

    rows = []
    for hour, g in sub.groupby('hour'):
        lam_total = len(g) / n_weeks
        # credit+ включает mortgage: senior может делать всё, что и middle (вложенность навыков)
        lam_credit = len(g[g['skill'].isin(['credit', 'mortgage'])]) / n_weeks
        lam_mortgage = len(g[g['skill'] == 'mortgage']) / n_weeks
        # эффективное среднее время обслуживания В ЭТОТ КОНКРЕТНЫЙ час (наблюдаемое
        # по факту, а не общее среднее по отделению) — учитывает, что микс типов
        # операций может отличаться по часам (например, больше кредитных консультаций
        # в обед, когда у людей есть время)
        avg_service = g['service_time_min'].mean()

        r_total = min_windows_erlang(lam_total, avg_service)
        r_credit = min_skill_servers(lam_credit, SERVICE_MIN['credit'])
        r_mortgage = min_skill_servers(lam_mortgage, SERVICE_MIN['mortgage'])

        # физический потолок окон в отделении — жёсткое ограничение сверху
        r_total_capped = min(r_total, n_windows_max)
        # гарантируем вложенность: под-требования не могут превышать общее
        r_credit = min(r_credit, r_total_capped)
        r_mortgage = min(r_mortgage, r_credit)

        # если даже на физическом максимуме окон порог недостижим — считаем
        # реальный Wq на этом максимуме, чтобы явно показать масштаб проблемы
        # (а не просто "порог не достигнут", а "насколько именно не достигнут")
        mu_total = 60.0 / avg_service if avg_service > 0 else np.nan
        wq_at_cap = (erlang_c_wait_minutes(lam_total, mu_total, r_total_capped)
                     if lam_total > 0 else 0.0)
        reachable = wq_at_cap <= THRESHOLD_MIN

        rows.append(dict(hour=hour, lambda_total=round(lam_total, 2), lambda_credit=round(lam_credit, 2),
                          lambda_mortgage=round(lam_mortgage, 2), avg_service_min=round(avg_service, 1),
                          R_total=r_total_capped, R_credit=r_credit, R_mortgage=r_mortgage,
                          Wq_at_cap_min=round(wq_at_cap, 1), threshold_reachable=reachable))
    return pd.DataFrame(rows).sort_values('hour').reset_index(drop=True)


if __name__ == '__main__':
    # Демонстрация: требуемые окна по часам для понедельника, все 3 отделения.
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')

    for bid in ['BR01', 'BR02', 'BR03']:
        n_win = int(br.loc[bid, 'n_windows'])
        t = required_windows_table(ca, ops, bid, 'Monday', n_win)
        print(f"\n=== {bid} ({br.loc[bid,'name']}), окон физически: {n_win}, понедельник ===")
        print(t.to_string(index=False))
