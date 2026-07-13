import pandas as pd
import numpy as np
from prop import *
import heapq
from collections import deque


q_weights = {
    "OP01":  5,
    "OP02": 10,
    "OP03": 15,
    "OP04": 25,
    "OP05": 40,
    "OP06": 50
}

class Client:
    def __init__(self, arrival_datetime, weekday, operation_id, service_time_min, branch_id):
        self.arrival_m = arrival_datetime.minute + arrival_datetime.hour*60
        self.day = weekday
        self.op_id = operation_id
        self.serv_time = service_time_min
        self.branch_id = branch_id

        self.wait_time = 0
    
    def free_time(self): return self.arrival_m+self.wait_time+self.serv_time

class Window:
    def __init__(self, branch_id):
        self.branch_id = branch_id
        self.free_time = 540
        self.employee = None
        
        self.is_open = False
        self.open_time = 0

class Employee:
    def __init__(self, emp_id, branch_id, name, grade):
        self.emp_id = emp_id
        self.branch_id = branch_id
        self.name = name
        self.grade = grade

def get_media_serv_time(path, branch_id):
    """не хочу сечас тратить на это время"""
    pass

def get_data_from_client_arrivals(path, branch_id):
    """получаем данные, сгруппированные по отделениям. получаем только по одному отделению"""
    df = pd.read_csv(path, parse_dates=['arrival_datetime'])

    grouped = df.groupby(["branch_id"])                                         
    return grouped.get_group((branch_id,))
                                                                                
def client_list(data):
    """
    data - данные; обЪект, возвращаемый get_data_from_arrivals() или срез такого объекта по дате
    """
    data = data.sort_values('arrival_datetime')
    return [Client(**row) for row in data.to_dict("records")]

def stat_simulate(clients, n_wins, branch_id = None):
    """
    clients - список клиентов, возвращаемый client_list()
    n_wins - число окон
    branch_id - номер отделения
    """
    if not clients: return 0, 0, 0
    # создаем окна
    wins = [Window(branch_id) for _ in range(n_wins)]
    # перебираем клиентов
    for client in clients:
        # получаем индекс окна, которое скорее освободится
        idx_near = wins.index(min(wins, key=lambda w: w.free_time))
        # счтитаем время ожидания
        client.wait_time = wins[idx_near].free_time - client.arrival_m
        if client.wait_time <0: client.wait_time = 0
        # меняем время освобождения окна
        wins[idx_near].free_time = client.free_time()
    
    # считаем метрики
    waits = [c.wait_time for c in clients]
    # среднее
    av_wait = sum(waits)/len(waits)
    # максимальное
    max_wait = max(waits)
    # процент ждавших более 15мин
    over15 = sum(1 for w in waits if w>15)/len(waits)*100

    return av_wait, max_wait, over15

def stat_case():
    # целевая дата
    target_date = pd.to_datetime('2025-06-02').date()
    # данные, взятые в целевой день
    one_d = get_data_from_client_arrivals(DATA+"client_arrivals.csv", "BR01")

    clients = client_list(
        one_d[one_d['arrival_datetime'].dt.date == target_date]
    )

    print(f"суммарное время обслуживания: {sum([c.serv_time for c in clients])}")

    print("кол-во окон|среднее ждание|максимальное ждание|процйент>15мин")
    for i in range(1, 6):
        l = stat_simulate(clients, i)
        print(f"{i} | {l[0]} | {l[1]} | {l[2]}")


def din_simulate(clients, branch_id, max_wins=6, 
                 open_threshold=30,
                 close_threshold=20,  # увеличено с 5 до 20
                 idle_threshold=10,
                 day_start=540,
                 day_end=1080):
    
    if not clients:
        return 0, 0, 0, 0
    
    # Создаем окна (все закрыты, кроме первого)
    wins = [Window(branch_id) for _ in range(max_wins)]
    wins[0].is_open = True
    wins[0].open_time = day_start
    wins[0].free_time = day_start
    
    # Очередь клиентов (FIFO)
    waiting_queue = deque()
    current_queue_weight = 0
    
    # Куча событий
    event_counter = 0
    event_queue = []
    
    # Добавляем все приходы клиентов
    for client in clients:
        heapq.heappush(event_queue, (client.arrival_m, 1, event_counter, "client_arrival", client))
        event_counter += 1
    
    # Метрики
    total_window_work_time = 0
    current_time = day_start
    
    # Обработка событий
    while event_queue:
        time, priority, counter, event_type, data = heapq.heappop(event_queue)
        current_time = time
        
        if event_type == "client_arrival":
            client = data
            weight = q_weights.get(client.op_id, 10)
            
            # Клиент всегда встает в конец очереди
            waiting_queue.append(client)
            current_queue_weight += weight
            
            # Пытаемся обслужить первого из очереди, если есть свободное окно
            if waiting_queue and any(w.is_open and w.free_time <= current_time for w in wins):
                next_client = waiting_queue.popleft()
                next_weight = q_weights.get(next_client.op_id, 10)
                
                # Находим первое свободное открытое окно
                for i, win in enumerate(wins):
                    if win.is_open and win.free_time <= current_time:
                        next_client.wait_time = current_time - next_client.arrival_m
                        win.free_time = current_time + next_client.serv_time
                        current_queue_weight -= next_weight
                        
                        heapq.heappush(event_queue, (win.free_time, 0, event_counter, "window_free", 
                                                     {"window_idx": i, "client": next_client, "weight": next_weight}))
                        event_counter += 1
                        break
            
            # Если не назначили — проверяем, нужно ли открыть новое окно
            if not any(w.is_open and w.free_time <= current_time for w in wins):
                if current_queue_weight >= open_threshold:
                    # Ищем первое закрытое окно
                    for i, win in enumerate(wins):
                        if not win.is_open:
                            win.is_open = True
                            win.open_time = current_time
                            win.free_time = current_time
                            
                            heapq.heappush(event_queue, (current_time, 2, event_counter, "window_open", {"window_idx": i}))
                            event_counter += 1
                            break
        
        elif event_type == "window_free":
            window_idx = data["window_idx"]
            client = data["client"]
            weight = data["weight"]
            win = wins[window_idx]
            
            # Проверяем: есть ли клиенты в очереди?
            if waiting_queue:
                next_client = waiting_queue.popleft()
                next_weight = q_weights.get(next_client.op_id, 10)
                
                next_client.wait_time = current_time - next_client.arrival_m
                win.free_time = current_time + next_client.serv_time
                current_queue_weight -= next_weight
                
                heapq.heappush(event_queue, (win.free_time, 0, event_counter, "window_free", 
                                             {"window_idx": window_idx, "client": next_client, "weight": next_weight}))
                event_counter += 1
            else:
                # Окно простаивает — добавляем событие проверки
                check_time = current_time + idle_threshold
                heapq.heappush(event_queue, (check_time, 2, event_counter, "window_check", {"window_idx": window_idx}))
                event_counter += 1
        
        elif event_type == "window_open":
            window_idx = data["window_idx"]
            win = wins[window_idx]
            
            # Проверяем: есть ли клиенты в очереди?
            if waiting_queue:
                next_client = waiting_queue.popleft()
                next_weight = q_weights.get(next_client.op_id, 10)
                
                next_client.wait_time = current_time - next_client.arrival_m
                win.free_time = current_time + next_client.serv_time
                current_queue_weight -= next_weight
                
                heapq.heappush(event_queue, (win.free_time, 0, event_counter, "window_free", 
                                             {"window_idx": window_idx, "client": next_client, "weight": next_weight}))
                event_counter += 1
            else:
                # Очередь пустая - запланировать проверку простаивания
                check_time = current_time + idle_threshold
                heapq.heappush(event_queue, (check_time, 2, event_counter, "window_check", {"window_idx": window_idx}))
                event_counter += 1
        
        elif event_type == "window_check":
            window_idx = data["window_idx"]
            
            # Не закрываем первое окно
            if window_idx == 0:
                continue
            
            win = wins[window_idx]
            
            # Проверяем: окно всё ещё простаивает?
            if win.free_time <= current_time:
                # Проверяем правило "до конца часа"
                next_hour = ((win.open_time // 60) + 1) * 60
                
                # Закрываем, если вес очереди маленький
                if current_queue_weight <= close_threshold:
                    if current_time >= next_hour or current_time >= day_end:
                        # Закрываем окно
                        win.is_open = False
                        total_window_work_time += (current_time - win.open_time)
    
    # Считаем время работы для окон, которые не закрылись до конца дня
    for win in wins:
        if win.is_open:
            total_window_work_time += (day_end - win.open_time)
    
    # Считаем метрики
    waits = [c.wait_time for c in clients]
    av_wait = sum(waits) / len(waits)
    max_wait = max(waits)
    over15 = sum(1 for w in waits if w > 15) / len(waits) * 100
    
    return av_wait, max_wait, over15, total_window_work_time

def din_case(s, br, v):
    target_date = pd.to_datetime(s).date()
    one_d = get_data_from_client_arrivals(DATA+"client_arrivals.csv", br)
    
    clients = client_list(one_d[one_d['arrival_datetime'].dt.date == target_date])
    
    print(f"Суммарное время обслуживания: {sum([c.serv_time for c in clients])}")
    print(f"Клиентов: {len(clients)}")
    print()
    
    # Тестируем с разными порогами
    print("open_thresh | close_thresh | idle_thresh | ср.ожидание | макс.ожидание | %>15мин | время работы окон")
    for open_t in [20, 30, 40]:
        for close_t in [10, 15, 20, 30]:
            av, mx, over, work = din_simulate(clients, br, max_wins=v,
                                               open_threshold=open_t, 
                                               close_threshold=close_t, 
                                               idle_threshold=10)
            print(f"{open_t} | {close_t} | 10 | {av:.1f} | {mx:.1f} | {over:.1f}% | {work} мин")

s = '2025-06-'
for d in range(2, 28):
    for br in ("BR01", "BR02", "BR03"):
        v = 6
        if br != "BR01": v = 4
        din_case(s+str(d), br, v)
