import pandas as pd
import numpy as np
from prop import *

days = ('Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб')

class Client():
    def __init__(self, day, arrival_min, op_id, branch_id, serv_t = 0):
        self.day = day
        self.arrival_min = arrival_min
        self.op_id = op_id
        self.branch_id = branch_id
        self.serv_time = serv_t
        self.wait_time = 0


class Employee():
    def __init__ (self, emp_id, branch_id, grade, name):
        self.emp_id = emp_id
        self.branch_id = branch_id
        self.grade = grade
        self.name = name
        self.worked_hours = 0


class Window():
    def __init__(self, branch_id):
        self.branch_id = branch_id
        self.free_time = 540
        self.employee = None
        

#===== СИМУЛЯЦИЯ С ДАННЫМИ ИЗ ДАТАСЕТА =====

# берем датасет
df = pd.read_csv(DATA + 'client_arrivals.csv', parse_dates=['arrival_datetime'])
#берем первую неделю
week1 = df[(df['arrival_datetime'].between('2025-06-01', '2025-06-07 23:59:59')) & (df["branch_id"] == "BR01")]

#формируем список клиентов
clients = [Client(i['weekday'], i['arrival_datetime'].minute + i['arrival_datetime'].hour*60, i['operation_id'], i['branch_id'], serv_t = i['service_time_min']) for i in week1.to_dict('records')]

#сортируем клиентов
clients.sort(key = lambda c: (days.index(c.day), c.arrival_min))


#перебираем клиентов
def stat_simulate(clients, n_wins):
    wins = [Window(None) for _ in range(n_wins)]
    #симулируем
    i = 0
    for day in days:
        while (i<len(clients))  and (clients[i].day == day):
            idx = wins.index(min(wins, key = lambda w: w.free_time))
            clients[i].wait_time = wins[idx].free_time - clients[i].arrival_min
            if clients[i].wait_time<0: clients[i].wait_time = 0
            wins[idx].free_time = clients[i].arrival_min + clients[i].wait_time + clients[i].serv_time
            i+=1
        for win in wins: win.free_time = 540
    
    #считаем метрики
    waits = [c.wait_time for c in clients]
    for i in clients: i.wait = 0
    av_wait = sum(waits)/len(waits)
    max_wait = max(waits)
    over15 = sum(1 for w in waits if w > 15)/len(waits)*100
    return av_wait, max_wait, over15

print("кол-во окон|среднее ждание|максимальное ждание|процйент>15мин")
for i in range(1, 6):
    l = stat_simulate(clients, i)
    print(f"{i} | {l[0]} | {l[1]} | {l[2]}")

#пробуем менять количество окон в симуляции
def cl_iter(one_day, wins):
    for cl in one_day:
        pass
        
        
        

def din_sim(clients, day):
    
    wins = [Window(None)]
    for day in days: pass
