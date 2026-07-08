from random import randint as rnd, choices as ch
import pandas as pd
import numpy as np
from prop import *


days = ('Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб')
ops = ('OP1', 'OP2', 'OP3', 'OP4', 'OP5', 'OP6')
opt = (5, 10, 15, 25, 40, 50)

opt_dickt = {ops[i]:opt[i] for i in range(6)}

count_clients = 600

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

#===== ПРОСТАЯ СИМУЛЯЦИЯ С РАНДОМОМ =====

#clients = [Client(ch(days)[0], rnd(0, 540), ch(ops, weights = (2018, 1362, 1042, 1018, 621, 334))[0], None) for _ in range(count_clients)]
#clients.sort(key = lambda c: (days.index(c.day), c.arrival_min))

#wins = [Window(None) for _ in range(5)]

#i = 0
#for day in days:
#    while (i<count_clients) and (clients[i].day == day):

#       idx = wins.index(min(wins, key = lambda w: w.free_time))
#       clients[i].wait = wins[idx].free_time - clients[i].arrival_min
#        if clients[i].wait<0: clients[i].wait = 0 
#        wins[idx].free_time = clients[i].arrival_min + clients[i].wait + opt_dickt[clients[i].op_id]
#        i+=1
#    for win in wins: win.free_time = 0
#for i in clients: print(i.wait, end = " ")
    


#===== СИМУЛЯЦИЯ С ДАННЫМИ ИЗ ДАТАСЕТА =====

# берем датасет
df = pd.read_csv(DATA + 'client_arrivals.csv', parse_dates=['arrival_datetime'])
#берем первую неделю
week1 = df[(df['arrival_datetime'].between('2025-06-01', '2025-06-07 23:59:59')) & (df["branch_id"] == "BR01")]

#формируем список клиентов
clients = [Client(i['weekday'], i['arrival_datetime'].minute + i['arrival_datetime'].hour*60, i['operation_id'], i['branch_id'], serv_t = i['service_time_min']) for i in week1.to_dict('records')]

#сортируем клиентов
clients.sort(key = lambda c: (days.index(c.day), c.arrival_min))

wins = [Window(None) for _ in range(5)]


#перебираем клиентов
i = 0
for day in days:
    while (i<len(clients))  and (clients[i].day == day):

       idx = wins.index(min(wins, key = lambda w: w.free_time))
       clients[i].wait_time = wins[idx].free_time - clients[i].arrival_min
       if clients[i].wait_time<0: clients[i].wait_time = 0
       wins[idx].free_time = clients[i].arrival_min + clients[i].wait_time + clients[i].serv_time
       i+=1
    for win in wins: win.free_time = 540

for i in clients: print(round(i.wait_time), end = " ")
