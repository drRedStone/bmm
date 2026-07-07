from random import randint as rnd, choice as ch

days = ('mon', 'tue', 'wed', 'thu', 'fri')
ops = ('OP1', 'OP2', 'OP3', 'OP4', 'OP5', 'OP6')
opt = (5, 10, 15, 25, 40, 50)

class Client():
    def __init__(self, day, arrival_min, op_id, branch_id):
        self.day = day
        self.arrival_min = arrival_min
        self.op_id = op_id
        self.branch_id = branch_id
        self.serv_time = 0 #генерим рандом, либо из датасекса
        self.wait_time = 0


class Employee():
    def __init__ (self, emp_id, branch_id, grade, name):
        self.emp_id = emp_id
        self.branch_id = branch_id
        self.grade = grade
        self.name = name
        self.worked_hours = 0


class Window():
    def __init__(self, branch_id, free_time):
        self.branch_id = branch_id
        self.free_time = free_time
        self.employee = None


clients = [Client(ch(days), rnd(0, 540), 
