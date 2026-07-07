from random import randint as rnd
nw = 2 #количество окно, выявить, от чего зависит
nc = 250 #коичество людей за день, придумать, как задавать

wins = [0 for _ in range(nw)]

clients = [[rnd(5, 30), rnd(0, 540), 0] for _ in range(nc)] #обслуживание рандом, но должно зависить от типа операции, а чтобы генерить правдоподобно, нужно в анализ типов операций

clients.sort(key=lambda x: x[1])

for i in clients:
    idx = wins.index(min(wins))
    
    wait = wins[idx] - i[1]
    if wait<0: wait = 0

    i[2] = wait

    wins[idx] = sum(i)

    print(wait, end = " ")
print()
