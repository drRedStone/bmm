def simulate_day(clients, window_schedule):
    """
    Симулирует день с переменным числом окон по часам.
    Возвращает: {(день, час): среднее_ожидание}
    """
    max_windows = max(window_schedule.values()) if window_schedule else 1
    wins = [Window(None) for _ in range(max_windows)]
    
    results = {}
    hour_clients = []
    current_hour = None
    
    for client in clients:
        client_hour = client.arrival_min // 60
        
        # Если сменился час — обрабатываем предыдущий
        if client_hour != current_hour and current_hour is not None:
            waits = [c.wait_time for c in hour_clients]
            avg_wait = sum(waits) / len(waits) if waits else 0
            results[(current_hour)] = avg_wait
            hour_clients = []
        
        current_hour = client_hour
        hour_clients.append(client)
        
        # Определяем, сколько окон открыто в этот час
        n_open = window_schedule.get(client_hour, 1)
        
        # Находим свободное окно среди открытых
        available_wins = wins[:n_open]
        idx = available_wins.index(min(available_wins, key=lambda w: w.free_time))
        
        # Обслуживаем клиента
        client.wait_time = max(0, wins[idx].free_time - client.arrival_min)
        wins[idx].free_time = client.arrival_min + client.wait_time + client.serv_time
    
    # Обрабатываем последний час
    if hour_clients:
        waits = [c.wait_time for c in hour_clients]
        avg_wait = sum(waits) / len(waits) if waits else 0
        results[(current_hour)] = avg_wait
    
    return results


# Итеративный подбор числа окон
window_schedule = {hour: 1 for hour in range(9, 19)}

for iteration in range(20):
    results = simulate_day(clients, window_schedule)
    
    needs_more = [hour for hour, avg_wait in results.items() if avg_wait > 10]
    
    if not needs_more:
        print(f"Сходимость на итерации {iteration}")
        break
    
    for hour in needs_more:
        window_schedule[hour] += 1
    
    print(f"Итерация {iteration}: добавлены окна в часах {needs_more}")

# Сохраняем результат
print("\nФинальное расписание окон:")
for hour in range(9, 19):
    print(f"{hour}:00 - {window_schedule[hour]} окон")
