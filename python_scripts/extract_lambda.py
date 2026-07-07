import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Чтобы работало без дисплея (headless)
import matplotlib.pyplot as plt
import seaborn as sns

path = "/home/you/bmm/dataset/"

# Загружаем данные
clients = pd.read_csv(path + 'client_arrivals.csv')

# Превращаем строку с датой в нормальный datetime
clients['arrival_datetime'] = pd.to_datetime(clients['arrival_datetime'])

# Извлекаем час и день недели
clients['hour'] = clients['arrival_datetime'].dt.hour
clients['weekday'] = clients['arrival_datetime'].dt.day_name()

# Считаем число клиентов для каждой пары (день_недели, час)
hourly_load = clients.groupby(['weekday', 'hour', 'branch_id']).size().reset_index(name='lambda_count')

hourly_load['lph'] = hourly_load['lambda_count']/4
# Сохраняем в CSV — это ваш параметр λ(t)
hourly_load.to_csv('lambda_parameters.csv', index=False)
print("Параметр λ(t) сохранён в lambda_parameters.csv")
print(hourly_load)
for j in ('lambda_count', 'lph'):
    for i in ('BR01', 'BR02', 'BR03'):
        # Строим heatmap и сохраняем в PNG
        pivot = hourly_load[hourly_load['branch_id'] == i].pivot(index='weekday', columns='hour', values=j)
        # Упорядочиваем дни недели
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        pivot = pivot.reindex(day_order)

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f')
        plt.title('Интенсивность потока клиентов λ(t)')
        plt.xlabel('Час дня')
        plt.ylabel('День недели')
        plt.tight_layout()
        plt.savefig('/home/you/bmm/heatmaps/ +i+ f'_lambda_heatmap{j}.png', dpi=150)
        print(f"Heatmap сохранён в {i} lambda_heatmap{j}.png")
