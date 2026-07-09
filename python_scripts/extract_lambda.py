import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Чтобы работало без дисплея (headless)
import matplotlib.pyplot as plt
import seaborn as sns

path = "/home/you/bmm/"

# Загружаем данные
clients = pd.read_csv(path + 'dataset/client_arrivals.csv')

# Превращаем строку с датой в нормальный datetime
clients['arrival_datetime'] = pd.to_datetime(clients['arrival_datetime'])

# Извлекаем час и день недели
clients['hour'] = clients['arrival_datetime'].dt.hour
clients['weekday'] = clients['arrival_datetime'].dt.day_name()

# Считаем число клиентов для каждой пары набора признаков
hourly_load = clients.groupby(['weekday', 'hour', 'operation_id', 'branch_id']).size().reset_index(name='lambda_count')

hourly_load['av_lambda'] = hourly_load['lambda_count']/4
# Сохраняем в CSV — это ваш параметр λ(t)
hourly_load.to_csv(path+'heatmaps/lambda_parameters.csv', index=False)
print("Параметр λ(t) сохранён в heatmaps/lambda_parameters.csv")
print(hourly_load)

for branch in ('BR01', 'BR02', 'BR03'):
    for op in (1, 2, 3, 4, 5, 6):
        op = f'OP0{op}'
        for val in ('lambda_count', 'av_lambda'):
            # Строим heatmap и сохраняем в PNG
            pivot = hourly_load[(hourly_load['branch_id'] == branch) & (hourly_load['operation_id'] == op)].pivot(index='weekday', columns='hour', values=val)
            # Упорядочиваем дни недели
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            pivot = pivot.reindex(day_order)

            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f')
            plt.title(f'Интенсивность потока операций {op} в отделении {branch}')
            plt.xlabel('Час дня')
            plt.ylabel('День недели')
            plt.tight_layout()
            plt.savefig(path + f'heatmaps/heatmap_{branch}_{op}_{val}.png', dpi=150)
            plt.close()
            print(f"Heatmap_{branch}_{op}_{val} сохранён")
