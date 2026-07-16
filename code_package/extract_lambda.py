"""
Строит тепловые карты интенсивности потока клиентов (heatmaps) на основе
данных client_arrivals.csv.

Для каждого отделения и типа операции вычисляется среднее число клиентов
в час (lambda_count) и среднее за 4 недели (av_lambda), после чего
сохраняются в папку heatmaps/:
- отдельные heatmaps для каждой операции,
- агрегированные по всем операциям для каждого отделения.

Также создаётся сводная таблица lambda_parameters.csv.
"""
from pathlib import Path

import pandas as pd
import matplotlib
from config import WEEKDAYS
matplotlib.use('Agg')  # Чтобы работало без дисплея (headless)
import matplotlib.pyplot as plt
import seaborn as sns

path = Path(__file__).parent.parent

# Загружаем данные
clients = pd.read_csv(path / 'dataset/client_arrivals.csv')

# Превращаем строку с датой в нормальный datetime
clients['arrival_datetime'] = pd.to_datetime(clients['arrival_datetime'])

# Извлекаем час и день недели
clients['hour'] = clients['arrival_datetime'].dt.hour
clients['weekday'] = clients['arrival_datetime'].dt.day_name()

# Считаем число клиентов для каждой пары набора признаков
hourly_load = clients.groupby(['weekday', 'hour', 'operation_id', 'branch_id']).size().reset_index(name='lambda_count')

hourly_load['av_lambda'] = hourly_load['lambda_count']/4
# Сохраняем в CSV — это ваш параметр λ(t)
hourly_load.to_csv(path / 'heatmaps/lambda_parameters.csv', index=False)
print("Параметр λ(t) сохранён в heatmaps/lambda_parameters.csv")
print(hourly_load)

for branch in ('BR01', 'BR02', 'BR03'):
    for op in (1, 2, 3, 4, 5, 6):
        op = f'OP0{op}'
        for val in ('lambda_count', 'av_lambda'):
            # Строим heatmap и сохраняем в PNG
            pivot = hourly_load[(hourly_load['branch_id'] == branch) & (hourly_load['operation_id'] == op)].pivot(index='weekday', columns='hour', values=val)
            # Упорядочиваем дни недели
            pivot = pivot.reindex(WEEKDAYS)

            plt.figure(figsize=(12, 6))
            sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f')
            plt.title(f'Интенсивность потока операций {op} в отделении {branch}')
            plt.xlabel('Час дня')
            plt.ylabel('День недели')
            plt.tight_layout()
            plt.savefig(path / f'heatmaps/heatmap_{branch}_{op}_{val}.png', dpi=150)
            plt.close()
            print(f"Heatmap_{branch}_{op}_{val} сохранён")
for branch in ('BR01', 'BR02', 'BR03'):
    for val in ('lambda_count', 'av_lambda'):
        # Строим heatmap и сохраняем в PNG
        aggregated = (hourly_load[hourly_load['branch_id'] == branch] \
            .groupby(['weekday', 'hour'], as_index=False)[val] \
            .sum())
        pivot = aggregated.pivot(index='weekday', columns='hour', values=val)
        # Упорядочиваем дни недели
        pivot = pivot.reindex(WEEKDAYS)

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f')
        plt.title(f'Интенсивность потока операций в отделении {branch}')
        plt.xlabel('Час дня')
        plt.ylabel('День недели')
        plt.tight_layout()
        plt.savefig(path / f'heatmaps/heatmap_{branch}_{val}.png', dpi=150)
        plt.close()
        print(f"Heatmap_{branch}_{val} сохранён")
