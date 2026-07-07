#!/bin/python
# Данный файл нужен для получения зависимости
# интенсивности от типв операции
from enum import auto
from typing import List, Optional, Union

from matplotlib.pylab import Enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import seaborn as sns


from model.operation import Operation
from model.client_arrival import ClientArrival
from model.branch import Branch




DATASET_PATH = Path(__file__).parent.parent.parent / 'dataset'


operations = list()
branchs = list()
client_arrivals = list()
def read_csv_structures():
	"""
	Печать структуры каждого csv файла
	"""
	for el in os.listdir(DATASET_PATH):
		fullpath = os.path.join(DATASET_PATH, el)
		if	os.path.exists(fullpath) and\
			os.path.isfile(fullpath) and\
			fullpath.endswith('.csv') :
			with open(fullpath, mode='r', encoding='utf-8') as file:
				df = pd.read_csv(file)
				print(el)
				print(df.head)
def load_all():
	"""
	Загрузка всех моделей
	"""
	global operations, client_arrivals, branchs
	operations = Operation.from_csv_file(
		os.path.join(DATASET_PATH, 'operations.csv')
	)
	client_arrivals = ClientArrival.from_csv_file(
		os.path.join(DATASET_PATH, 'client_arrivals.csv')
	)
	branchs = Branch.from_csv_file(
		os.path.join(DATASET_PATH, 'branches.csv')
	)


def intensity_for_operation_type():
	"""
	Зависимость интенсивности от типа операции
	"""
	operations_labels = [el.name for el in operations]
	operations_intensity = np.zeros(len(operations))
	for el in client_arrivals:
		idx = Operation.get_number_from_str_id(operations, el.operation_id)
		operations_intensity[idx] += 1
	print(operations_intensity)

	fig, ax = plt.subplots()
	bars = ax.bar(operations_labels, operations_intensity)
	ax.bar_label(bars, label_type='edge', color='black')
	ax.set_yticks(np.arange(0,3000,50))
	plt.grid(True)
	plt.show()
	

# Существующий Enum для группировки
class GroupBy(Enum):
	DATE = auto()
	WEEKDAY = auto()
	WEEKDAY_NAME = auto()

# Новый Enum для типа визуализации
class ViewType(Enum):
	HEATMAP = auto()            # тепловая карта: дата/день недели vs час (по умолчанию)
	BAR_BY_DATE = auto()        # столбчатая диаграмма: операции на конкретную дату
	HEATMAP_OP_VS_DAY = auto()  # тепловая карта: дата vs операция (без часов)

# ... предыдущие импорты и Enum ...

def intensity_operation_type_of_date_and_time(
	group_by: GroupBy = GroupBy.DATE,
	branch_id: Optional[int] = None,
	operation_id: Optional[Union[str, List[str]]] = None,
	view_type: ViewType = ViewType.HEATMAP_OP_VS_DAY,
	target_date: Optional[str] = None,
	start_date: Optional[str] = None,
	end_date: Optional[str] = None
) -> None:
	"""
	Универсальная функция для визуализации интенсивности операций.

	Параметры
	----------
	group_by : GroupBy
		Способ группировки по дням (для HEATMAP_OP_VS_DAY и HEATMAP).
	branch_id : int, optional
		Фильтр по ветке.
	operation_id : str или список str, optional
		Фильтр по типу операции. Если None – все операции.
	view_type : ViewType
		- HEATMAP_OP_VS_DAY    : тепловая карта (операция vs дата/день недели) – ОСНОВНОЙ
		- HEATMAP              : тепловая карта (дата/день недели vs час)
		- BAR_BY_DATE          : столбчатая диаграмма по операциям за target_date
	target_date : str, optional
		Дата для BAR_BY_DATE (формат 'YYYY-MM-DD').
	start_date, end_date : str, optional
		Диапазон дат для HEATMAP_OP_VS_DAY (если не указаны – все даты).
	"""
	# --- Сбор данных ---
	raw_data = []
	for el in client_arrivals:
		if branch_id is not None and el.branch_id != branch_id:
			continue
		if operation_id is not None:
			if isinstance(operation_id, list):
				if el.operation_id not in operation_id:
					continue
			else:
				if el.operation_id != operation_id:
					continue

		dt = pd.to_datetime(el.arrival_datetime, format='%Y-%m-%d %H:%M')
		date = dt.date()
		hour = dt.hour
		op = el.operation_id
		raw_data.append({'date': date, 'hour': hour, 'operation': op})

	if not raw_data:
		print("Нет данных для выбранных фильтров.")
		return

	df = pd.DataFrame(raw_data)

	# --- Словарь для читаемых названий операций ---
	# operations – глобальный список объектов с полями operation_id и name
	op_id_to_name = {op.operation_id: op.name for op in operations}
	# На всякий случай: если operation_id нет в словаре, оставляем как есть
	def get_op_name(op_id):
		return op_id_to_name.get(op_id, op_id)

	# --- Обработка в зависимости от view_type ---
	if view_type == ViewType.HEATMAP_OP_VS_DAY:
		if start_date is not None and end_date is not None:
			start = pd.to_datetime(start_date).date()
			end = pd.to_datetime(end_date).date()
			df = df[(df['date'] >= start) & (df['date'] <= end)]
			if df.empty:
				print(f"Нет данных за период {start_date} - {end_date}")
				return

		if group_by == GroupBy.DATE:
			df['group_key'] = df['date']
		elif group_by == GroupBy.WEEKDAY:
			df['group_key'] = df['date'].apply(lambda d: d.weekday())
		else:  # WEEKDAY_NAME
			df['group_key'] = df['date'].apply(lambda d: d.strftime('%a'))

		pivot = df.pivot_table(index='group_key', columns='operation', aggfunc='size', fill_value=0)

		# Переименовываем столбцы из operation_id в названия
		pivot.columns = [get_op_name(col) for col in pivot.columns]

		if group_by in (GroupBy.WEEKDAY, GroupBy.WEEKDAY_NAME):
			if group_by == GroupBy.WEEKDAY:
				order = range(7)
			else:
				order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
			pivot = pivot.reindex(order, fill_value=0)
			pivot.index = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']

		plt.figure(figsize=(12, max(6, len(pivot) * 0.3)))
		sns.heatmap(pivot, cmap='inferno', annot=True, fmt='d',
					cbar_kws={'label': 'Количество клиентов'})

		period = f"{start_date} – {end_date}" if start_date and end_date else "весь период"
		title = f'Интенсивность операций по дням ({period})'
		if operation_id is not None:
			title += ' (фильтр по операциям)'
		if branch_id is not None:
			title += f', ветка {branch_id}'
		plt.title(title)
		plt.xlabel('Тип операции')
		ylabel = 'День недели' if group_by != GroupBy.DATE else 'Дата'
		plt.ylabel(ylabel)
		plt.tight_layout()
		plt.show()

	elif view_type == ViewType.HEATMAP:
		if group_by == GroupBy.DATE:
			df['group_key'] = df['date']
		elif group_by == GroupBy.WEEKDAY:
			df['group_key'] = df['date'].apply(lambda d: d.weekday())
		else:
			df['group_key'] = df['date'].apply(lambda d: d.strftime('%a'))

		pivot = df.pivot_table(index='group_key', columns='hour', aggfunc='size', fill_value=0)

		if group_by in (GroupBy.WEEKDAY, GroupBy.WEEKDAY_NAME):
			if group_by == GroupBy.WEEKDAY:
				order = range(7)
			else:
				order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
			pivot = pivot.reindex(order, fill_value=0)
			pivot.index = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']

		figsize = (14, 6) if group_by != GroupBy.DATE else (14, 10)
		plt.figure(figsize=figsize)
		sns.heatmap(pivot, cmap='inferno', annot=True, fmt='d',
					cbar_kws={'label': 'Количество клиентов'})
		title = 'Тепловая карта интенсивности (часы vs дни)'
		if operation_id is None:
			title += ' (все операции)'
		elif isinstance(operation_id, list):
			# для заголовка тоже можно имена, но оставим для краткости
			title += f' (операции {", ".join([get_op_name(o) for o in operation_id])})'
		else:
			title += f' (операция {get_op_name(operation_id)})'
		if branch_id is not None:
			title += f', ветка {branch_id}'
		plt.title(title)
		plt.xlabel('Час дня')
		plt.ylabel('День недели' if group_by != GroupBy.DATE else 'Дата')
		plt.tight_layout()
		plt.show()

	elif view_type == ViewType.BAR_BY_DATE:
		if target_date is None:
			print("Для BAR_BY_DATE необходимо указать target_date (формат 'YYYY-MM-DD')")
			return
		target = pd.to_datetime(target_date).date()
		filtered = df[df['date'] == target]
		if filtered.empty:
			print(f"Нет данных за {target_date}")
			return
		counts = filtered.groupby('operation').size().reset_index(name='count')
		counts = counts.sort_values('count', ascending=False)

		# Заменяем ID на названия
		counts['operation'] = counts['operation'].apply(get_op_name)

		plt.figure(figsize=(10, 6))
		sns.barplot(data=counts, x='operation', y='count', palette='Blues_d')
		plt.title(f'Интенсивность операций за {target_date}')
		if branch_id is not None:
			plt.title(plt.title().get_text() + f', ветка {branch_id}')
		plt.xlabel('Тип операции')
		plt.ylabel('Количество клиентов')
		for i, v in enumerate(counts['count']):
			plt.text(i, v + 0.5, str(v), ha='center')
		plt.tight_layout()
		plt.show()





def main():
	if not DATASET_PATH.exists():
		raise Exception('Не найдена директория исходных данных')
	load_all()
	intensity_for_operation_type()
	intensity_operation_type_of_date_and_time(GroupBy.WEEKDAY)
	input()

if __name__ == '__main__':
	main()