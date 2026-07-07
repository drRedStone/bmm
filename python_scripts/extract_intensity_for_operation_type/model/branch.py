# Файл модели "Отдел"
from __future__ import annotations # Для линтера
import csv
from typing import List




class Branch:
	def __init__(
			self,
			branch_id,
			name,
			type,
			n_windows,
			hours_weekday,
			hours_saturday,
			hours_sunday
		):
			self.branch_id = branch_id
			self.name = name
			self.type = type
			self.n_windows = n_windows
			self.hours_weekday = hours_weekday
			self.hours_saturday = hours_saturday
			self.hours_sunday = hours_sunday

	


	@staticmethod
	def from_csv_file(
		filename: str
	) -> List[Branch]:
		"""
		Данный метод предназначен для формирования объектов
		из csv файла напрямую
		"""
		with open(filename, encoding='utf-8') as file:
			reader = csv.DictReader(file)
			return Branch.from_csv(reader)
		
	@staticmethod
	def from_csv(
		dict_reader: csv.DictReader
	) -> List[Branch]:
		"""
		Данный метод предназначен для формирования объектов
		из csv файла напрямую
		"""
		result = []
		for row in dict_reader:
			result.append(Branch(
				row['branch_id'],
				row['name'],
				row['type'],
				row['n_windows'],
				row['hours_weekday'],
				row['hours_saturday'],
				row['hours_sunday']
			))
		return result