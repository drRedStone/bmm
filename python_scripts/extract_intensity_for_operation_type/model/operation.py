# Файл модели "Типа операции"
from __future__ import annotations # Для линтера
from typing import List, Optional
import csv




class Operation:
	def __init__(
			self,
			operation_id,
			name,
			avg_service_min,
			std_service_min,
			min_service_min,
			frequency_weight,
			required_skill  
		):
		self.operation_id = operation_id
		self.name = name
		self.avg_service_min = avg_service_min
		self.std_service_min = std_service_min
		self.min_service_min = min_service_min
		self.frequency_weight = frequency_weight
		self.required_skill = required_skill




	@staticmethod
	def from_csv(
		dict_reader: csv.DictReader
	) -> List[Operation]:
		"""
		Данный метод предназначен для формирования объектов
		из csv файла напрямую
		"""
		result = []
		for row in dict_reader:
			result.append(Operation(
				row['operation_id'],
				row['name'],
				row['avg_service_min'],
				row['std_service_min'],
				row['min_service_min'],
				row['frequency_weight'],
				row['required_skill']
			))
		return result
	
	@staticmethod
	def from_csv_file(
		filename: str
	) -> List[Operation]:
		"""
		Данный метод предназначен для формирования объектов
		из csv файла напрямую
		"""
		with open(filename, encoding='utf-8') as file:
			reader = csv.DictReader(file)
			return Operation.from_csv(reader)


	@staticmethod
	def get_number_from_str_id(
		operations_list: List[Operation],
		str_id: str
	) -> Optional[int]:
		"""
		Данный метод предназначен для поиска номера элемента
		при формипровании данных для визуализации
		"""
		for i, el in enumerate(operations_list):
			if el.operation_id == str_id:
				return i
		return None