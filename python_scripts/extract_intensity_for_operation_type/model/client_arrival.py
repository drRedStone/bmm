# Файл модели "Визит клиентв"
from __future__ import annotations # Для линтера
import csv
from typing import List




class ClientArrival:
	def __init__(
			self,
			arrival_datetime,
			weekday,
			operation_id,
			service_time_min,
			branch_id
		):
			self.arrival_datetime = arrival_datetime
			self.weekday = weekday
			self.operation_id = operation_id
			self.service_time_min = service_time_min
			self.branch_id = branch_id




	@staticmethod
	def from_csv_file(
		filename: str
	) -> List[ClientArrival]:
		"""
		Данный метод предназначен для формирования объектов
		из csv файла напрямую
		"""
		with open(filename, encoding='utf-8') as file:
			reader = csv.DictReader(file)
			return ClientArrival.from_csv(reader)
		
	@staticmethod
	def from_csv(
		dict_reader: csv.DictReader
	) -> List[ClientArrival]:
		"""
		Данный метод предназначен для формирования объектов
		из csv файла напрямую
		"""
		result = []
		for row in dict_reader:
			result.append(ClientArrival(
				row['arrival_datetime'],
				row['weekday'],
				row['operation_id'],
				row['service_time_min'],
				row['branch_id']
			))
		return result
		