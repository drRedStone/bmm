# Оптимизация графика работы отделений банка — код проекта

Все модули полностью документированы: у каждой функции — docstring с
Args/Returns, у нетривиальных мест внутри функций — построчные комментарии.

## Порядок запуска / зависимостей

1. `required_windows.py` — оценка λ(t), формула Эрланга C, требуемое число окон по часам (basic/credit/mortgage)
2. `schedule_greedy.py` — жадный алгоритм составления расписания (+ schedule_week для масштабирования на неделю)
3. `schedule_ilp.py` — точный ILP через scipy.optimize.milp (жёсткая и мягкая версии, + schedule_week_ilp)
4. `queue_simulation.py` — дискретно-событийная симуляция (SimPy), валидация формулы Эрланга C
5. `compare_policies.py` — сравнение naive/greedy/ILP на одном дне/отделении
6. `compare_branches.py` — то же самое по всем трём отделениям
7. `week_pipeline.py` — масштабирование сравнения на всю неделю
8. `br03_deep_dive.py` — диагностика структурных/физических ограничений BR03 (учитывает реальный состав сотрудников)
9. `estimate_lambda.py` — отдельный скрипт-утилита для оценки λ с выгрузкой в CSV

Все файлы ожидают датасет в подпапке `dataset/` (client_arrivals.csv, operations.csv,
employees.csv, branches.csv, hourly_load.csv) рядом с собой.

Запуск любого файла напрямую (`python3 schedule_ilp.py` и т.д.) выполняет его
`__main__`-блок с демонстрацией на реальных данных.

## Стек
Python 3.10+, pandas, numpy, matplotlib, scipy (scipy.optimize.milp — солвер HiGHS,
входит в scipy "из коробки"), simpy (для дискретно-событийной симуляции).
