# Скипай на 20 строку!!!

datasex - папка с данными нам датасетами, ничего интересного

puthon_scripts - папка с кодом:

extract_parametrs.py - скрипт берет client_arrivals.csv, группирует данные по дню недели, часу прихода и отделению. по полученной таблие строит 3 тепловые каты по каждому отделению, на которых отображена средняя нагрузка в каждый час каждого дня
    
lambda_parameters.csv - сгруппировання таблица, использованная для создания тепловых карт    

queue.py - простая модель очереди, в которой можно задать количество окон, количество людей за день, время прихода и время обслживания генерируются случайно с ограничениями (от 5 до 30 минут на обслуживание, приход от 0 до 540 минуты, где рабочий день начинается с нулевой минуты). выводит в консоль по порядку время ожидания каждого клиента

model.py - модель работы банка с сотрудниками, клиентами, отделениями и окнами.
    
    




# Система оптимизации штата и расписаний отделений банка

Данный репозиторий содержит набор Python-модулей для расчёта оптимального состава сотрудников (junior/middle/senior) и формирования недельных расписаний с минимизацией времени ожидания клиентов.  
**Цель для разработчика:** за 8 рабочих часов превратить этот аналитический бэкенд в полноценный веб-сайт, через который бизнес-пользователь сможет запускать расчёты и просматривать результаты.

---

## 1. Что уже реализовано (обзор модулей)

| Модуль | Назначение | Ключевые функции |
|--------|-----------|-----------------|
| `optimize_staff.py` | Подбор оптимального штата (J/M/S) перебором + ILP | `optimize_staff_for_branch()` – перебор комбинаций, параллельный запуск, Парето-фронт |
| `schedule_ilp.py` | Точное ILP-планирование расписания (HiGHS) | `schedule_day_ilp_soft()`, `schedule_week_ilp()` |
| `schedule_greedy.py` | Жадный алгоритм расписания | `schedule_day()`, `schedule_week()` |
| `required_windows.py` | Расчёт требуемого числа окон по часам (Эрланг C) | `required_windows_table()` |
| `compare_policies.py` | Сравнение трёх политик: naive/greedy/ILP, метрики, графики | `get_weekly_metrics()`, `plot_*()`, `weekly_schedule_to_json()` |
| `queue_simulation.py` | Имитационное моделирование очереди (SimPy) | `simulate_day_continuous()`, `validate_schedule_with_simulation()` |
| `config.py` | Все константы (ставки, часы работы, цвета, имена файлов, параметры оптимизации) | Импортируется во все модули |
| `week_pipeline.py` | Недельное сравнение трёх политик | `week_summary()` |
| `compare_branches.py` | Сравнение отделений за один день | `compare_branch()` |
| `extract_lambda.py` / `estimate_lambda.py` | Оценка интенсивности потока клиентов | – |
| `срет_расписанием.py` (опечатка) | Генерация Excel с расписаниями | – |

**Важные изменения, внесённые в процессе отладки:**
- Все магические числа вынесены в `config.py` (и в сам `optimize_staff.py` как константы).
- Добавлена многопроцессная обработка комбинаций с прогресс-баром (`tqdm`) и fallback-режимом.
- Исправлены ошибки с ширинами столбцов таблиц и переполнением графиков при большом количестве отделений.
- Для отделения BR03 автоматически ослабляется целевое время ожидания (с 10 до 15 мин.), так как при жёстком ограничении допустимых вариантов нет.
- Параметр `STAFF_MULTIPLIER` управляет максимальным соотношением «всего сотрудников / число окон»; по умолчанию = 3.

Все модули уже **работают из командной строки**, генерируют `.json`, `.csv` и `.png` с отчётами.  
Главный «движок» для подбора штата – `optimize_staff.py`, он и будет ядром веб-сервиса.

---

## 2. Быстрый старт (локальный запуск)

### 2.1. Требования
- Python 3.9+
- Установленные пакеты (желательно в виртуальном окружении):
  ```bash
  pip install pandas numpy scipy matplotlib simpy tqdm openpyxl
  ```
- Наличие папки `dataset/` с CSV-файлами:
  - `client_arrivals.csv`
  - `operations.csv`
  - `employees.csv`
  - `branches.csv`
  - `hourly_load.csv`

### 2.2. Проверка работы
```bash
python optimize_staff.py
```
Ожидаемый результат: в консоли появится прогресс-бар перебора комбинаций, затем лучшие составы по трём отделениям и графики. В папке `optimize_staff/` (рядом с кодом) будут созданы файлы отчётов.

```bash
python compare_policies.py   # генерация отчётов сравнения политик в report/
python queue_simulation.py   # валидация симуляцией в sim_report/
```

---

## 3. Что требуется от веб-сайта (MVP за 8 часов)

Пользователь (менеджер отделения) должен:
1. Выбрать отделение (BR01/BR02/BR03) и задать ограничения: максимальное количество сотрудников по грейдам, целевое время ожидания, бюджет.
2. Запустить расчёт оптимального штата (функция `optimize_staff_for_branch`).
3. Увидеть:
   - Лучший состав (таблица junior/middle/senior, ФОТ, среднее ожидание, недобор окон).
   - Парето-фронт в виде графика (scatter plot).
   - Таблицу всех проверенных вариантов (или Парето-оптимальных).
4. Возможность скачать CSV с результатами.

**Веб-фреймворк:** рекомендуем **FastAPI** (проще для интеграции с Python-функциями) или Flask.  
**Фронтенд:** минимальный – одна HTML-страница с формой, таблицами и встроенными изображениями (графики можно отдавать как base64 или статические файлы).

---

## 4. Архитектура веб-приложения (предлагаемая)

```
project/
├── dataset/                     # CSV-файлы (неизменны)
├── modules/                     # все текущие .py файлы (можно оставить как есть)
│   ├── config.py
│   ├── optimize_staff.py        # главный модуль
│   ├── schedule_ilp.py
│   ├── schedule_greedy.py
│   ├── required_windows.py
│   ├── compare_policies.py      # для визуализаций и метрик
│   └── ...
├── webapp/
│   ├── app.py                   # FastAPI приложение
│   ├── static/                  # CSS, JS (опционально)
│   └── templates/
│       └── index.html           # основной интерфейс
├── reports/                     # сюда будут сохраняться временные отчёты
└── README.md
```

### 4.1. API эндпоинты (FastAPI)

```python
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import optimize_staff as opt
import config
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt

app = FastAPI()
templates = Jinja2Templates(directory="webapp/templates")

# Загрузка данных один раз при старте (глобально)
@app.on_event("startup")
def load_data():
    global ca, ops, br
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
```

**Основной endpoint (POST) – запуск оптимизации:**
```python
@app.post("/optimize")
async def run_optimization(
    branch_id: str = Form(...),
    max_junior: int = Form(6),
    max_middle: int = Form(6),
    max_senior: int = Form(6),
    target_wait: float = Form(15.0),
    budget: Optional[float] = Form(None)
):
    # Вызов функции из optimize_staff.py
    result = opt.optimize_staff_for_branch(
        ca, ops, br, branch_id,
        max_junior=max_junior, max_middle=max_middle, max_senior=max_senior,
        target_wait=target_wait, budget=budget, verbose=False
    )
    if result['best'] is None:
        return {"error": "Нет допустимых вариантов"}
    
    # Строим график (scatter + выделенный лучший)
    buf = io.BytesIO()
    # код построения графика, сохранение в buf
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    
    # Преобразуем DataFrame в HTML-таблицы
    pareto_html = result['pareto'].to_html(classes='table', index=False)
    
    return {
        "best": result['best'][:3],  # (junior, middle, senior)
        "cost": result['best'][3],
        "wait": result['best'][4],
        "shortfall": result['best'][5],
        "plot": img_base64,
        "pareto_table": pareto_html,
        "download_url": "/download_csv"  # сгенерировать временный файл
    }
```

**Страница с формой (GET `/`)** – рендерит `index.html` с полями.

**Скачивание CSV** – временный файл или генерация на лету.

### 4.2. Фронтенд (минимальный)

В шаблоне `index.html`:
- Выпадающий список отделений (можно захардкодить BR01/BR02/BR03).
- Поля ввода для max_junior/middle/senior, target_wait, budget.
- Кнопка «Рассчитать».
- После ответа от сервера (AJAX) отображается:
  - Блок с лучшим составом (карточка).
  - Изображение графика (в `<img src="data:image/png;base64, ...">`).
  - Таблица Парето-вариантов (вставляется через innerHTML).
  - Кнопка «Скачать CSV».

Всё это можно сделать на чистом HTML + Vanilla JS (fetch API) менее чем за час.

---

## 5. Пошаговый план реализации за 8 часов

### **Час 1-2: Подготовка и обёртка функции**
- Настройка FastAPI проекта, установка зависимостей.
- Перенос всех модулей в подпапку `modules/`, корректировка импортов (добавить `sys.path` или сделать пакет).
- Создание `app.py` с загрузкой данных на старте.
- Реализация endpoint `/optimize` с вызовом `optimize_staff_for_branch` (пока без графиков и таблиц, просто возвращает JSON).

### **Час 3-4: Генерация графиков и таблиц**
- Интеграция функций визуализации из `optimize_staff.py`: `plot_branch_detail`, но с возвратом изображения в base64.
- Формирование HTML-таблиц из DataFrame (метод `.to_html()`).
- Эндпоинт для скачивания CSV (можно генерировать временный файл или отдавать прямо из памяти).

### **Час 5-6: Фронтенд**
- Создание HTML-страницы с формой (используем Bootstrap для скорости).
- Написание JavaScript для отправки формы через `fetch`, обработки ответа, отображения данных.
- Вставка изображения и таблицы в DOM.
- Кнопка «Скачать CSV» (переход по ссылке, сгенерированной сервером).

### **Час 7: Обработка ошибок и улучшения UX**
- Показывать индикатор загрузки (спиннер) во время расчёта.
- Обработка случая «нет допустимых вариантов» (сообщение пользователю).
- Добавить валидацию вводимых чисел (min, max, step).
- Кеширование результатов? (не обязательно)

### **Час 8: Тестирование, документация и деплой**
- Проверить работу с разными отделениями и параметрами.
- Написать краткое руководство пользователя (можно прямо на странице).
- Упаковка в Docker (опционально) или инструкция по запуску.
- Финальный коммит и демонстрация.

---

## 6. Важные нюансы

1. **Производительность:** `optimize_staff_for_branch` может выполняться несколько минут. На вебе это неприемлемо в синхронном режиме. На MVP допустимо повесить обработчик и ждать, но для production нужен асинхронный воркер (Celery, Dramatiq, фоновая задача FastAPI). В рамках 8 часов можно использовать `BackgroundTasks` в FastAPI и поллинг статуса, либо оставить синхронный вызов, предупредив пользователя о длительности.

2. **Параллельность:** функция уже использует `ProcessPoolExecutor` внутри. При запуске из веб-сервера могут быть проблемы с доступом к глобальным данным в форкнутых процессах. Решение: передавать все данные (ca, ops, br) как аргументы в `_eval_one_combination` (уже сделано). Однако при использовании `ProcessPoolExecutor` внутри синхронного веб-запроса могут быть ограничения. Если будут падать ошибки, на время MVP можно временно переключить на последовательный режим: в `optimize_staff_for_branch` заменить `num_workers` на 0 или запускать без пула.

3. **Графики:** При сохранении в BytesIO не забудьте закрыть фигуру (`plt.close()`), иначе будут утечки памяти.

4. **Пути к данным:** Жёстко заданные `'dataset/...'` должны работать относительно корня проекта. При деплое убедитесь, что рабочая директория правильная.

5. **Конфигурация:** Все константы в `config.py`. Если нужно менять ставки или пороги для веба, можно сделать их настройку через переменные окружения или отдельный конфиг веб-приложения.

---

## 7. Примерный код для `app.py` (скелет)

```python
import sys
import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io, base64, matplotlib.pyplot as plt

# добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from modules import optimize_staff as opt
from modules.config import *

app = FastAPI()
templates = Jinja2Templates(directory="webapp/templates")

ca = ops = br = None

@app.on_event("startup")
def startup():
    global ca, ops, br
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/optimize")
async def optimize(
    branch_id: str = Form(...),
    max_junior: int = Form(6),
    max_middle: int = Form(6),
    max_senior: int = Form(6),
    target_wait: float = Form(15.0)
):
    result = opt.optimize_staff_for_branch(
        ca, ops, br, branch_id,
        max_junior=max_junior, max_middle=max_middle, max_senior=max_senior,
        target_wait=target_wait, verbose=False
    )
    if result['best'] is None:
        return {"error": "Нет допустимых вариантов."}
    
    # Генерация графика
    fig, ax = plt.subplots()
    df = result['all']
    ax.scatter(df['weekly_cost']/1000, df['avg_wait_min'], c='lightgray')
    best = result['best']
    ax.scatter(best[3]/1000, best[4], c='orange', s=120)
    ax.set_xlabel('Недельный ФОТ, тыс. руб.')
    ax.set_ylabel('Среднее ожидание, мин')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    
    pareto_html = result['pareto'].to_html(classes='table table-striped', index=False)
    
    return {
        "best_j": best[0],
        "best_m": best[1],
        "best_s": best[2],
        "cost": f"{best[3]:,.0f}",
        "wait": f"{best[4]:.1f}",
        "plot": img_b64,
        "pareto_table": pareto_html
    }
```

---

## 8. Итог

За 8 часов middle-разработчик способен обернуть уже готовый аналитический движок в простой, но функциональный веб-интерфейс. Ключевая задача – аккуратная интеграция существующих функций и представление результатов в понятном виде. Дальнейшее развитие может включать асинхронную очередь задач, кеширование, более продвинутую визуализацию и настройку параметров сотрудников.

**Дерзайте!** 🚀
