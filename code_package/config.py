from pathlib import Path
from typing import List, Tuple


WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
MAX_WORK_TIME = 9
MAX_HOURS_WEEK = 40
MAX_WORK_TIME_BEFORE_LAUNCH = 4
MIN_WORK_TIME = 2
MIN_TIME_BETWEEN_BREAK_AND_END = 1
WEEKDAY_HOURS = {  # (open_hour, close_hour) — из branches.csv: будни 09-19, суббота 09-16, воскресенье закрыто
    'Monday': (9, 19), 'Tuesday': (9, 19), 'Wednesday': (9, 19),
    'Thursday': (9, 19), 'Friday': (9, 19), 'Saturday': (9, 16),
}

WERGES = {"junior": 350, "middle": 500, "senior": 700}
SKILLS = {
    "junior": ["basic"],
    "middle": ["basic", "credit"],
    "senior": ["basic", "credit", "mortgage"]
}

OPEN_HOUR = 9 # чекать, где используется и менять на обращение к словарю

REPORT_SAVE_DIRECTORY = Path(__file__).parent.parent / 'report'
SIM_REPORT_SAVE_DIRECTORY = Path(__file__).parent.parent / 'sim_report'
OPTIMIZE_STAFF_REPORT_SAVE_DIRECTORY = Path(__file__).parent.parent / 'optimize_staff'
GRADE_COLORS = {'junior': 'limegreen', 'middle': 'orange', 'senior': 'tomato'}



# ═══════════════════════════════════════════
#  Константы (бывшие магические числа)
# ═══════════════════════════════════════════
# --- Перебор составов ---
DEFAULT_MAX_JUNIOR: int = 6
DEFAULT_MAX_MIDDLE: int = 6
DEFAULT_MAX_SENIOR: int = 6
DEFAULT_TARGET_WAIT: float = 40.0          # минут
STAFF_MULTIPLIER: int = 3                 # общее число сотрудников ≤ STAFF_MULTIPLIER * n_win

# --- Параметры сотрудников (значения по умолчанию для create_staff) ---
DEFAULT_MAX_HOURS_WEEK: int = 40
DEFAULT_MAX_HOURS_DAY: int = 9
DEFAULT_LUNCH_MIN: int = 60

# --- Визуализация: детальный график ---
DETAIL_FIGSIZE: Tuple[int, int] = (16, 6)
DETAIL_WIDTH_RATIOS: List[int] = [3, 2]
SCATTER_ALL_COLOR: str = 'lightgray'
SCATTER_ALL_SIZE: int = 30
SCATTER_BEST_COLOR: str = 'orange'
SCATTER_BEST_SIZE: int = 120
SCATTER_BEST_EDGECOLOR: str = 'black'
SCATTER_BEST_ZORDER: int = 5
# Ширины столбцов таблицы детального графика (6 колонок)
DETAIL_TABLE_COLWIDTHS: List[float] = [0.1, 0.1, 0.1, 0.18, 0.15, 0.12]

# --- Визуализация: сводный отчёт ---
COMPARATIVE_FIGSIZE: Tuple[int, int] = (16, 12)
BAR_WIDTH: float = 0.35
BAR_COLOR_RED: str = 'red'
LEGEND_LOC: str = 'lower center'
LEGEND_BBOX: Tuple[float, float] = (0.5, -0.25)
LEGEND_NCOL: int = 2
# Ширины столбцов таблицы лучших составов (5 колонок)
TABLE_COLWIDTHS: List[float] = [0.12, 0.25, 0.18, 0.22, 0.18]
TABLE_FONTSIZE: int = 9
TABLE_TITLE_FONTSIZE: int = 12
PARETO_COLOR: str = 'steelblue'
PARETO_SIZE: int = 50
ANNOTATE_XYTEXT: Tuple[int, int] = (5, 5)
ANNOTATE_FONTSIZE: int = 7
SAVE_DPI: int = 150

# --- Имена выходных файлов ---
BEST_CSV: str = 'best_compositions.csv'
PARETO_CSV_PATTERN: str = 'pareto_{bid}.csv'
ALL_VARIANTS_CSV_PATTERN: str = 'all_variants_{bid}.csv'
DETAIL_PNG_PATTERN: str = 'detail_{bid}.png'
COMPARATIVE_PNG: str = 'comparative_report.png'
COMPARATIVE_PARETO_PNG: str = 'comparative_pareto.png'
REPORT_JSON: str = 'report.json'

# --- Ключи JSON ---
JSON_BEST_KEY: str = 'best_compositions'
JSON_PARETO_KEY: str = 'pareto_fronts'
JSON_ALL_KEY: str = 'all_variants'

# --- Главный запуск (значения можно менять при вызове) ---
MAIN_MAX_JUNIOR: int = 6
MAIN_MAX_MIDDLE: int = 6
MAIN_MAX_SENIOR: int = 6
MAIN_TARGET_WAIT: float = 40.0