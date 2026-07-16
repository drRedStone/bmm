"""
optimize_staff.py
=================
Подбор оптимального состава штата (junior, middle, senior) для
отделения банка с использованием существующих модулей проекта.

Все числовые параметры вынесены в именованные константы в начале модуля.
Реализована многопроцессная проверка комбинаций с индикатором прогресса.
"""
import math
import os
import traceback
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm   # <-- потребуется установить: pip install tqdm

from config import (
    WERGES, SKILLS, WEEKDAY_HOURS,
    OPTIMIZE_STAFF_REPORT_SAVE_DIRECTORY
)
from schedule_ilp import schedule_week_ilp
from compare_policies import get_weekly_metrics
from required_windows import required_windows_unconstrained

# ═══════════════════════════════════════════
#  Константы (бывшие магические числа)
# ═══════════════════════════════════════════
# --- Перебор составов ---
DEFAULT_MAX_JUNIOR   = 10
DEFAULT_MAX_MIDDLE   = 10
DEFAULT_MAX_SENIOR   = 10
DEFAULT_TARGET_WAIT  = 40.0          # минут
STAFF_MULTIPLIER     = 3             # общее число сотрудников ≤ STAFF_MULTIPLIER * n_win

# --- Параметры сотрудников (значения по умолчанию для create_staff) ---
DEFAULT_MAX_HOURS_WEEK = 40
DEFAULT_MAX_HOURS_DAY  = 9
DEFAULT_LUNCH_MIN      = 60

# --- Визуализация: детальный график ---
DETAIL_FIGSIZE         = (16, 6)
DETAIL_WIDTH_RATIOS    = [3, 2]
SCATTER_ALL_COLOR      = 'lightgray'
SCATTER_ALL_SIZE       = 30
SCATTER_BEST_COLOR     = 'orange'
SCATTER_BEST_SIZE      = 120
SCATTER_BEST_EDGECOLOR = 'black'
SCATTER_BEST_ZORDER    = 5
# Ширины столбцов таблицы детального графика (6 колонок)
DETAIL_TABLE_COLWIDTHS = [0.1, 0.1, 0.1, 0.18, 0.15, 0.12]

# --- Визуализация: сводный отчёт ---
COMPARATIVE_FIGSIZE    = (16, 12)
BAR_WIDTH              = 0.35
BAR_COLOR_RED          = 'red'
LEGEND_LOC             = 'lower center'
LEGEND_BBOX            = (0.5, -0.25)
LEGEND_NCOL            = 2
# Ширины столбцов таблицы лучших составов (5 колонок)
TABLE_COLWIDTHS        = [0.12, 0.25, 0.18, 0.22, 0.18]
TABLE_FONTSIZE         = 9
TABLE_TITLE_FONTSIZE   = 12
PARETO_COLOR           = 'steelblue'
PARETO_SIZE            = 50
ANNOTATE_XYTEXT        = (5, 5)
ANNOTATE_FONTSIZE      = 7
SAVE_DPI               = 150

# --- Имена выходных файлов ---
BEST_CSV             = 'best_compositions.csv'
PARETO_CSV_PATTERN   = 'pareto_{bid}.csv'
ALL_VARIANTS_CSV_PATTERN = 'all_variants_{bid}.csv'
DETAIL_PNG_PATTERN   = 'detail_{bid}.png'
COMPARATIVE_PNG      = 'comparative_report.png'
COMPARATIVE_PARETO_PNG = 'comparative_pareto.png'   # доп. файл для Парето-фронтов
REPORT_JSON          = 'report.json'

# --- Ключи JSON ---
JSON_BEST_KEY   = 'best_compositions'
JSON_PARETO_KEY = 'pareto_fronts'
JSON_ALL_KEY    = 'all_variants'

# --- Главный запуск (значения можно менять при вызове) ---
MAIN_MAX_JUNIOR = 6
MAIN_MAX_MIDDLE = 6
MAIN_MAX_SENIOR = 6
MAIN_TARGET_WAIT = 40.0


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def create_staff(junior: int, middle: int, senior: int, branch_id: str) -> pd.DataFrame:
    """Создаёт синтетический штат сотрудников для одного отделения."""
    employees = []
    emp_id = 0
    for grade, count in [('junior', junior), ('middle', middle), ('senior', senior)]:
        for _ in range(count):
            employees.append({
                'employee_id': f'synth_{emp_id}',
                'branch_id': branch_id,
                'grade': grade,
                'skills': SKILLS[grade],
                'hourly_cost_rub': WERGES[grade],
                'max_hours_week': DEFAULT_MAX_HOURS_WEEK,
                'max_hours_day': DEFAULT_MAX_HOURS_DAY,
                'lunch_min': DEFAULT_LUNCH_MIN,
                'name': None
            })
            emp_id += 1
    if not employees:
        return pd.DataFrame(columns=['employee_id', 'branch_id', 'name', 'grade',
                                     'skills', 'hourly_cost_rub', 'max_hours_week',
                                     'max_hours_day', 'lunch_min'])
    return pd.DataFrame(employees)


def compute_staff_lower_bounds(
    client_arrivals: pd.DataFrame,
    operations: pd.DataFrame,
    branch_id: str
) -> Tuple[int, int, int]:
    """Возвращает минимально необходимое количество сотрудников каждого грейда."""
    max_total = 0
    max_credit = 0
    max_mortgage = 0
    for day in WEEKDAY_HOURS:
        req = required_windows_unconstrained(client_arrivals, operations, branch_id, day)
        if req.empty:
            continue
        max_total = max(max_total, req['R_total'].max())
        max_credit = max(max_credit, req['R_credit'].max())
        max_mortgage = max(max_mortgage, req['R_mortgage'].max())

    min_senior = max_mortgage
    min_middle_plus_senior = max_credit
    min_middle = max(0, min_middle_plus_senior - min_senior)
    min_total = max_total
    min_junior = max(0, min_total - min_middle - min_senior)

    return int(min_junior), int(min_middle), int(min_senior)


def _eval_one_combination(
    args: Tuple[int, int, int, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, int]
) -> Optional[Dict]:
    """
    Вычисляет метрики для одной комбинации (junior, middle, senior).
    Возвращает словарь с результатами или None, если комбинация невалидна.
    Эта функция вызывается параллельно в пуле процессов.
    """
    j, m, s, client_arrivals, operations, branches, branch_id, n_win = args
    try:
        staff = create_staff(j, m, s, branch_id)
        week_res, summary = schedule_week_ilp(
            client_arrivals, operations, staff, branches, branch_id
        )
        metrics = get_weekly_metrics(week_res, summary, client_arrivals,
                                     operations, branch_id, branches)
        return {
            'junior': j, 'middle': m, 'senior': s,
            'weekly_cost': metrics['total_cost'],
            'avg_wait_min': metrics['avg_wait_weighted'],
            'total_shortfall': metrics['total_shortfall']
        }
    except Exception:
        # В реальном коде можно логировать, но для чистоты просто пропускаем
        return None


# ----------------------------------------------------------------------
# Основная функция подбора штата
# ----------------------------------------------------------------------
def optimize_staff_for_branch(
    client_arrivals: pd.DataFrame,
    operations: pd.DataFrame,
    branches: pd.DataFrame,
    branch_id: str,
    max_junior: int = DEFAULT_MAX_JUNIOR,
    max_middle: int = DEFAULT_MAX_MIDDLE,
    max_senior: int = DEFAULT_MAX_SENIOR,
    target_wait: Optional[float] = DEFAULT_TARGET_WAIT,
    budget: Optional[float] = None,
    verbose: bool = True
) -> Dict:
    """Подбирает оптимальный состав штата для одного отделения."""
    min_j, min_m, min_s = compute_staff_lower_bounds(client_arrivals, operations, branch_id)
    if verbose:
        print(f"Минимальные требования: junior>={min_j}, middle>={min_m}, senior>={min_s}")

    n_win = int(branches.loc[branch_id, 'n_windows'])
    max_total = STAFF_MULTIPLIER * n_win

    # Автоматически сужаем верхние границы, чтобы не перебирать заведомо недопустимые значения
    max_j = min(max_junior, max_total - min_m - min_s)
    max_m = min(max_middle, max_total - min_j - min_s)
    max_s = min(max_senior, max_total - min_j - min_m)
    max_j = max(max_j, min_j)
    max_m = max(max_m, min_m)
    max_s = max(max_s, min_s)

    if verbose:
        print(f"Пределы перебора: junior {min_j}..{max_j}, middle {min_m}..{max_m}, senior {min_s}..{max_s}")

    # Собираем комбинации
    combos = []
    for j in range(min_j, max_j + 1):
        for m in range(min_m, max_m + 1):
            for s in range(min_s, max_s + 1):
                if j + m + s <= max_total:
                    combos.append((j, m, s, client_arrivals, operations, branches, branch_id, n_win))

    if not combos:
        if verbose:
            print("Нет комбинаций, удовлетворяющих базовым ограничениям.")
        return {'best': None, 'pareto': pd.DataFrame(), 'all': pd.DataFrame()}

    # Определяем число процессов (не более 8 и не более числа комбинаций)
    num_workers = min(multiprocessing.cpu_count() * 2, len(combos), 16)
    if num_workers <= 0:
        num_workers = 1

    results = []
    if verbose:
        print(f"Обработка {len(combos)} комбинаций (процессов: {num_workers})...")

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_eval_one_combination, combo): combo for combo in combos}
            for future in tqdm(as_completed(futures), total=len(combos), desc="Перебор"):
                res = future.result()
                if res is not None:
                    results.append(res)
    except RuntimeError:
        # Fallback для сред, где ProcessPoolExecutor не работает (например, интерактивный режим)
        if verbose:
            print("Многопроцессность недоступна, переходим на последовательный режим.")
        for combo in tqdm(combos, desc="Последовательный перебор"):
            res = _eval_one_combination(combo)
            if res is not None:
                results.append(res)

    # Применяем фильтры по бюджету и целевому ожиданию
    filtered = []
    for r in results:
        if budget is not None and r['weekly_cost'] > budget:
            continue
        if target_wait is not None and r['avg_wait_min'] > target_wait:
            continue
        filtered.append(r)

    if not filtered:
        if verbose:
            print("Не найдено ни одного допустимого варианта.")
        return {'best': None, 'pareto': pd.DataFrame(), 'all': pd.DataFrame()}

    df_all = pd.DataFrame(filtered).sort_values('total_shortfall')

    # Парето-фронт
    pareto_mask = np.ones(len(df_all), dtype=bool)
    for i, (s1, c1) in enumerate(zip(df_all['total_shortfall'], df_all['weekly_cost'])):
        for j, (s2, c2) in enumerate(zip(df_all['total_shortfall'], df_all['weekly_cost'])):
            if s2 <= s1 and c2 <= c1 and (s2 < s1 or c2 < c1):
                pareto_mask[i] = False
                break
    df_pareto = df_all[pareto_mask].sort_values('total_shortfall')
    best_row = df_pareto.iloc[0]
    best = (int(best_row['junior']), int(best_row['middle']), int(best_row['senior']),
            best_row['weekly_cost'], best_row['avg_wait_min'], best_row['total_shortfall'])

    return {'best': best, 'pareto': df_pareto, 'all': df_all}


# ----------------------------------------------------------------------
# Анализ и визуализация
# ----------------------------------------------------------------------
def analyze_optimization_results(all_results: Dict[str, Dict]) -> Dict:
    """Собирает лучшие составы и Парето-фронты по всем отделениям."""
    best_rows = []
    pareto_dict = {}
    full_dict = {}
    for bid, res in all_results.items():
        if res['best'] is None:
            continue
        j, m, s, cost, wait, sf = res['best']
        best_rows.append({
            'branch': bid,
            'junior': j, 'middle': m, 'senior': s,
            'weekly_cost': cost,
            'avg_wait_min': wait,
            'total_shortfall': sf
        })
        pareto_dict[bid] = res['pareto']
        full_dict[bid] = res['all']
    return {
        'best': pd.DataFrame(best_rows),
        'pareto': pareto_dict,
        'full': full_dict
    }


def plot_branch_detail(branch_id: str, branch_results: Dict, save_path: Optional[str] = None) -> None:
    """
    Для одного отделения строит подробный график:
    - Слева: диаграмма рассеяния всех вариантов (лучший выделен оранжевым),
    - Справа: таблица всех вариантов.
    """
    df_all = branch_results['all']
    best = branch_results['best']
    if df_all.empty:
        print(f"Нет данных для детального графика {branch_id}")
        return

    fig, (ax_scatter, ax_table) = plt.subplots(1, 2, figsize=DETAIL_FIGSIZE,
                                               gridspec_kw={'width_ratios': DETAIL_WIDTH_RATIOS})

    # Диаграмма рассеяния
    ax_scatter.scatter(df_all['weekly_cost']/1000, df_all['avg_wait_min'],
                       c=SCATTER_ALL_COLOR, s=SCATTER_ALL_SIZE, label='Все варианты')
    if best is not None:
        j, m, s, cost, wait, _ = best
        ax_scatter.scatter(cost/1000, wait, c=SCATTER_BEST_COLOR, s=SCATTER_BEST_SIZE,
                           edgecolors=SCATTER_BEST_EDGECOLOR,
                           label=f'Лучший ({j},{m},{s})', zorder=SCATTER_BEST_ZORDER)
    ax_scatter.set_xlabel('Недельный ФОТ, тыс. руб.')
    ax_scatter.set_ylabel('Среднее время ожидания, мин')
    ax_scatter.set_title(f'{branch_id} – все проверенные составы')
    ax_scatter.legend(loc='upper right')
    ax_scatter.grid(True, linestyle='--', alpha=0.7)

    # Таблица
    ax_table.axis('off')
    columns = ['Junior', 'Middle', 'Senior', 'ФОТ/нед', 'Ожидание', 'Недобор']
    cell_text = []
    for _, row in df_all.iterrows():
        cell_text.append([
            str(int(row['junior'])),
            str(int(row['middle'])),
            str(int(row['senior'])),
            f"{row['weekly_cost']:,.0f}",
            f"{row['avg_wait_min']:.1f}",
            f"{row['total_shortfall']:.0f}"
        ])
    table = ax_table.table(cellText=cell_text, colLabels=columns,
                           cellLoc='center', loc='center',
                           colWidths=DETAIL_TABLE_COLWIDTHS)
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONTSIZE)
    ax_table.set_title(f'{branch_id} – таблица всех вариантов', pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=SAVE_DPI)
        print(f"Детальный график {branch_id} сохранён: {save_path}")
    else:
        plt.show()


def plot_comparative_report(analysis: Dict, save_path: Optional[str] = None) -> None:
    """
    Сводный отчёт по всем отделениям:
    - Столбцы стоимости и ожидания (легенда под графиком),
    - Таблица лучших составов,
    - Парето-фронты для каждого отделения (адаптивная сетка).
    При большом числе отделений Парето-фронты выводятся на отдельной фигуре.
    """
    best_df = analysis['best']
    if best_df.empty:
        print("Нет данных для графиков.")
        return

    branches = best_df['branch'].tolist()
    pareto_dict = analysis['pareto']

    # ---------- Основная фигура: столбцы + таблица ----------
    fig1 = plt.figure(figsize=(14, 6))
    ax1 = fig1.add_subplot(1, 2, 1)
    x = np.arange(len(branches))
    ax1.bar(x - BAR_WIDTH/2, best_df['weekly_cost']/1000, BAR_WIDTH,
            label='Недельный ФОТ (тыс.руб)')
    ax1_twin = ax1.twinx()
    ax1_twin.bar(x + BAR_WIDTH/2, best_df['avg_wait_min'], BAR_WIDTH,
                 color=BAR_COLOR_RED, label='Среднее время ожидания (мин)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(branches)
    ax1.set_ylabel('ФОТ, тыс. руб')
    ax1_twin.set_ylabel('Среднее ожидание, мин')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc=LEGEND_LOC,
               bbox_to_anchor=LEGEND_BBOX, ncol=LEGEND_NCOL)
    ax1.set_title('Лучшие составы по отделениям')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Таблица
    ax_table = fig1.add_subplot(1, 2, 2)
    ax_table.axis('off')
    table_data = []
    for _, row in best_df.iterrows():
        table_data.append([
            row['branch'],
            f"{int(row['junior'])} / {int(row['middle'])} / {int(row['senior'])}",
            f"{row['weekly_cost']:,.0f}",
            f"{row['avg_wait_min']:.1f}",
            f"{row['total_shortfall']:.0f}"
        ])
    columns = ['Отделение', 'Junior / Middle / Senior',
               'ФОТ, руб/нед', 'Среднее ожидание, мин', 'Недобор, окно-ч']
    table = ax_table.table(cellText=table_data, colLabels=columns,
                           cellLoc='center', loc='center',
                           colWidths=TABLE_COLWIDTHS)
    table.auto_set_font_size(False)
    table.set_fontsize(TABLE_FONTSIZE)
    ax_table.set_title('Лучшие найденные составы', fontsize=TABLE_TITLE_FONTSIZE, pad=20)

    plt.tight_layout()
    if save_path:
        base = str(save_path).replace('.png', '')
        fig1.savefig(f"{base}.png", dpi=SAVE_DPI)
        print(f"Сводный отчёт сохранён: {base}.png")
    else:
        fig1.show()

    # ---------- Парето-фронты (отдельная фигура) ----------
    if len(branches) > 0:
        n = len(branches)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        fig2, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                                  squeeze=False)
        for i, bid in enumerate(branches):
            ax = axes[i//ncols][i%ncols]
            df = pareto_dict[bid]
            ax.scatter(df['weekly_cost']/1000, df['avg_wait_min'],
                       c=PARETO_COLOR, s=PARETO_SIZE, label='Парето-оптимальные составы')
            for _, row in df.iterrows():
                ax.annotate(f"({int(row['junior'])}, {int(row['middle'])}, {int(row['senior'])})",
                            (row['weekly_cost']/1000, row['avg_wait_min']),
                            textcoords="offset points", xytext=ANNOTATE_XYTEXT, fontsize=ANNOTATE_FONTSIZE)
            ax.set_xlabel('Недельный ФОТ, тыс. руб.')
            ax.set_ylabel('Среднее время ожидания, мин')
            ax.set_title(f'{bid} – лучшие варианты')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        for j in range(n, nrows*ncols):
            fig2.delaxes(axes[j//ncols][j%ncols])
        plt.tight_layout()
        if save_path:
            base = str(save_path).replace('.png', '')
            fig2.savefig(f"{base}_pareto.png", dpi=SAVE_DPI)
            print(f"Парето-фронты сохранены: {base}_pareto.png")
        else:
            fig2.show()


def save_report_files(analysis: Dict, output_dir: str) -> None:
    """Сохраняет все отчётные файлы, включая детальные плоты для каждого отделения."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_df = analysis['best']
    best_df.to_csv(f"{output_dir}/{BEST_CSV}", index=False)

    for bid, df in analysis['pareto'].items():
        df.to_csv(f"{output_dir}/{PARETO_CSV_PATTERN.format(bid=bid)}", index=False)
    for bid, df in analysis['full'].items():
        df.to_csv(f"{output_dir}/{ALL_VARIANTS_CSV_PATTERN.format(bid=bid)}", index=False)

    full_dict = analysis['full']
    pareto_dict = analysis['pareto']
    best_dict = {bid: (analysis['best'][analysis['best']['branch'] == bid].iloc[0]
                       if not analysis['best'].empty and bid in analysis['best']['branch'].values
                       else None)
                 for bid in full_dict}

    for bid in full_dict:
        branch_res = {
            'all': full_dict[bid],
            'pareto': pareto_dict[bid],
            'best': None
        }
        if best_dict[bid] is not None:
            row = best_dict[bid]
            branch_res['best'] = (int(row['junior']), int(row['middle']), int(row['senior']),
                                  row['weekly_cost'], row['avg_wait_min'], row['total_shortfall'])
        plot_branch_detail(bid, branch_res,
                           save_path=f"{output_dir}/{DETAIL_PNG_PATTERN.format(bid=bid)}")

    plot_comparative_report(analysis, save_path=f"{output_dir}/{COMPARATIVE_PNG}")

    json_str = generate_json_report(analysis)
    with open(f"{output_dir}/{REPORT_JSON}", 'w', encoding='utf-8') as f:
        f.write(json_str)

    print(f"Все файлы отчёта сохранены в {output_dir}")


def generate_json_report(analysis: Dict) -> str:
    best_df = analysis['best']
    report = {
        JSON_BEST_KEY: best_df.to_dict(orient='records'),
        JSON_PARETO_KEY: {
            bid: df.to_dict(orient='records')
            for bid, df in analysis['pareto'].items()
        },
        JSON_ALL_KEY: {
            bid: df.to_dict(orient='records')
            for bid, df in analysis['full'].items()
        }
    }
    return json.dumps(report, indent=2, ensure_ascii=False, default=str)


# ----------------------------------------------------------------------
# Запуск
# ----------------------------------------------------------------------
if __name__ == '__main__':
    ca = pd.read_csv('dataset/client_arrivals.csv')
    ops = pd.read_csv('dataset/operations.csv')
    br = pd.read_csv('dataset/branches.csv').set_index('branch_id')

    all_results = {}
    for bid in ['BR01', 'BR02', 'BR03']:
        print(f"\n===== Оптимизация штата для {bid} =====")
        if bid == 'BR03':
            res = optimize_staff_for_branch(
                ca, ops, br, bid,
                max_junior=MAIN_MAX_JUNIOR,
                max_middle=MAIN_MAX_MIDDLE,
                max_senior=MAIN_MAX_SENIOR,
                target_wait=15.0,
                verbose=True
            )
        else:
            res = optimize_staff_for_branch(
                ca, ops, br, bid,
                max_junior=MAIN_MAX_JUNIOR,
                max_middle=MAIN_MAX_MIDDLE,
                max_senior=MAIN_MAX_SENIOR,
                target_wait=MAIN_TARGET_WAIT,
                verbose=True
            )
        all_results[bid] = res
        if res['best'] is not None:
            j, m, s, cost, wait, sf = res['best']
            print(f"Лучший состав: junior={j}, middle={m}, senior={s}")
            print(f"  Недельный ФОТ: {cost:,.0f} руб.")
            print(f"  Среднее время ожидания: {wait:.1f} мин")
            print(f"  Недобор окон: {sf:.1f} окно-часов")
            print("\nВсе допустимые варианты:")
            display_df = res['all'].rename(columns={
                'junior': 'junior', 'middle': 'middle', 'senior': 'senior',
                'weekly_cost': 'ФОТ/нед', 'avg_wait_min': 'Ожидание, мин',
                'total_shortfall': 'Недобор, окно-ч'
            })
            print(display_df.to_string(index=False))
        else:
            print("Не найдено допустимых вариантов. "
                  "Возможные причины: слишком жёсткий порог ожидания или недостаточный диапазон перебора.")

    analysis = analyze_optimization_results(all_results)
    if not analysis['best'].empty:
        save_report_files(analysis, str(OPTIMIZE_STAFF_REPORT_SAVE_DIRECTORY))
        plot_comparative_report(analysis)
    else:
        print("Нет данных для построения отчётов.")
    input()