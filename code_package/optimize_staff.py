"""
Подбор оптимального состава штата (junior, middle, senior) для
отделения банка с использованием существующих модулей проекта.

Все числовые параметры вынесены в именованные константы в начале модуля.
Реализована многопроцессная проверка комбинаций с индикатором прогресса.
Функции снабжены полными аннотациями типов и документирующими строками.
"""
import math
import os
import traceback
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, Optional, List, Tuple, Any
import matplotlib.pyplot as plt
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

from config import (
    WERGES, SKILLS, WEEKDAY_HOURS,
    OPTIMIZE_STAFF_REPORT_SAVE_DIRECTORY
)
from schedule_ilp import schedule_week_ilp
from compare_policies import get_weekly_metrics
from required_windows import required_windows_unconstrained

from config import (
    DEFAULT_MAX_JUNIOR,
    DEFAULT_MAX_MIDDLE,
    DEFAULT_MAX_SENIOR,
    DEFAULT_TARGET_WAIT,
    STAFF_MULTIPLIER,
    DEFAULT_MAX_HOURS_WEEK,
    DEFAULT_MAX_HOURS_DAY,
    DEFAULT_LUNCH_MIN,
    DETAIL_FIGSIZE,
    DETAIL_WIDTH_RATIOS,
    SCATTER_ALL_COLOR,
    SCATTER_ALL_SIZE,
    SCATTER_BEST_COLOR,
    SCATTER_BEST_SIZE,
    SCATTER_BEST_EDGECOLOR,
    SCATTER_BEST_ZORDER,
    DETAIL_TABLE_COLWIDTHS,
    COMPARATIVE_FIGSIZE,
    BAR_WIDTH,
    BAR_COLOR_RED,
    LEGEND_LOC,
    LEGEND_BBOX,
    LEGEND_NCOL,
    TABLE_COLWIDTHS,
    TABLE_FONTSIZE,
    TABLE_TITLE_FONTSIZE,
    PARETO_COLOR,
    PARETO_SIZE,
    ANNOTATE_XYTEXT,
    ANNOTATE_FONTSIZE,
    SAVE_DPI,
    BEST_CSV,
    PARETO_CSV_PATTERN,
    ALL_VARIANTS_CSV_PATTERN,
    DETAIL_PNG_PATTERN,
    COMPARATIVE_PNG,
    COMPARATIVE_PARETO_PNG,
    REPORT_JSON,
    JSON_BEST_KEY,
    JSON_PARETO_KEY,
    JSON_ALL_KEY,
    MAIN_MAX_JUNIOR,
    MAIN_MAX_MIDDLE,
    MAIN_MAX_SENIOR,
    MAIN_TARGET_WAIT,
)


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def create_staff(junior: int, middle: int, senior: int, branch_id: str) -> pd.DataFrame:
    """
    Создаёт синтетический штат сотрудников для одного отделения.

    Args:
        junior: Количество сотрудников грейда junior.
        middle: Количество сотрудников грейда middle.
        senior: Количество сотрудников грейда senior.
        branch_id: Идентификатор отделения (например, 'BR01').

    Returns:
        DataFrame с колонками, соответствующими структуре employees.csv.
        Если все три параметра равны нулю, возвращается пустой DataFrame
        с корректными колонками.
    """
    employees: List[Dict[str, Any]] = []
    emp_id: int = 0
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
    """
    Возвращает минимально необходимое количество сотрудников каждого грейда,
    основанное на пиковых требованиях к окнам (без учёта физического числа окон).

    Просматривает все рабочие дни недели, определяет максимальное количество
    требуемых окон (общих, кредитных, ипотечных) и вычисляет минимальное
    число junior, middle и senior, способное покрыть эти пики.

    Args:
        client_arrivals: DataFrame с данными о прибытии клиентов.
        operations: DataFrame с информацией об операциях.
        branch_id: Идентификатор отделения.

    Returns:
        Кортеж из трёх целых чисел: (min_junior, min_middle, min_senior).
    """
    max_total: int = 0
    max_credit: int = 0
    max_mortgage: int = 0
    for day in WEEKDAY_HOURS:
        req = required_windows_unconstrained(client_arrivals, operations, branch_id, day)
        if req.empty:
            continue
        max_total = max(max_total, req['R_total'].max())
        max_credit = max(max_credit, req['R_credit'].max())
        max_mortgage = max(max_mortgage, req['R_mortgage'].max())

    min_senior: int = max_mortgage
    min_middle_plus_senior: int = max_credit
    min_middle: int = max(0, min_middle_plus_senior - min_senior)
    min_total: int = max_total
    min_junior: int = max(0, min_total - min_middle - min_senior)

    return int(min_junior), int(min_middle), int(min_senior)


def _eval_one_combination(
    args: Tuple[int, int, int, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, int]
) -> Optional[Dict[str, Any]]:
    """
    Вычисляет метрики для одной комбинации (junior, middle, senior).
    Функция предназначена для запуска в параллельных процессах.

    Args:
        args: Кортеж, содержащий:
            j (int): число junior,
            m (int): число middle,
            s (int): число senior,
            client_arrivals (pd.DataFrame),
            operations (pd.DataFrame),
            branches (pd.DataFrame),
            branch_id (str),
            n_win (int) – не используется, оставлено для совместимости.

    Returns:
        Словарь с ключами 'junior', 'middle', 'senior', 'weekly_cost',
        'avg_wait_min', 'total_shortfall', либо None, если произошла ошибка
        (например, ILP не нашёл решения).
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
        # Ошибки внутри процесса не должны рушить общий цикл
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
) -> Dict[str, Any]:
    """
    Подбирает оптимальный состав штата для одного отделения.

    Перебирает все комбинации (junior, middle, senior) в заданных диапазонах,
    для каждой запускает недельное ILP-планирование, собирает метрики и
    строит Парето-фронт по недобору окон и недельному ФОТ.

    Args:
        client_arrivals: Данные о прибытии клиентов.
        operations: Данные об операциях.
        branches: Информация об отделениях (индексирован по branch_id).
        branch_id: Идентификатор отделения.
        max_junior: Верхняя граница перебора для junior.
        max_middle: Верхняя граница перебора для middle.
        max_senior: Верхняя граница перебора для senior.
        target_wait: Максимально допустимое среднее время ожидания (мин).
            Варианты с большим ожиданием отфильтровываются.
        budget: Максимальный недельный ФОТ; варианты дороже отбрасываются.
        verbose: Печатать ли промежуточные сообщения.

    Returns:
        Словарь с ключами:
        - 'best': кортеж (junior, middle, senior, cost, wait, shortfall)
           лучшего (первого по Парето) варианта.
        - 'pareto': DataFrame Парето-оптимальных вариантов.
        - 'all': DataFrame всех допустимых вариантов, прошедших фильтры.
        Если не найдено ни одного допустимого варианта, 'best' = None,
        а DataFrame'ы пусты.
    """
    min_j, min_m, min_s = compute_staff_lower_bounds(client_arrivals, operations, branch_id)
    if verbose:
        print(f"Минимальные требования: junior>={min_j}, middle>={min_m}, senior>={min_s}")

    n_win: int = int(branches.loc[branch_id, 'n_windows'])
    max_total: int = STAFF_MULTIPLIER * n_win

    # Автоматически сужаем верхние границы, чтобы не перебирать заведомо недопустимые значения
    max_j: int = min(max_junior, max_total - min_m - min_s)
    max_m: int = min(max_middle, max_total - min_j - min_s)
    max_s: int = min(max_senior, max_total - min_j - min_m)
    max_j = max(max_j, min_j)
    max_m = max(max_m, min_m)
    max_s = max(max_s, min_s)

    if verbose:
        print(f"Пределы перебора: junior {min_j}..{max_j}, middle {min_m}..{max_m}, senior {min_s}..{max_s}")

    # Собираем комбинации
    combos: List[Tuple] = []
    for j in range(min_j, max_j + 1):
        for m in range(min_m, max_m + 1):
            for s in range(min_s, max_s + 1):
                if j + m + s <= max_total:
                    combos.append((j, m, s, client_arrivals, operations, branches, branch_id, n_win))

    if not combos:
        if verbose:
            print("Нет комбинаций, удовлетворяющих базовым ограничениям.")
        return {'best': None, 'pareto': pd.DataFrame(), 'all': pd.DataFrame()}

    # Определяем число процессов (не более 16 и не более числа комбинаций)
    num_workers: int = min(multiprocessing.cpu_count(), len(combos), 16)
    if num_workers <= 0:
        num_workers = 1

    results: List[Dict[str, Any]] = []
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
        if verbose:
            print("Многопроцессность недоступна, переходим на последовательный режим.")
        for combo in tqdm(combos, desc="Последовательный перебор"):
            res = _eval_one_combination(combo)
            if res is not None:
                results.append(res)

    # Применяем фильтры по бюджету и целевому ожиданию
    filtered: List[Dict[str, Any]] = []
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
def analyze_optimization_results(all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Собирает лучшие составы и Парето-фронты по всем отделениям.

    Args:
        all_results: Словарь, где ключ — branch_id, значение — результат
            optimize_staff_for_branch (словарь с best, pareto, all).

    Returns:
        Словарь с ключами:
        - 'best': DataFrame с лучшими составами всех отделений.
        - 'pareto': dict {branch_id: DataFrame} Парето-фронтов.
        - 'full': dict {branch_id: DataFrame} всех вариантов.
    """
    best_rows: List[Dict[str, Any]] = []
    pareto_dict: Dict[str, pd.DataFrame] = {}
    full_dict: Dict[str, pd.DataFrame] = {}
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


def plot_branch_detail(branch_id: str, branch_results: Dict[str, Any],
                       save_path: Optional[str] = None) -> None:
    """
    Для одного отделения строит подробный график:
    - Слева: диаграмма рассеяния всех вариантов (лучший выделен оранжевым),
    - Справа: таблица всех вариантов.

    Args:
        branch_id: Идентификатор отделения.
        branch_results: Словарь с ключами 'all' (DataFrame), 'best' (кортеж),
            'pareto' (DataFrame) — результат optimize_staff_for_branch.
        save_path: Если указан, график сохраняется в файл; иначе показывается на экране.
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


def plot_comparative_report(analysis: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Сводный отчёт по всем отделениям:
    - Столбцы стоимости и ожидания (легенда под графиком),
    - Таблица лучших составов,
    - Парето-фронты для каждого отделения (адаптивная сетка).

    При большом числе отделений Парето-фронты выводятся на отдельной фигуре.

    Args:
        analysis: Результат работы analyze_optimization_results.
        save_path: Базовое имя файла для сохранения (без расширения).
            Основной отчёт сохраняется как save_path.png,
            Парето-фронты — как save_path_pareto.png.
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


def save_report_files(analysis: Dict[str, Any], output_dir: str) -> None:
    """
    Сохраняет все отчётные файлы: CSV с лучшими составами, Парето-фронтами,
    всеми вариантами, а также графики и JSON.

    Args:
        analysis: Результат analyze_optimization_results.
        output_dir: Путь к папке для сохранения файлов.
    """
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


def generate_json_report(analysis: Dict[str, Any]) -> str:
    """
    Генерирует JSON-строку с полными результатами.

    Args:
        analysis: Результат analyze_optimization_results.

    Returns:
        Строка с отформатированным JSON.
    """
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

    all_results: Dict[str, Dict[str, Any]] = {}
    for bid in ['BR01', 'BR02', 'BR03']:
        print(f"\n===== Оптимизация штата для {bid} =====")
        if bid == 'BR03':
            # Для отделения с ограниченными ресурсами ослабляем целевое ожидание
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
            pareto_df = res['pareto']
            print(f"\nПарето-оптимальные варианты ({len(pareto_df)} шт.):")
            print(pareto_df.rename(columns={
                'junior': 'junior', 'middle': 'middle', 'senior': 'senior',
                'weekly_cost': 'ФОТ/нед', 'avg_wait_min': 'Ожидание, мин',
                'total_shortfall': 'Недобор, окно-ч'
            }).to_string(index=False))
            print("(Полный список всех вариантов сохранён в CSV-файлах)")
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