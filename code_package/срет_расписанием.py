import pandas as pd
from schedule_ilp import schedule_week_ilp

# Загрузка данных
ca = pd.read_csv('dataset/client_arrivals.csv')
ops = pd.read_csv('dataset/operations.csv')
br = pd.read_csv('dataset/branches.csv').set_index('branch_id')
emp = pd.read_csv('dataset/employees.csv')
emp['skills'] = emp['skills'].str.split(',')

all_rows = []

for branch_id in ['BR01', 'BR02', 'BR03']:
    print(f"Генерация расписания для {branch_id}...")
    results, _ = schedule_week_ilp(ca, ops, emp, br, branch_id)

    for weekday, data in results.items():
        for a in data['assignments']:
            all_rows.append({
                'branch_id': branch_id,
                'weekday': weekday,
                'employee_id': a['employee_id'],
                'start': a['start'],
                'end': a['end'],
                'lunch_hour': a.get('lunch_hour'),
                'serving_hours': ', '.join(map(str, a['serving_hours'])),
                'paid_hours': a['paid_hours'],
                'shift_cost': a['shift_cost']
            })

df_all = pd.DataFrame(all_rows)
df_all.to_excel('schedules_all_branches.xlsx', index=False)

print(f"Готово! Всего назначений: {len(df_all)}")
print("Файл: schedules_all_branches.xlsx")