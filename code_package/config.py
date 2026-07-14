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

OPEN_HOUR = 9 # чекать, где используется и менять на обращение к словарю