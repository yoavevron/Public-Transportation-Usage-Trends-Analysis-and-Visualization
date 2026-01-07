import re

# A dictionary that maps each day number to its hebrew name for example "1:ראשון"
day_names_map = {
    1: "ראשון",
    2: "שני",
    3: "שלישי",
    4: "רביעי",
    5: "חמישי",
    6: "שישי",
    7: "שבת",
}

month_map = {
    1: 'ינואר', 2: 'פבואר', 3: 'מרץ', 4: 'אפריל', 5: 'מאי', 6: 'יוני',
    7: 'יולי', 8: 'אוגוסט', 9: 'ספטמבר', 10: 'אוקטובר', 11: 'נובמבר', 12: 'דצמבר'
}


# Formatting
def format_millions(x):
    if x >= 1_000_000:
        return f'{x / 1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x / 1_000:.0f}K'
    return "" if x == 0 else str(int(x))


def format_comma(x):
    return f"{int(x):,}"

CUSTOM_BLUE_SCALE = ['#BDD7EE', '#6BAED6', '#3182BD', '#08519C']

YEARS_IN_DATA = 5
ESTIMATED_NON_SAT_DAYS = 1566


# Time Parsing
def parse_time_range(desc):
    matches = re.findall(r'(\d{2}):(\d{2})', str(desc))
    if len(matches) >= 2:
        start_h, start_m = int(matches[0][0]), int(matches[0][1])
        end_h, end_m = int(matches[1][0]), int(matches[1][1])
        start_decimal = start_h + (start_m / 60.0)
        end_decimal = end_h + (end_m / 60.0)
        if end_decimal < start_decimal: end_decimal += 24
        duration = end_decimal - start_decimal
        if duration <= 0: duration = 1
        return start_decimal, duration
    return 0, 1


def get_time_range_only(desc):
    match = re.search(r'(\d{2}:\d{2}\s*-\s*\d{2}:\d{2})', str(desc))
    if match: return match.group(1).strip()
    return str(desc)

time_order = [
    "06:00 - 08:59 - שיא בוקר",
    "09:00 - 11:59 - שפל יום 1",
    "12:00 - 14:59 - שפל יום 2",
    "15:00 - 18:59 - שיא ערב",
    "19:00 - 23:59 - שפל ערב",
]

time_labels = {
    "06:00 - 08:59 - שיא בוקר": "06:00 - 08:59 ",
    "09:00 - 11:59 - שפל יום 1": "09:00 - 11:59 ",
    "12:00 - 14:59 - שפל יום 2": "12:00 - 14:59 ",
    "15:00 - 18:59 - שיא ערב": "15:00 - 18:59",
    "19:00 - 23:59 - שפל ערב": "19:00 - 23:59",
}