HOUR_ORDER = ["Night", "Morning", "Afternoon", "Evening"]
SEASON_ORDER = ["winter", "spring", "summer", "fall"]
DOW_ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

STRATA_SPECS = [
    ("hour_group", "Hour", HOUR_ORDER),
    ("season", "Season", SEASON_ORDER),
    ("day_of_week_name", "Day of Week", DOW_ORDER),
]

DEFAULT_THRESHOLDS = (8, 10, 30, 60)

DEFAULT_CENSOR_TIME = 60.0