import calendar
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import dateparser
import holidays
import pytimeparse
from nltk import MWETokenizer, pos_tag
from parsedatetime import Constants
from results.models import Visualisation

from .types import DateTuple, RegexInterval, Tags, TimeInfo
from .utils import find_regex, get_visual_text

YEARS = ["2015", "2016", "2018", "2019", "2020"]
MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
SHORT_MONTHS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sept",
    "oct",
    "nov",
    "dec",
]


MORE_TIMEOFDAY = {
    "early; morning": ["dawn", "sunrise", "daybreak"],
    "morning": ["breakfast"],
    "evening": ["nightfall", "dusk", "dinner", "dinnertime", "sunset", "twilight"],
    "midday": ["lunchtime", "lunch"],
    "night": ["nighttime"],
    "afternoon": ["supper", "suppertime", "teatime"],
}
SPECIAL_TIMEOFDAY = [x for y in MORE_TIMEOFDAY.values() for x in y]

TIMEOFDAY = {
    "early; morning": "5am-10am",
    "late; morning": "11am-12pm",
    "morning": "5am-12pm",
    "early; afternoon": "1pm-3pm",
    "late; afternoon": "4pm-5pm",
    "midafternoon": "2pm-3pm",
    "afternoon": "12pm-5pm",
    "early; evening": "5pm-7pm",
    "midevening": "7pm-9pm",
    "evening": "5pm-9pm",
    "night": "9pm-4am",
    "noon": "11am-1pm",
    "midday": "10am-2pm",
    "midnight": "11pm-1am",
    "bedtime": "8pm-1am",
}

SEASONS = {
    "spring": ["march", "april", "may"],
    "summer": ["june", "july", "august"],
    "fall": ["september", "october", "november"],
    "autumn": ["september", "october", "november"],
    "winter": ["december", "january", "february"],
}


holiday_names = set()
all_holidays = dict()
for holiday in holidays.Ireland(years=[int(y) for y in YEARS]).items():  # type: ignore
    all_holidays[f"{holiday[1]}, {holiday[0].year}".lower()] = holiday[0]
    holiday_names.add(holiday[1])
HOLIDAY_NAMES = list(holiday_names)

for t in MORE_TIMEOFDAY:
    for synonym in MORE_TIMEOFDAY[t]:
        TIMEOFDAY[synonym] = TIMEOFDAY[t]


# Parse period expression
def parse_period_expression(time_expression: str) -> Optional[int]:
    """
    Parse a period expression into seconds
    """
    dt = pytimeparse.parse(time_expression)
    if dt is None:
        dt = None
    else:
        dt = int(dt)
    return dt


class TimeTagger:
    def __init__(self):
        regex_lib = Constants()
        self.all_regexes = []
        self.all_regexes.append(
            ("DATE", r"(?:\bthe )?\b\d{1,2}(?:st|nd|rd|th)? of(?: year)? \d{4}\b")
        )  # The 3rd of 2019
        self.all_regexes.append(
            ("HOLIDAY", r"\b(" + "|".join(holiday_names) + r")\b(?: in)?( \d{4}\b)?")
        )  # Christmas 2019
        self.all_regexes.append(
            ("DATE", r"\b(" + "|".join(MONTHS) + r")\b(?: in)?( \d{4}\b)?")
        )  # August 2019
        self.all_regexes.append(
            (
                "DATE",
                r"(?:\bthe )?\d{1,2}(?:st|nd|rd|th)?(?: of)? ("
                + "|".join(MONTHS)
                + r")\b((?: in)?( \d{4}\b)?)?",
            )
        )  # 5th Aust 20gu19

        for key, r in regex_lib.cre_source.items():
            # if key in ["CRE_MODIFIER"]:
            #     self.all_regexes.append(("TIMEPREP", r))
            if key in ["CRE_TIMEHMS", "CRE_TIMEHMS2", "CRE_RTIMEHMS", "CRE_RTIMEHMS"]:
                # TIME (proper time oclock)
                self.all_regexes.append(("TIME", r))
            elif key in [
                "CRE_DATE",
                "CRE_DATE3",
                "CRE_DATE4",
                "CRE_MONTH",
                "CRE_DAY",
                "",
                "CRE_RDATE",
                "CRE_RDATE2",
            ]:
                self.all_regexes.append(("DATE", r))  # DATE (day in a month)
            elif key in [
                "CRE_TIMERNG1",
                "CRE_TIMERNG2",
                "CRE_TIMERNG3",
                "CRE_TIMERNG4",
                "CRE_DATERNG1",
                "CRE_DATERNG2",
                "CRE_DATERNG3",
            ]:
                self.all_regexes.append(("TIMERANGE", r))  # TIMERANGE
            elif key in ["CRE_UNITS", "CRE_QUNITS"]:
                self.all_regexes.append(("PERIOD", r))  # PERIOD
            elif key in ["CRE_UNITS_ONLY"]:
                self.all_regexes.append(("TIMEUNIT", r))  # TIMEUNIT
        for word in [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]:
            self.all_regexes.append(("WEEKDAY", word))  # WEEKDAY
        for word in SEASONS:
            self.all_regexes.append(("SEASON", word))  # SEASON
        # Added by myself
        timeofday_regex = set()
        for t in TIMEOFDAY:
            if ";" in t:
                t = t.split("; ")[-1]
            timeofday_regex.add(t)

        timeofday_regex = "|".join(timeofday_regex)
        self.all_regexes.append(("TIMEOFDAY", r"\b(" + timeofday_regex + r")\b"))
        self.all_regexes.append(
            (
                "DATEPREP",
                r"\b((last|first)[ ]day[ ]of|(a|the)[ ]day[ ](before|after))\b",
            )
        )
        self.all_regexes.append(
            (
                "TIMEPREP",
                r"\b(before|after|while|late|early|later[ ]than|earlier[ ]than|sooner[ ]than)\b",
            )
        )
        self.all_regexes.append(("DATE", r"\b(2015|2016|2018|2019|2020)\b"))
        self.tags = [t for t, r in self.all_regexes]

    def merge_interval(self, intervals: List[RegexInterval]) -> List[RegexInterval]:
        if intervals:
            intervals.sort(key=lambda interval: interval.start)
            merged = [intervals[0]]
            for current in intervals:
                previous = merged[-1]
                if current.start <= previous.end and current.tag == previous.tag:
                    if current.end > previous.end:
                        previous.end = current.end
                        previous.tag = current.tag
                else:
                    merged.append(current)
            return merged
        return []

    def find_time(self, sent: str) -> List[RegexInterval]:
        results = []
        for kind, r in self.all_regexes:
            for t in find_regex(r, sent):
                t.tag = kind
                results.append(t)
        return self.merge_interval(results)

    def tag(self, sent: str) -> List[Tags]:
        times = self.find_time(sent)
        intervals = dict([(time.start, time.end) for time in times])
        tag_dict = dict([(time.text, time.tag) for time in times])

        tokenizer = MWETokenizer(separator="__")
        for a in times:
            if a.text:
                tokenizer.add_mwe(a.text.split())

        # --- FIXED ---
        original_tokens = tokenizer.tokenize(sent.split())
        original_tags = pos_tag(original_tokens)
        # --- END FIXED ---

        tokens = tokenizer.tokenize(sent.split())
        tags = pos_tag(tokens)

        new_tags = []
        for word, tag in tags:
            if "__" in word or word in tag_dict:
                word = word.replace("__", " ")
                try:
                    new_tags.append((word, tag_dict[word]))
                except KeyError:
                    new_tags.append((word, "O"))
            else:
                tag = [t[1] for t in original_tags if t[0] == word][0]  # FIXED
                new_tags.append((word, tag))
        return new_tags


def get_day_month(
    date_string: str,
) -> DateTuple:
    date_tuple = DateTuple()
    if (date_string) in YEARS:
        date_tuple.year = int(date_string)
        return date_tuple
    today = dateparser.parse("today")
    if today is None:
        today = datetime.now()
    date = dateparser.parse(date_string, settings={"DATE_ORDER": "DMY"})
    if date is None:
        return date_tuple
    y, m, d = date.year, date.month, date.day

    date_string = date_string.lower()
    for ex_year in YEARS:
        if ex_year in date_string:
            y = int(ex_year)
            break
    if str(y) not in date_string:
        y = None
    if y:
        date_string = date_string.replace(str(y), "")
    if m == today.month:
        if (
            re.search(r"\b0?" + str(m) + r"\b", date_string)
            or re.search(r"\b" + str(MONTHS[m - 1]) + r"\b", date_string)
            or re.search(r"\b" + str(SHORT_MONTHS[m - 1]) + r"\b", date_string)
        ):
            pass
        else:
            m = None
    if str(d) not in date_string:
        d = None
    date_tuple.year, date_tuple.month, date_tuple.day = y, m, d
    return date_tuple


def am_pm_to_num(hour_string: str) -> Tuple[int, int]:
    minute = 0
    if ":" in hour_string:
        minute = re.compile(r"\d+(:\d+).*").findall(hour_string)[0]
        hour = hour_string.replace(minute, "")
        minute = int(minute[1:])
    if "am" in hour_string:
        hour = int(hour_string.replace("am", ""))
        if hour == 12:
            hour = 0
    elif "pm" in hour_string:
        hour = int(hour_string.replace("pm", "")) + 12
        if hour == 24:
            hour = 12
    else:
        hour = int(hour_string)
    return hour, minute


def adjust_start_end(
    mode: str, original: Tuple[int, int], hour: int, minute: int
) -> Tuple[int, int]:
    if mode == "start":
        if original[0] == hour:
            return hour, max(original[1], minute)
        elif hour > original[0]:
            return hour, minute
        else:
            return original
    if original[0] == hour:
        return hour, min(original[1], minute)
    elif hour < original[0]:
        return hour, minute
    else:
        return original


def holiday_text_to_datetime(
    text: str,
) -> DateTuple:
    regex = r"\b(" + "|".join(holiday_names) + r")\b(?: in)?( \d{4}\b)?"
    res = re.findall(regex, text, re.IGNORECASE)
    date_tuple = DateTuple()
    if res:
        holiday_name = res[0][0]
        year = res[-1][-1]

        false_year = False
        if not year:
            if holiday_name.lower() in [
                "christmas day",
                "new year's day",
                "st. patrick's day",
                "st. stephen's day",
            ]:
                year = 2020
                false_year = True
        if year:
            year = int(year)
            if false_year:
                year = None
            date_tuple.year = year
            if f"{holiday_name}, {year}" in all_holidays:
                datetime = all_holidays[f"{holiday_name}, {year}"]
                date_tuple.month = datetime.month
                date_tuple.day = datetime.day
    return date_tuple


def search_for_time(
    time_tagger: TimeTagger, text: str, disabled_times: List[str] = []
) -> Tuple[str, TimeInfo, Visualisation]:
    """Search for time info in a text"""
    tags = time_tagger.tag(text)
    processed_words = set()

    weekdays = []
    dates = []
    timestamps = [] # for before, after a certain date
    times = []
    start = (0, 0)
    end = (24, 0)
    duration = None

    matches = {"WEEKDAY": [], "TIMEOFDAY": [], "DURATION": []}

    for i, (word, tag) in enumerate(tags):
        if word in disabled_times:
            continue
        if tag in [
            "WEEKDAY",
            "TIMERANGE",
            "DATEPREP",
            "DATE",
            "TIME",
            "TIMEOFDAY",
            "PERIOD",
        ]:
            # These can be deleted directly
            if word not in SPECIAL_TIMEOFDAY:
                processed_words.add(i)
        # ============================================== #
        # ================== WEEKDAYS ================== #
        # ============================================== #
        if tag == "WEEKDAY":
            weekdays.append(word)
            matches["WEEKDAY"].append(word)

        # ============================================== #
        # ================= TIMERANGE ================== #
        # ============================================== #
        elif tag == "TIMERANGE":
            s, e = word.split("-")
            start = adjust_start_end("start", start, *am_pm_to_num(s))
            end = adjust_start_end("end", end, *am_pm_to_num(e))

        # ============================================== #
        # =================== TIME ===================== #
        # ============================================== #
        elif tag == "TIME":
            if word in YEARS:
                # Sometimes years are tagged as times
                dates.append(get_day_month(word))
            else:
                # Check if these are time prepositions (before, after, etc.)
                timeprep = ""
                if i >= 1 and tags[i - 1][1] == "TIMEPREP":
                    timeprep = tags[i - 1][0]
                    processed_words.add(i - 1)
                if timeprep in ["before", "earlier than", "sooner than"]:
                    end = adjust_start_end("end", end, *am_pm_to_num(word))
                elif timeprep in ["after", "later than"]:
                    start = adjust_start_end("start", start, *am_pm_to_num(word))
                else:
                    h, m = am_pm_to_num(word)
                    start = adjust_start_end("start", start, h - 1, m)
                    end = adjust_start_end("end", end, h + 1, m)

        # ============================================== #
        # =================== DATE ===================== #
        # ============================================== #

        elif tag in ["DATE", "HOLIDAY"]:
            if tag == "DATE":
                date_tuple = get_day_month(word)
            elif tag == "HOLIDAY":
                date_tuple = holiday_text_to_datetime(word)
            else:
                date_tuple = DateTuple()

            dateprep = ""
            if i >= 1 and tags[i - 1][1] in ["TIMEPREP", "DATEPREP"]:
                dateprep = tags[i - 1][0]

            print("Found date", word, tags[i - 1], dateprep)

            # Timestamps
            if dateprep in ["before", "after"]:
                if date_tuple.year and date_tuple.month and date_tuple.day:
                    dt_object = datetime(
                        date_tuple.year, date_tuple.month, date_tuple.day
                    )
                    time_stamp = dt_object.timestamp()
                    timestamps.append([time_stamp, dateprep])
                    continue

            if "first day of" in dateprep:
                date_tuple.day = 1
            elif "last day of" in dateprep:
                if date_tuple.year and date_tuple.month:
                    monthrange = calendar.monthrange(date_tuple.year, date_tuple.month)
                    date_tuple.day = monthrange[1]
                elif date_tuple.month and date_tuple.month != 2:
                    monthrange = calendar.monthrange(2020, date_tuple.month)
                    date_tuple.day = monthrange[1]
                else:
                    date_tuple.day = 29
                    dates.append(date_tuple)
            elif "day after" in dateprep:
                original_year = date_tuple.year
                if not date_tuple.year:
                    date_tuple.year = 2024  # using the current year as a default
                if date_tuple.month and date_tuple.day:
                    dt_object = datetime(
                        date_tuple.year, date_tuple.month, date_tuple.day
                    )
                    dt_object += timedelta(days=1)
                    if date_tuple.year != dt_object.year and original_year:
                        original_year += 1

                    date_tuple.year, date_tuple.month, date_tuple.day = (
                        dt_object.year,
                        dt_object.month,
                        dt_object.day,
                    )
                date_tuple.year = original_year
            elif "day before" in dateprep:
                original_year = date_tuple.year
                if not date_tuple.year:
                    date_tuple.year = 2024
                if date_tuple.month and date_tuple.day:
                    dt_object = datetime(
                        date_tuple.year, date_tuple.month, date_tuple.day
                    )
                    dt_object -= timedelta(days=1)
                    if date_tuple.year != dt_object.year and original_year:
                        original_year -= 1
                    date_tuple.year, date_tuple.month, date_tuple.day = (
                        dt_object.year,
                        dt_object.month,
                        dt_object.day,
                    )
                date_tuple.year = original_year
            else:
                dates.append(date_tuple)

        # ============================================== #
        # ================= SEASONS ==================== #
        # ============================================== #
        elif tag == "SEASON":
            # Heuristic
            for month in SEASONS[word]:
                dates.append((None, MONTHS.index(month) + 1, None))

        # ============================================== #
        # ================= TIMEOFDAY ================= #
        # ============================================== #
        elif tag == "TIMEOFDAY":
            # Not deleting meals or sun events because these can be visual cues
            if word not in ["lunch", "breakfast", "dinner", "sunrise", "sunset"]:
                processed_words.add(i)
            else:
                matches["TIMEOFDAY"].append(word)
            timeprep = ""
            if i > 1 and tags[i - 1][1] == "TIMEPREP":
                timeprep = tags[i - 1][0]
                processed_words.add(i - 1)
            if "early" in timeprep:
                if "early; " + word in TIMEOFDAY:
                    word = "early; " + word
            elif "late" in timeprep:
                if "late; " + word in TIMEOFDAY:
                    word = "late; " + word
            if word in TIMEOFDAY:
                s, e = TIMEOFDAY[word].split("-")
                start = adjust_start_end("start", start, *am_pm_to_num(s))
                end = adjust_start_end("end", end, *am_pm_to_num(e))
            else:
                print(word, f"is not a registered time of day ({TIMEOFDAY})")
        elif tag == "PERIOD":
            duration = parse_period_expression(word)
            matches["DURATION"].append(f"{word}")

    start = start[0] * 3600 + start[1] * 60
    end = end[0] * 3600 + end[1] * 60

    # Remove processed words
    unprocessed_words = [
        (word, tag) for i, (word, tag) in enumerate(tags) if i not in processed_words
    ]

    # Clean_query
    clean_query = get_visual_text(text, unprocessed_words)

    # Post-processing
    info = TimeInfo(
        time=(start, end),
        duration=duration,
        weekdays=weekdays,
        dates=dates,
        timestamps=timestamps,
        original_texts=matches,
    )

    # Visualisation
    query_visualisation = Visualisation()
    for key, value in matches.items():
        for v in value:
            query_visualisation.time_hints.extend(f"{key}: {v}")

    return clean_query, info, query_visualisation


def add_time(timeinfo: TimeInfo, extra_timeinfo: TimeInfo):
    """
    Add time information to a TimeInfo object
    """

    if timeinfo is None:
        return extra_timeinfo
    if extra_timeinfo is None:
        return timeinfo

    if not timeinfo.weekdays:
        timeinfo.weekdays = extra_timeinfo.weekdays
    if not timeinfo.dates:
        timeinfo.dates = extra_timeinfo.dates

    return timeinfo


def calculate_duration(
    start_time: Optional[datetime], end_time: Optional[datetime]
) -> str:
    """
    Calculate the duration between two datetimes
    and convert to a human-readable format
    """
    if start_time is None or end_time is None:
        return ""
    time_delta = end_time - start_time
    time = ""
    if time_delta.seconds > 0:
        hours = time_delta.seconds // 3600
        if time_delta.days > 0:
            if hours > 0:
                duration = f"{time_delta.days} days and {hours} hours"
            else:
                duration = f"{time_delta.days} days"
        else:
            if hours > 0:
                minutes = (time_delta.seconds - hours * 3600) // 60
                duration = f"{hours} hours and {minutes} minutes"
            elif time_delta.seconds < 60:
                duration = f"{time_delta.seconds} seconds"
            else:
                minutes = time_delta.seconds // 60
                duration = f"{minutes} minutes"
        duration = f", lasted for about {duration}"
        time = f"from {start_time} to {end_time}"
    else:
        duration = ""
        time = f"at {start_time}"
    return time + duration
