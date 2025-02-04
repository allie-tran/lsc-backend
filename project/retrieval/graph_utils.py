import logging
from collections import Counter
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from database.utils import get_unique_values
from query_parse.types.requests import Data
from query_parse.visual import photo_ids
from results.models import HeatmapResults

logger = logging.getLogger(__name__)


# Convert each date to its corresponding week format
def to_week(date):
    """Convert date to formatted week string."""
    week_str = datetime.strftime(date, "W%W - %Y")
    return "W52 - 2019" if week_str == "W00 - 2020" else week_str


def to_month(week):
    """Convert week to month-year string."""
    week_start = datetime.strptime(f"Mon {week}", "%a W%W - %Y")
    return week_start.strftime("%b %Y")


def get_heatmap_data(data: Data, scores: List[float], high_score_indices: List[int]):
    """
    Get the heatmap data for a list of scores
    """
    if data == Data.Deakin:
        return get_deakin_heatmap_data(data, scores, high_score_indices)
    return [get_lsc_heatmap_data(data, scores, high_score_indices)]


def get_lsc_heatmap_data(
    data: Data, scores: List[float], high_score_indices: List[int]
):
    # Create a day-to-count mapping using Counter for efficiency
    days = ["/".join(photo_id.split("/")[0:2]) for photo_id in photo_ids(data)]
    # initiall all 0s
    day_to_count = Counter()
    for day in set(days):
        day_to_count[day] = 0
    for day in high_score_indices:
        day_to_count[days[day]] += 1

    # Convert to datetime for easier handling
    datetimes = [datetime.strptime(day, "%Y%m/%d") for day in day_to_count.keys()]

    weeks = [to_week(date) for date in datetimes]
    days_of_week = [date.weekday() for date in datetimes]  # Monday = 0, Sunday = 6

    # Sort weeks based on the starting date
    unique_weeks = sorted(
        set(weeks), key=lambda w: datetime.strptime(f"Mon {w}", "%a W%W - %Y")
    )

    # Get the months corresponding to the weeks
    months = [to_month(week) for week in unique_weeks]
    unique_months = sorted(set(months), key=lambda x: datetime.strptime(x, "%b %Y"))

    # ignore December 2018
    if "Dec 2018" in unique_months:
        unique_months.remove("Dec 2018")

    # Create tick positions for months
    x_ticks = [months.index(month) + 0.5 for month in unique_months]

    # Create a DataFrame for visualization
    df = pd.DataFrame(
        {
            "week_index": [unique_weeks.index(week) for week in weeks],
            "week": weeks,
            "day_of_week": days_of_week,
            "count": list(day_to_count.values()),
        }
    )

    # Pivot the DataFrame
    df = df.pivot_table(
        index="day_of_week", columns="week_index", values="count", fill_value=0
    )

    values = df.to_dict("split")["data"]
    # replace 0s with None
    for i in range(len(values)):
        for j in range(len(values[i])):
            if values[i][j] == 0:
                values[i][j] = None

    # Hover info: the actual date
    hover_info = []
    for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        hover_text = []
        for week in unique_weeks:
            date = datetime.strptime(f"{day} {week}", "%a W%W - %Y")
            hover_text.append(date.strftime("%d %b %Y"))
        hover_info.append(hover_text)

    return HeatmapResults(
        name="Lifelogger",
        values=values,
        hover_info=hover_info,
        x_ticks=x_ticks,
        x_labels=unique_months,
        y_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )


def get_deakin_heatmap_data(
    data: Data, scores: List[float], high_score_indices: List[int]
):
    """
    Deakin data has multiple users (patientID)
    We map a heatmap of:
    - x-axis: patientID
    - y-axis: 2 weeks (Monday to Sunday - 14 days)
    """
    patient_ids = set(get_unique_values(data, "patient.id"))
    # patient_ids = sorted(patient_ids)

    # For each user, create a heatmap
    heatmaps = []
    full_photo_ids = photo_ids(data)
    full_days = ["/".join(photo_id.split("/")[1:4]) for photo_id in full_photo_ids]

    patient_ids = patient_ids.union(
        set([photo_id.split("/")[0] for photo_id in photo_ids(data)])
    )
    patient_ids = sorted(list(patient_ids))

    patient_counts = Counter()
    all_patient_data = []
    hover_info = []
    patient_list = []

    for patient_id in patient_ids:
        # Filter the data for the patient
        patient_photo_ids = [
            photo_id for photo_id in full_photo_ids if photo_id.startswith(patient_id)
        ]
        patient_days = [
            "/".join(photo_id.split("/")[1:4]) for photo_id in patient_photo_ids
        ]

        if len(patient_days) > 0:
            day_to_count = Counter()
            for day in set(patient_days):
                day_to_count[day] = 0

            high_count = 0
            for day in high_score_indices:
                if full_days[day] in patient_days:
                    high_count += 1
                    day_to_count[full_days[day]] += 1

            if high_count > 0:
                # Convert to datetime for easier handling
                datetimes = [
                    datetime.strptime(day, "%Y/%m/%d") for day in day_to_count.keys()
                ]
                first_date = min(datetimes)
                # get the monday before (or on) the first date
                first_monday = first_date - pd.DateOffset(days=first_date.weekday())
                recording_days = [
                    first_monday + pd.DateOffset(days=i) for i in range(14)
                ]

                patient_data = []
                hover = []
                for day in recording_days:
                    day_str = day.strftime("%Y/%m/%d")
                    count = day_to_count[day_str]
                    if count:
                        patient_data.append(count)
                    else:
                        patient_data.append(None)

                    new_day = day.strftime("%d %b %Y")
                    hover.append(f"{patient_id} - {new_day} ({count})")

                patient_counts[patient_id] = high_count
                all_patient_data.append(patient_data)
                patient_list.append(patient_id)
                hover_info.append(hover)
                continue

        # Empty data
        patient_counts[patient_id] = 0
        all_patient_data.append([None] * 14)
        patient_list.append(patient_id)
        hover_info.append(
            ["" for _ in range(14)]
        )

    # Tranpose the data
    all_patient_data = np.array(all_patient_data).T.tolist()
    hover_info = np.array(hover_info).T.tolist()

    heatmaps.append(
        HeatmapResults(
            name="All Patients",
            values=all_patient_data,
            hover_info=hover_info,
            x_ticks=list(range(len(patient_list))),
            x_labels=patient_list,
            y_labels=[
                "Mon",
                "Tue",
                "Wed",
                "Thu",
                "Fri",
                "Sat",
                "Sun",
                "Mon",
                "Tue",
                "Wed",
                "Thu",
                "Fri",
                "Sat",
                "Sun",
            ],
        )
    )

    # Sort the heatmaps based on the number of high scores
    return heatmaps

def get_hour_key(photo_id: str):
    day = "/".join(photo_id.split("/")[1:4])
    hour = photo_id.split("_")[2][:2]
    return f"{day} {hour}"

def get_deakin_heatmap_per_hours(data: Data, scores: List[float], high_score_indices: List[int], patient_ids: list[str]):
    """
    Visualise a patient's heatmap per hour (for 7 days of recording)
    x-axis is the hours of the day (0-23)
    y-axis is the 7 days of recording
    """
    heatmaps = []
    for patient_id in patient_ids:
        full_keys = [get_hour_key(photo_id) for photo_id in photo_ids(data)]
        patient_photo_ids = [
            photo_id for photo_id in photo_ids(data) if photo_id.startswith(patient_id)
        ]
        patient_keys = [get_hour_key(photo_id) for photo_id in patient_photo_ids]

        if len(patient_keys) > 0:
            key_to_count = Counter()
            for key in set(patient_keys):
                key_to_count[key] = 0

            high_count = 0
            for idx in high_score_indices:
                if full_keys[idx] in patient_keys:
                    high_count += 1
                    key_to_count[full_keys[idx]] += 1

            if high_count > 0:
                patient_data = []
                hover = []
                recording_days = sorted(set([key.split(" ")[0] for key in key_to_count.keys()]))
                for day in recording_days:
                    for hour in range(24):
                        key = f"{day} {hour:02d}"
                        count = key_to_count[key]
                        if count:
                            patient_data.append(count)
                        else:
                            patient_data.append(None)
                        hover.append(f"{patient_id} - {key} ({count})")

                heatmaps.append(
                    HeatmapResults(
                        name=patient_id,
                        values=[patient_data],
                        hover_info=[hover],
                        x_ticks=list(range(24)),
                        x_labels=[str(i) for i in range(24)],
                        y_labels=recording_days,
                    )
                )
    return heatmaps
