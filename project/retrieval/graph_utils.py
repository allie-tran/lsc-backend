import logging
from collections import Counter
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

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


def get_heatmap_data(
    data: Data, scores: List[float], threshold: Optional[float] = None
):
    """
    Get the heatmap data for a list of scores
    """
    if threshold is None:
        np_threshold = np.percentile(scores, 90)
    else:
        np_threshold = np.float32(threshold)
    good_indices = np.where(scores > np_threshold)[0]

    if data == Data.Deakin:
        raise NotImplementedError("Deakin data not implemented")

    # Create a day-to-count mapping using Counter for efficiency
    days = ["/".join(photo_id.split("/")[0:2]) for photo_id in photo_ids(data)]
    day_to_count = Counter([days[i] for i in good_indices])

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
        values=values,
        hover_info=hover_info,
        x_ticks=x_ticks,
        x_labels=unique_months,
        y_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )
