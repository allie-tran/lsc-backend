# Filter relevant fields from the query
from llm.prompts import SCHEMA


RELEVANT_FIELDS_PROMPT = (
    SCHEMA
    + """

And these are valid gap units:
- time_gap: hour, day, week, month, year
- gps_gap: km, meter

These are valid sort fields:
- score: relevance score
- time
- date
- weekday
- year
- duration

Given this query: "{query}", what are the relevant fields that I should consider? Try to choose the most important ones. If two fields are similar, choose the one that is more specific.
For merge_by criteria, consider how far apart two events should be split into two different occasions. Some examples:
For aggregated answer, consider what math operation should be used to aggregate the data in different events (sum, average, etc.) to answer the question. Applicable fields are: duration, count, time, date, year

Query: "How many times did I go to the gym last month?"
Merge by: day. Max gap: 3 hours.
Sort by: score<desc>

Query: "How many days did I spend in Paris last year?"
Merge by: day. Max gap: none
Sort by: date<desc>
Reason: If the lifelogger spent multiple days in Paris, each day should be considered a separate occasion. There is no need to consider the max gap as it's embedded in the merge_by criteria.
Aggregated answer: count<sum>

Query: "In which city did I spend the most time in 2019?"
Merge by: city, month. Max gap: 1 week
Reason: Events that happen in the same city within a month are considered the same occasion. If two events are in the same city but 1 week apart, they are considered different occasions.
Sort by: duration<desc>
Aggregated answer: duration<argmax>

Query: "What is my car's brand and model?"
Merge by: none. Max gap: none
Sort by: score<desc>
Aggregated answer: None

Query: "How long did I go running last night? Assume I only went running once."
Merge by: none. Max gap: 1 hour
Sort by: score<desc>
Aggregated answer: None

Query: "How many hours did I spend at work last week?"
Merge by: day. Max gap: 1 hour
Sort by: score<desc>
Aggregated answer: duration<sum>

Query: "How long is my average commute time to work?"
Merge by: none. Max gap: none
Sort by: score<desc>
Aggregated answer: duration<average>

Reminder of the query: {query}

Answer in this format.
```json
{{
    "relevant_fields": ["place", "date"],
    "max_gap": {{
                    "time_gap": {{"unit": "hour", "value": 5}},
                    "gps_gap": {{"unit": "km", "value": 1}}
                }},
    "merge_by": ["day", "country"], // or any other relevant fields
    "sort_by": [{{ "field": "score", "order": "desc" }}]
    "aggregated_answer": {{"operation": "count", "field": "duration"}}
}}
```
"""
)

GRAPH_QUERY = (
    SCHEMA
    + """
Generate a vegalite schema (with encoding, without data) to answer this lifelog question:
{question}

Here is the data:
{data}

Answer in this format:
```json
{{
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "title": "Vega-Lite Schema",
    "description": "Vega-Lite Schema to answer the lifelog question",
    "mark": "bar",
    "encoding": {{
        "x": {{"field": "date", "type": "temporal"}},
        "y": {{"field": "count", "type": "quantitative"}}
    }},
    "data": {{ // example data
        "values": [
            "date": "2022-01-01",
            "count": 10
            ]
    }}
    ... // other fields
}}
```
"""
)
