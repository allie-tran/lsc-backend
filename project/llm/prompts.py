"""
Prompts for GPT-4
"""

INSTRUCTIONS = "You are a helpful assistant. Always include one single valid JSON object. Do not return multiple JSON objects."

# Rewrite the question into a search query
REWRITE_QUESTION = """
Rewrite the following question into a statement as a retrieval query: {question}
It should be a statement in natural language to search for the relevant information in the lifelog retrieval system. It should include all the information from the question, excluding the question phrase

Avoid using "search for", "find", or "retrieve" in the query. Keep it descriptive.

Some examples:
Question: "How many times did I go to the gym last month?"
Rewrite: "I went to the gym last month."

Question: "What did I eat for breakfast on Monday?"
Rewrite: "I ate breakfast on Monday."

Question: "When was the last time I went to the park, assuming today is 2022-03-01?"
Rewrite: "I went to the park, before 2022-03-01."

Question: "Where did I buy the MacBook Pro in summer 2021?"
Rewrite: "I bought the MacBook Pro in summer 2021."

Answer in this format:

```json
{{
    "text": "A statement that is a search query",
}}
```
"""

# Automatically parse the query into a different format
PARSE_QUERY = """
I need to parse information from a text query to find relevant information from the lifelog retrieval system. As my parser is rule-based, I need you to provide me with the relevant fields that I should consider. Be as specific as possible, because the parser is not very smart.

The fields are: location, time, date, visual, after-query (same fields with after-when), before-query (same fields with before-when).

If some location details are not specific (names of places, cities, countries), you should include them in visual field as they can be intepreted visually. Similarly, time information such as sunrise, sunset, or meal times can be included in the visual field. Only include the most specific information in the fields.

Some examples:

Query: "I was biking in the park near my house in the early morning."
Response:
```json
{{
    "visual": "biking in the park", // notice that the location is not specific
    "time": "from 5am to 8am",
    "location": "in the park near my house"
}}
```

Query: "I was having an Irish beer on St. Patrick's Day last year. Assuming this year is 2022."
Response:
```json
{{
    "visual": "having an Irish beer",
    "date": "17th March 2021"
}}
```

Query: "Waiting for a flight to Paris at the airport."
Response:
```json
{{
    "visual": "waiting for a flight at the airport",
    "location": "airport",
    "after-query": {{
        "after-when": "3 hours",
        "location": "Paris"
    }}
}}
```

Query: "It was a very cold winter day in New York City. I walked to the bus at sunrise."
Response:
```json
{{
    "visual": "very cold winter day, walking at sunrise",
    "location": "New York City",
    "date": "January, February, December"
    "time": "from 9am to 10am" // because it is at winter so sunrise is late
    "after-query": {{
        "after-when": "1 hour",
        "visual": "walking to the bus",
    }}
}}
```

Now it's your turn. Use no comments.
Provide the relevant fields for the following query:
Query:{query}
Response:
"""

# Parse before and after information
PARSE_BEFORE_AFTER = """
I need to find the relevant information from the lifelog retrieval system based on the before and after information. The system will retrieve the information based on the before and after information. The system will consider the before information as the query and the after information as the answer.
How many hours before and after each event should I consider to find the relevant information? The time unit is in hours. If you don't need to consider before or after, just put 0.

For example, for the events "I am going to the airport to catch a flight to Paris". The before and after information could be:
```json
{{
    "main_query": "Going to the airport",
    "after": 3-4, # 3-4 hours after the event
    "after_query": "I am at Paris"
}}
```

Example 2: "I was working at a cafe to wait for my friend. After that, we went to the cinema."
```json
{{
    "before_query": "working at a cafe",
    "before": 0-1, # waited for 0-1 hour
    "main_query": "meeting my friend at the cafe",
    "after": 1-2, # went to the cinema 1-2 hours after
    "after_query": "went to the cinema"
}}
```

Example 3: "I had a long walk in the park in the morning because I stayed out late the night before."
```json
{{
    "before": 5-7, # it was the morning so, 7-8am, so last night was 10pm-2am, so 5-10 hours before
    "before_query": "stayed out late at night",
    "main_query": "long walk in the park",
}}
```

These are rational guesses. Now, provide the before and after information for the following query:
Query: {query}
"""

# Answer the question based on text
QA_PROMPT = """Answer the following question based on the text provided: {question}
There are {num_events} retrieved using the lifelog retrieval system. Here are the non-visual information of each event. Bear in mind that some location data might be wrong, but the order of relevance is mostly correct (using the system's best guess).
{events}
Give one or more of your best, educated guesses to the question in the following format, with each answer being the key. The explanation should be brief. You don't have to give a full sentence, just list the reasons.

Reminder of the question: {question}

Use the following schema in a valid JSON format:
Answers: List[Answer]
Answer:
- answer: str
- explanation: str
- evidence: List[EventLink]
EventLink:
- event_id: int # the event id, starting from 1

```json
{{
    "answers": [
        {{
            "answer": "something",
            "explanation": "brief explanation for why the answer is `something`",
            "evidence": [1, 2, 3]
        }},
        {{
            "answer": "one",
            "explanation": "brief explanation for why the answer is `one`",
            "evidence": [4, 5, 6]
        }},
        {{
            "answer": "5 days",
            "explanation": "brief explanation for why the answer is `5 days`",
            "evidence": [1, 7, 8]
        }}
    ]
}}
```
"""

MIXED_PROMPTS = [
    """Answer the following question based on the text and images provided: {question}
There are {num_events} retrieved using the lifelog retrieval system. Here are some information of each event. Bear in mind that some location data might be wrong, but the order of relevance is mostly correct (using the system's best guess).""",
    """
Reminder of the question: {question}.
Give one or more of your best guess to the question in the following format, with each answer being the key. The explantion should be brief. You don't have to give a full sentence, just list the reasons.

Use the following schema in a valid JSON format:
Answers: List[Answer]
Answer:
- answer: str
- explanation: str
- evidence: List[EventLink]
EventLink:
- event_id: int # the event id, starting from 1

```json
{{
    answers: [
        {{
            "answer": "something",
            "explanation": "brief explanation for why the answer is `something`",
            "evidence": [1, 2, 3]
        }},
        {{
            "answer": "one",
            "explanation": "brief explanation for why the answer is `one`",
            "evidence": [4, 5, 6]
        }},
        {{
            "answer": "5 days",
            "explanation": "brief explanation for why the answer is `5 days`",
            "evidence": [1, 7, 8]
        }}
    ]
}}
```
""",
]


VISUAL_PROMPT = """<ImageHere>Answer the following question: {question}
Some extra information (non visual): {extra_info}. But focus on the visual information more. Be brief and concise. If you can't answer the question, just say so."""


# Filter relevant fields from the query
RELEVANT_FIELDS_PROMPT = """My database has this schema:

```
# Valid location fields
place: str # home, work, a restaurant's name, etc. This is the most specific location
place_info: str # type of the semantic_location (restaurant, university, etc.)
region: str # cities, states
country: str

# Valid time fields
start_time: datetime
end_time: datetime
date: datetime
month: str
weekday: str
year: int
duration: str # duration of the event

# Visual information
ocr: List[str] # text extracted from images
```

And these are valid gap units:
- time_gap: hour, day, week, month, year
- gps_gap: km, meter

Given this query: "{query}", what are the relevant fields that I should consider? Try to choose the most important ones. If two fields are similar, choose the one that is more specific.
For merge_by criteria, consider how far apart two events should be split into two different occasions  For example, if two events are in the same city but 1 week apart, should they be grouped together or not? How about if they are 5 hours apart but in different cities?

The relevant fields should include all the fields mentioned in the other fields.

Answer in this format.
```json
{{
    "merge_by": ["hour", "place"],
    "max_gap": {{
                    "time_gap": {{"unit": "hour", "value": 5}},
                    "gps_gap": {{"unit": "km", "value": 1}}
                }},
    "sort_by": [{{"field": "time", "order": "desc"}}],
    "relevant_fields": ["place", "start_time", "end_time", "date", "ocr"],
}}
```
"""

PARSING_QUERY = """I need to find the answer for this question using my lifelog retrieval system. In my system, a flow of processes is needed:
1. Segmentation: this function takes two parameters: max_time, time_gap, and loc_change, where max_time is the maximum time for each segment, time_gap is the maximum time gap between two segments, and loc_change is the type of location change (semantic_location, city, country, continent). The function returns a list of segments, where each segment is a list of events.
2. Retrieval: this function takes a list of segments and a question. It returns a list of events that are relevant to the question. The function takes the top-K events that are relevant to the question.
3. Extraction: this function takes a list of events and a question. It returns the answer to the question. The function extracts the information from the events to answer the question.
4. Answering: this function takes the answer and the question. It returns the answer to the question.
5. Post-processing: re-organize the events (merge, split, or filter) and the answer to the question. Events with the same answers can be grouped together (if it makes sense).

For example, if the question is "What is my favourite airlines to fly with in 2019?", this is what I'm looking for:
- Segmentation: max_time=1 day, time_gap=1 day, loc_change=city
- Retrieval: query="airlines name on boarding pass or brochure', K=50
- Extraction: metadata=["start_city", "end_city"]
- Answering: needs Visual Question Answering=yes, needs OCR=yes, expected answer type=a name, possible answers=["Delta", "United", "American", "Southwest", "JetBlue"], sort by time=no
- Post-processing: sort=time, group=airlines

Now, the question is "{question}". I need you to define these paramters:
Please provide the following JSON structure:
```
{{
    "segmentation": {{
        "max_time": [a time unit in the following: "year", "month", "week", "day", "hour"],
        "time_gap": [in hours],
        "loc_change": [a location unit,  one of the following: "country", "city", "location_name", "continent"]
    }},
    "retrieval": {{
        "search query": [a search query to find the events],
        "K": [number of events to retrieve and extract answers from]
    }},
    "extraction": {{
        "metadata": [a list of metadata to extract from each event, one of the following: "start_time", "end_time", "semantic_location", "duration", "country", "city", "continent" that might be useful to answer the question],
        "needs Visual Question Answering": [true/false],
        "needs OCR": [true/false],
    }},
    "answering": {{
        "expected answer type": "explanation of what the answer should look like",
        "possible answers": [a list of possible answers]
        }},
    "post-processing": {{
        "group": [a way to group the events, one of the following: any time unit, any location unit, "answer"],
        "sort": [a way to sort the events, one of the following: any time unit, any location_unit, "most_common_answer"]
        "aggregate": [a way to aggregate the events, one of the following: "sum", "average, "max", "min"]
        }}
}}
No explanation is needed. Just provide the JSON structure.
```
"""
